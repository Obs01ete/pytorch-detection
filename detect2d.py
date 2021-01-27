# Author: Dmitrii Khizbullin
# A script to train a neural network for 2d detection task.
# Some code is borrowed from pytorch examples and torchvision.

import os
import sys
import time
import pickle
import argparse
import importlib
import numpy as np
from termcolor import colored

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from name_list_dataset import NameListDataset
from summary_writer_opt import SummaryWriterOpt
from helpers import *
from average_meter import AverageMeter
import detection_models
from extended_collate import extended_collate
import image_anno_transforms
import average_precision
from debug_tools import dump_images


def default_input_traits():
    """
    Default resolutions for training and evaluation.
    """
    return {
        "resolution": (256, 512)
        #"resolution": (512, 1024)
        #"resolution": (384, 1152)
        #"resolution": (256, 768)
        #"resolution": (384, 768)
    }


def train_image_transform():
    """PIL image transformation for training."""
    return image_anno_transforms.ComposeVariadic([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])


def train_image_and_annotation_transform():
    """Image+annotation synchronous transformation for training."""
    return image_anno_transforms.ComposeVariadic([
        image_anno_transforms.RandomHorizontalFlipWithAnno(),
        image_anno_transforms.RandomCropWithAnno(0.3),
    ])


class BuildTargetFunctor:
    """Functor to delegate model's target construction to preprocessing threads."""

    def __init__(self, model):
        self.model = model

    def __call__(self, *args):
        return self.model.build_target(*args)


def clip_gradient(model, clip_val, mode):
    """Clip the gradient."""

    assert mode in ('by_max', 'by_norm')

    if mode is 'by_max':
        for p in model.parameters():
            if p.grad is not None:
                mv = torch.max(torch.abs(p.grad.data))
                if mv > clip_val:
                    print(colored("Grad max {:.3f}".format(mv), "red"))
                p.grad.data.clamp_(-clip_val, clip_val)
    elif mode is 'by_norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

    pass


class Config:
    def __init__(self, attr_dict):
        for n, v in attr_dict.items():
            self.__dict__[n] = v


def import_config_by_name(config_name):
    config_module = importlib.import_module('configs.' + config_name)
    cfg_dict = {key: value for key, value in config_module.__dict__.items() if
                not (key.startswith('__') or key.startswith('_'))}
    cfg = Config(cfg_dict)
    return cfg


class Trainer:
    """Class that performs train-validation loop to train a detection neural network."""

    def __init__(self, config_name):
        """
        Args:
            config_name: name of a configuration module to import
        """

        print('Config name: {}'.format(config_name))

        self.cfg = import_config_by_name(config_name)
        print(self.cfg)

        print('Start preparing dataset')
        self.prepare_dataset()
        print('Finished preparing dataset')

        print("torch.__version__=", torch.__version__)
        torchvision.set_image_backend('accimage')
        print("torchvision.get_image_backend()=", torchvision.get_image_backend())

        self.epochs_to_train = 500
        self.base_learning_rate = 0.02
        self.lr_scales = (
            (0, 0.1), # perform soft warm-up to reduce chance of divergence
            (2, 0.2),
            (4, 0.3),
            (6, 0.5),
            (8, 0.7),
            (10, 1.0), # main learning rate multiplier
            (int(0.90 * self.epochs_to_train), 0.1),
            (int(0.95 * self.epochs_to_train), 0.01),
        )

        self.train_batch_size = 32
        self.val_batch_size = 32

        num_workers_train = 12
        num_workers_val = 12

        input_traits = default_input_traits()

        labelmap = NameListDataset.getLabelmap()

        model = detection_models.SingleShotDetector(
            self.cfg.backbone_specs,
            self.cfg.multibox_specs,
            input_traits['resolution'],
            labelmap)

        if True:
            model_dp = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        else:
            model_dp = model

        if torch.cuda.is_available():
            model_dp.cuda()

        self.model = model
        self.model_dp = model_dp

        build_target = BuildTargetFunctor(model)
        map_to_network_input = image_anno_transforms.MapImageAndAnnoToInputWindow(input_traits['resolution'])

        def load_list(name):
            path = os.path.join(self.cfg.train_val_split_dir, name + '.pkl')
            with open(path, 'rb') as input:
                return pickle.load(input)

        self.train_dataset = NameListDataset(
            dataset_list=load_list('train_list'),
            image_transform=train_image_transform(),
            image_and_anno_transform=train_image_and_annotation_transform(),
            map_to_network_input=map_to_network_input,
            build_target=build_target
        )

        self.balanced_val_dataset = NameListDataset(
            dataset_list=load_list('val_list'),
            image_transform=None,
            image_and_anno_transform=None,
            map_to_network_input=map_to_network_input,
            build_target=build_target
        )

        # Data loading and augmentation pipeline for training
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
            num_workers=num_workers_train, collate_fn=extended_collate, pin_memory=True)

        # Data loading and augmentation pipeline for validation
        self.val_loader = torch.utils.data.DataLoader(
            self.balanced_val_dataset, batch_size=self.val_batch_size, shuffle=False,
            num_workers=num_workers_val, collate_fn=extended_collate, pin_memory=True)

        self.optimizer = None
        self.learning_rate = None

        self.train_iter = 0
        self.epoch = 0
        self.best_performance_metric = None

        self.print_freq = 10

        self.writer = None

        self.run_dir = os.path.join('runs', self.cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.snapshot_path = os.path.join(self.run_dir, self.cfg.run_name + '.pth.tar')

        pass

    def prepare_dataset(self):
        """Prepare dataset for training the detector. Done only once."""

        if os.path.exists(self.cfg.train_val_split_dir):
            return

        image_list = NameListDataset.list_all_images()
        assert len(image_list) > 0

        os.makedirs(self.cfg.train_val_split_dir)

        NameListDataset.train_val_split(image_list, self.cfg.train_val_split_dir)

    @staticmethod
    def wrap_sample_with_variable(input, target, **kwargs):
        """Wrap tensor with Variable and push to cuda."""
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = [t.cuda(non_blocking=True) for t in target]
        return input, target

    def train_epoch(self):
        """
        Train the model for one epoch.
        """

        print("-------------- Train epoch ------------------")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        forward_time = AverageMeter()
        loss_time = AverageMeter()
        backward_time = AverageMeter()
        loss_total_am = AverageMeter()
        loss_loc_am = AverageMeter()
        loss_cls_am = AverageMeter()

        # switch to training mode
        self.model_dp.train()

        is_lr_change = self.epoch in [epoch for epoch, _ in self.lr_scales]
        if self.optimizer is None or is_lr_change:
            scale = None
            if self.optimizer is None:
                scale = 1.0
            if is_lr_change:
                scale = [sc for epoch, sc in self.lr_scales if epoch == self.epoch][0]
            self.learning_rate = self.base_learning_rate * scale
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(
                    self.model_dp.parameters(), self.learning_rate,
                    momentum=0.9,
                    weight_decay=0.0001)
            else:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

        do_dump_train_images = False
        detection_train_dump_dir = None
        if do_dump_train_images:
            detection_train_dump_dir = os.path.join(self.run_dir, 'detection_train_dump')
            clean_dir(detection_train_dump_dir)

        end = time.time()
        for batch_idx, sample in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input, target, names, pil_images, annotations, stats = sample

            if do_dump_train_images: # and random.random() < 0.01:
                dump_images(
                    names, pil_images, annotations, None, stats,
                    self.model.labelmap,
                    detection_train_dump_dir)

            input_var, target_var = self.wrap_sample_with_variable(input, target)

            # compute output
            forward_ts = time.time()
            encoded_tensor = self.model_dp(input_var)
            forward_time.update(time.time() - forward_ts)
            loss_ts = time.time()
            loss, loss_details = self.model.get_loss(encoded_tensor, target_var)
            loss_time.update(time.time() - loss_ts)

            # record loss
            loss_total_am.update(loss_details["loss"], input.size(0))
            loss_loc_am.update(loss_details["loc_loss"], input.size(0))
            loss_cls_am.update(loss_details["cls_loss"], input.size(0))

            # compute gradient and do SGD step
            backward_ts = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            clip_gradient(self.model, 2.0, 'by_max')
            self.optimizer.step()
            backward_time.update(time.time() - backward_ts)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                      'LossTime {loss_time.val:.3f} ({loss_time.avg:.3f})\t'
                      'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                      'Loss {loss_total_am.val:.4f} ({loss_total_am.avg:.4f})\t'
                      'Loss_loc {loss_loc_am.val:.4f} ({loss_loc_am.avg:.4f})\t'
                      'Loss_cls {loss_cls_am.val:.4f} ({loss_cls_am.avg:.4f})\t'
                    .format(
                        self.epoch, batch_idx, len(self.train_loader),
                        batch_time=batch_time, data_time=data_time,
                        forward_time=forward_time, loss_time=loss_time, backward_time=backward_time,
                        loss_total_am=loss_total_am, loss_loc_am=loss_loc_am, loss_cls_am=loss_cls_am
                    ))

            if self.train_iter % self.print_freq == 0:
                self.writer.add_scalar('train/loss', loss_total_am.avg, self.train_iter)
                self.writer.add_scalar('train/loss_loc', loss_loc_am.avg, self.train_iter)
                self.writer.add_scalar('train/loss_cls', loss_cls_am.avg, self.train_iter)
                self.writer.add_scalar('train/lr', self.learning_rate, self.train_iter)

                num_prints = self.train_iter // self.print_freq
                # print('num_prints=', num_prints)
                num_prints_rare = num_prints // 100
                # print('num_prints_rare=', num_prints_rare)
                if num_prints_rare == 0 and num_prints % 10 == 0 or num_prints % 100 == 0:
                    print('save historgams')
                    if self.train_iter > 0:
                        import itertools
                        named_parameters = itertools.chain(
                            self.model.multibox_layers.named_parameters(),
                            self.model.extra_layers.named_parameters(),
                            )
                        for name, param in named_parameters:
                            self.writer.add_histogram(name, param.detach().cpu().numpy(), self.train_iter, bins='fd')
                            self.writer.add_histogram(name+'_grad', param.grad.detach().cpu().numpy(), self.train_iter, bins='fd')

                    first_conv = list(self.model.backbone._modules.items())[0][1]._parameters['weight']
                    image_grid = torchvision.utils.make_grid(first_conv.detach().cpu(), normalize=True, scale_each=True)
                    image_grid_grad = torchvision.utils.make_grid(first_conv.grad.detach().cpu(), normalize=True, scale_each=True)
                    self.writer.add_image('layers0_conv', image_grid, self.train_iter)
                    self.writer.add_image('layers0_conv_grad', image_grid_grad, self.train_iter)

            self.train_iter += 1
            pass

        self.epoch += 1

    def to_class_grouped_anno(self, batch_anno):
        """
        Since annotations have all classes mixed together, need to group them by class to
        pass it to average precision calculation function.
        """

        all_annotations = []
        for anno in batch_anno:
            classes = [[] for i in range(len(self.model.labelmap))]
            for obj in anno:
                object_id = self.model.labelmap.index(obj["type"])
                classes[object_id].append(obj["bbox"])
            classes = [
                np.stack(objs, axis=0) if len(objs) > 0 else np.empty((0, 4), dtype=np.float64) \
                for objs in classes]
            all_annotations.append(classes)
        return all_annotations

    def validate(self, do_dump_images=False, save_checkpoint=False):
        """
        Run validation on the current network state.
        """

        print("-------------- Validation ------------------")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_total_am = AverageMeter()
        loss_loc_am = AverageMeter()
        loss_cls_am = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        detection_val_dump_dir = os.path.join(self.run_dir, 'detection_val_dump')
        if do_dump_images:
            clean_dir(detection_val_dump_dir)

        iou_threshold_perclass = [0.7 if i == 0 else 0.5 for i in range(len(self.model.labelmap))]  # Kitti

        ap_estimator = average_precision.AveragePrecision(self.model.labelmap, iou_threshold_perclass)

        end = time.time()
        for batch_idx, sample in enumerate(self.val_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            input, target, names, pil_images, annotations, stats = sample

            with torch.no_grad():
                input_var, target_var = self.wrap_sample_with_variable(input, target, volatile=True)

                # Compute output tensor of the network
                encoded_tensor = self.model_dp(input_var)
                # Compute loss for logging only
                _, loss_details = self.model.get_loss(encoded_tensor, target_var)

            # Save annotation and detection results for further AP calculation
            class_grouped_anno = self.to_class_grouped_anno(annotations)
            detections_all = self.model.get_detections(encoded_tensor, 0.0)
            ap_estimator.add_batch(class_grouped_anno, detections_all)

            # Record loss
            loss_total_am.update(loss_details["loss"], input.size(0))
            loss_loc_am.update(loss_details["loc_loss"], input.size(0))
            loss_cls_am.update(loss_details["cls_loss"], input.size(0))

            # Dump validation images with overlays for developer to subjectively estimate accuracy
            if do_dump_images:
                overlay_conf_threshold = 0.3
                detections_thr = self.model.get_detections(encoded_tensor, overlay_conf_threshold)
                dump_images(
                    names, pil_images, annotations, detections_thr, stats,
                    self.model.labelmap,
                    detection_val_dump_dir)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss_total_am.val:.4f} ({loss_total_am.avg:.4f})\t'
                      'Loss_loc {loss_loc_am.val:.4f} ({loss_loc_am.avg:.4f})\t'
                      'Loss_cls {loss_cls_am.val:.4f} ({loss_cls_am.avg:.4f})\t'
                    .format(
                        batch_idx, len(self.val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss_total_am = loss_total_am, loss_loc_am = loss_loc_am, loss_cls_am = loss_cls_am
                    ))

        # After coming over the while validation set, calculate individual average precision values and total mAP
        mAP, AP_list = ap_estimator.calculate_mAP()

        for ap, label in zip(AP_list, self.model.labelmap):
            print('{} {:.3f}'.format(label.ljust(20), ap))
        print('   mAP - {mAP:.3f}'.format(mAP=mAP))
        performance_metric = AP_list[self.model.labelmap.index('Car')]

        # Log to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('val/mAP', mAP, self.train_iter)
            self.writer.add_scalar('val/performance_metric', performance_metric, self.train_iter)
            self.writer.add_scalar('val/loss', loss_total_am.avg, self.train_iter)
            self.writer.add_scalar('val/loss_loc', loss_loc_am.avg, self.train_iter)
            self.writer.add_scalar('val/loss_cls', loss_cls_am.avg, self.train_iter)

        if save_checkpoint:
            # Remember best accuracy and save checkpoint
            is_best = performance_metric > self.best_performance_metric

            if is_best:
                self.best_performance_metric = performance_metric
                torch.save(
                    {'state_dict': self.model.state_dict()},
                    self.snapshot_path)

        pass

    def load_checkpoint(self, checkpoint_path):
        """Load spesified snapshot to the network."""
        if checkpoint_path is None:
            checkpoint_path = self.snapshot_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Checkpoint not found:", checkpoint_path)
            assert False, "No sense to test random weights"
        pass

    def print_anchor_coverage(self):
        from anchor_coverage import AnchorCoverage

        anchor_coverage = AnchorCoverage()

        for batch_idx, sample in enumerate(self.val_loader):
            input, target, names, pil_images, annotations, stats = sample
            anchor_coverage.add_batch(annotations, stats)

        anchor_coverage.print()

    def export(self):
        resolution_hw = default_input_traits()["resolution"]
        example_input = torch.rand((1, 3, *resolution_hw))
        example_input_cuda = example_input.cuda()
        traced_model = torch.jit.trace(self.model, (example_input_cuda,))
        print(traced_model)

        self.model.cpu()
        path = os.path.join(self.run_dir, self.cfg.run_name+'.onnx')
        torch.onnx.export(self.model, (example_input,), path, verbose=False)
        if torch.cuda.is_available():
            self.model.cuda()
        # assert False
        pass

    def run(self):
        """
        Launch training procedure. Performs training interleaved
        by validation according to the training schedule.
        """

        self.print_anchor_coverage()

        self.best_performance_metric = 0.0

        do_dump_images = False

        self.writer = SummaryWriterOpt(enabled=True, suffix=self.cfg.run_name)

        # self.validate(do_dump_images=do_dump_images, save_checkpoint=False)

        num_epochs = 0
        do_process = True
        while do_process:
            for i in range(self.cfg.epochs_before_val):
                if num_epochs >= self.epochs_to_train:
                    do_process = False
                    break
                self.train_epoch()
                num_epochs += 1
            self.validate(do_dump_images=do_dump_images, save_checkpoint=True)
        pass


def main():
    """Entry point."""

    default_config = 'resnet34_pretrained'
    # default_config = 'resnet34_custom'
    # default_config = 'simple_model'
    # default_config = 'resnet50_pretrained'

    parser = argparse.ArgumentParser(description="Training script for 2D detection")
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()

    trainer = Trainer(args.config)

    if args.validate:

        print('Start validation')
        trainer.print_anchor_coverage()
        trainer.load_checkpoint(args.checkpoint_path)
        trainer.export()
        trainer.validate(do_dump_images=True, save_checkpoint=False)
        print('Finished validation. Done!')

    else:

        print('Start training')
        # trainer.load_checkpoint("runs/simple_model_1x1/simple_model_1x1_epochs452.pth.tar")
        trainer.run()
        print('Finished training. Done!')

    pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
