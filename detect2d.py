# Author: Dmitry Khizbullin
# A script to train a neural network for 2d detection task.
# Some code is borrowed from pytorch examples and torchvision.

import os
import sys
import time
import pickle
import argparse
import numpy as np
from termcolor import colored

import torch
import torch.utils.data as data
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


def list_all_images():
    """Scan over all samples in the dataset"""

    print('Start generation of a file list')

    names = []
    dir = 'training/image_2'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                nameonly = os.path.splitext(fname)[0]
                names.append(nameonly)

    print('End generation of a file list')

    return names


def train_val_split(image_list, dataset_dir, fraction_for_val=0.05):
    """Prepare file lists for training and validation."""

    train_num = int(len(image_list) * (1.0 - fraction_for_val))
    train_list = image_list[:train_num]
    val_list = image_list[train_num:]

    def save_object(name, obj):
        path = os.path.join(dataset_dir, name + '.pkl')
        with open(path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    save_object('train_list', train_list)
    save_object('val_list', val_list)

    pass


def prepare_dataset(dataset_dir):
    """Prepare dataset for training the detector. Done only once."""

    if os.path.exists(dataset_dir):
        return

    image_list = list_all_images()
    assert len(image_list) > 0

    os.makedirs(dataset_dir)

    train_val_split(image_list, dataset_dir)


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
    return transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)


def train_image_and_annotation_transform():
    """Image+annotation synchronous transformation for training."""
    return image_anno_transforms.ComposeVariadic([
        image_anno_transforms.RandomHorizontalFlipWithAnno(),
        image_anno_transforms.RandomCropWithAnno(0.3)
    ])



class BuildTargetFunctor(object):
    """Functor to delegate model's target construction to preprocessing threads."""

    def __init__(self, model):
        self.model = model

    def __call__(self, *args):
        return self.model.build_target(*args)


def create_detection_model(input_traits):
    labelmap = NameListDataset.getLabelmap()
    model = detection_models.SingleShotDetector(input_traits['resolution'], labelmap)
    return model


def clip_gradient(model, clip_val):
    """Clip the gradient."""
    for p in model.parameters():
        if p.grad is not None:
            mv = torch.max(torch.abs(p.grad.data))
            if mv > clip_val:
                print(colored("Grad max {:.3f}".format(mv), "red"))
            p.grad.data.clamp_(-clip_val, clip_val)



class Trainer():
    """Class that performs train-validation loop to train a detection neural network."""

    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir: directory with detection dataset
        """

        self.epochs_to_train = 1000
        self.base_learning_rate = 0.05 # 0.01
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

        model = detection_models.SingleShotDetector(input_traits['resolution'], labelmap)
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
            path = os.path.join(dataset_dir, name + '.pkl')
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

        self.train_iter = 0
        self.epoch = 0

        self.print_freq = 10

        self.writer = SummaryWriterOpt(enabled=True)

        pass


    @staticmethod
    def wrap_sample_with_variable(input, target, **kwargs):
        """Wrap tensor with Variable and push to cuda."""
        if torch.cuda.is_available():
            input = input.cuda()
            target = [t.cuda() for t in target]
        input_var = torch.autograd.Variable(input, **kwargs)
        target_var = [torch.autograd.Variable(t, **kwargs) for t in target]
        return input_var, target_var


    def train_epoch(self):
        """
        Train the model for one epoch.
        """

        print("-------------- Train epoch ------------------")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_total_am = AverageMeter()
        loss_loc_am = AverageMeter()
        loss_cls_am = AverageMeter()

        # switch to training mode
        self.model_dp.train()

        is_lr_change = self.epoch in [epoch for epoch, _ in self.lr_scales]
        if self.optimizer is None or is_lr_change:
            if self.optimizer is None:
                scale = 1.0
            if is_lr_change:
                scale = [sc for epoch, sc in self.lr_scales if epoch == self.epoch][0]
            self.learning_rate = self.base_learning_rate * scale
            self.optimizer = torch.optim.SGD(
                self.model_dp.parameters(), self.learning_rate,
                momentum=0.9,
                weight_decay=0.0001)

        detection_train_dump_dir = 'detection_train_dump'
        clean_dir(detection_train_dump_dir)

        end = time.time()
        for batch_idx, sample in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input, target, names, pil_images, annotations, stats = sample

            if False: # and random.random() < 0.01:
                dump_images(
                    names, pil_images, annotations, None, stats,
                    self.model.labelmap,
                    detection_train_dump_dir)

            input_var, target_var = self.wrap_sample_with_variable(input, target)

            # compute output
            encoded_tensor = self.model_dp(input_var)
            loss, loss_details = self.model.get_loss(encoded_tensor, target_var)

            # record loss
            loss_total_am.update(loss_details["loss"], input.size(0))
            loss_loc_am.update(loss_details["loc_loss"], input.size(0))
            loss_cls_am.update(loss_details["cls_loss"], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            clip_gradient(self.model, 2.0)
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss_total_am.val:.4f} ({loss_total_am.avg:.4f})\t'
                      'Loss_loc {loss_loc_am.val:.4f} ({loss_loc_am.avg:.4f})\t'
                      'Loss_cls {loss_cls_am.val:.4f} ({loss_cls_am.avg:.4f})\t'
                    .format(
                        self.epoch, batch_idx, len(self.train_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss_total_am=loss_total_am, loss_loc_am=loss_loc_am, loss_cls_am=loss_cls_am
                    ))

            if self.train_iter % self.print_freq == 0:
                self.writer.add_scalar('train/loss', loss_total_am.avg, self.train_iter)
                self.writer.add_scalar('train/loss_loc', loss_loc_am.avg, self.train_iter)
                self.writer.add_scalar('train/loss_cls', loss_cls_am.avg, self.train_iter)

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

        detection_val_dump_dir = 'detection_val_dump'
        if do_dump_images:
            clean_dir(detection_val_dump_dir)

        iou_threshold_perclass = [0.7 if i == 0 else 0.5 for i in range(len(self.model.labelmap))]  # Kitti

        ap_estimator = average_precision.AveragePrecision(self.model.labelmap, iou_threshold_perclass)

        end = time.time()
        for batch_idx, sample in enumerate(self.val_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            input, target, names, pil_images, annotations, stats = sample

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
                    'detection_2d_best.pth.tar')

        pass


    def load_checkpoint(self, checkpoint_path):
        """Load spesified snapshot to the network."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        pass


    def print_anchor_coverage(self):
        from anchor_coverage import AnchorCoverage

        anchor_coverage = AnchorCoverage()

        for batch_idx, sample in enumerate(self.val_loader):
            input, target, names, pil_images, annotations, stats = sample
            anchor_coverage.add_batch(annotations, stats)

        anchor_coverage.print()


    def run(self):
        """
        Launch training procedure. Performs training interleaved
        by validation according to the training schedule.
        """

        self.print_anchor_coverage()

        self.best_performance_metric = 0.0

        do_dump_images = False

        self.validate(do_dump_images=do_dump_images, save_checkpoint=False)

        for epoch in range(self.epochs_to_train):
            self.train_epoch()
            self.validate(do_dump_images=do_dump_images, save_checkpoint=True)
        pass




def main():
    """Entry point."""

    dataset_dir = 'detection_dataset'

    parser = argparse.ArgumentParser(description="Training script for 2D detection")
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()

    if args.checkpoint_path != None:

        print('Start validation')
        trainer = Trainer(dataset_dir)
        trainer.print_anchor_coverage()
        trainer.load_checkpoint(args.checkpoint_path)
        trainer.validate(do_dump_images=True, save_checkpoint=False)
        print('Finished validation. Done!')

    else:

        print('Start preparing dataset')
        prepare_dataset(dataset_dir)
        print('Finished preparing dataset')

        print('Start training')
        trainer = Trainer(dataset_dir)
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
