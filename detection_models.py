import os
import sys
import math
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import imagenet_models as models
import box_utils
import decode_detection


class ConvLayer(nn.Module):
    """Basic convolution block of ResNet."""
    def __init__(self, in_planes, out_planes, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ExtraBlock(nn.Module):
    """Basic block of extra layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        bottleneck = out_channels // 2
        self.conv1 = ConvLayer(in_channels, bottleneck, 1, 0, 1)
        self.conv2 = ConvLayer(bottleneck, out_channels, 3, 1, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ExtraLayers(nn.Module):
    """Extra layers in VGG style to finish feature pyramid."""

    def __init__(self, cfg, in_channels):
        super().__init__()

        layers = []
        for k, out_channels in enumerate(cfg):
            extra_block = ExtraBlock(in_channels, out_channels)
            layers.append(extra_block)
            in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        branches = []
        for block in self.blocks:
            x = block(x)
            branches.append(x)
        return branches


class MultiboxLayers(nn.Module):
    """Detection head of SSD. It is customized for regular anchors across pyramid's scales."""

    def __init__(self, final_branch_channels, labelmap, use_ohem=False):
        super().__init__()

        self.final_branch_channels = final_branch_channels
        self.labelmap = labelmap
        self.num_classes = len(self.labelmap)
        self.use_ohem = use_ohem

        sizes = [math.pow(2.0, 0.25), math.pow(2.0, 0.75)]
        self.iou_anchor_and_gt = 0.3

        aspect_ratios = [1.0, 1/2.0, 2.0] # AR=width/height

        anchor_list = []
        for size in sizes:
            for aspect_ratio in aspect_ratios:
                anchor_list.append(np.array([size * aspect_ratio, size / aspect_ratio], dtype=np.float32))
        self.anchors = np.stack(anchor_list, axis=0)

        self.variances = (0.1, 0.2) # (cxcy loc scale, wh loc scale)

        self.num_anchors_per_cell = 6
        self.bbox_regresion_size = 4
        self.detection_size = self.bbox_regresion_size + self.num_classes + 1
        self.cell_num_channels = self.num_anchors_per_cell*self.detection_size

        for i in range(len(self.final_branch_channels)):
            name = "multibox_branch_{}".format(i)
            conv = nn.Conv2d(final_branch_channels[i], self.cell_num_channels, kernel_size=3, padding=1)
            self.add_module(name, conv)

        self.detect_functor = decode_detection.Detect(self.num_classes, 0, 200, 0.45, self.variances)

        self.branch_resolutions = None

        pass

    def forward(self, tensor_list, is_probe=False):
        """
        Forward.
        :param tensor_list: list of tensors (feature maps) to which to apply loc & cls regression
        :param is_probe: if the pass is a test one to generate anchors. Must be run once.
        :return: joint tensor of all encoded detections
        """

        encoded_branches = []
        for i, in_tensor in zip(range(len(self.final_branch_channels)), tensor_list):
            name = "multibox_branch_{}".format(i)
            conv = self.__getattr__(name)
            encoded_tensor = conv(in_tensor)
            encoded_branches.append(encoded_tensor)

        if is_probe:
            self.branch_resolutions = [v.size()[2:] for v in encoded_branches]
            self._generate_anchors()

        single_tensor = self._reshape_and_concat(encoded_branches)

        return single_tensor

    def _generate_anchors(self):
        """Generate anchors according to number of branches and featuremap resolutions."""

        anchor_list = []
        for resolution in self.branch_resolutions:
            for anchor in self.anchors:
                height = resolution[0]
                width = resolution[1]
                cell_width = 1.0 / width
                cell_height = 1.0 / height
                for row in range(height):
                    for col in range(width):
                        anchor_cx = (col + 0.5) * cell_width
                        anchor_cy = (row + 0.5) * cell_height
                        anchor_width = anchor[0] * cell_width
                        anchor_height = anchor[1] * cell_height

                        anchor_cxcywh = np.array([anchor_cx, anchor_cy, anchor_width, anchor_height], dtype=np.float32)
                        anchor_list.append(anchor_cxcywh)
        anchors_cxcywh = np.stack(anchor_list, axis=0)
        anchors_cxcywh = torch.from_numpy(anchors_cxcywh)
        self.anchors_cxcywh = anchors_cxcywh
        self.register_buffer("anchors_cxcywh_cuda", anchors_cxcywh.clone())
        pass

    def _reshape_and_concat(self, encoded_branches):
        """Transform separate branch outputs to a joint tensor."""

        reshaped_branches = []
        for branch in encoded_branches:
            s = branch.size()
            b = s[0]
            a = self.num_anchors_per_cell
            d = self.detection_size
            assert s[1] == a * d
            h = s[2]
            w = s[3]
            branch = branch.view(b, a, d, h, w)
            branch = branch.permute(0, 3, 4, 1, 2).contiguous()
            branch = branch.view(b, h*w*a, d)

            reshaped_branches.append(branch)

        encoded_tensor = torch.cat(reshaped_branches, dim=1)
        return encoded_tensor

    def build_target(self, anno):
        """
        Building a target for loss calculation is incapsulated into the detection model class.
        Method to be called outside - in data loader threads. Must have no side effects on self object.

        :param anno: list of boxes with class ids
        :return:
            (loc, cls): encoded target: location regression and classification class
                loc: float tensor of shape (A, 4), A - total number of anchors
                cls: int tensor of shape (A,) of class labels, where 0 - background, 1 - class 0, etc
            matches: statistics of coverage of GT boxes by anchors
        """

        anno = self._anno_class_names_to_ids(anno)

        if len(anno) > 0:
            gt_boxes = np.stack([obj['bbox'] for obj in anno], axis=0)
            gt_classes = np.stack([obj['class_id'] for obj in anno], axis=0).astype(np.int32)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int32)

        gt_boxes = torch.from_numpy(gt_boxes)
        gt_classes = torch.from_numpy(gt_classes).long()

        loc, cls, matches = box_utils.match(self.iou_anchor_and_gt, gt_boxes,
            self.anchors_cxcywh, self.variances, gt_classes)

        return (loc, cls), matches

    def calculate_loss(self, encoded_prediction, encoded_target):
        """
        Calculate total classification & localization loss of SSD.

        :param encoded_prediction: tensor [N, A, D], N-batch size, A-total number of anchors, D-detection size
        :param encoded_target: pair of (loc, cls).
            loc: shape [N, A, R], R - bbox regression size = 4
            cls: shape [N, A, C], C - number of classes including background
        :return:
            loss: loss variable to optimize
            losses: dict of scalars to post to graphs
        """

        pred_xywh = encoded_prediction[:, :, 0:4].contiguous()
        pred_class = encoded_prediction[:, :, 4:].contiguous()
        assert pred_class.shape[2] == 1 + self.num_classes

        target_xywh = encoded_target[0]
        target_class_indexes = encoded_target[1]
        if torch.cuda.is_available():
            target_xywh = target_xywh.to(encoded_prediction.device)
            target_class_indexes = target_class_indexes.to(encoded_prediction.device)

        # determine positives
        bbox_matches_byte = target_class_indexes > 0
        bbox_matches = bbox_matches_byte.long()
        batch_size = bbox_matches.size(0)
        num_matches = bbox_matches.sum().item()

        # bbox loss only for positives
        bbox_mask = bbox_matches_byte.unsqueeze(2).expand_as(pred_xywh)
        bbox_denom = max(num_matches, 1)
        loc_loss = F.smooth_l1_loss(pred_xywh[bbox_mask], target_xywh[bbox_mask], reduction='sum') / bbox_denom

        pred_class_flat = pred_class.view(-1, pred_class.shape[-1])

        target_class_indexes_flat = target_class_indexes.view(-1)

        # calculate cls losses for positives and negative without reduction
        cls_loss_vec = F.cross_entropy(pred_class_flat, target_class_indexes_flat, reduction='none')
        cls_loss_vec = cls_loss_vec.view(batch_size, -1)

        if self.use_ohem:
            # Online hard sample mining (OHEM)
            neg_to_pos_ratio = 3 # the same as in the original SSD
            virtual_min_positive_matches = 100 # value for NN to learn on images without annotations

            # determine negatives with biggest loss
            cls_loss_neg = cls_loss_vec * (bbox_matches_byte.float() - 1.0)
            _, idx = cls_loss_neg.sort(1)
            _, rank_idxes = idx.sort(1)
            num_pos = bbox_matches.sum(1)
            num_neg = neg_to_pos_ratio * num_pos
            neg_idx = rank_idxes < num_neg[:, None]

            # combine losses from positives and negatives
            num_bbox_matches = bbox_matches.sum(dim=1)
            contributors_to_loss_mask = bbox_matches_byte | neg_idx
            contributors_to_loss_mask = contributors_to_loss_mask.float()
            contributors_to_loss = cls_loss_vec * contributors_to_loss_mask.float()
            cls_loss_batch_total = contributors_to_loss.sum(dim=1)
            cls_loss_total = cls_loss_batch_total.sum()
            num_bbox_matches_total = num_bbox_matches.sum()
            cls_denom = max(num_bbox_matches_total.float().item(), virtual_min_positive_matches)
            cls_loss = cls_loss_total / cls_denom
            pass
        else:
            # Average loss over all anchors (worse convergence than with OHEM)
            cls_loss = cls_loss_vec.sum() / cls_loss_vec.shape[1]

        loc_loss_mult = 1.0 #0.2
        cls_loss_mult = 1.0 if self.use_ohem else 8.0
        loc_loss_weighted = loc_loss_mult * loc_loss
        cls_loss_weighted = cls_loss_mult * cls_loss
        loss = loc_loss_weighted + cls_loss_weighted

        loss_details = {
            "loc_loss": loc_loss_weighted,
            "cls_loss": cls_loss_weighted,
            "loss": loss
        }
        loss_details = {name: float(var.item()) for (name, var) in loss_details.items()}

        return loss, loss_details

    def calculate_detections(self, encoded_tensor, threshold):
        """

        :param encoded_tensor: tensor [N, A, D], N-batch size, A-total number of anchors, D-detection size
        :param threshold: minimum confidence threshold for generated detections
        :return: list [N] of list [C] of numpy arrays [Q, 5], where N - batch size,
            C - number of object classes (i.e. no including background), Q - quantity of detected objects.
            Dimention of size 5 is decoded as [0] - confidence, [1:5] - bbox in fractional
            left-top-right-bottom (LTRB) format.
        """

        #encoded_tensor = encoded_tensor.cpu()

        loc_var = encoded_tensor[:, :, :4]
        conf_var = encoded_tensor[:, :, 4:]

        loc_data = loc_var.data
        conf_data = F.softmax(conf_var, dim=2).data

        conf_data = conf_data[:, :, 1:].contiguous() # throw away BG row after softmax

        anchors_cxcywh = self.anchors_cxcywh_cuda

        detections = self.detect_functor.forward(loc_data, conf_data, anchors_cxcywh, threshold)

        detections = detections.cpu().numpy()

        det_varsize = []
        for s in detections:
            c_varsize = []
            for c in s:
                c = c[c[:, 0] > 0.0]
                c_varsize.append(c)
            det_varsize.append(c_varsize)

        return det_varsize

    def _anno_class_names_to_ids(self, anno):
        anno_out = []
        for obj in anno:
            obj_out = {
                'class_id': self.labelmap.index(obj['type']),
                'bbox': obj['bbox'].astype(np.float32)
            }
            anno_out.append(obj_out)
        return anno_out

    def export_model_to_caffe(self, input_resolution):
        """
        Export to Caffe.
        """

        sys.path.insert(0, os.path.join("~/git/pytorch2caffe/"))
        sys.path.insert(0, "~/git/caffe_ssd_py3/build/install/python/")
        from pytorch2caffe import pytorch2caffe

        input_var = torch.rand(1, 3, int(input_resolution[0]), int(input_resolution[1]))
        encoded_var = self(input_var)
        pytorch2caffe(
            input_var, encoded_var,
            'model.prototxt',
            'model.caffemodel')
        pass


class SingleShotDetector(nn.Module):
    def __init__(self, backbone_specs, multibox_specs, input_resolution, labelmap):
        """
        Ctor.

        :param input_resolution: input resolution (H, W)
        :param labelmap: list [C] of class name strings, where C - number of object classes (not including background)
        """

        super().__init__()

        for c in input_resolution:
            assert c % 256 == 0

        self.labelmap = labelmap

        backbone_module = importlib.import_module(backbone_specs['backbone_module'])

        # Use Resnet-XX as a backbone
        backbone_create_func = getattr(backbone_module, backbone_specs['backbone_function'])
        self.backbone = backbone_create_func(**backbone_specs['kwargs'])
        channel_multiplier = backbone_specs['head_channel_multiplier']

        self.backbone.eval()

        # probe backbone
        input_batch_shape = (1, 3, *input_resolution)
        input_tensor = torch.autograd.Variable(torch.rand(input_batch_shape))
        backbone_out = self.backbone(input_tensor)
        backbone_last = backbone_out[-1]
        backbone_last_channels = backbone_last.shape[1]

        # create additional layers
        # extras_config = [512, 256, 256]
        extras_config = [v*channel_multiplier for v in (2, 2, 2)]
        self.extra_layers = ExtraLayers(extras_config, backbone_last_channels)
        self.extra_layers.eval()

        # probe extra layers
        extra_layers_out = self.extra_layers(backbone_last)

        # take only these last branches from backbone, all other branches come from additional layers
        self.num_last_backbone_branches = 3

        print("----- SSD branch configuration -----")
        for i, t in enumerate(backbone_out):
            print(t.shape, " <- branch" if len(backbone_out)-i <= self.num_last_backbone_branches else "")
        for t in extra_layers_out:
            print(t.shape, " <- branch")
        print("------------------------------------")

        # collect all branches in a tuple
        final_branches = (*backbone_out[-self.num_last_backbone_branches:], *extra_layers_out)
        final_branch_channels = [b.shape[1] for b in final_branches]

        # add multi-branch detection head on top of all branches
        self.multibox_layers = MultiboxLayers(final_branch_channels, self.labelmap, multibox_specs['use_ohem'])
        self.multibox_layers.eval()
        # probe multibox, save branch resolutions, generate anchors
        self.multibox_layers(final_branches, is_probe=True)

        if False:
            # probe the whole net
            encoded_tensor = self.forward(input_tensor)
            detections = self.get_detections(encoded_tensor, threshold=0.15)

        # export model
        if False:
            self.export_model_to_caffe(input_resolution)

        pass

    def forward(self, input_tensor_batch):
        """
        Forward.

        :param input_tensor_batch: input image of shape [N, H, W, 3], where N - batch size, H - height, W - width
        :return: target - single tensor of shape [b=32, cat(flat_anchors=A*H*W, for all branches), D=4+1+num_classes]
        """

        backbone_branches = self.backbone(input_tensor_batch)

        # automatically derive resolution for extra layers
        backbone_last_branch = backbone_branches[-1]
        extra_branches = self.extra_layers(backbone_last_branch)

        # collect all branch feature maps in a tuple
        final_branches = (*backbone_branches[-self.num_last_backbone_branches:], *extra_branches)

        encoded_tensor = self.multibox_layers(final_branches)

        return encoded_tensor

    def get_loss(self, encoded_tensor, target):
        """Get loss for optimization."""
        return self.multibox_layers.calculate_loss(encoded_tensor, target)

    def get_detections(self, encoded_tensor, threshold):
        """Get bbox detections in finally decoded format."""
        return self.multibox_layers.calculate_detections(encoded_tensor, threshold)

    def build_target(self, anno):
        # Forward to multibox component
        return self.multibox_layers.build_target(anno)
