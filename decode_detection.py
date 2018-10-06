# From https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/detection.py
# with minor modifications.

import torch
from box_utils import decode, nms


class Detect():
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, nms_thresh, variance):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.variance = variance

    def forward(self, loc_data, conf_data, prior_data, conf_thresh):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        if loc_data.is_cuda:
            output = output.cuda()
        conf_preds = conf_data.transpose(2, 1) # group by classes

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(self.num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                thresholded_boxes = decoded_boxes[l_mask]
                if len(thresholded_boxes) > 0:
                    boxes = thresholded_boxes.view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                    output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                   boxes[ids[:count]]), 1)
        flt = output.contiguous().view(batch_size, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
