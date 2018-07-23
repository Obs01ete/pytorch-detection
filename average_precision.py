import numpy as np


def iou_point_np(box, boxes):
    """
    Find intersection over union
    :param box: (tensor) One box [xmin, ymin, xmax, ymax], shape: [4].
    :param boxes: (tensor) Shape:[N, 4].
    :return: intersection over union. Shape: [N]
    """

    A = np.maximum(box[:2], boxes[:, :2])
    B = np.minimum(box[2:], boxes[:, 2:])
    interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])

    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = boxArea + boxesArea - interArea
    iou = interArea / union
    return iou



class AveragePrecision(object):
    """Average precision calculation using sort-and-iterate algorithm (VOC12)"""

    def __init__(self, labelmap, iou_threshold_perclass):
        """
        Ctor.

        :param labelmap: list of strings - class names
        :param iou_threshold_perclass: intersection over union thresholds for each class
        """

        self.labelmap = labelmap
        self.num_classes = len(labelmap)
        self.iou_threshold_perclass = iou_threshold_perclass
        self.annotation_list = []
        self.detection_list = []

    def add_batch(self, annotations, detections):
        """
        Accumulate detection results and annotations from one batch.

        :param annotations: list [N] of list [C] of numpy arrays [Q, 4], where N - batch size,
            C - number of object classes (i.e. no including background), Q - quantity of annotated objects.
            Dimension of size 4 is decoded as a bbox in fractional left-top-right-bottom (LTRB) format.

        :param detections: list [N] of list [C] of numpy arrays [Q, 5], where N - batch size,
            C - number of object classes (i.e. no including background), Q - quantity of detected objects.
            Dimension of size 5 is decoded as [0] - confidence, [1:5] - bbox in fractional
            left-top-right-bottom (LTRB) format.
        """

        self.annotation_list.extend(annotations)
        self.detection_list.extend(detections)

    def calculate_mAP(self):
        """Perform calculation of mAP and per-class APs"""

        AP_list = np.zeros((self.num_classes,), dtype=np.float64)

        for cls_idx in range(self.num_classes):

            true_positive_list = []
            positive_list = []
            conf_list = []
            for det, anno in zip(self.detection_list, self.annotation_list):

                annotation = anno[cls_idx]
                prediction = det[cls_idx]
                iou_threshold = self.iou_threshold_perclass[cls_idx]

                if len(prediction) == 0:
                    continue

                matched_gt = np.zeros((len(annotation),), dtype=np.int32)
                true_positives = np.zeros((len(prediction),), dtype=np.int32)

                predicted_confs = prediction[:, 0]
                predicted_boxes = prediction[:, 1:]

                for idx, true_bbox in enumerate(annotation):

                    iou = iou_point_np(true_bbox, predicted_boxes)

                    # find matching
                    iou_max = np.max(iou)
                    if iou_max > iou_threshold:
                        matched_gt[idx] = 1
                        true_positives[np.argmax(iou)] = 1

                true_positive_list.append(true_positives)
                positive_list.append(len(annotation))
                conf_list.append(predicted_confs)

            # end loop over images

            true_positive = np.concatenate(true_positive_list, axis=0)
            positive = np.array(positive_list, dtype=np.int).sum()
            conf = np.concatenate(conf_list, axis=0)

            idx_sort = np.argsort(-conf)
            fn = 1 - true_positive[idx_sort]
            true_positive = np.cumsum(true_positive[idx_sort])
            false_negative = np.cumsum(fn)

            precision = true_positive / (true_positive + false_negative + 1e-4)
            recall = true_positive / (positive + 1e-4)
            AP_val = np.sum((recall[1:] - recall[:-1]) * precision[1:])
            AP_list[cls_idx] = AP_val

            pass

        # end for cls_idx

        mAP = float(AP_list.mean())

        return mAP, AP_list
