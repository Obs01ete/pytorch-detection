import os
import cv2
import itertools
import numpy as np


def dump_images(
        names, pil_images, annotations, detections, stats,
        labelmap, dir):
    """
    Dumps images with bbox overlays to disk.

    :param names: batch of sample names
    :param pil_images: batch of original PIL images
    :param annotations: batch of annotations
    :param detections: batch of detections from NN
    :param stats: batch of debug info from a network. Keeps number of anchors that match particular GT box.
    :param labelmap: names of classes
    :param dir: destination directory to save images
    :return: None
    """

    det_color = (0, 255, 0)
    anno_color = (255, 0, 0)

    if annotations is None: annotations = []
    if detections is None: detections = []
    if stats is None: stats = []

    try:
        for ib, (name, pil_img, anno, detection, stat) in \
                enumerate(itertools.zip_longest(names, pil_images, annotations, detections, stats)):

            img = np.asarray(pil_img).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            if detection is not None:
                for icls, cls_det in enumerate(detection):
                    for det in cls_det:
                        conf = det[0]
                        if conf > 0.0:
                            bbox = det[1:]
                            bbox_pix = bbox * scale
                            type = labelmap[icls]
                            cv2.rectangle(
                                img,
                                (int(bbox_pix[0]), int(bbox_pix[1])),
                                (int(bbox_pix[2]), int(bbox_pix[3])),
                                det_color, 1)
                            cv2.putText(
                                img,
                                '{} {:.2f}'.format(type, conf),
                                (int(bbox_pix[0]), int(bbox_pix[1])+10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                det_color)

            if anno is not None and stat is not None:
                for obj, num_matches in zip(anno, stat):
                    bbox = obj['bbox']
                    bbox_pix = bbox * scale
                    cv2.rectangle(
                        img,
                        (int(bbox_pix[0]), int(bbox_pix[1])),
                        (int(bbox_pix[2]), int(bbox_pix[3])),
                        anno_color, 1)
                    cv2.putText(
                        img,
                        obj['type'] + " M{}".format(num_matches), # M - number of matching anchors
                        (int(bbox_pix[0]), int(bbox_pix[1])+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        anno_color)

            filename = name + '.png'
            cv2.imwrite(os.path.join(dir, filename), img)
            pass
    except Exception as e:
        pass

    pass
