# Author: Dmitrii Khizbullin
# Builds a table of numbers of anchor boxes which cover each ground truth box

import numpy as np


class AnchorCoverage:
    def __init__(self):
        self.annotations = []
        self.stats = []

    def add_batch(self, annotations, stats):
        self.annotations.extend(annotations)
        self.stats.extend(stats)

    def print(self):
        import math
        from tabulate import tabulate

        num_height_bins = 12
        max_quantity = 20

        hist = np.zeros((num_height_bins, max_quantity), dtype=np.int)
        for frame_anno, frame_stat in zip(self.annotations, self.stats):
            for anno, stat in zip(frame_anno, frame_stat):
                height = anno['bbox'][3] - anno['bbox'][1]
                height_bin = min(math.floor(-math.log2(height)*2), num_height_bins-1)
                quantity_bin = min(stat, max_quantity-1)
                hist[height_bin, quantity_bin] += 1

        table = tabulate(hist, headers=list(range(max_quantity)), tablefmt="fancy_grid")
        print(table)
        fn = hist[:, 0].sum()
        pos = hist.sum()
        fnr = fn / pos
        print('Anchor FNR {:6f} = {}/{}'.format(fnr, fn, pos))
        pass
