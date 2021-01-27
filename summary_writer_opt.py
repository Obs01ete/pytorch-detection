# Author: Dmitrii Khizbullin

import os
import time
from tensorboardX import SummaryWriter

class SummaryWriterOpt(object):
    def __init__(self, enabled=True, log_root_dir="logs", suffix=None):
        self.enabled = enabled
        self.writer = None
        time_str = time.strftime("%Y.%m.%d_%H-%M-%S", time.gmtime())
        if suffix is not None:
            log_dir = time_str + "_" + suffix
        else:
            log_dir = time_str
        self.log_dir = os.path.join(log_root_dir, log_dir)

    def _create_writer(self):
        if self.writer is None and self.enabled:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalar(self, *args):
        self._create_writer()
        if self.writer is not None:
            self.writer.add_scalar(*args)

    def add_image(self, *args, **kwargs):
        self._create_writer()
        if self.writer is not None:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        self._create_writer()
        if self.writer is not None:
            self.writer.add_histogram(*args, **kwargs)

