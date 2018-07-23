"""Load Kitti samples"""

import numpy as np
from PIL import Image


def get_image(filename):
    """Function to read image files into arrays."""
    return np.asarray(Image.open(filename), np.uint8)


def get_image_pil(filename):
    """Function to read image files into arrays."""
    return Image.open(filename)


def get_velo_scan(filename):
    """Function to parse velodyne binary files into arrays."""
    scan = np.fromfile(filename, dtype=np.float32)
    return scan.reshape((-1, 4))


def get_calib(filename):
    """Function to parse calibration text files into a dictionary."""
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(7):
            key, value = lines[i].split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
    return data


def get_label(filename):
    """Function to parse label text files into a dictionary."""
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            values = line.split()
            assert len(values) == 15
            obj = {
                'type': str(values[0]),
                'truncated': float(values[1]),
                'occluded': int(values[2]),
                'alpha': float(values[3]),
                'bbox': np.array(values[4:8], dtype=float),
                'dimensions': np.array(values[8:11], dtype=float),
                'location': np.array(values[11:14], dtype=float),
                'rotation_y': float(values[14]),
            }
            data.append(obj)
    return data


