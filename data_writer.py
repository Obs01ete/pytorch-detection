#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides 'Writer', which which writes predictions to file."""

from __future__ import absolute_import, print_function
import os
import numpy as np


class Writer(object):
    """Write results into same format as label .txt files."""

    def __init__(self, data_path):
        """
        Set the folder into which prediction file will be written to.
        """
        if not os.path.exists(data_path):
            print("Data path %s created." % data_path)
            os.makedirs(data_path)
        self.data_path = data_path
        self._defaults = {'type': 'DontCare',
                          'truncated': np.nan,
                          'occluded': 3,
                          'alpha': np.nan,
                          'bbox': np.array([np.nan, np.nan, np.nan, np.nan]),
                          'dimensions': np.array([np.nan, np.nan, np.nan]),
                          'location': np.array([np.nan, np.nan, np.nan]),
                          'rotation_y': np.nan}

    def write(self, filename, labels):
        """
        Function to write labels to file provided by filename (i.e. '000000.txt')
        labels, just like in Parser, is a list of dictionaries with the keys below.
        N.B. You need not provide all the keys! For example, if your task is to do 2D
        bounding box detection, you can simply add your predicted 'type' and 'bbox' to dict,
        and ignore the rest. These will be padded with 'nan'.

        #Values     Key        Description
        ----------------------------------------------------------------------------
        1           type        Describes the type of object: 'Car', 'Van', 'Truck',
                                'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                'Misc' or 'DontCare'

        1           truncated   Float from 0 (non-truncated) to 1 (truncated), where
                                truncated refers to the object leaving image boundaries

        1           occluded    Integer (0,1,2,3) indicating occlusion state:
                                0 = fully visible, 1 = partly occluded
                                2 = largely occluded, 3 = unknown

        1           alpha       Observation angle of object, ranging [-pi..pi]

        4           bbox        2D bounding box of object in the image (0-based index):
                                contains left, top, right, bottom pixel coordinates

        3           dimensions  3D object dimensions: height, width, length (in meters)

        3           location    3D object location x,y,z in camera coordinates (in meters)

        1           rotation_y  Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        assert isinstance(filename, str)
        assert isinstance(labels, list)
        for l in labels:
            assert isinstance(l, dict)
        keys = ['type', 'truncated', 'occluded', 'alpha',
                'bbox', 'dimensions', 'location', 'rotation_y']

        with open(os.path.join(self.data_path, filename), 'w+') as f:
            for obj in labels:
                out = []
                for key in keys:
                    if key in obj:
                        self._checkvalidity(key, obj[key])
                        out.append(self._tostring(obj[key]))
                    else:
                        out.append(self._tostring(self._getdefault(key)))
                line = ' '.join(out) + '\n'
                f.write(line)

    def _tostring(self, value):
        if isinstance(value, str):
            return value
        else:
            try:
                return ' '.join([str(x) for x in value])
            except TypeError:
                return str(value)

    def _checkvalidity(self, key, value):
        if key == 'type':
            assert value in {'Car', 'Van', 'Truck', 'Pedestrian',
                                'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'}
        elif key == 'truncated':
            assert isinstance(value, float)
            assert value <= 1.0
            assert value >= 0.0
        elif key == 'occluded':
            assert isinstance(value, int)
            assert value in {0, 1, 2, 3}
        elif key == 'alpha':
            assert isinstance(value, float)
            assert value <= np.pi
            assert value >= -np.pi
        elif key == 'bbox':
            assert isinstance(value, np.ndarray)
            assert np.all(value >= 0)
        elif key == 'dimensions':
            assert isinstance(value, np.ndarray)
            assert np.all(value >= 0)
        elif key == 'location':
            assert isinstance(value, np.ndarray)
            assert np.all(value >= 0)
        elif key == 'rotation_y':
            assert isinstance(value, float)
            assert value <= np.pi
            assert value >= -np.pi
        else:
            raise IndexError

    def _getdefault(self, key):
        return self._defaults[key]
