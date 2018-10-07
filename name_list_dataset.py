import os

import torch.utils.data as data

import kitti_randomaccess


class NameListDataset(data.Dataset):
    """Class to load the custom dataset with PyTorch's DataLoader"""

    def __init__(
            self,
            dataset_list,
            image_transform,
            image_and_anno_transform,
            map_to_network_input,
            build_target
        ):
        """
        Ctor.

        :param dataset_list: list of strings
        :param image_transform: transformer for images only (anno not altered)
        :param image_and_anno_transform: transformer that alters anno as well
        :param map_to_network_input: transformer to input tensor
        :param build_target: functor to build target from anno
        """

        self.dataset_list = dataset_list
        self.image_transform = image_transform
        self.image_and_anno_transform = image_and_anno_transform
        self.map_to_network_input = map_to_network_input
        self.build_target = build_target

        self._is_pil_image = True
        self.data_path = 'kitti/training/'
        self.image_path = os.path.join(self.data_path, 'image_2')
        self.velo_path = os.path.join(self.data_path, 'velodyne')
        self.calib_path = os.path.join(self.data_path, 'calib')
        self.label_path = os.path.join(self.data_path, 'label_2')

        pass

    @staticmethod
    def getLabelmap():
        return ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

    @staticmethod
    def leave_required_fields(anno):
        required_fields = ['type', 'bbox']
        anno_out = []
        for obj in anno:
            if obj['type'] != 'DontCare':
                obj_out = {}
                for f in obj.items():
                    if f[0] in required_fields:
                        obj_out[f[0]] = f[1]
                anno_out.append(obj_out)
        return anno_out

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of a sample

        Returns:
            input_tensor: tensor to feed into neural network
            built_target: target tuple of tensors for loss calculation
            name: string name of the sample
            image: PIL image (to render overlays)
            anno: annotation prior to encoding
            stats: debug information (number of anchor overlaps for every GT box)
        """

        name = self.dataset_list[index]
        image, velo, calib, anno = self._getitem(name)

        anno = self.leave_required_fields(anno)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.image_and_anno_transform is not None:
            image, anno = self.image_and_anno_transform(image, anno)

        input_tensor, anno = self.map_to_network_input(image, anno)

        built_target, stats = self.build_target(anno)

        return input_tensor, built_target, name, image, anno, stats

    def _getitem(self, name, load_image=True, load_velodyne=False, load_calib=True, load_label=True):
        image = None
        if load_image:
            path = os.path.join(self.image_path, name+'.png')
            if self._is_pil_image:
                image = kitti_randomaccess.get_image_pil(path)
            else:
                image = kitti_randomaccess.get_image(path)

        velo = None
        if load_velodyne:
            path = os.path.join(self.velo_path, name+'.bin')
            velo = kitti_randomaccess.get_velo_scan(path)

        calib = None
        if load_calib:
            path = os.path.join(self.calib_path, name+'.txt')
            calib = kitti_randomaccess.get_calib(path)

        label = None
        if load_label:
            path = os.path.join(self.label_path, name+'.txt')
            label = kitti_randomaccess.get_label(path)

        return image, velo, calib, label

    def __len__(self):
        """
        Args:
            none

        Returns:
            int: number of images in the dataset
        """

        return len(self.dataset_list)


