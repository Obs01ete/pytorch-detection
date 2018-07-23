import random
from PIL import Image, ImageOps
import torchvision.transforms as transforms


def normalize_image():
    """Zero-mean input image for faster training."""
    return transforms.Normalize(
        mean=[0.5]*3,
        std=[0.25]*3)


class RandomCropWithAnno(object):
    """
    Crop the given PIL.Image at a random location along with annotations.
    Do not drop or crop annotations.

    Args:
        pad_val: percent of image width or height to pad
    """

    def __init__(self, pad_val):
        self.pad_val = pad_val

    def __call__(self, img, anno):
        """
        Args:
            img (PIL.Image): Image to be cropped.
            anno: corresponding annotation.

        Returns:
            PIL.Image: Cropped image + annotation.
        """

        ow, oh = img.size

        border = tuple([int(ow*self.pad_val), int(oh*self.pad_val)] * 2)
        img_padded = ImageOps.expand(img, border=border, fill=0)

        w, h = img_padded.size

        x1 = random.randint(0, w - ow)
        y1 = random.randint(0, h - oh)
        img_out = img_padded.crop((x1, y1, x1 + ow, y1 + oh))

        anno_offs_x = border[0] - x1
        anno_offs_y = border[1] - y1

        for obj in anno:
            bbox = obj['bbox']
            bbox += [anno_offs_x, anno_offs_y] * 2

        return img_out, anno


class RandomHorizontalFlipWithAnno(object):
    """Horizontally flip the given PIL.Image + annotation randomly with a probability of 0.5."""

    def __call__(self, img, anno):
        """
        Args:
            img (PIL.Image): Image to be flipped.
            anno: corresponding annotation.

        Returns:
            PIL.Image: Randomly flipped image + annotation.
        """
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            width = img.width
            for obj in anno:
                bbox = obj['bbox']
                new_left = width - bbox[2]
                new_right = width - bbox[0]
                bbox[0] = new_left
                bbox[2] = new_right
        return img, anno


class ComposeVariadic(object):
    """
    Composes several transforms together. Processes lists of image + annotation + whatever.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class MapImageAndAnnoToInputWindow(object):
    """Transform to map any image + anno to network input format."""

    def __init__(self, input_resolution):
        self.input_resolution = input_resolution
        self.transform = transforms.Compose([
            transforms.Resize(input_resolution, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            normalize_image(),
        ])

    def __call__(self, img, anno):
        img_out = self.transform(img)
        if anno is not None:
            anno_out = []
            for obj in anno:
                obj_out = {
                    'type': obj['type'],
                    'bbox': obj['bbox'] / ([img.width, img.height] * 2) # left top right bottom
                }
                anno_out.append(obj_out)
        else:
            anno_out = None
        return img_out, anno_out

