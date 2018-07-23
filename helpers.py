import os
import glob
from PIL import Image

IMG_EXTENSIONS = ['.png', '.PNG']

def is_image_file(filename):
    """Borrowed helper."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    """Borrowed helper"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def clean_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        files = glob.glob(os.path.join(dir, '*'))
        for f in files:
            os.remove(f)

