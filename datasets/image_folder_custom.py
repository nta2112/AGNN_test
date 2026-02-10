import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

@register('image-folder-custom')
class ImageFolderCustom(Dataset):

    def __init__(self, root_path, image_size=224, box_size=256, **kwargs):
        if box_size is None:
            box_size = image_size

        self.filepaths = []
        self.label = []
        
        # Read split file if provided
        if kwargs.get('split'):
            path = kwargs.get('split_file')
            if path is None:
                path = os.path.join(
                        os.path.dirname(root_path.rstrip('/')), 'split.json')
            if os.path.exists(path):
                split = json.load(open(path, 'r'))
                classes = sorted(split[kwargs['split']])
            else:
                # Fallback if split file not found but arg provided (should error usually, but let's be safe)
                print(f"Warning: Split file {path} not found. Using all folders.")
                classes = sorted(os.listdir(root_path))
        else:
             classes = sorted(os.listdir(root_path))

        # Filter classes to ensure they are directories
        classes = [c for c in classes if os.path.isdir(os.path.join(root_path, c))]
        print(f"DEBUG: ImageFolderCustom root={root_path}, split={kwargs.get('split')}, found {len(classes)} classes: {classes[:5]}...")

        for i, c in enumerate(classes):
            class_path = os.path.join(root_path, c)
            files = sorted(os.listdir(class_path))
            for filename in files:
                if is_image_file(filename):
                    self.filepaths.append(os.path.join(class_path, filename))
                    self.label.append(i)
        
        print(f"DEBUG: Loaded {len(self.filepaths)} images across {len(classes)} classes.")
        if len(self.filepaths) == 0:
             print(f"ERROR: No images found! Check root_path and split file.")
        
        self.n_classes = max(self.label) + 1 if self.label else 0

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        if kwargs.get('augment'):
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(box_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        while True:
            try:
                path = self.filepaths[i]
                img = Image.open(path).convert('RGB')
                return self.transform(img), self.label[i]
            except Exception as e:
                print(f"Warning: Error loading image {path}: {e}. Skipping...")
                # Pick a new random index to retry
                import random
                i = random.randint(0, len(self.filepaths) - 1)
