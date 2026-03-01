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

        # Support for max_samples_per_class
        max_samples_per_class = kwargs.get('max_samples_per_class')

        for i, c in enumerate(classes):
            class_path = os.path.join(root_path, c)
            class_files = []
            
            # Gather all images for this class
            for filename in sorted(os.listdir(class_path)):
                if is_image_file(filename):
                    class_files.append(os.path.join(class_path, filename))
                    
            # Apply limit if specified and necessary
            if max_samples_per_class is not None and len(class_files) > max_samples_per_class:
                import random
                rng = random.Random(42) # fixed seed for reproducibility across runs if needed
                class_files = rng.sample(class_files, max_samples_per_class)
                
            for filepath in class_files:
                self.filepaths.append(filepath)
                self.label.append(i)
        
        if max_samples_per_class is not None:
             print(f"Applying class balancing: max {max_samples_per_class} samples per class.")

        self.n_classes = max(self.label) + 1 if self.label else 0

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        if kwargs.get('augment'):
            # self.transform = transforms.Compose([

            #     transforms.Resize((image_size, image_size)), # Chỉ nén/kéo dãn nhẹ

            #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Nghịch màu

            #     transforms.RandomRotation(15),       # Xoay nhẹ

            #     transforms.RandomHorizontalFlip(),

            #     transforms.ToTensor(),

            #     normalize,
            self.transform = transforms.Compose([
                # Cắt ngẫu nhiên 70-100% diện tích ảnh ban đầu, sau đó resize về kích thước chuẩn.
                # Ép mô hình nhìn vào các chi tiết zoom cận cảnh thay vì nhìn tổng thể.
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)), 
                
                # Tăng mạnh biên độ thay đổi màu sắc (sáng, tương phản, độ bão hòa) để mô hình bớt phụ thuộc vào màu.
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                
                # Tăng góc xoay ngẫu nhiên lên 30 độ
                transforms.RandomRotation(30),       
                
                # Lật ngang ảnh
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
