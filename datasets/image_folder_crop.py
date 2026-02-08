import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

@register('image-folder-crop')
class ImageFolderCrop(Dataset):

    def __init__(self, root_path, image_size=224, box_size=256, **kwargs):
        if box_size is None:
            box_size = image_size
            
        self.root_path = root_path
        self.samples = [] # List of (image_path, xmin, ymin, xmax, ymax, label)
        self.label = [] # To keep track of labels for sampling

        # Assuming root_path points to "images" folder, e.g. data/tlu-states/images
        # Annotations should be in a sibling folder "annotations"
        images_dir = root_path
        parent_dir = os.path.dirname(root_path.rstrip(os.sep))
        annotations_dir = os.path.join(parent_dir, 'annotations')

        if not os.path.exists(annotations_dir):
             # Fallback: maybe root_path is the dataset root itself?
             if os.path.exists(os.path.join(root_path, 'annotations')):
                 annotations_dir = os.path.join(root_path, 'annotations')
                 images_dir = os.path.join(root_path, 'images')
             else:
                 raise ValueError(f"Could not find annotations directory at {annotations_dir} or {os.path.join(root_path, 'annotations')}")

        print(f"Scanning for images in: {images_dir}")
        print(f"Scanning for annotations in: {annotations_dir}")

        classes = sorted(os.listdir(images_dir))
        classes = [c for c in classes if os.path.isdir(os.path.join(images_dir, c))]
        
        # Build samples
        for i, class_name in enumerate(classes):
            class_img_dir = os.path.join(images_dir, class_name)
            class_xml_dir = os.path.join(annotations_dir, class_name)
            
            if not os.path.exists(class_xml_dir):
                print(f"Warning: No annotations folder for class {class_name}. Skipping.")
                continue

            for filename in sorted(os.listdir(class_img_dir)):
                if is_image_file(filename):
                    image_path = os.path.join(class_img_dir, filename)
                    
                    # Construct corresponding XML path
                    basename = os.path.splitext(filename)[0]
                    xml_path = os.path.join(class_xml_dir, basename + '.xml')
                    
                    if os.path.exists(xml_path):
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            has_object = False
                            for obj in root.findall('object'):
                                bndbox = obj.find('bndbox')
                                if bndbox is not None:
                                    xmin = int(bndbox.find('xmin').text)
                                    ymin = int(bndbox.find('ymin').text)
                                    xmax = int(bndbox.find('xmax').text)
                                    ymax = int(bndbox.find('ymax').text)
                                    
                                    # Append sample
                                    self.samples.append((image_path, xmin, ymin, xmax, ymax, i))
                                    self.label.append(i)
                                    has_object = True
                            
                        except Exception as e:
                            print(f"Error parsing {xml_path}: {e}")
                    # else: skip images without XML
        
        self.n_classes = max(self.label) + 1 if self.label else 0
        print(f"Found {len(self.samples)} cropped objects across {self.n_classes} classes.")

        # Transforms
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        
        if kwargs.get('augment'):
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)), # Resize slightly larger before random crop
                transforms.RandomCrop(image_size), # Or RandomResizedCrop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((box_size, box_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        try:
            image_path, xmin, ymin, xmax, ymax, label = self.samples[i]
            
            img = Image.open(image_path).convert('RGB')
            
            # Crop the object
            # Validation of coordinates could be added here
            crop = img.crop((xmin, ymin, xmax, ymax))
            
            return self.transform(crop), label
            
        except Exception as e:
            print(f"Error loading sample {i}: {e}. Retrying a random sample.")
            import random
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
