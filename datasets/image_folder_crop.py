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
                            
                            if not has_object:
                                # Fallback: XML exists but no object? Use whole image?
                                # Ideally we should check image size, but for now let's rely on PIL lazy loading later
                                # We mark special crop -1 to indicate full image
                                print(f"Warning: {xml_path} has no objects. Using full image.")
                                self.samples.append((image_path, -1, -1, -1, -1, i))
                                self.label.append(i)

                        except Exception as e:
                            print(f"Error parsing {xml_path}: {e}")
                    else: 
                        # Fallback: No XML found. Use whole image.
                        # print(f"Info: No XML for {filename}. Using full image.")
                        self.samples.append((image_path, -1, -1, -1, -1, i))
                        self.label.append(i)
                    # else: skip images without XML
        
        # Undersampling (Class Balancing)
        max_samples_per_class = kwargs.get('max_samples_per_class')
        if max_samples_per_class is not None:
            # ... (balancing logic remains same, just need to be careful with remapping)
            # Actually, let's remap labels FIRST, then balance, or balance then remap.
            # Best is:
            # 1. Load all valid samples as (path, box..., class_name_index)
            # 2. Filter out classes with 0 samples
            # 3. Create a mapping old_index -> new_contiguous_index
            # 4. Update samples with new index
            pass

        # RE-IMPLEMENTATION OF LABEL MAPPING TO FIX "EMPTY CLASS" BUG
        # The previous loop used `i` from `enumerate(classes)` as label.
        # If a class had no samples (e.g. no XML), that `i` would never appear in `self.label`.
        # But `n_classes = max(label) + 1` would include it, creating a gap.
        # CategoriesSampler iterates `range(n_classes)`, hitting the gap -> Empty list -> Error.

        # 1. Extract unique labels present in the data
        present_labels = sorted(list(set(s[-1] for s in self.samples)))
        
        # 2. Create mapping: old_label -> new_contiguous_label
        label_map = {old_l: new_l for new_l, old_l in enumerate(present_labels)}
        
        # 3. Update samples with new labels
        new_samples = []
        new_labels = []
        for s in self.samples:
            # s is tuple, immutable. construct new one
            # s = (path, xmin, ymin, xmax, ymax, old_label)
            old_label = s[-1]
            new_label = label_map[old_label]
            new_samples.append(s[:-1] + (new_label,))
            new_labels.append(new_label)
            
        self.samples = new_samples
        self.label = new_labels
        
        self.n_classes = len(present_labels)
        print(f"DEBUG: Remapped {len(present_labels)} non-empty classes to contiguous labels 0-{self.n_classes-1}.")

        # Undersampling (Class Balancing) - Moved after remapping for safety
        max_samples_per_class = kwargs.get('max_samples_per_class')
        if max_samples_per_class is not None:
            print(f"Applying class balancing: max {max_samples_per_class} samples per class.")
            import random
            balanced_samples = []
            balanced_labels = []
            
            # Group samples by class
            class_samples = {}
            for s in self.samples:
                label = s[-1]
                if label not in class_samples:
                    class_samples[label] = []
                class_samples[label].append(s)
            
            for label, samples in class_samples.items():
                if len(samples) > max_samples_per_class:
                    selected = random.sample(samples, max_samples_per_class)
                else:
                    selected = samples
                balanced_samples.extend(selected)
                balanced_labels.extend([label] * len(selected))
            
            self.samples = balanced_samples
            self.label = balanced_labels
            print(f"After balancing: {len(self.samples)} samples across {self.n_classes} classes.")

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
            if xmin == -1:
                # Full image fallback
                crop = img
                # Optional: maybe check if image is too large?
            else:
                crop = img.crop((xmin, ymin, xmax, ymax))
            
            return self.transform(crop), label
            
        except Exception as e:
            print(f"Error loading sample {i}: {e}. Retrying a random sample.")
            import random
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
