import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

import cv2
import numpy as np

def is_good_crop(crop_img, min_size=30, max_ratio=4.0, min_blur_var=50.0):
    """
    Check if the cropped image is good enough.
    Criteria: Not too small, not too narrow, not too blurry.
    """
    w, h = crop_img.size
    
    # 1. Check size (too small)
    if w < min_size or h < min_size:
        return False
        
    # 2. Check aspect ratio (too narrow)
    ratio = max(w/h, h/w) if h > 0 and w > 0 else float('inf')
    if ratio > max_ratio:
        return False
        
    # 3. Check blurriness using Variance of Laplacian
    try:
        img_cv = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        if variance < min_blur_var:
            return False
    except Exception:
        pass # If we fail to compute blur, just let it pass
        
    return True

def get_saliency_crop(img_pil):
    """
    Detects the most salient object in the image and returns a crop.
    Uses OpenCV Saliency (Spectral Residual) or Thresholding as fallback.
    """
    try:
        # Convert PIL to OpenCV
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Method 1: Simple Thresholding (Fastest for clear fruit on background)
        # Convert to HSV to detect colored objects (fruits usually pop out)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1] # Saturation channel
        _, thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Constraints: Ignore tiny noise or full image
            img_h, img_w = img.shape[:2]
            if w > 10 and h > 10 and (w < img_w or h < img_h):
                # Add padding
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(img_w - x, w + 2*pad)
                h = min(img_h - y, h + 2*pad)
                
                return img_pil.crop((x, y, x+w, y+h))
                
    except Exception as e:
        print(f"Saliency detection error: {e}")
        
    return None # Fallback to full image if detection fails

def preprocess(args):
    input_root = args.input
    output_root = args.output
    target_size = args.size
    
    # Check structure: strict "images" and "annotations" folder expected
    images_dir = os.path.join(input_root, 'images')
    annotations_dir = os.path.join(input_root, 'annotations')
    
    if not os.path.exists(images_dir):
        # Fallback: maybe input_root IS the images folder?
        # But for this script to work with XMLs, we need structure.
        print(f"Error: {images_dir} does not exist. Please provide root path containing 'images' and 'annotations'.")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    classes = sorted(os.listdir(images_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(images_dir, c))]
    
    # Optional: Filter by split file if provided
    if args.split_file and args.split_name:
        import json
        try:
            with open(args.split_file, 'r') as f:
                split_data = json.load(f)
                
                if args.split_name == 'all':
                   target_classes = set()
                   for split_key in split_data:
                       target_classes.update(split_data[split_key])
                   classes = [c for c in classes if c in target_classes]
                   print(f"Filtered to {len(classes)} classes from ALL splits in {args.split_file}")
                
                elif args.split_name in split_data:
                    target_classes = set(split_data[args.split_name])
                    classes = [c for c in classes if c in target_classes]
                    print(f"Filtered to {len(classes)} classes from split '{args.split_name}' in {args.split_file}")
                else:
                    print(f"Warning: Split name '{args.split_name}' not found in {args.split_file}. Using all classes.")
        except Exception as e:
            print(f"Error reading split file: {e}")
            return

    print(f"Found {len(classes)} classes in {images_dir} to process.")
    
    count_crop = 0
    count_full = 0
    count_auto = 0
    count_err = 0
    count_discarded = 0
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_img_dir = os.path.join(images_dir, class_name)
        # Handle missing annotation folder gracefully
        class_xml_dir = os.path.join(annotations_dir, class_name)
        
        class_out_dir = os.path.join(output_root, class_name)
        if not os.path.exists(class_out_dir):
            os.makedirs(class_out_dir)
            
        images = [f for f in os.listdir(class_img_dir) if is_image_file(f)]
        
        for filename in images:
            img_path = os.path.join(class_img_dir, filename)
            basename = os.path.splitext(filename)[0]
            
            # Look for XML
            xml_path = os.path.join(class_xml_dir, basename + '.xml')
            
            try:
                img = Image.open(img_path).convert('RGB')
                crops = []
                
                # 1. Try XML Crop
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
                                
                                # Validate box
                                if xmax > xmin and ymax > ymin:
                                    crop = img.crop((xmin, ymin, xmax, ymax))
                                    if is_good_crop(crop):
                                        crops.append(crop)
                                        has_object = True
                                    else:
                                        # Box was valid but crop is discarded due to quality
                                        pass
                        
                        if not has_object:
                            # valid XML but no object -> Auto crop later
                            pass 
                            
                    except Exception as e:
                        print(f"Error parsing XML {xml_path}: {e}")
                
                # 2. If No XML Crops, Try Auto Saliency Crop
                if len(crops) == 0:
                    auto_crop = get_saliency_crop(img)
                    if auto_crop is not None and is_good_crop(auto_crop):
                        crops.append(auto_crop)
                        count_auto += 1
                    else:
                        # 3. Last Resort: Full Image
                        if is_good_crop(img):
                            if args.keep_size:
                                img_to_save = img
                            else:
                                img_to_save = ImageOps.pad(img, (target_size, target_size), color=(0, 0, 0))
                            
                            save_name = f"{basename}_full.jpg"
                            img_to_save.save(os.path.join(class_out_dir, save_name), quality=90)
                            count_full += 1
                        else:
                            # Both crops and full image failed quality checks
                            count_discarded += 1
                else:
                    count_crop += len(crops)

                # Save collected crops (from XML or Auto)
                if len(crops) > 0:
                     for i, crop in enumerate(crops):
                        if args.keep_size:
                            crop_to_save = crop
                        else:
                            crop_to_save = ImageOps.pad(crop, (target_size, target_size), color=(0, 0, 0))
                            
                        # Use _auto suffix if it came from auto crop to distinguish
                        suffix = "auto" if (len(crops)==1 and count_auto > 0 and "_full" not in filename) else f"crop_{i}" 
                        
                        save_name = f"{basename}_{suffix}.jpg"
                        crop_to_save.save(os.path.join(class_out_dir, save_name), quality=90)
                        
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                count_err += 1

    print("Preprocessing Complete!")
    print(f"Total XML Crops saved: {count_crop}")
    print(f"Total Auto Crops saved: {count_auto}")
    print(f"Total Full Images saved: {count_full}")
    print(f"Total Discarded (Too small/narrow/blurry): {count_discarded}")
    print(f"Errors: {count_err}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input root path containing images/ and annotations/')
    parser.add_argument('--output', type=str, required=True, help='Output root path')
    parser.add_argument('--size', type=int, default=256, help='Target size for resize (default: 256)')
    parser.add_argument('--keep-size', action='store_true', help='Skip padding/resizing and save crop at original size')
    parser.add_argument('--split-file', type=str, default=None, help='Path to split.json (optional)')
    parser.add_argument('--split-name', type=str, default='train', help='Key in split.json to filter (default: train)')
    args = parser.parse_args()
    preprocess(args)
