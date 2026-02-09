import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    classes = sorted(os.listdir(images_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(images_dir, c))]
    
    print(f"Found {len(classes)} classes in {images_dir}.")
    
    count_crop = 0
    count_full = 0
    count_err = 0
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_img_dir = os.path.join(images_dir, class_name)
        # Handle missing annotation folder gracefully
        class_xml_dir = os.path.join(annotations_dir, class_name)
        
        class_out_dir = os.path.join(output_dir, class_name)
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
                                    crops.append(crop)
                                    has_object = True
                        
                        if not has_object:
                            # valid XML but no object -> Full image
                            pass 
                            
                    except Exception as e:
                        print(f"Error parsing XML {xml_path}: {e}")
                
                # Fallback or Save
                if len(crops) == 0:
                    # No XML or no objects found -> Use full image
                    img_resized = img.resize((target_size, target_size))
                    save_name = f"{basename}_full.jpg"
                    img_resized.save(os.path.join(class_out_dir, save_name), quality=90)
                    count_full += 1
                else:
                    # Save crops
                    for i, crop in enumerate(crops):
                        crop_resized = crop.resize((target_size, target_size))
                        save_name = f"{basename}_crop_{i}.jpg"
                        crop_resized.save(os.path.join(class_out_dir, save_name), quality=90)
                        count_crop += 1
                        
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                count_err += 1

    print("Preprocessing Complete!")
    print(f"Total Crops: {count_crop}")
    print(f"Total Full Images (Fallback): {count_full}")
    print(f"Errors: {count_err}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input root path containing images/ and annotations/')
    parser.add_argument('--output', type=str, required=True, help='Output root path')
    parser.add_argument('--size', type=int, default=256, help='Target size for resize (default: 256)')
    args = parser.parse_args()
    preprocess(args)
