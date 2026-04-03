import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

def is_image_file(filename):
    """Check if file is an image"""
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(IMG_EXTENSIONS)

def create_preprocessor(image_size=224, box_size=256):
    """Create preprocessing pipeline (same as ImageFolderCustom without augment)"""
    norm_params = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
    normalize = transforms.Normalize(**norm_params)
    
    transform = transforms.Compose([
        transforms.Resize(box_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return transform

def preprocess_dataset(root_path, output_path, image_size=224, box_size=256):
    """
    Preprocess all images in folders and save them
    
    Args:
        root_path: Path to root folder containing class folders
        output_path: Path to save preprocessed data
        image_size: Target image size (default 224)
        box_size: Box size for resize before crop (default 256)
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all class folders
    classes = sorted([c for c in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, c))])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create preprocessor
    transform = create_preprocessor(image_size, box_size)
    
    # Process each class
    total_images = 0
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(root_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        
        # Get all images in this class
        image_files = sorted([f for f in os.listdir(class_path) 
                            if is_image_file(f)])
        
        print(f"\n[{class_idx+1}/{len(classes)}] Processing class '{class_name}' with {len(image_files)} images")
        
        # Process each image
        for img_idx, img_file in enumerate(tqdm(image_files, desc=class_name, leave=False)):
            try:
                img_path = os.path.join(class_path, img_file)
                
                # Open and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                
                # Save as tensor (.pt file)
                output_name = os.path.splitext(img_file)[0] + '.pt'
                output_file = os.path.join(class_output_path, output_name)
                torch.save(img_tensor, output_file)
                
                total_images += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"✓ Processed {len(image_files)} images from '{class_name}'")
    
    print(f"\n{'='*60}")
    print(f"✓ Preprocessing complete!")
    print(f"✓ Total images processed: {total_images}")
    print(f"✓ Output directory: {output_path}")
    print(f"✓ Image size: {image_size}×{image_size}")
    print(f"{'='*60}")

def load_and_verify(output_path, num_samples=3):
    """Load and verify preprocessed data"""
    print(f"\n{'='*60}")
    print("Verifying preprocessed data...")
    print(f"{'='*60}")
    
    classes = sorted([c for c in os.listdir(output_path) 
                     if os.path.isdir(os.path.join(output_path, c))])
    
    for class_name in classes[:min(len(classes), 5)]:  # Check first 5 classes
        class_path = os.path.join(output_path, class_name)
        tensor_files = sorted([f for f in os.listdir(class_path) if f.endswith('.pt')])
        
        print(f"\nClass: '{class_name}' - {len(tensor_files)} samples")
        
        # Load first few samples
        for tensor_file in tensor_files[:num_samples]:
            tensor_path = os.path.join(class_path, tensor_file)
            tensor = torch.load(tensor_path)
            print(f"  {tensor_file}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                  f"min={tensor.min():.4f}, max={tensor.max():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset offline')
    parser.add_argument('--root_path', type=str, default='./data/tlu-states/images',
                       help='Path to root folder containing class folders')
    parser.add_argument('--output_path', type=str, default='./preprocess_data',
                       help='Path to save preprocessed data')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Target image size (default 224)')
    parser.add_argument('--box_size', type=int, default=256,
                       help='Box size for resize before crop (default 256)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify preprocessed data after processing')
    
    args = parser.parse_args()
    
    # Preprocess dataset
    if os.path.exists(args.root_path):
        preprocess_dataset(args.root_path, args.output_path, 
                          args.image_size, args.box_size)
        
        # Verify if requested
        if args.verify:
            load_and_verify(args.output_path)
    else:
        print(f"Error: Root path '{args.root_path}' does not exist!")
