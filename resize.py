import os
import argparse
from PIL import Image
from tqdm import tqdm

def is_image_file(filename):
    """Check if file is an image"""
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jfif')
    return filename.lower().endswith(IMG_EXTENSIONS)


def collect_image_files(class_path):
    """Collect image files recursively under one class folder."""
    image_paths = []
    for dirpath, _, filenames in os.walk(class_path):
        for filename in filenames:
            if is_image_file(filename):
                image_paths.append(os.path.join(dirpath, filename))
    return sorted(image_paths)

def resize_dataset(root_path, output_path, target_size=224):
    """
    Resize all images to target size (224x224)
    
    Args:
        root_path: Path to root folder containing class folders
        output_path: Path to save resized images
        target_size: Target size (default 224)
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all class folders
    classes = sorted([c for c in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, c))])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Process each class
    total_images = 0
    total_failed = 0
    failed_logs = []
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(root_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        
        # Get all images in this class
        image_files = collect_image_files(class_path)
        
        print(f"\n[{class_idx+1}/{len(classes)}] Processing class '{class_name}' with {len(image_files)} images")
        
        # Process each image
        class_success = 0
        class_failed = 0
        for img_path in tqdm(image_files, desc=class_name, leave=False):
            try:
                # Open image
                img = Image.open(img_path).convert('RGB')
                
                # Resize to target size
                img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                
                # Preserve class subfolder structure to avoid overwriting files with same name.
                rel_path = os.path.relpath(img_path, class_path)
                output_file = os.path.join(class_output_path, rel_path)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                img_resized.save(output_file, quality=95)
                
                total_images += 1
                class_success += 1
                
            except Exception as e:
                class_failed += 1
                total_failed += 1
                failed_logs.append((class_name, img_path, str(e)))
                print(f"  Error processing {img_path}: {e}")
        
        print(
            f"✓ Class '{class_name}': found={len(image_files)}, "
            f"success={class_success}, failed={class_failed}"
        )

    if failed_logs:
        os.makedirs(output_path, exist_ok=True)
        error_log_path = os.path.join(output_path, 'resize_errors.txt')
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for class_name, img_path, err in failed_logs:
                f.write(f"[{class_name}] {img_path} :: {err}\n")
        print(f"⚠ Saved failure list to: {error_log_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Resizing complete!")
    print(f"✓ Total images resized: {total_images}")
    print(f"✓ Total images failed: {total_failed}")
    print(f"✓ Output directory: {output_path}")
    print(f"✓ Target size: {target_size}×{target_size}")
    print(f"{'='*60}")

def get_size_statistics(root_path):
    """Get statistics about input image sizes"""
    print(f"\n{'='*60}")
    print("Input image size statistics:")
    print(f"{'='*60}")
    
    classes = sorted([c for c in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, c))])
    
    all_sizes = []
    for class_name in classes:
        class_path = os.path.join(root_path, class_name)
        image_files = collect_image_files(class_path)
        
        class_sizes = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                class_sizes.append(img.size)
                all_sizes.append(img.size)
            except:
                pass
        
        if class_sizes:
            widths = [s[0] for s in class_sizes]
            heights = [s[1] for s in class_sizes]
            print(f"{class_name}: W={min(widths)}-{max(widths)}, H={min(heights)}-{max(heights)} (count={len(class_sizes)})")
    
    # Overall statistics
    if all_sizes:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        print(f"\nOverall: W={min(widths)}-{max(widths)} (avg={sum(widths)//len(widths)})")
        print(f"         H={min(heights)}-{max(heights)} (avg={sum(heights)//len(heights)})")
        print(f"Performance: {(widths.count(224) if 224 in widths else 0)} images already 224×224")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize images to 224*224')
    parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/Do_an_Data/tlu-states/images',
                       help='Path to root folder containing class folders')
    parser.add_argument('--output_path', type=str, default='/content/resize_image',
                       help='Path to save resized images')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target image size (default 224)')
    parser.add_argument('--stats', action='store_true',
                       help='Show size statistics before resizing')
    
    args = parser.parse_args()
    
    # Check if root path exists
    if os.path.exists(args.root_path):
        # Show statistics if requested
        if args.stats:
            get_size_statistics(args.root_path)
        
        # Resize dataset
        resize_dataset(args.root_path, args.output_path, args.target_size)
    else:
        print(f"Error: Root path '{args.root_path}' does not exist!")
