import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random # Added for random sampling

import models
import utils
import utils.few_shot as fs

def load_image(path, transform):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='AGNN Custom Prediction Script')
    parser.add_argument('--config', default='./configs/train_meta_agnn_resnet50.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--support-path', required=True, help='Path to support set folder (can be the full dataset root)')
    parser.add_argument('--query-path', required=True, help='Path to query images folder (containing images to predict)')
    parser.add_argument('--n-shot', type=int, default=5, help='Number of support images to randomly select per class')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()

    # Setup Setup
    utils.set_gpu(args.gpu)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    # Override parallel to avoid DataParallel issues in inference if running single GPU
    config['_parallel'] = False 
    
    # 1. Setup Transform (same as validation)
    image_size = config.get('model_args', {}).get('encoder_args', {}).get('image_size', 224)
    # config might use 'image_size' in dataset args too, let's stick to 224 or what's in config
    if 'val_dataset_args' in config and 'image_size' in config['val_dataset_args']:
        image_size = config['val_dataset_args']['image_size']
        
    print(f"Using image size: {image_size}")
    
    norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)), # Resize to keep ratio roughly
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(**norm_params)
    ])

    # 2. Load Support Set
    print(f"Loading Support Set from {args.support_path}...")
    print(f"Randomly sampling {args.n_shot} shots per class.")
    
    class_names = sorted([d for d in os.listdir(args.support_path) if os.path.isdir(os.path.join(args.support_path, d))])
    
    n_way = len(class_names)
    if n_way == 0:
        print("Error: No class folders found in support path.")
        return

    n_shot = args.n_shot
    x_shot_tensor = torch.zeros(1, n_way, n_shot, 3, image_size, image_size)
    
    print(f"Found {n_way} classes: {class_names}")
    
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(args.support_path, class_name)
        img_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        if len(img_files) < n_shot:
            print(f"Warning: Class {class_name} has only {len(img_files)} images. Duplicating to fill {n_shot}.")
            # Simple replacement sampling if not enough images
            selected_files = np.random.choice(img_files, n_shot, replace=True)
        else:
            selected_files = random.sample(img_files, n_shot)
            
        for j, img_file in enumerate(selected_files):
            img_path = os.path.join(class_dir, img_file)
            img = load_image(img_path, transform)
            if img is not None:
                x_shot_tensor[0, i, j] = img
            else:
                print(f"Warning: Failed to load {img_path}")

    # 3. Load Query Set
    print("Loading Query Set...")
    query_files = sorted([f for f in os.listdir(args.query_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(query_files) == 0:
        print("Error: No query images found.")
        return
        
    x_query_tensor = torch.stack([load_image(os.path.join(args.query_path, f), transform) for f in query_files])
    x_query_tensor = x_query_tensor.unsqueeze(0) # (1, N_query, C, H, W)
    
    n_query = x_query_tensor.shape[1]
    
    # 4. Load Model
    print("Loading Model...")
    
    # Inject n_way into config for model creation (AGNN needs this)
    config['model_args']['n_way'] = n_way
    
    model = models.make(config['model'], **config['model_args'])
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model_sd' in checkpoint:
        model_sd = checkpoint['model_sd']
    else:
        model_sd = checkpoint
    
    # Handle parallel trained weights loaded into non-parallel model
    if list(model_sd.keys())[0].startswith('module.'):
        model_sd = {k[len('module.'):]: v for k, v in model_sd.items()}
        
    model.load_state_dict(model_sd, strict=False) # strict=False to be safe with partial loads/legacy
    model.cuda()
    model.eval()
    
    # 5. Inference
    x_shot = x_shot_tensor.cuda()
    x_query = x_query_tensor.cuda()
    
    # Create tr_label: 0, 1, 2... for support
    # (n_way * n_shot) flat labels
    tr_label = torch.arange(n_way).repeat_interleave(n_shot).cuda() # [0,0... 1,1...]
    
    with torch.no_grad():
        logits = model(x_shot, x_query, tr_label) 
        # Output shape: (1, n_query, n_way)
        
        probs = F.softmax(logits, dim=2)
        preds = torch.argmax(probs, dim=2).squeeze(0) # (n_query)
        probs = probs.squeeze(0)

    # 6. Output
    print("-" * 60)
    print(f"{'Image Name':<30} | {'Prediction':<20} | {'Conf':<10}")
    print("-" * 60)
    
    for i, f in enumerate(query_files):
        pred_idx = preds[i].item()
        confidence = probs[i, pred_idx].item()
        print(f"{f:<30} | {class_names[pred_idx]:<20} | {confidence:.4f}")

if __name__ == '__main__':
    main()
