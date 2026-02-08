import os
import matplotlib.pyplot as plt
import datasets 
from datasets import image_folder_crop
from PIL import Image
import torch
import numpy as np

def visualize_crops(root_path, num_samples=5):
    print(f"Visualizing crops from: {root_path}")
    
    # 1. Instantiate the dataset
    # Note: We use image_size=224 for standard size, but for visualization of "raw" crop
    # we might want to see the original crop content. 
    # However, the dataset transform resizes it. 
    # To see "before" and "after", we need to manually access the sample info.
    
    try:
        dataset = datasets.make('image-folder-crop', root_path=root_path, image_size=224)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Check paths.")
        return

    # Randomly select samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i in indices:
        image_path, xmin, ymin, xmax, ymax, label = dataset.samples[i]
        
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Draw bounding box on original image (copy for viz)
        import cv2
        img_np = np.array(original_img)
        # cv2 rectangle takes (x1, y1), (x2, y2), color, thickness
        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
        
        # Get processed crop from dataset (with transforms applied)
        crop_tensor, _ = dataset[i]
        
        # Denormalize for visualization
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_viz = crop_tensor * std + mean
        crop_viz = crop_viz.permute(1, 2, 0).numpy()
        crop_viz = np.clip(crop_viz, 0, 1)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Image (Box: {xmin},{ymin} to {xmax},{ymax})")
        axes[0].axis('off')
        
        axes[1].imshow(crop_viz)
        axes[1].set_title("Processed Crop (224x224)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        print("-" * 50)

if __name__ == "__main__":
    # Colab typical path after unzip
    # ROOT_PATH = '/content/data/tlu-states/images' 
    # Local Windows path for testing
    ROOT_PATH = r'c:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\tlu-states\images'
    
    # Adjust path if running on Colab
    if os.path.exists('/content/data/tlu-states/images'):
        ROOT_PATH = '/content/data/tlu-states/images'
        
    visualize_crops(ROOT_PATH)
