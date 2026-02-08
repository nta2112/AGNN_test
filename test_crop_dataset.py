import os
import datasets
from datasets import image_folder_crop
from torchvision.utils import save_image
import torch

def test_dataset():
    # Setup path - typical path in this project
    root_path = os.path.abspath('data/tlu-states/images')
    
    print(f"Testing ImageFolderCrop with root: {root_path}")
    
    try:
        # Instantiate dataset
        dataset = datasets.make('image-folder-crop', root_path=root_path, image_size=224)
        
        print(f"Dataset created successfully.")
        print(f"Number of classes: {dataset.n_classes}")
        print(f"Number of samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("Error: No samples found! Check paths.")
            return

        # Create output directory for visualization
        os.makedirs('test_crops', exist_ok=True)
        
        # Save first 20 samples
        print("Saving first 20 cropped samples to 'test_crops/'...")
        for i in range(min(20, len(dataset))):
            img, label = dataset[i]
            # img is float tensor normalized. Unnormalize probably needed for good viz, 
            # but for checking content, saving normalized is often visible enough or we can denormalize.
            # Let's just save simple normalized (might be dark/bright) or simple un-normalized.
            
            # Simple denormalization for visualization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # img = (img * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            viz_img = img * std + mean
            
            save_image(viz_img, f'test_crops/sample_{i}_label_{label}.jpg')
            
        print("Done. Check 'test_crops' folder.")
        
    except Exception as e:
        print(f"Failed to verify dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataset()
