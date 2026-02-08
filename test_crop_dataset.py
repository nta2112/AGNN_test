import os
import datasets
from datasets import image_folder_crop
from torchvision.utils import save_image
import torch
from collections import Counter

def test_dataset():
    # Setup path - typical path in this project
    root_path = os.path.abspath('data/tlu-states/images')
    
    print(f"Testing ImageFolderCrop with root: {root_path}")
    
    try:
        # Default run
        print("\n--- Test 1: Normal Loading ---")
        dataset = datasets.make('image-folder-crop', root_path=root_path, image_size=224)
        print(f"Dataset created successfully.")
        print(f"Total samples: {len(dataset)}")
        
        # Test Balancing
        print("\n--- Test 2: Balancing (max 10 samples per class) ---")
        dataset_balanced = datasets.make('image-folder-crop', root_path=root_path, image_size=224, max_samples_per_class=10)
        print(f"Balanced Dataset created successfully.")
        print(f"Total samples: {len(dataset_balanced)}")
        
        # Verify counts
        labels = [s[-1] for s in dataset_balanced.samples]
        counts = Counter(labels)
        print("Counts per class (should be <= 10):")
        print(dict(sorted(counts.items())))
        
        max_count = max(counts.values()) if counts else 0
        if max_count <= 10:
            print("SUCCESS: Balancing logic works!")
        else:
            print(f"FAILURE: Found class with {max_count} samples > 10")

    except Exception as e:
        print(f"Failed to verify dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataset()
