import os
import json

def generate_split(root_path, threshold=100, save_path='split.json'):
    classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    counts = {}
    for c in classes:
        counts[c] = len(os.listdir(os.path.join(root_path, c)))
    
    base_classes = sorted([c for c, n in counts.items() if n >= threshold])
    novel_classes = sorted([c for c, n in counts.items() if n < threshold])
    
    # Standard format often uses 'train', 'val', 'test'
    # For now, we put all novel classes in BOTH val and test so we can test on them.
    # Or split novel into val/test if enough classes. 
    # With 15 novel classes, let's just use all 15 for 'test' (and 'val' for monitoring).
    
    split_dict = {
        "train": base_classes,
        "val": novel_classes,
        "test": novel_classes
    }
    
    with open(save_path, 'w') as f:
        json.dump(split_dict, f, indent=4)
    
    print(f"Generated {save_path}")
    print(f"Train (Base): {len(base_classes)} classes")
    print(f"Val/Test (Novel): {len(novel_classes)} classes")

if __name__ == "__main__":
    root = r"c:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\tlu-states\images" 
    generate_split(root, threshold=100, save_path=r"c:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\split.json")
