import os
import json

def analyze_split(root_path, thresholds=[100, 200, 300]):
    classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    counts = {}
    for c in classes:
        items = os.listdir(os.path.join(root_path, c))
        # Filter only images if needed, but simple count is likely enough
        counts[c] = len(items)
    
    print(f"Total Classes: {len(classes)}")
    print("-" * 30)
    
    for t in thresholds:
        base_classes = [c for c, n in counts.items() if n >= t]
        novel_classes = [c for c, n in counts.items() if n < t]
        print(f"Threshold >= {t} images:")
        print(f"  Base Classes (Train): {len(base_classes)}")
        print(f"  Novel Classes (Val/Test): {len(novel_classes)}")
        if len(base_classes) < 10:
            print(f"  WARNING: Base classes count ({len(base_classes)}) is very low for pre-training.")
        print("-" * 30)

    # Specific check for 300
    base_300 = [c for c, n in counts.items() if n >= 300]
    print(f"Classes with >= 300 images: {base_300}")

if __name__ == "__main__":
    # Adjust path as seen in list_dir output
    root = r"c:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\tlu-states\images" 
    analyze_split(root)
