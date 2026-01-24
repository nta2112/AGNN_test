import os
import matplotlib.pyplot as plt
import glob

def visualize_dataset_distribution(root_path):
    print(f"Scanning directory: {root_path}")
    
    if not os.path.exists(root_path):
        print(f"Error: Directory '{root_path}' does not exist.")
        return

    # Dictionary to store count of images per class
    class_counts = {}
    
    # Get all subdirectories (assuming each folder is a class)
    classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    classes.sort()  # Sort classes alphabetically

    if not classes:
        print("No class folders found in the specified directory.")
        return

    print(f"Found {len(classes)} classes.")

    for class_name in classes:
        class_dir = os.path.join(root_path, class_name)
        # Count all files in the directory (you can filter by extension if needed, e.g., *.jpg, *.png)
        # Here we count all files for simplicity
        num_images = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        class_counts[class_name] = num_images

    # Preparing data for plotting
    names = list(class_counts.keys())
    values = list(class_counts.values())

    # Plotting
    plt.figure(figsize=(15, 8)) # Make the figure large enough to read labels
    plt.bar(names, values, color='skyblue')
    
    plt.xlabel('Class Names', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Dataset Distribution per Class', fontsize=16)
    
    # Rotate x-axis labels if there are many classes or long names
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    
    # Save the plot
    save_path = 'dataset_distribution.png'
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")
    
    # Show plot (optional, might not work well in headless colab environment without specific setup, but saving works)
    # plt.show()

if __name__ == "__main__":
    # USER SPECIFIED PATH
    DATA_PATH = '/content/drive/MyDrive/tlu-states/images'
    visualize_dataset_distribution(DATA_PATH)
