import re
import os
import sys

def parse_log_file(log_path):
    """
    Parses the log.txt file to extract metrics for each epoch.
    Returns a list of dictionaries.
    """
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return []

    data = []
    # Regex to match the log line format:
    # epoch 1, train 0.8293|0.6480, tval 1.3893|0.4532, val 1.3893|0.4532, ...
    # structure: epoch {epoch}, train {loss}|{acc}, tval {loss}|{acc}, val {loss}|{acc}
    pattern = re.compile(r'epoch\s+(\d+),\s+train\s+([\d\.]+)\|([\d\.]+),\s+tval\s+([\d\.]+)\|([\d\.]+),\s+val\s+([\d\.]+)\|([\d\.]+)')

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_acc = float(match.group(3))
                tval_loss = float(match.group(4))
                tval_acc = float(match.group(5))
                val_loss = float(match.group(6))
                val_acc = float(match.group(7))

                data.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'tval_loss': tval_loss,
                    'tval_acc': tval_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                })
    return data

def analyze_best_epoch(log_dir):
    log_path = os.path.join(log_dir, 'log.txt')
    
    # Check if log.txt exists, if not try to find it in the dir
    if not os.path.exists(log_path):
        # Fallback: check if user provided the direct path to log.txt
        if log_dir.endswith('.txt') and os.path.exists(log_dir):
            log_path = log_dir
        else:
            print(f"Could not find log.txt in {log_dir}")
            return

    print(f"Analyzing log file: {log_path}")
    data = parse_log_file(log_path)
    
    if not data:
        print("No training data found in log.txt")
        return

    # Find best epoch based on Validation Accuracy (val_acc)
    # You can change this to 'tval_acc' if you prefer meta-validation accuracy
    best_epoch_data = max(data, key=lambda x: x['val_acc'])
    
    print("=" * 40)
    print(f"KẾT QUẢ TỐT NHẤT TẠI EPOCH: {best_epoch_data['epoch']}")
    print("=" * 40)
    
    metrics_map = {
        'train_loss': 'Train Loss',
        'train_acc': 'Train Accuracy',
        'tval_loss': 'TVal Loss',
        'tval_acc': 'TVal Accuracy',
        'val_loss': 'Val Loss',
        'val_acc': 'Val Accuracy'
    }
    
    for key, name in metrics_map.items():
        print(f"{name:<20}: {best_epoch_data[key]:.4f}")
            
    print("=" * 40)

if __name__ == '__main__':
    # Default usage: current directory
    log_directory = '.'
    
    # Colab usage example:
    # log_directory = '/content/drive/MyDrive/AGNN/save/meta_tlu-30shot_gnn-resnet12'
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='Path to log folder or log.txt file')
    args = parser.parse_args()
    
    analyze_best_epoch(args.path)
