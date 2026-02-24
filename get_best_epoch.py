import argparse
import os
import torch

def get_best_epoch(save_dir):
    """
    Reads the max-va.pth file from the given directory to find the best epoch.
    Fallback to checking trlog.pth if max-va.pth doesn't have epoch info.
    """
    if not os.path.isdir(save_dir):
        print(f"Error: Directory '{save_dir}' does not exist.")
        return

    max_va_path = os.path.join(save_dir, 'max-va.pth')
    if os.path.isfile(max_va_path):
        try:
            # Load checkpoint on CPU
            checkpoint = torch.load(max_va_path, map_location='cpu')
            
            best_epoch = None
            max_val_acc = None

            # Check standard saving format from train_meta_custom.py
            if 'training' in checkpoint and 'epoch' in checkpoint['training']:
                best_epoch = checkpoint['training']['epoch']
            elif 'epoch' in checkpoint:
                best_epoch = checkpoint['epoch']

            if best_epoch is not None:
                print(f"[{save_dir}] Best Epoch: {best_epoch}")
                return
        except Exception as e:
            print(f"Warning: Could not read epoch from max-va.pth: {e}")

    # Fallback to trlog.pth
    trlog_path = os.path.join(save_dir, 'trlog.pth')
    if os.path.isfile(trlog_path):
        try:
            trlog = torch.load(trlog_path, map_location='cpu')
            if 'va' in trlog and len(trlog['va']) > 0:
                # Find the epoch with the maximum validation accuracy (1-indexed)
                val_accs = trlog['va']
                best_epoch_idx = val_accs.index(max(val_accs))
                best_epoch = best_epoch_idx + 1 # Epochs usually start at 1
                max_val_acc = val_accs[best_epoch_idx]
                
                print(f"[{save_dir}] Best Epoch: {best_epoch} (Val Acc: {max_val_acc:.4f})")
                return
        except Exception as e:
            print(f"Warning: Could not read trlog.pth: {e}")
            
    print(f"[{save_dir}] Cannot determine best epoch. Make sure max-va.pth or trlog.pth exists and is valid.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get the best training epoch from a save directory.")
    parser.add_argument('save_dir', type=str, help='Path to the training save directory (e.g., ./save/meta_baseline_gnn)')
    args = parser.parse_args()

    get_best_epoch(args.save_dir)
