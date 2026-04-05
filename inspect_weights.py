"""
inspect_weights.py
------------------
Phân tích trọng số của checkpoint .pth
- In cấu trúc checkpoint
- Phân tích phân phối (mean, std, min, max) của từng layer
- Tách riêng encoder (ResNet50) vs GNN layers
- Vẽ histogram so sánh nếu có matplotlib
"""

import torch
import numpy as np
import sys
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"C:\Users\HP\Downloads\max-va.pth"
TOP_N_LAYERS = 20          # Số layer muốn hiển thị chi tiết nhất
HIST_BINS    = 50          # Số bin histogram
SAVE_PLOT    = True        # Lưu ảnh histogram ra file
PLOT_OUT     = r"C:\Users\HP\Downloads\weight_analysis.png"
# ──────────────────────────────────────────────────────────────────────────────


def load_checkpoint(path):
    print(f"\n{'='*60}")
    print(f" LOAD CHECKPOINT: {os.path.basename(path)}")
    print(f"{'='*60}")
    
    # Thử load bình thường (zip/modern format)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"  [OK] Loaded via torch.load (modern zip format)")
    except RuntimeError as e:
        if 'failed finding central directory' in str(e) or 'PytorchStreamReader' in str(e):
            # File lưu theo format pickle cũ — dùng pickle trực tiếp
            print(f"  [!] Zip format thất bại, thử legacy pickle format...")
            import pickle
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            print(f"  [OK] Loaded via pickle (legacy format)")
        else:
            raise
    
    print(f"  Keys trong checkpoint: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    
    # Thông tin meta
    if 'training' in ckpt:
        t = ckpt['training']
        print(f"  Epoch đã lưu      : {t.get('epoch', 'N/A')}")
        print(f"  Optimizer         : {t.get('optimizer', 'N/A')}")
        opt_args = t.get('optimizer_args', {})
        print(f"  LR                : {opt_args.get('lr', 'N/A')}")
    if 'model' in ckpt:
        print(f"  Model type        : {ckpt['model']}")
    if 'model_args' in ckpt:
        print(f"  Model args        : {ckpt['model_args']}")
    
    return ckpt


def analyze_state_dict(state_dict):
    """Phân tích từng tensor trong state_dict."""
    print(f"\n{'='*60}")
    print(f" PHÂN TÍCH TRỌNG SỐ ({len(state_dict)} tensors)")
    print(f"{'='*60}")
    
    encoder_layers = {}
    gnn_layers     = {}
    other_layers   = {}
    
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue  # Bỏ qua buffer integer (ví dụ: num_batches_tracked)
        
        t = tensor.float()
        info = {
            'shape': tuple(t.shape),
            'numel': t.numel(),
            'mean' : t.mean().item(),
            'std'  : t.std().item(),
            'min'  : t.min().item(),
            'max'  : t.max().item(),
            'abs_mean': t.abs().mean().item(),
        }
        
        # Phân loại layer
        if name.startswith('encoder'):
            encoder_layers[name] = info
        elif any(k in name for k in ['gnn', 'graph', 'edge', 'node', 'agg', 'fc', 'att', 'linear']):
            gnn_layers[name] = info
        else:
            other_layers[name] = info
    
    return encoder_layers, gnn_layers, other_layers


def print_layer_table(layers_dict, title, top_n=None):
    if not layers_dict:
        return
    print(f"\n{'─'*80}")
    print(f"  {title}  ({len(layers_dict)} layers)")
    print(f"{'─'*80}")
    print(f"  {'Layer Name':<45} {'Shape':<20} {'Mean':>8} {'Std':>8} {'AbsMean':>8}")
    print(f"  {'─'*45} {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    
    items = list(layers_dict.items())
    if top_n:
        items = items[:top_n]
    
    for name, info in items:
        shape_str = str(info['shape'])
        # Rút gọn tên nếu quá dài
        disp_name = name if len(name) <= 44 else '...' + name[-41:]
        print(f"  {disp_name:<45} {shape_str:<20} {info['mean']:>8.4f} {info['std']:>8.4f} {info['abs_mean']:>8.4f}")
    
    if top_n and len(layers_dict) > top_n:
        print(f"  ... (còn {len(layers_dict) - top_n} layers nữa)")


def print_summary(encoder_layers, gnn_layers, other_layers):
    """In tổng kết theo nhóm."""
    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT THEO NHÓM")
    print(f"{'='*60}")
    
    for group_name, layers in [
        ('🔵 ENCODER (ResNet backbone - ImageNet pretrained)', encoder_layers),
        ('🟠 GNN / HEAD LAYERS (khởi tạo ngẫu nhiên)', gnn_layers),
        ('⚪ OTHER LAYERS', other_layers),
    ]:
        if not layers:
            continue
        all_tensors = []
        total_params = 0
        for info in layers.values():
            total_params += info['numel']
        
        all_vals = []
        for name, info in layers.items():
            all_vals.append(info['mean'])
        
        means = [i['mean'] for i in layers.values()]
        stds  = [i['std']  for i in layers.values()]
        
        print(f"\n  {group_name}")
        print(f"    Số layers    : {len(layers)}")
        print(f"    Tổng params  : {total_params:,}")
        print(f"    Mean trung bình: {np.mean(means):.6f}  (lý tưởng ~ 0)")
        print(f"    Std trung bình : {np.mean(stds):.6f}")
        print(f"    Std min/max    : {np.min(stds):.6f} / {np.max(stds):.6f}")


def plot_histograms(state_dict, save_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\n  [INFO] matplotlib không có, bỏ qua vẽ biểu đồ.")
        print("         Cài đặt: pip install matplotlib")
        return
    
    # Tách tensor encoder và GNN
    encoder_vals = []
    gnn_vals     = []
    
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        vals = tensor.float().numpy().flatten()
        if name.startswith('encoder'):
            encoder_vals.append(vals)
        else:
            gnn_vals.append(vals)
    
    encoder_all = np.concatenate(encoder_vals) if encoder_vals else np.array([])
    gnn_all     = np.concatenate(gnn_vals)     if gnn_vals     else np.array([])
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Weight Distribution Analysis\n(Best Model - max-va.pth)', 
                 fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # ── Plot 1: Encoder weights histogram
    ax1 = fig.add_subplot(gs[0, 0])
    if len(encoder_all) > 0:
        ax1.hist(encoder_all, bins=HIST_BINS, color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.3)
        ax1.set_title(f'Encoder (ResNet50) Weights\nmean={encoder_all.mean():.4f}, std={encoder_all.std():.4f}')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Count')
        ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, label='zero')
        ax1.legend()
    
    # ── Plot 2: GNN weights histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if len(gnn_all) > 0:
        ax2.hist(gnn_all, bins=HIST_BINS, color='darkorange', alpha=0.8, edgecolor='white', linewidth=0.3)
        ax2.set_title(f'GNN / Head Layers Weights\nmean={gnn_all.mean():.4f}, std={gnn_all.std():.4f}')
        ax2.set_xlabel('Weight Value')
        ax2.set_ylabel('Count')
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='zero')
        ax2.legend()
    
    # ── Plot 3: Std per layer (GNN)
    ax3 = fig.add_subplot(gs[1, :])
    
    layer_names = []
    layer_stds  = []
    layer_colors = []
    
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point() or tensor.numel() < 4:
            continue
        short = name if len(name) <= 30 else '...' + name[-27:]
        layer_names.append(short)
        layer_stds.append(tensor.float().std().item())
        layer_colors.append('steelblue' if name.startswith('encoder') else 'darkorange')
    
    y = np.arange(len(layer_names))
    bars = ax3.barh(y, layer_stds, color=layer_colors, alpha=0.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels(layer_names, fontsize=6)
    ax3.set_xlabel('Weight Std (độ phân tán)')
    ax3.set_title('Std của từng Layer  (🔵 Encoder | 🟠 GNN/Head)')
    ax3.axvline(0.01, color='green', linestyle='--', linewidth=1, label='std=0.01')
    ax3.axvline(0.1,  color='red',   linestyle='--', linewidth=1, label='std=0.1')
    ax3.legend()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Đã lưu biểu đồ: {save_path}")
    
    try:
        plt.show()
    except Exception:
        pass


def find_suspicious_layers(state_dict):
    """Tìm các layer có dấu hiệu bất thường."""
    print(f"\n{'='*60}")
    print(f"  KIỂM TRA LAYER BẤT THƯỜNG")
    print(f"{'='*60}")
    
    warnings = []
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point() or tensor.numel() < 4:
            continue
        t = tensor.float()
        std  = t.std().item()
        mean = t.mean().item()
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        
        if has_nan:
            warnings.append(f"  ⛔ NaN detected        : {name}")
        if has_inf:
            warnings.append(f"  ⛔ Inf detected        : {name}")
        if std < 1e-6:
            warnings.append(f"  ⚠️  Std quá nhỏ (<1e-6): {name}  std={std:.2e}  (có thể frozen/dead)")
        if std > 2.0:
            warnings.append(f"  ⚠️  Std quá lớn (>2.0) : {name}  std={std:.4f}  (exploding?)")
        if abs(mean) > 1.0:
            warnings.append(f"  ⚠️  Mean lệch xa 0     : {name}  mean={mean:.4f}")
    
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  ✅ Không phát hiện layer bất thường!")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Không tìm thấy file: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    # 1. Load checkpoint
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    
    # 2. Lấy state_dict
    if 'model_sd' in ckpt:
        state_dict = ckpt['model_sd']
        print(f"\n  [OK] Dùng key 'model_sd' — {len(state_dict)} tensors")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"\n  [OK] Dùng key 'state_dict' — {len(state_dict)} tensors")
    else:
        # Checkpoint chính là state_dict
        state_dict = ckpt
        print(f"\n  [OK] Checkpoint là state_dict thẳng — {len(state_dict)} tensors")
    
    # 3. Phân tích
    encoder_layers, gnn_layers, other_layers = analyze_state_dict(state_dict)
    
    # 4. In bảng chi tiết
    print_layer_table(encoder_layers, "🔵 ENCODER LAYERS (ResNet50 backbone)", top_n=TOP_N_LAYERS)
    print_layer_table(gnn_layers,     "🟠 GNN / HEAD LAYERS",                  top_n=TOP_N_LAYERS)
    print_layer_table(other_layers,   "⚪ OTHER LAYERS",                         top_n=TOP_N_LAYERS)
    
    # 5. Tổng kết
    print_summary(encoder_layers, gnn_layers, other_layers)
    
    # 6. Kiểm tra bất thường
    find_suspicious_layers(state_dict)
    
    # 7. Vẽ biểu đồ
    if SAVE_PLOT:
        plot_histograms(state_dict, PLOT_OUT)
    
    print(f"\n{'='*60}")
    print("  PHÂN TÍCH HOÀN TẤT")
    print(f"{'='*60}\n")
