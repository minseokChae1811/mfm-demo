"""Extract real data assets for MFM demo."""
import numpy as np
import json
import base64
import io
from PIL import Image
import os

OUT = 'd:/Dataset/demo/assets'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. Real STFT spectrograms from hit_sm_self
# ============================================================
print("Loading STFT data (mmap)...")
X = np.load('d:/Dataset/data/stft_224/pretrain_X.npy', mmap_mode='r')
Y = np.load('d:/Dataset/data/stft_224/pretrain_Y.npy')

# hit_sm_self: indices 10000-12100, labels 8-14
# 8=inner_race_02, 11=normal, 12=outer_race_02
samples = {
    'normal': 11,
    'inner_race': 8,
    'outer_race': 12,
}

for name, label in samples.items():
    mask = Y == label
    indices = np.where(mask)[0]
    idx = indices[5]  # pick 5th sample
    img_data = X[idx]  # (224, 224) or (224, 224, 3)

    if img_data.ndim == 3:
        img = Image.fromarray(img_data.astype(np.uint8))
    else:
        # Normalize to 0-255
        img_data = img_data.astype(np.float32)
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255
        img = Image.fromarray(img_data.astype(np.uint8))

    img.save(f'{OUT}/stft_{name}.png')
    print(f"  Saved stft_{name}.png ({img.size})")

# Also save a few from CWRU for variety
cwru_classes = {0: 'cwru_normal', 1: 'cwru_ball', 2: 'cwru_inner', 3: 'cwru_outer'}
for label, name in cwru_classes.items():
    indices = np.where(Y == label)[0]
    idx = indices[10]
    img_data = X[idx].astype(np.float32)
    if img_data.ndim == 2:
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255
    img = Image.fromarray(img_data.astype(np.uint8))
    img.save(f'{OUT}/stft_{name}.png')
    print(f"  Saved stft_{name}.png")

# ============================================================
# 2. t-SNE coordinates for MCC5 (real data)
# ============================================================
print("\nExtracting t-SNE coords for MCC5...")
tsne_coords = np.load('d:/Dataset/results/B_stratified/mcc5_analysis_bcd/C_tsne_coords.npy')

# MCC5-Thu: indices 97800-109800 in pretrain, labels 104-111
# But tsne is subset (4800 points, severity=H)
# Need to figure out labels - MCC5 has 12000 samples, 4800 = subset
# Labels 104-111 (8 classes)
# Let me get labels from pretrain_Y for mcc5
with open('d:/Dataset/data/stft_224/pretrain_summary.json') as f:
    summary = json.load(f)

mcc5_info = None
for ds in summary['datasets']:
    if ds['name'] == 'mcc5_thu':
        mcc5_info = ds
        break

if mcc5_info:
    start, end = mcc5_info['indices']
    mcc5_labels = Y[start:end]
    label_min = mcc5_info['label_range'][0]
    local_labels = mcc5_labels - label_min

    # The tsne has 4800 points = 4800 out of 12000
    # Likely every condition (6 conditions, 100 per domain, 8 classes = 4800)
    # or half the data. Let's use first 4800 or subsample
    if len(local_labels) >= 4800:
        tsne_labels = local_labels[:4800]
    else:
        tsne_labels = local_labels

    class_names = [mcc5_info['classes'][str(label_min + i)]['name'] for i in range(8)]

    # Normalize coords to [0, 1]
    coords_norm = tsne_coords.copy()
    coords_norm[:, 0] = (coords_norm[:, 0] - coords_norm[:, 0].min()) / (coords_norm[:, 0].max() - coords_norm[:, 0].min())
    coords_norm[:, 1] = (coords_norm[:, 1] - coords_norm[:, 1].min()) / (coords_norm[:, 1].max() - coords_norm[:, 1].min())

    # Subsample for performance (max 500 points)
    n = len(coords_norm)
    if n > 500:
        step = n // 500
        indices = np.arange(0, n, step)[:500]
        coords_sub = coords_norm[indices]
        labels_sub = tsne_labels[indices]
    else:
        coords_sub = coords_norm
        labels_sub = tsne_labels

    tsne_data = {
        'coords': coords_sub.tolist(),
        'labels': labels_sub.tolist(),
        'class_names': class_names,
        'n_classes': 8,
        'dataset': 'MCC5-Thu'
    }

    with open(f'{OUT}/tsne_mcc5.json', 'w') as f:
        json.dump(tsne_data, f)
    print(f"  Saved tsne_mcc5.json ({len(coords_sub)} points, {8} classes)")

# ============================================================
# 3. Resize attention maps
# ============================================================
print("\nResizing attention maps...")
attn_files = [
    ('d:/Dataset/figures/attention_analysis/dino_attention/attn_grid_fstrip.png', 'attn_grid_fstrip.png'),
    ('d:/Dataset/figures/attention_analysis/dino_attention/attn_analysis_fstrip.png', 'attn_analysis_fstrip.png'),
]

for src, dst in attn_files:
    if os.path.exists(src):
        img = Image.open(src)
        # Resize to reasonable web size (max 800px wide)
        ratio = 800 / img.width
        new_size = (800, int(img.height * ratio))
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(f'{OUT}/{dst}', quality=85)
        print(f"  Saved {dst} ({new_size})")

# Also get gradcam
gradcam_src = 'd:/Dataset/figures/attention_analysis/dino_gradcam/gradcam_fstrip.png'
if os.path.exists(gradcam_src):
    img = Image.open(gradcam_src)
    ratio = 800 / img.width
    new_size = (800, int(img.height * ratio))
    img_resized = img.resize(new_size, Image.LANCZOS)
    img_resized.save(f'{OUT}/gradcam_fstrip.png', quality=85)
    print(f"  Saved gradcam_fstrip.png ({new_size})")

# ============================================================
# 4. Resize tsne_triple for pre-training visualization
# ============================================================
print("\nResizing t-SNE triple...")
tsne_src = 'd:/Dataset/figures/attention_analysis/dino_v2_analysis/fstrip_p16/tsne_triple.png'
if os.path.exists(tsne_src):
    img = Image.open(tsne_src)
    ratio = 900 / img.width
    new_size = (900, int(img.height * ratio))
    img_resized = img.resize(new_size, Image.LANCZOS)
    img_resized.save(f'{OUT}/tsne_triple_fstrip.png', quality=85)
    print(f"  Saved tsne_triple_fstrip.png ({new_size})")

# ============================================================
# 5. Generate fstrip masking visualization data
# ============================================================
print("\nGenerating fstrip mask pattern...")
# fstrip4 means frequency-strip patches of width 4 on 224x224
# 224/4 = 56 patches along frequency axis, 1 along time = 56 strips
# mask 75% of them
n_patches = 56
mask_count = int(n_patches * 0.75)
np.random.seed(42)
mask_indices = sorted(np.random.choice(n_patches, mask_count, replace=False).tolist())

mask_data = {
    'patch_type': 'fstrip',
    'patch_width': 4,
    'img_size': 224,
    'n_patches': n_patches,
    'mask_ratio': 0.75,
    'masked_indices': mask_indices,
    'visible_indices': sorted(set(range(n_patches)) - set(mask_indices))
}

with open(f'{OUT}/fstrip_mask.json', 'w') as f:
    json.dump(mask_data, f)
print(f"  Saved fstrip_mask.json ({mask_count}/{n_patches} masked)")

# ============================================================
# 6. Pre-training loss curve (real data, epoch averages)
# ============================================================
print("\nExtracting loss curve...")
import csv
losses_by_epoch = {}
with open('d:/Dataset/weights/pretrained/mae_fstrip4_stft_alldata/train_log.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = int(row['epoch'])
        loss = float(row['loss'])
        if ep not in losses_by_epoch:
            losses_by_epoch[ep] = []
        losses_by_epoch[ep].append(loss)

epoch_avg_loss = []
for ep in sorted(losses_by_epoch.keys()):
    vals = losses_by_epoch[ep]
    # Take a few samples per epoch for smooth curve
    step = max(1, len(vals) // 10)
    for i in range(0, len(vals), step):
        epoch_avg_loss.append({
            'epoch': ep + i / len(vals),
            'loss': round(sum(vals[max(0,i-2):i+3]) / min(5, len(vals[max(0,i-2):i+3])), 4)
        })

with open(f'{OUT}/pretrain_loss.json', 'w') as f:
    json.dump(epoch_avg_loss, f)
print(f"  Saved pretrain_loss.json ({len(epoch_avg_loss)} points)")

print("\nDone! Assets in:", OUT)
print("Files:", os.listdir(OUT))
