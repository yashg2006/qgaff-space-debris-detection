"""
experiment_6_yolo_scalability.py
==================================
Evaluates QGAFF scalability across YOLOv8 model sizes:
  - YOLOv8n (nano,  ~3.2M params)
  - YOLOv8m (medium, ~25.9M params)
  - YOLOv8x (extra-large, ~68.2M params)

For each backbone:
  1. Extracts features using hook at an equivalent C2f block
  2. Trains FusionMLP (QGAFF proxy) on the extracted features
  3. Reports silhouette score before and after QGAFF

Note: YOLOv8m/x weight files must be available or will be downloaded by Ultralytics.
      If weights are not present, the script will attempt to use YOLO('<size>.pt').

Run from project root:
    python revision_experiments/experiment_6_yolo_scalability.py

Outputs:
    revision_experiments/results/yolo_scalability.json
    revision_experiments/results/yolo_scalability.png
"""

import sys
import os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

OUT_DIR   = "revision_experiments/results"
SPARK_IMG = "spark_yolo/images/val"  # SPARK validation images
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model configs
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CONFIGS = [
    {
        'name':        'YOLOv8n (nano)',
        'weight_file': 'yolov8n.pt',
        'hook_layer':  8,
        'feat_dim':    256,
        'params':      '3.2M',
    },
    {
        'name':        'YOLOv8s (small)',
        'weight_file': 'yolov8s.pt',
        'hook_layer':  8,
        'feat_dim':    512,
        'params':      '11.2M',
    },
    {
        'name':        'YOLOv8m (medium)',
        'weight_file': 'yolov8m.pt',
        'hook_layer':  8,
        'feat_dim':    576,
        'params':      '25.9M',
    },
    {
        'name':        'YOLOv8x (extra-large)',
        'weight_file': 'yolov8x.pt',
        'hook_layer':  8,
        'feat_dim':    640,
        'params':      '68.2M',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# QGAFF Fusion Architecture (backbone-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        bc = self.centers[labels]
        return 0.5 * ((features - bc) ** 2).sum(dim=1).mean()


class FusionMLP(nn.Module):
    def __init__(self, in_dim, embed_dim=32, num_classes=11):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, embed_dim), nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embed = self.encoder(x)
        return embed, self.classifier(embed)


def train_fusion(feats_raw, labels_cont, K, seed=42, epochs=80, batch_size=64):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feats_t  = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)
    in_dim   = feats_raw.shape[1]

    model     = FusionMLP(in_dim=in_dim, embed_dim=32, num_classes=K)
    center_fn = CenterLoss(num_classes=K, feat_dim=32)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt_m = optim.Adam(model.parameters(),    lr=3e-3, weight_decay=1e-4)
    opt_c = optim.Adam(center_fn.parameters(), lr=5e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=epochs, eta_min=5e-5)

    model.train()
    center_fn.train()

    for epoch in range(epochs):
        perm = torch.randperm(feats_t.size(0))
        fs, ls = feats_t[perm], labels_t[perm]
        for i in range(0, fs.size(0), batch_size):
            bx = fs[i:i+batch_size]
            by = ls[i:i+batch_size]
            if len(bx) < 2:
                continue
            opt_m.zero_grad(); opt_c.zero_grad()
            embed, logits = model(bx)
            loss = ce_fn(logits, by) + 0.5 * center_fn(embed, by)
            loss.backward()
            for p in center_fn.parameters():
                if p.grad is not None:
                    p.grad.data *= 1.0 / 1.5
            opt_m.step(); opt_c.step()
        sched.step()

    # Extract continuous latent embeddings (not softmax probabilities)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk = feats_t[i:i+256]
            embed, _ = model(chunk)
            normed = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())

    out_feats = torch.cat(all_embeds).numpy()
    return float(silhouette_score(out_feats, labels_cont, metric='cosine'))


def extract_features_from_backbone(weight_file, hook_layer, img_folder,
                                    subset=500, seed=42):
    """Extract backbone features using a YOLO model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [!] Ultralytics not installed.")
        return None, None

    if not os.path.exists(img_folder):
        print(f"  [!] Image folder not found: {img_folder}")
        return None, None

    print(f"  Loading YOLO model: {weight_file} ...")

    # Use pre-trained weights if available, else use downloaded
    if os.path.exists(weight_file):
        model = YOLO(weight_file)
    else:
        model = YOLO(weight_file)  # Ultralytics will auto-download

    model.model.eval()

    features_list = []
    labels_list   = []

    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu().numpy())

    hook = model.model.model[hook_layer].register_forward_hook(hook_fn)

    img_files = sorted([
        f for f in os.listdir(img_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    np.random.seed(seed)
    if len(img_files) > subset:
        img_files = list(np.random.choice(img_files, subset, replace=False))

    print(f"  Extracting features from {len(img_files)} images...")

    for i, img_name in enumerate(img_files):
        path = os.path.join(img_folder, img_name)
        try:
            results = model.predict(source=path, verbose=False)
            cls = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            labels_list.append(int(cls[0]) if len(cls) > 0 else -1)
        except Exception as e:
            labels_list.append(-1)

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(img_files)}] done")

    hook.remove()

    if len(features_list) == 0:
        print("  [!] No features extracted.")
        return None, None

    feats = np.vstack([f.mean(axis=(2, 3)) for f in features_list])
    labs  = np.array(labels_list)

    # Filter invalid
    mask  = labs != -1
    feats = feats[mask]
    labs  = labs[mask]

    if len(feats) == 0:
        print("  [!] No valid detections found.")
        return None, None

    # Remap labels
    unique_cls  = sorted(np.unique(labs).tolist())
    remap       = {old: new for new, old in enumerate(unique_cls)}
    labs_cont   = np.array([remap[l] for l in labs])

    print(f"  Extracted: {feats.shape}, Classes: {len(unique_cls)}")
    return feats, labs_cont


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 6: YOLOv8 Scalability Study")
    print("=" * 60)

    results = []

    for cfg in YOLO_CONFIGS:
        print(f"\n{'─'*50}")
        print(f"  Model: {cfg['name']}")
        print(f"{'─'*50}")

        # Extract features
        feats, labs = extract_features_from_backbone(
            weight_file=cfg['weight_file'],
            hook_layer=cfg['hook_layer'],
            img_folder=SPARK_IMG,
            subset=150,
            seed=42
        )

        if feats is None:
            print(f"  [!] Skipping {cfg['name']} — extraction failed.")
            results.append({
                'model': cfg['name'],
                'params': cfg['params'],
                'feat_dim': cfg['feat_dim'],
                'sil_before': None,
                'sil_after':  None,
                'improvement': None,
                'status': 'FAILED'
            })
            continue

        K = len(np.unique(labs))
        feats_norm = normalize(feats, norm='l2')

        if K < 2:
            print(f"  [!] Only {K} class found — not enough for silhouette. Skipping.")
            results.append({
                'model': cfg['name'],
                'params': cfg['params'],
                'feat_dim': cfg['feat_dim'],
                'sil_before': None,
                'sil_after':  None,
                'improvement': None,
                'status': 'INSUFFICIENT_CLASSES'
            })
            continue

        # Silhouette before QGAFF
        sil_before = float(silhouette_score(feats_norm, labs, metric='cosine'))
        print(f"  Silhouette BEFORE QGAFF: {sil_before:.4f}")

        # Train QGAFF + silhouette after
        sil_after = train_fusion(feats, labs, K, seed=42, epochs=80)
        print(f"  Silhouette AFTER  QGAFF: {sil_after:.4f}")

        improvement = (sil_after - sil_before) / (abs(sil_before) + 1e-9) * 100
        print(f"  Improvement: {improvement:+.1f}%")

        results.append({
            'model':       cfg['name'],
            'params':      cfg['params'],
            'feat_dim':    cfg['feat_dim'],
            'sil_before':  sil_before,
            'sil_after':   sil_after,
            'improvement': improvement,
            'n_samples':   int(len(feats)),
            'n_classes':   int(K),
            'status':      'OK'
        })

    # Save
    results_path = os.path.join(OUT_DIR, "yolo_scalability.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Plot
    valid = [r for r in results if r['status'] == 'OK']
    if valid:
        fig, ax = plt.subplots(figsize=(9, 5))
        names    = [r['model'] for r in valid]
        bef      = [r['sil_before'] for r in valid]
        aft      = [r['sil_after']  for r in valid]
        x        = np.arange(len(names))
        width    = 0.35

        ax.bar(x - width/2, bef, width, label='Before QGAFF', color='#90caf9', edgecolor='black')
        ax.bar(x + width/2, aft, width, label='After QGAFF',  color='#ef5350', edgecolor='black')

        for i, (b, a) in enumerate(zip(bef, aft)):
            ax.text(i - width/2, b + 0.005, f'{b:.4f}', ha='center', fontsize=8.5)
            ax.text(i + width/2, a + 0.005, f'{a:.4f}', ha='center', fontsize=8.5)

        ax.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Silhouette Score (cosine metric)', fontsize=12)
        ax.set_title('QGAFF Scalability Across YOLOv8 Model Sizes\n(SPARK-2022 Cross-Domain)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "yolo_scalability.png")
        plt.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {out_path}")

    # Summary table
    print("\n" + "="*70)
    print("  SCALABILITY SUMMARY")
    print("="*70)
    print(f"  {'Model':<30} {'Params':>8} {'Dim':>6} {'Before':>10} {'After':>10} {'Δ%':>8}")
    print("  " + "-"*70)
    for r in results:
        if r['status'] == 'OK':
            print(f"  {r['model']:<30} {r['params']:>8} {r['feat_dim']:>6} "
                  f"{r['sil_before']:>10.4f} {r['sil_after']:>10.4f} {r['improvement']:>7.1f}%")
        else:
            print(f"  {r['model']:<30} {r['params']:>8} {r['feat_dim']:>6} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
    print("="*70)
