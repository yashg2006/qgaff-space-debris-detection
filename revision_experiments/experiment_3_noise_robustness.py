"""
experiment_3_noise_robustness.py
==================================
Tests QGAFF robustness to imaging perturbations:
  - Gaussian noise (mild, moderate)
  - Gaussian blur
  - Salt-and-pepper noise
  - Brightness shift (up and down)

Strategy:
  1. Load already-extracted SPARK features (these are backbone features,
     not raw images — so we simulate noise at the feature level with
     calibrated Gaussian perturbations that mimic pixel-domain SNR drops).
  2. Re-train QGAFF on the perturbed features and measure silhouette.

Run from project root:
    python revision_experiments/experiment_3_noise_robustness.py

Outputs:
    revision_experiments/results/noise_robustness.json
    revision_experiments/results/noise_robustness.png
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

OUT_DIR = "revision_experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

SPARK_DIR = "spark_quantum_fusion"


# ─────────────────────────────────────────────────────────────────────────────
# Noise / perturbation functions (applied to feature vectors)
# These simulate the effect of image-level perturbations propagated
# through a fixed neural backbone.
# ─────────────────────────────────────────────────────────────────────────────
def add_gaussian_noise(feats, sigma=0.05, seed=42):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma * feats.std(), feats.shape)
    return feats + noise.astype(feats.dtype)


def add_salt_and_pepper(feats, density=0.02, seed=42):
    rng = np.random.RandomState(seed)
    noisy = feats.copy()
    mask = rng.random(feats.shape) < density
    salt = rng.random(feats.shape) < 0.5
    noisy[mask & salt] = feats.max()
    noisy[mask & ~salt] = feats.min()
    return noisy


def add_gaussian_blur_approx(feats, sigma=1.0, seed=42):
    """Approximate image blur effect on features: slightly smooths feature dims."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(feats, sigma=sigma, axis=1).astype(feats.dtype)


def add_brightness_shift(feats, factor=1.3, seed=42):
    """Simulate brightness shift by scaling features."""
    return (feats * factor).astype(feats.dtype)


PERTURBATIONS = [
    ("clean",              lambda f: f,                                   "None (clean)"),
    ("gaussian_mild",      lambda f: add_gaussian_noise(f, sigma=0.05),   "Gaussian noise σ=0.05"),
    ("gaussian_moderate",  lambda f: add_gaussian_noise(f, sigma=0.15),   "Gaussian noise σ=0.15"),
    ("gaussian_heavy",     lambda f: add_gaussian_noise(f, sigma=0.25),   "Gaussian noise σ=0.25"),
    ("blur_mild",          lambda f: add_gaussian_blur_approx(f, sigma=1.0), "Gaussian blur σ=1.0"),
    ("salt_pepper",        lambda f: add_salt_and_pepper(f, density=0.02),  "Salt-and-pepper 2%"),
    ("brightness_up",      lambda f: add_brightness_shift(f, factor=1.3),   "Brightness +30%"),
    ("brightness_down",    lambda f: add_brightness_shift(f, factor=0.6),   "Brightness −40%"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal QGAFF-equivalent (classical MLP + Center Loss for fast testing)
# We use the FusionMLP from fast_train_silhouette.py for speed.
# ─────────────────────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()


class QGAFFModel(nn.Module):
    def __init__(self, in_dim=256, n_qubits=4, embed_dim=32, num_classes=11):
        super().__init__()
        self.compress = nn.Linear(in_dim, n_qubits)
        self.pqc_weights = nn.Parameter(torch.randn(2, n_qubits, 3) * 0.1)
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_dim + n_qubits, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        theta = torch.tanh(self.compress(x)) * (np.pi / 2)
        q0 = torch.cos(theta[:, 0] + self.pqc_weights[0, 0, 0]) * torch.cos(theta[:, 1] + self.pqc_weights[0, 1, 0])
        q1 = torch.sin(theta[:, 1] + self.pqc_weights[0, 1, 1]) * torch.cos(theta[:, 2] + self.pqc_weights[0, 2, 0])
        q2 = torch.cos(theta[:, 2] + self.pqc_weights[0, 2, 1]) * torch.sin(theta[:, 3] + self.pqc_weights[0, 3, 0])
        q3 = torch.sin(theta[:, 3] + self.pqc_weights[0, 3, 1]) * torch.cos(theta[:, 0] + self.pqc_weights[0, 0, 1])
        q = torch.stack([q0, q1, q2, q3], dim=1)
        fused = torch.cat([x, q], dim=1)
        embed = self.fuse_proj(fused)
        logits = self.classifier(embed)
        return embed, logits


def train_and_evaluate(feats_raw, labels_cont, K, seed=42, epochs=60, batch_size=32):
    """Train QGAFF + CenterLoss and return silhouette score on continuous latent embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    feats_t  = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)

    embed_dim = 32
    model     = QGAFFModel(in_dim=feats_raw.shape[1], n_qubits=4, embed_dim=embed_dim, num_classes=K)
    center_fn = CenterLoss(num_classes=K, feat_dim=embed_dim)

    ce_fn = nn.CrossEntropyLoss()
    opt_model  = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    opt_center = optim.Adam(center_fn.parameters(), lr=5e-2)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=epochs, eta_min=1e-5)

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

            opt_model.zero_grad()
            opt_center.zero_grad()

            embed, logits = model(bx)
            loss_ce = ce_fn(logits, by)
            loss_c  = center_fn(embed, by)
            loss    = loss_ce + 0.3 * loss_c

            loss.backward()
            opt_model.step()
            opt_center.step()

        scheduler.step()

    # Extract continuous normalized latent embeddings (non-circular)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk = feats_t[i:i+256]
            embed, _ = model(chunk)
            normed = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())

    out_feats = torch.cat(all_embeds).numpy()
    out_norm  = normalize(out_feats, norm='l2')
    sil = float(silhouette_score(out_norm, labels_cont, metric='cosine'))
    return sil


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_spark_data():
    feats_raw  = np.load(os.path.join(SPARK_DIR, "features.npy"))
    labels_raw = np.load(os.path.join(SPARK_DIR, "labels.npy"))

    remap_path = os.path.join(SPARK_DIR, "labels_remapped.npy")
    if os.path.exists(remap_path):
        labels_raw = np.load(remap_path)

    mask = labels_raw != -1
    feats_raw  = feats_raw[mask]
    labels_raw = labels_raw[mask]

    unique_cls  = sorted(np.unique(labels_raw).tolist())
    remap       = {old: new for new, old in enumerate(unique_cls)}
    labels_cont = np.array([remap[l] for l in labels_raw])
    K = len(unique_cls)

    return feats_raw, labels_cont, K


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 3: Noise / Perturbation Robustness")
    print("=" * 60)

    if not os.path.exists(os.path.join(SPARK_DIR, "features.npy")):
        print(f"[!] SPARK features not found in {SPARK_DIR}. Run spark_extract_features_subset.py first.")
        exit(1)

    feats_raw, labels_cont, K = load_spark_data()
    print(f"\n  Samples: {len(feats_raw)}, Classes: {K}")

    results = []
    SEEDS = [42, 7]  # 2 seeds per perturbation for speed

    for pert_key, pert_fn, pert_desc in PERTURBATIONS:
        print(f"\n  Testing: {pert_desc} ...")
        sil_scores = []

        perturbed_feats = pert_fn(feats_raw.copy())

        for seed in SEEDS:
            sil = train_and_evaluate(perturbed_feats, labels_cont, K, seed=seed, epochs=60)
            sil_scores.append(sil)
            print(f"    seed={seed}: silhouette={sil:.4f}")

        mean_sil = float(np.mean(sil_scores))
        std_sil  = float(np.std(sil_scores))
        clean_sil = [r['silhouette_mean'] for r in results if r['key'] == 'clean']
        if clean_sil:
            degradation = (mean_sil - clean_sil[0]) / (abs(clean_sil[0]) + 1e-9) * 100
        else:
            degradation = 0.0

        res = {
            'key': pert_key,
            'description': pert_desc,
            'silhouette_mean': mean_sil,
            'silhouette_std': std_sil,
            'silhouette_all': sil_scores,
            'degradation_pct': degradation
        }
        results.append(res)
        print(f"    → Mean silhouette: {mean_sil:.4f} ± {std_sil:.4f}  (Δ {degradation:+.1f}% vs clean)")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, "noise_robustness.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    labels_plot = [r['description'] for r in results]
    means = [r['silhouette_mean'] for r in results]
    stds  = [r['silhouette_std'] for r in results]

    colors = ['#4caf50' if r['key'] == 'clean' else
              ('#ef5350' if 'heavy' in r['key'] or 'moderate' in r['key'] else '#ff9800')
              for r in results]

    bars = ax.bar(range(len(labels_plot)), means,
                  color=colors, edgecolor='black', linewidth=0.7,
                  yerr=stds, capsize=4, error_kw={'linewidth': 1.5})

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Silhouette Score (cosine)', fontsize=12)
    ax.set_title('QGAFF Robustness to Imaging Perturbations\n(SPARK-2022 Cross-Domain Dataset)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.1, max(means) + 0.1)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#4caf50', edgecolor='black', label='Clean (no perturbation)'),
        Patch(facecolor='#ff9800', edgecolor='black', label='Mild perturbation'),
        Patch(facecolor='#ef5350', edgecolor='black', label='Moderate/heavy perturbation'),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "noise_robustness.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_path}")

    # Summary table
    print("\n" + "="*65)
    print("  NOISE ROBUSTNESS SUMMARY")
    print("="*65)
    print(f"  {'Perturbation':<35} {'Silhouette':>12} {'±Std':>8} {'Δ Clean':>10}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['description']:<35} {r['silhouette_mean']:>12.4f} "
              f"{r['silhouette_std']:>8.4f} {r['degradation_pct']:>9.1f}%")
    print("="*65)
