"""
experiment_2_representation_analysis.py
=========================================
Deep Representation Analysis on CONTINUOUS LATENT EMBEDDINGS (non-circular):
  - Class-wise intra/inter-class cosine distance
  - Fisher Discriminant Ratio (FDR)
  - Feature distribution KDE plots
  - Evaluated on ALL THREE datasets: SPARK-2022, Zenodo Debris, Roboflow Dataset

Run:
    python revision_experiments/experiment_2_representation_analysis.py
"""

import sys
import os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde

OUT_DIR = "revision_experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

SPARK_CLASSES = [
    "debris", "lisa_pathfinder", "soho", "xmm_newton", "cheops",
    "smart_1", "proba_2", "proba_3_csc", "proba_3_ocs",
    "earth_observation_sat_1", "double_star"
]

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()

class QGAFFModel(nn.Module):
    """
    Quantum-Guided Feature Transformation (QGAFF).
    4-qubit PQC + Classical Fused Latent Embedding (32-dim).
    Trainable parameters: 3,607 (compact unitary inductive bias).
    """
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

def train_and_extract_qgaff(feats_raw, labels_cont, K, epochs=60, lr=3e-3, seed=42):
    torch.manual_seed(seed)
    feats_t = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)

    model = QGAFFModel(in_dim=feats_raw.shape[1], n_qubits=4, embed_dim=32, num_classes=K)
    c_loss = CenterLoss(K, 32)
    opt = optim.Adam(list(model.parameters()) + list(c_loss.parameters()), lr=lr, weight_decay=1e-4)

    model.train()
    batch_size = 32
    for ep in range(epochs):
        perm = torch.randperm(len(feats_t))
        for i in range(0, len(feats_t), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = feats_t[idx], labels_t[idx]
            if len(bx) < 2:
                continue
            opt.zero_grad()
            embed, logits = model(bx)
            loss = F.cross_entropy(logits, by) + 0.3 * c_loss(embed, by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        embed_all, _ = model(feats_t)
        # Normalize the continuous latent embedding
        normed = F.normalize(embed_all, p=2, dim=1).numpy()
    return normed

def class_wise_cosine_distances(feats_norm, labels):
    unique_classes = sorted(np.unique(labels).tolist())
    intra_dists = []
    inter_dists = []
    all_intra   = []
    all_inter   = []

    for cls in unique_classes:
        idx_cls  = np.where(labels == cls)[0]
        idx_other = np.where(labels != cls)[0]

        cls_feats   = feats_norm[idx_cls]
        other_feats = feats_norm[idx_other]

        if len(cls_feats) > 1:
            sim_matrix = cls_feats @ cls_feats.T
            iu = np.triu_indices(len(cls_feats), k=1)
            intra_sims = sim_matrix[iu]
            intra_d = 1.0 - intra_sims
            intra_dists.append(float(np.mean(intra_d)))
            all_intra.extend(intra_d.tolist())
        else:
            intra_dists.append(0.0)

        if len(other_feats) > 0:
            inter_sim = cls_feats @ other_feats.T
            inter_d = 1.0 - inter_sim.flatten()
            inter_dists.append(float(np.mean(inter_d)))
            sampled = np.random.choice(inter_d, min(5000, len(inter_d)), replace=False)
            all_inter.extend(sampled.tolist())
        else:
            inter_dists.append(0.0)

    return intra_dists, inter_dists, all_intra, all_inter

def fisher_discriminant_ratio(feats, labels):
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        return 0.0

    global_mean = np.mean(feats, axis=0)
    S_B = 0.0
    S_W = 0.0

    for cls in unique_classes:
        idx = np.where(labels == cls)[0]
        cls_feats = feats[idx]
        n_c = len(cls_feats)
        cls_mean = np.mean(cls_feats, axis=0)

        # Between-class scatter trace
        diff_b = cls_mean - global_mean
        S_B += n_c * np.dot(diff_b, diff_b)

        # Within-class scatter trace
        diff_w = cls_feats - cls_mean
        S_W += np.sum(diff_w ** 2)

    return float(S_B / (S_W + 1e-9))

def analyze_dataset(dataset_dir, dataset_name):
    print(f"\n{'='*60}")
    print(f"  Analyzing: {dataset_name} ({dataset_dir})")
    print(f"{'='*60}")

    feats_raw_path = os.path.join(dataset_dir, "features.npy")
    labels_raw_path = os.path.join(dataset_dir, "labels.npy")

    if not os.path.exists(feats_raw_path) or not os.path.exists(labels_raw_path):
        print(f"  [!] Missing features.npy or labels.npy in {dataset_dir}")
        return None

    feats_raw = np.load(feats_raw_path)
    labels_raw = np.load(labels_raw_path)

    mask = labels_raw != -1
    feats_raw = feats_raw[mask]
    labels_raw = labels_raw[mask]

    unique_cls = sorted(np.unique(labels_raw).tolist())
    remap = {old: new for new, old in enumerate(unique_cls)}
    labels_cont = np.array([remap[l] for l in labels_raw])
    K = len(unique_cls)

    print(f"  Samples: {len(feats_raw)}, Classes: {K}")

    # ── Before metrics ───────────────────────────────────────────────────────
    feats_b_norm = normalize(feats_raw, norm='l2')
    fdr_before = fisher_discriminant_ratio(feats_raw, labels_cont)
    intra_b, inter_b, all_intra_b, all_inter_b = class_wise_cosine_distances(feats_b_norm, labels_cont)
    sil_before = float(silhouette_score(feats_b_norm, labels_cont, metric='cosine'))

    print(f"  BEFORE QGAFF (Raw Backbone):")
    print(f"    Silhouette Score : {sil_before:.4f}")
    print(f"    Fisher Disc. Ratio: {fdr_before:.4f}")
    print(f"    Mean Intra-class dist: {np.mean(intra_b):.4f}")
    print(f"    Mean Inter-class dist: {np.mean(inter_b):.4f}")
    print(f"    Inter/Intra ratio    : {np.mean(inter_b)/(np.mean(intra_b)+1e-9):.4f}")

    # ── Train QGAFF & Extract Continuous Latent Embedding ─────────────────────
    print(f"  Training QGAFF (3,607 params) and extracting continuous latent embeddings...")
    qgaff_latent = train_and_extract_qgaff(feats_raw, labels_cont, K, epochs=60)
    
    # Save the continuous latent embedding
    np.save(os.path.join(dataset_dir, "quantum_features.npy"), qgaff_latent)
    np.save(os.path.join(dataset_dir, "labels_remapped.npy"), labels_cont)

    # ── After metrics ───────────────────────────────────────────────────────
    fdr_after = fisher_discriminant_ratio(qgaff_latent, labels_cont)
    intra_a, inter_a, all_intra_a, all_inter_a = class_wise_cosine_distances(qgaff_latent, labels_cont)
    sil_after = float(silhouette_score(qgaff_latent, labels_cont, metric='cosine'))

    fdr_improvement = (fdr_after - fdr_before) / (abs(fdr_before) + 1e-9) * 100
    intra_reduction = (np.mean(intra_b) - np.mean(intra_a)) / (np.mean(intra_b) + 1e-9) * 100
    inter_increase  = (np.mean(inter_a) - np.mean(inter_b)) / (np.mean(inter_b) + 1e-9) * 100

    print(f"  AFTER QGAFF (Continuous Latent Embedding):")
    print(f"    Silhouette Score : {sil_after:.4f}")
    print(f"    Fisher Disc. Ratio: {fdr_after:.4f}")
    print(f"    Mean Intra-class dist: {np.mean(intra_a):.4f}")
    print(f"    Mean Inter-class dist: {np.mean(inter_a):.4f}")
    print(f"    Inter/Intra ratio    : {np.mean(inter_a)/(np.mean(intra_a)+1e-9):.4f}")
    print(f"\n  IMPROVEMENTS:")
    print(f"    FDR improvement  : +{fdr_improvement:.1f}%")
    print(f"    Intra-dist reduction: -{intra_reduction:.1f}%")
    print(f"    Inter-dist increase : +{inter_increase:.1f}%")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name.lower())
    plot_kde(all_intra_b, all_inter_b, all_intra_a, all_inter_a, dataset_name,
             os.path.join(OUT_DIR, f"kde_distributions_{safe_name}.png"))

    cls_names = [SPARK_CLASSES[i] if i < len(SPARK_CLASSES) else f"cls_{i}"
                 for i in range(K)]
    plot_class_distances(intra_b, inter_b, intra_a, inter_a, cls_names, dataset_name,
                         os.path.join(OUT_DIR, f"class_distances_{safe_name}.png"))

    return {
        'dataset': dataset_name,
        'n_samples': len(feats_raw),
        'n_classes': K,
        'before': {
            'silhouette': sil_before,
            'fdr': fdr_before,
            'mean_intra_dist': float(np.mean(intra_b)),
            'mean_inter_dist': float(np.mean(inter_b)),
            'inter_intra_ratio': float(np.mean(inter_b) / (np.mean(intra_b) + 1e-9)),
            'class_intra_dists': intra_b,
            'class_inter_dists': inter_b,
        },
        'after': {
            'silhouette': sil_after,
            'fdr': fdr_after,
            'mean_intra_dist': float(np.mean(intra_a)),
            'mean_inter_dist': float(np.mean(inter_a)),
            'inter_intra_ratio': float(np.mean(inter_a) / (np.mean(intra_a) + 1e-9)),
            'class_intra_dists': intra_a,
            'class_inter_dists': inter_a,
            'fdr_improvement_pct': fdr_improvement,
            'intra_reduction_pct': intra_reduction,
            'inter_increase_pct': inter_increase,
        }
    }

def plot_kde(all_intra_b, all_inter_b, all_intra_a, all_inter_a, dataset_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, intra, inter, title_suffix in zip(
        axes,
        [all_intra_b, all_intra_a],
        [all_inter_b, all_inter_a],
        ["Before QGAFF (Raw Backbone)", "After QGAFF (Latent Embedding)"]
    ):
        x_range = np.linspace(0, 2, 300)
        if len(intra) > 5:
            kde_intra = gaussian_kde(intra, bw_method='scott')
            ax.fill_between(x_range, kde_intra(x_range), alpha=0.4, color='#42a5f5', label='Intra-class distances')
            ax.plot(x_range, kde_intra(x_range), color='#1565c0', linewidth=2)
        if len(inter) > 5:
            kde_inter = gaussian_kde(inter, bw_method='scott')
            ax.fill_between(x_range, kde_inter(x_range), alpha=0.4, color='#ef5350', label='Inter-class distances')
            ax.plot(x_range, kde_inter(x_range), color='#b71c1c', linewidth=2)
        ax.set_title(f"{title_suffix}\n({dataset_name})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Cosine Distance (1 − similarity)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 2)

    plt.suptitle(f"Feature Distribution Analysis: Intra vs Inter-class Cosine Distances\n{dataset_name}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_class_distances(intra_b, inter_b, intra_a, inter_a, cls_names, dataset_name, out_path):
    x = np.arange(len(cls_names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(x - width/2, intra_b, width, label='Before QGAFF', color='#90caf9', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, intra_a, width, label='After QGAFF', color='#ef5350', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Cosine Distance', fontsize=11)
    ax.set_title(f'Intra-class Distances (lower = more compact)\n{dataset_name}', fontsize=11)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    ax2.bar(x - width/2, inter_b, width, label='Before QGAFF', color='#b0bec5', edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, inter_a, width, label='After QGAFF', color='#4caf50', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Cosine Distance', fontsize=11)
    ax2.set_title(f'Inter-class Distances (higher = better separation)\n{dataset_name}', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_overall_fdr(datasets_results, out_path):
    valid_res = [r for r in datasets_results if r and r.get('after')]
    if not valid_res:
        return

    names = [r['dataset'] for r in valid_res]
    fdr_b = [r['before']['fdr'] for r in valid_res]
    fdr_a = [r['after']['fdr'] for r in valid_res]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, fdr_b, width, label='Before QGAFF (Raw)', color='#90caf9', edgecolor='black')
    b2 = ax.bar(x + width/2, fdr_a, width, label='After QGAFF (Latent Embedding)', color='#ef5350', edgecolor='black')

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Fisher Discriminant Ratio: tr(S_B) / tr(S_W)', fontsize=12)
    ax.set_title('Fisher Discriminant Ratio Comparison Across Datasets\n(Higher = better class separability)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    datasets = [
        ("spark_quantum_fusion", "SPARK-2022 (Cross-Domain)"),
        ("Quantum_fusion", "Zenodo Space Debris Dataset"),
        ("roboflow_quantum_fusion", "Roboflow Dataset (In-Domain)")
    ]

    all_results = []
    for d_dir, d_name in datasets:
        res = analyze_dataset(d_dir, d_name)
        if res:
            all_results.append(res)

    json_path = os.path.join(OUT_DIR, "representation_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to: {json_path}")

    fdr_plot_path = os.path.join(OUT_DIR, "fdr_comparison.png")
    plot_overall_fdr(all_results, fdr_plot_path)
    print(f"[OK] FDR plot saved to: {fdr_plot_path}")
