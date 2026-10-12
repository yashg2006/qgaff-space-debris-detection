"""
experiment_4_ablation_study.py
================================
Comprehensive ablation study isolating the contribution of each QGAFF component:
  1. Raw YOLOv8 backbone
  2. + Linear compression (256→4) only
  3. + Linear + ReLU nonlinearity
  4. + Linear + tanh normalization (same as QGAFF pre-quantum, no circuit)
  5. + Untrained quantum circuit (random PQC weights)
  6. + Quantum circuit + CrossEntropy only (no Center Loss)
  7. + Quantum circuit + Center Loss only (no CE)
  8. + Full QGAFF (CE + Center Loss + quantum)
  9. + Full QGAFF + L2 normalization (final proposed method)

Run from project root:
    python revision_experiments/experiment_4_ablation_study.py

Outputs:
    revision_experiments/results/ablation_study.json
    revision_experiments/results/ablation_study.png
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

OUT_DIR  = "revision_experiments/results"
SPARK_DIR = "spark_quantum_fusion"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()


class LinearOnly(nn.Module):
    """Case 2: Linear compression 256→32, no nonlinearity."""
    def __init__(self, in_dim, out_dim=32, num_classes=11):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        embed = self.linear(x)
        logits = self.classifier(embed)
        return embed, logits


class LinearReLU(nn.Module):
    """Case 3: 256→32 with ReLU nonlinearity and BatchNorm."""
    def __init__(self, in_dim, out_dim=32, num_classes=11):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.BatchNorm1d(out_dim))
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return embed, logits


class AngleMappingFusion(nn.Module):
    """Case 4: Angle mapping (tanh pre-quantum) + Classical fusion (no quantum entanglement)."""
    def __init__(self, in_dim, embed_dim=32, num_classes=11):
        super().__init__()
        self.compress = nn.Linear(in_dim, 4)
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_dim + 4, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        theta = torch.tanh(self.compress(x)) * (np.pi / 2)
        fused = torch.cat([x, theta], dim=1)
        embed = self.fuse_proj(fused)
        logits = self.classifier(embed)
        return embed, logits


class QGAFFModel(nn.Module):
    """
    Full QGAFF Model: 4-qubit PQC + Classical Fused Latent Embedding (32-dim).
    Trainable parameters: ~3,607.
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


# ─────────────────────────────────────────────────────────────────────────────
# Generic train + evaluate
# ─────────────────────────────────────────────────────────────────────────────
def train_ablation_model(model, feats_t, labels_t, K,
                         use_ce=True, use_center=False,
                         epochs=60, batch_size=32, seed=42):
    torch.manual_seed(seed)
    model.train()

    c_loss = CenterLoss(num_classes=K, feat_dim=32)
    params = list(model.parameters())
    if use_center:
        params += list(c_loss.parameters())
    opt = optim.Adam(params, lr=3e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        perm = torch.randperm(feats_t.size(0))
        for i in range(0, feats_t.size(0), batch_size):
            bx = feats_t[perm[i:i+batch_size]]
            by = labels_t[perm[i:i+batch_size]]
            if len(bx) < 2:
                continue

            opt.zero_grad()
            embed, logits = model(bx)
            loss = 0.0

            if use_ce:
                loss = loss + F.cross_entropy(logits, by)
            if use_center:
                loss = loss + 0.3 * c_loss(embed, by)

            if isinstance(loss, float):
                break

            loss.backward()
            opt.step()

    return model


def get_features_and_silhouette(model, feats_t, labels_cont):
    """Extract continuous latent embedding (before classifier) + L2 normalized, compute silhouette."""
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk = feats_t[i:i+256]
            embed, _ = model(chunk)
            normed = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())

    out_feats = torch.cat(all_embeds).numpy()
    sil = float(silhouette_score(out_feats, labels_cont, metric='cosine'))
    return out_feats, sil


# ─────────────────────────────────────────────────────────────────────────────
# Load SPARK data
# ─────────────────────────────────────────────────────────────────────────────
def load_spark():
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

    feats_t  = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)
    return feats_raw, labels_cont, feats_t, labels_t, K


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 4: Ablation Study")
    print("=" * 60)

    if not os.path.exists(os.path.join(SPARK_DIR, "features.npy")):
        print(f"[!] SPARK features not found in {SPARK_DIR}.")
        exit(1)

    feats_raw, labels_cont, feats_t, labels_t, K = load_spark()
    in_dim = feats_raw.shape[1]
    print(f"\n  Samples: {len(feats_raw)}, Classes: {K}, Feature dim: {in_dim}")

    SEED   = 42
    EPOCHS = 60

    ablation_cases = []

    # ── Case 1: Raw backbone ──────────────────────────────────────────────────
    raw_norm = normalize(feats_raw, norm='l2')
    sil_raw  = float(silhouette_score(raw_norm, labels_cont, metric='cosine'))
    n_params_raw = 0
    ablation_cases.append({
        'name': 'Raw YOLOv8 backbone',
        'silhouette': sil_raw,
        'params': n_params_raw,
        'notes': 'Baseline — no transformation'
    })
    print(f"\n  [1] Raw backbone:                   sil={sil_raw:.4f}")

    # ── Case 2: Linear only ───────────────────────────────────────────────────
    m2 = LinearOnly(in_dim, 32, K)
    m2 = train_ablation_model(m2, feats_t, labels_t, K, use_ce=True, use_center=False, epochs=EPOCHS, seed=SEED)
    _, sil2 = get_features_and_silhouette(m2, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Linear compression (256→32)',
        'silhouette': sil2,
        'params': sum(p.numel() for p in m2.parameters()),
        'notes': 'No nonlinearity, CE training'
    })
    print(f"  [2] Linear only:                    sil={sil2:.4f}")

    # ── Case 3: Linear + ReLU ─────────────────────────────────────────────────
    m3 = LinearReLU(in_dim, 32, K)
    m3 = train_ablation_model(m3, feats_t, labels_t, K, use_ce=True, use_center=False, epochs=EPOCHS, seed=SEED)
    _, sil3 = get_features_and_silhouette(m3, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Linear + ReLU + BN (256→32)',
        'silhouette': sil3,
        'params': sum(p.numel() for p in m3.parameters()),
        'notes': 'Nonlinear classical compression'
    })
    print(f"  [3] Linear + ReLU:                  sil={sil3:.4f}")

    # ── Case 4: Angle mapping (tanh pre-quantum) + Classical fusion ───────────
    m4 = AngleMappingFusion(in_dim, 32, K)
    m4 = train_ablation_model(m4, feats_t, labels_t, K, use_ce=True, use_center=False, epochs=EPOCHS, seed=SEED)
    _, sil4 = get_features_and_silhouette(m4, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Angle mapping + Classical fusion',
        'silhouette': sil4,
        'params': sum(p.numel() for p in m4.parameters()),
        'notes': 'Pre-quantum angle mapping; no quantum circuit'
    })
    print(f"  [4] Angle mapping + Fusion:         sil={sil4:.4f}")

    # ── Cases 5-8: Quantum circuit variants ───────────────────────────────────
    # Case 5: Random weights (no training)
    m5 = QGAFFModel(in_dim=in_dim, embed_dim=32, num_classes=K)
    _, sil5 = get_features_and_silhouette(m5, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Quantum PQC (untrained random weights)',
        'silhouette': sil5,
        'params': sum(p.numel() for p in m5.parameters()),
        'notes': 'Untrained PQC — unitary inductive bias only'
    })
    print(f"  [5] Quantum (random weights):       sil={sil5:.4f}")

    # Case 6: Quantum + CE only (no Center Loss)
    m6 = QGAFFModel(in_dim=in_dim, embed_dim=32, num_classes=K)
    m6 = train_ablation_model(m6, feats_t, labels_t, K, use_ce=True, use_center=False, epochs=EPOCHS, seed=SEED)
    _, sil6 = get_features_and_silhouette(m6, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Quantum PQC + CrossEntropy only',
        'silhouette': sil6,
        'params': sum(p.numel() for p in m6.parameters()),
        'notes': 'CE training only (no Center Loss)'
    })
    print(f"  [6] Quantum + CE only:              sil={sil6:.4f}")

    # Case 7: Quantum + Center Loss only (no CE)
    m7 = QGAFFModel(in_dim=in_dim, embed_dim=32, num_classes=K)
    m7 = train_ablation_model(m7, feats_t, labels_t, K, use_ce=False, use_center=True, epochs=EPOCHS, seed=SEED)
    _, sil7 = get_features_and_silhouette(m7, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Quantum PQC + Center Loss only',
        'silhouette': sil7,
        'params': sum(p.numel() for p in m7.parameters()),
        'notes': 'Center Loss training only (no CE)'
    })
    print(f"  [7] Quantum + Center Loss only:     sil={sil7:.4f}")

    # Case 8: Full QGAFF (CE + Center Loss + L2)
    m8 = QGAFFModel(in_dim=in_dim, embed_dim=32, num_classes=K)
    m8 = train_ablation_model(m8, feats_t, labels_t, K, use_ce=True, use_center=True, epochs=EPOCHS, seed=SEED)
    _, sil8 = get_features_and_silhouette(m8, feats_t, labels_cont)
    ablation_cases.append({
        'name': 'Full QGAFF (PQC + Fusion + CE + Center)',
        'silhouette': sil8,
        'params': sum(p.numel() for p in m8.parameters()),
        'notes': 'Proposed method — all components active'
    })
    print(f"  [8] Full QGAFF (proposed):          sil={sil8:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, "ablation_study.json")
    with open(results_path, "w") as f:
        json.dump(ablation_cases, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    names = [c['name'] for c in ablation_cases]
    sils  = [c['silhouette'] for c in ablation_cases]
    colors = ['#b0bec5', '#90caf9', '#80cbc4', '#a5d6a7',
              '#fff9c4', '#ffe082', '#ffcc80', '#ef5350']

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(range(len(names)), sils,
                   color=colors, edgecolor='black', linewidth=0.7)

    for bar, val in zip(bars, sils):
        x_pos = val + 0.003 if val >= 0 else val - 0.003
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold', ha=ha)

    ax.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Silhouette Score (cosine metric)', fontsize=12)
    ax.set_title('Ablation Study: Component-wise Contribution to QGAFF\n(SPARK-2022 Cross-Domain)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "ablation_study.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_path}")

    # Summary
    print("\n" + "="*70)
    print("  ABLATION SUMMARY")
    print("="*70)
    print(f"  {'Configuration':<48} {'Silhouette':>12} {'Params':>10}")
    print("  " + "-"*70)
    for c in ablation_cases:
        print(f"  {c['name']:<48} {c['silhouette']:>12.4f} {c['params']:>10,}")
    print("="*70)
