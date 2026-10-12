"""
experiment_4b_missing_ablation_controls.py
============================================
Adds the two MISSING ablation rows identified by Reviewer 1 (Major Comment 6):
  1. "Angle mapping + Classical fusion + Center Loss" (no quantum circuit)
  2. "Linear + BN + Center Loss" (parameter-matched, no quantum circuit)

These are the decisive controls: if they score close to Full QGAFF,
the quantum block adds nothing beyond Center Loss.

Run from project root:
    python revision_experiments/experiment_4b_missing_ablation_controls.py

Outputs:
    revision_experiments/results/ablation_study_extended.json
    revision_experiments/results/ablation_study_extended.png
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
SPARK_DIR = "spark_quantum_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 60


# ─────────────────────────────────────────────────────────────────────────────
# Shared components
# ─────────────────────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()


class LinearBN(nn.Module):
    """Linear + BN (256→32) trained with Center Loss. ~8.5k params."""
    def __init__(self, in_dim, out_dim=32, num_classes=11):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        embed  = self.encoder(x)
        logits = self.classifier(embed)
        return embed, logits


class AngleMappingFusionCL(nn.Module):
    """
    Angle mapping + classical fusion + Center Loss (no quantum circuit).
    Matches QGAFF topology: compress(256→4) → tanh scale → fuse([x; theta]) → 32-dim.
    ~9.7k params — identical to QGAFF minus the PQC weights (24 floats).
    """
    def __init__(self, in_dim, embed_dim=32, num_classes=11):
        super().__init__()
        self.compress = nn.Linear(in_dim, 4)
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_dim + 4, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        theta  = torch.tanh(self.compress(x)) * (np.pi / 2)
        fused  = torch.cat([x, theta], dim=1)
        embed  = self.fuse_proj(fused)
        logits = self.classifier(embed)
        return embed, logits


class QGAFFModel(nn.Module):
    """Full QGAFF (analytic surrogate of PQC) + Center Loss + CE."""
    def __init__(self, in_dim=256, n_qubits=4, embed_dim=32, num_classes=11):
        super().__init__()
        self.compress    = nn.Linear(in_dim, n_qubits)
        self.pqc_weights = nn.Parameter(torch.randn(2, n_qubits, 3) * 0.1)
        self.fuse_proj   = nn.Sequential(
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
        q  = torch.stack([q0, q1, q2, q3], dim=1)
        fused  = torch.cat([x, q], dim=1)
        embed  = self.fuse_proj(fused)
        logits = self.classifier(embed)
        return embed, logits


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, feats_t, labels_t, K,
                use_ce=True, use_center=True,
                epochs=60, batch_size=32, seed=42):
    torch.manual_seed(seed)
    model.train()

    c_loss = CenterLoss(num_classes=K, feat_dim=32)
    params = list(model.parameters()) + (list(c_loss.parameters()) if use_center else [])
    opt    = optim.Adam(params, lr=3e-3, weight_decay=1e-4)

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


def get_silhouette(model, feats_t, labels_cont):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk     = feats_t[i:i+256]
            embed, _  = model(chunk)
            normed    = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())
    out = torch.cat(all_embeds).numpy()
    return float(silhouette_score(out, labels_cont, metric='cosine')), sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Load SPARK
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
    print("=" * 65)
    print("  EXPERIMENT 4b: Missing Ablation Controls")
    print("  (Reviewer 1, Major Comment 6)")
    print("=" * 65)

    if not os.path.exists(os.path.join(SPARK_DIR, "features.npy")):
        print(f"[!] SPARK features not found in {SPARK_DIR}. Run experiment_2 first.")
        exit(1)

    feats_raw, labels_cont, feats_t, labels_t, K = load_spark()
    in_dim = feats_raw.shape[1]
    print(f"\n  Samples: {len(feats_raw)}, Classes: {K}, Feature dim: {in_dim}")

    # Load existing ablation results (to append)
    existing_path = os.path.join(OUT_DIR, "ablation_study.json")
    existing_cases = json.load(open(existing_path)) if os.path.exists(existing_path) else []

    new_cases = []

    # ── NEW CASE A: Linear + BN + Center Loss (no quantum) ───────────────────
    print("\n  [A] Training: Linear + BN + Center Loss (no quantum circuit)...")
    ma = LinearBN(in_dim, 32, K)
    ma = train_model(ma, feats_t, labels_t, K, use_ce=True, use_center=True,
                     epochs=EPOCHS, seed=SEED)
    sil_a, params_a = get_silhouette(ma, feats_t, labels_cont)
    print(f"      Silhouette: {sil_a:.4f}  |  Params: {params_a:,}")
    new_cases.append({
        'name': 'Linear + BN + Center Loss (no PQC)',
        'silhouette': sil_a,
        'params': params_a,
        'notes': 'CE + Center Loss, no quantum block — classical control for Center Loss effect'
    })

    # ── NEW CASE B: Angle mapping + fusion + Center Loss (no quantum) ─────────
    print("\n  [B] Training: Angle mapping + Classical fusion + Center Loss (no circuit)...")
    mb = AngleMappingFusionCL(in_dim, 32, K)
    mb = train_model(mb, feats_t, labels_t, K, use_ce=True, use_center=True,
                     epochs=EPOCHS, seed=SEED)
    sil_b, params_b = get_silhouette(mb, feats_t, labels_cont)
    print(f"      Silhouette: {sil_b:.4f}  |  Params: {params_b:,}")
    new_cases.append({
        'name': 'Angle mapping + Fusion + Center Loss (no PQC)',
        'silhouette': sil_b,
        'params': params_b,
        'notes': 'CE + Center Loss + angle fusion; no quantum circuit — closest classical analog to QGAFF'
    })

    # ── FULL QGAFF for comparison (re-run to ensure same config) ─────────────
    print("\n  [C] Re-running: Full QGAFF (PQC + CE + Center Loss)...")
    mc = QGAFFModel(in_dim=in_dim, embed_dim=32, num_classes=K)
    mc = train_model(mc, feats_t, labels_t, K, use_ce=True, use_center=True,
                     epochs=EPOCHS, seed=SEED)
    sil_c, params_c = get_silhouette(mc, feats_t, labels_cont)
    print(f"      Silhouette: {sil_c:.4f}  |  Params: {params_c:,}")

    # ── Save extended ablation ────────────────────────────────────────────────
    extended_cases = existing_cases + new_cases
    out_path = os.path.join(OUT_DIR, "ablation_study_extended.json")
    with open(out_path, "w") as f:
        json.dump(extended_cases, f, indent=2)
    print(f"\n  Extended ablation saved to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXTENDED ABLATION SUMMARY  (SPARK-2022, seed=42)")
    print("=" * 65)
    print(f"  {'Configuration':<48} {'Silhouette':>12} {'Params':>10}")
    print("  " + "-" * 65)
    for c in extended_cases:
        print(f"  {c['name']:<48} {c['silhouette']:>12.4f} {c['params']:>10,}")
    print("  " + "-" * 65)
    print(f"\n  Interpretation:")
    print(f"    Full QGAFF (PQC + CE + CenterLoss)        : {sil_c:.4f}")
    print(f"    Angle mapping + Fusion + CenterLoss (no PQC): {sil_b:.4f}")
    delta = sil_c - sil_b
    if abs(delta) < 0.02:
        print(f"    Δ = {delta:+.4f}  → PQC adds NEGLIGIBLE benefit over classical fusion")
    elif delta > 0:
        print(f"    Δ = {delta:+.4f}  → PQC shows a SMALL positive contribution")
    else:
        print(f"    Δ = {delta:+.4f}  → Angle mapping + fusion OUTPERFORMS PQC")
    print("=" * 65)

    # ── Plot (horizontal bar, showing all extended cases) ─────────────────────
    all_names = [c['name'] for c in extended_cases]
    all_sils  = [c['silhouette'] for c in extended_cases]

    colors = [
        '#b0bec5', '#90caf9', '#80cbc4', '#a5d6a7',
        '#fff9c4', '#ffe082', '#ffcc80', '#ef5350',
        '#4fc3f7',  # NEW Case A (linear + BN + CL)
        '#ab47bc',  # NEW Case B (angle + fusion + CL)
    ]
    while len(colors) < len(all_names):
        colors.append('#cccccc')

    fig, ax = plt.subplots(figsize=(14, max(6, len(all_names) * 0.65)))
    bars = ax.barh(range(len(all_names)), all_sils,
                   color=colors[:len(all_names)], edgecolor='black', linewidth=0.7)

    for bar, val in zip(bars, all_sils):
        x_pos = val + 0.003 if val >= 0 else val - 0.003
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold', ha=ha)

    ax.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=9)
    ax.set_xlabel('Silhouette Score (cosine metric)', fontsize=12)
    ax.set_title(
        'Extended Ablation: Component-wise Contribution to QGAFF\n'
        '(SPARK-2022 Cross-Domain) — NEW rows highlighted in purple/blue',
        fontsize=12, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "ablation_study_extended.png")
    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {fig_path}")
