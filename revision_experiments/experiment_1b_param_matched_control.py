"""
experiment_1b_param_matched_control.py
========================================
Adds the PARAMETER-MATCHED classical control to Table 3 (Reviewer 1, Major Comment 3):
  - TinyMLP + Center Loss: ~9.7k trainable params (same as QGAFF), no quantum circuit.
  
This answers: "Does QGAFF at ~9.7k params outperform a classical network of the same size?"

Run from project root:
    python revision_experiments/experiment_1b_param_matched_control.py

Outputs:
    revision_experiments/results/baselines_comparison_extended.json
    revision_experiments/results/baselines_comparison_spark_extended.png
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

SPARK_DIR    = "spark_quantum_fusion"
ROBOFLOW_DIR = "roboflow_quantum_fusion"
OUT_DIR      = "revision_experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS  = [0, 7, 21, 42, 99]
EPOCHS = 80


# ─────────────────────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()


class TinyMLPCenterLoss(nn.Module):
    """
    Parameter-matched classical control (~9.7k params).
    Architecture: 256 → 32 (Linear + BN + GELU) → classify.
    Trained with CE + Center Loss, same as QGAFF.

    Param count for SPARK (K=7, in_dim=256, embed_dim=32):
      Linear(256,32)     = 256*32 + 32       = 8,224
      BN(32)             = 32*2              = 64
      Linear(32, K)      = 32*K + K          = 32*7+7 = 231
      CenterLoss centers = K * 32            = 224
      Total              ≈ 8,743  (vs QGAFF: 9,699)
    """
    def __init__(self, in_dim=256, embed_dim=32, num_classes=11):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embed  = self.encoder(x)
        logits = self.classifier(embed)
        return embed, logits


class QGAFFModel(nn.Module):
    """Full QGAFF (analytic PQC surrogate) for comparison."""
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


def train_model(model, feats_t, labels_t, K, use_center=True,
                epochs=80, batch_size=128, seed=42):
    torch.manual_seed(seed)
    model.train()

    with torch.no_grad():
        sample   = feats_t[:2]
        emb_dim  = model(sample)[0].shape[1]

    c_loss   = CenterLoss(num_classes=K, feat_dim=emb_dim)
    ce_fn    = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt_m    = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    opt_c    = optim.Adam(c_loss.parameters(), lr=5e-2)
    sched    = optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        perm = torch.randperm(feats_t.size(0))
        fs, ls = feats_t[perm], labels_t[perm]
        for i in range(0, fs.size(0), batch_size):
            bx, by = fs[i:i+batch_size], ls[i:i+batch_size]
            if len(bx) < 2:
                continue
            opt_m.zero_grad()
            opt_c.zero_grad()
            embed, logits = model(bx)
            loss = ce_fn(logits, by)
            if use_center:
                loss = loss + 0.3 * c_loss(embed, by)
            loss.backward()
            opt_m.step()
            if use_center:
                opt_c.step()
        sched.step()
    return model


def extract_and_score(model, feats_t, labels_cont):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk     = feats_t[i:i+256]
            embed, _  = model(chunk)
            normed    = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())
    out = torch.cat(all_embeds).numpy()
    sil = float(silhouette_score(out, labels_cont, metric='cosine'))
    return sil


def load_data(dataset_dir):
    feats_raw  = np.load(os.path.join(dataset_dir, "features.npy"))
    labels_raw = np.load(os.path.join(dataset_dir, "labels.npy"))
    remap_path = os.path.join(dataset_dir, "labels_remapped.npy")
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


def run_dataset(dataset_dir, dataset_name):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    if not os.path.exists(os.path.join(dataset_dir, "features.npy")):
        print(f"  [!] features.npy not found in {dataset_dir}")
        return None

    feats_raw, labels_cont, feats_t, labels_t, K = load_data(dataset_dir)
    print(f"  Samples: {len(feats_raw)}, Classes: {K}")

    # ── Tiny MLP + Center Loss (NEW param-matched baseline) ──────────────────
    tiny_sils = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = TinyMLPCenterLoss(in_dim=feats_raw.shape[1], embed_dim=32, num_classes=K)
        m = train_model(m, feats_t, labels_t, K, use_center=True,
                        epochs=EPOCHS, batch_size=128, seed=seed)
        tiny_sils.append(extract_and_score(m, feats_t, labels_cont))
        print(f"    Seed {seed}: sil={tiny_sils[-1]:.4f}")

    n_params_tiny = sum(p.numel() for p in TinyMLPCenterLoss(
        feats_raw.shape[1], 32, K).parameters())
    print(f"  TinyMLP+CL params: {n_params_tiny:,}")
    print(f"  TinyMLP+CL mean ± std: {np.mean(tiny_sils):.4f} ± {np.std(tiny_sils):.4f}")

    # ── QGAFF for this dataset (re-run, same config) ──────────────────────────
    qgaff_sils = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = QGAFFModel(in_dim=feats_raw.shape[1], n_qubits=4, embed_dim=32, num_classes=K)
        m = train_model(m, feats_t, labels_t, K, use_center=True,
                        epochs=EPOCHS, batch_size=128, seed=seed)
        qgaff_sils.append(extract_and_score(m, feats_t, labels_cont))
        print(f"    [QGAFF] Seed {seed}: sil={qgaff_sils[-1]:.4f}")

    n_params_qgaff = sum(p.numel() for p in QGAFFModel(
        feats_raw.shape[1], 4, 32, K).parameters())

    return {
        'dataset': dataset_name,
        'tiny_mlp_center_loss': {
            'silhouette_mean': float(np.mean(tiny_sils)),
            'silhouette_std':  float(np.std(tiny_sils)),
            'silhouette_all':  tiny_sils,
            'params':          n_params_tiny,
            'description':     'TinyMLP + Center Loss (~same params as QGAFF, no quantum)'
        },
        'qgaff_rerun': {
            'silhouette_mean': float(np.mean(qgaff_sils)),
            'silhouette_std':  float(np.std(qgaff_sils)),
            'silhouette_all':  qgaff_sils,
            'params':          n_params_qgaff,
            'description':     'QGAFF (analytic PQC surrogate + Center Loss)'
        }
    }


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  EXPERIMENT 1b: Parameter-Matched Classical Control")
    print("  (Reviewer 1, Major Comment 3)")
    print("=" * 65)

    all_results = {}
    for ddir, dname, key in [
        (SPARK_DIR,    "SPARK-2022 (cross-domain)", "spark"),
        (ROBOFLOW_DIR, "Roboflow (in-domain)",       "roboflow"),
    ]:
        res = run_dataset(ddir, dname)
        if res:
            all_results[key] = res

    # Load existing baselines JSON to merge
    existing_path = os.path.join(OUT_DIR, "baselines_comparison.json")
    if os.path.exists(existing_path):
        existing = json.load(open(existing_path))
    else:
        existing = {}

    # Merge new rows into existing structure
    extended = {}
    for key in ['spark', 'roboflow']:
        extended[key] = dict(existing.get(key, {}))
        if key in all_results:
            extended[key]['tiny_mlp_center_loss'] = all_results[key]['tiny_mlp_center_loss']
            # Update QGAFF with fresh run (same epochs)
            extended[key]['qgaff_param_matched_run'] = all_results[key]['qgaff_rerun']

    out_path = os.path.join(OUT_DIR, "baselines_comparison_extended.json")
    with open(out_path, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"\n  Extended results saved to: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PARAMETER-MATCHED COMPARISON (SPARK-2022)")
    print("=" * 70)
    if 'spark' in all_results:
        res = all_results['spark']
        tiny = res['tiny_mlp_center_loss']
        qg   = res['qgaff_rerun']
        print(f"  {'Method':<45} {'Sil. Mean':>10} {'±Std':>7} {'Params':>10}")
        print("  " + "-" * 70)
        print(f"  {'TinyMLP + Center Loss (classical control)':<45} "
              f"{tiny['silhouette_mean']:>10.4f} {tiny['silhouette_std']:>7.4f} "
              f"{tiny['params']:>10,}")
        print(f"  {'QGAFF (analytic PQC + Center Loss)':<45} "
              f"{qg['silhouette_mean']:>10.4f} {qg['silhouette_std']:>7.4f} "
              f"{qg['params']:>10,}")
        delta = qg['silhouette_mean'] - tiny['silhouette_mean']
        print(f"\n  QGAFF vs TinyMLP+CL (param-matched): Δsil = {delta:+.4f}")
        if abs(delta) < 0.02:
            print("  → No meaningful difference: quantum block adds negligible benefit at this parameter scale.")
        elif delta > 0:
            print(f"  → QGAFF is BETTER by {delta:.4f} silhouette points despite same parameter count.")
        else:
            print(f"  → TinyMLP+CL is BETTER: quantum block HURTS at this scale (Δ={delta:.4f}).")
    print("=" * 70)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if 'spark' in all_results:
        res   = all_results['spark']
        tiny  = res['tiny_mlp_center_loss']
        qg    = res['qgaff_rerun']

        methods = ['TinyMLP + Center Loss\n(~same params, no PQC)', 'QGAFF\n(analytic PQC + CL)']
        means   = [tiny['silhouette_mean'], qg['silhouette_mean']]
        stds    = [tiny['silhouette_std'],  qg['silhouette_std']]
        colors  = ['#4fc3f7', '#ef5350']

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(methods, means, color=colors, edgecolor='black',
                      yerr=stds, capsize=6, linewidth=0.9,
                      error_kw={'linewidth': 1.5})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

        ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_ylabel('Silhouette Score (cosine, 5-seed mean ± SD)', fontsize=11)
        ax.set_title('Parameter-Matched Comparison: QGAFF vs Classical Control\n'
                     '(SPARK-2022, ~9.7k trainable parameters each)', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "param_matched_comparison_spark.png")
        plt.savefig(fig_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\n  Plot saved to: {fig_path}")
