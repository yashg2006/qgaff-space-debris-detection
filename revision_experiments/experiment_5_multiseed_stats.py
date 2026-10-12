"""
experiment_5_multiseed_stats.py
=================================
Multi-seed statistical validation:
  Runs QGAFF (using FusionMLP for speed) across 5 random seeds,
  reports mean ± std for silhouette, FDR, and inter/intra-class ratio.

Run from project root:
    python revision_experiments/experiment_5_multiseed_stats.py

Outputs:
    revision_experiments/results/multiseed_stats.json
    revision_experiments/results/multiseed_stats.png
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
ROBO_DIR  = "roboflow_quantum_fusion"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 7, 21, 42, 99]


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


def fisher_discriminant_ratio(feats, labels):
    unique_cls = np.unique(labels)
    N, D = feats.shape
    overall_mean = feats.mean(axis=0)
    S_W = np.zeros((D, D), dtype=np.float64)
    S_B = np.zeros((D, D), dtype=np.float64)
    for cls in unique_cls:
        idx = (labels == cls)
        n_k = idx.sum()
        cls_feats = feats[idx].astype(np.float64)
        cls_mean  = cls_feats.mean(axis=0)
        diff_w = cls_feats - cls_mean
        S_W += diff_w.T @ diff_w
        diff_b = (cls_mean - overall_mean).reshape(-1, 1)
        S_B += n_k * (diff_b @ diff_b.T)
    return float(np.trace(S_B) / (np.trace(S_W) + 1e-10))


def mean_intra_inter(feats_norm, labels):
    unique_cls = np.unique(labels)
    intra_list, inter_list = [], []
    for cls in unique_cls:
        idx_cls   = np.where(labels == cls)[0]
        idx_other = np.where(labels != cls)[0]
        if len(idx_cls) > 1:
            cls_f = feats_norm[idx_cls]
            sim = cls_f @ cls_f.T
            iu = np.triu_indices(len(cls_f), k=1)
            intra_list.extend((1.0 - sim[iu]).tolist())
        if len(idx_other) > 0:
            inter_sim = feats_norm[idx_cls] @ feats_norm[idx_other].T
            sample = np.random.choice((1.0 - inter_sim).flatten(),
                                      min(2000, len(idx_cls)*len(idx_other)), replace=False)
            inter_list.extend(sample.tolist())
    return float(np.mean(intra_list)), float(np.mean(inter_list))


def run_one_seed(feats_raw, labels_cont, K, seed, epochs=60, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feats_t  = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)

    model     = QGAFFModel(in_dim=feats_raw.shape[1], n_qubits=4, embed_dim=32, num_classes=K)
    center_fn = CenterLoss(num_classes=K, feat_dim=32)

    ce_fn = nn.CrossEntropyLoss()
    opt_m = optim.Adam(model.parameters(),    lr=3e-3, weight_decay=1e-4)
    opt_c = optim.Adam(center_fn.parameters(), lr=5e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=epochs, eta_min=1e-5)

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
            opt_m.zero_grad()
            opt_c.zero_grad()
            embed, logits = model(bx)
            loss = ce_fn(logits, by) + 0.3 * center_fn(embed, by)
            loss.backward()
            opt_m.step()
            opt_c.step()
        sched.step()

    # Extract continuous normalized latent embedding (non-circular)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, feats_t.size(0), 256):
            chunk = feats_t[i:i+256]
            embed, _ = model(chunk)
            normed = F.normalize(embed, p=2, dim=1)
            all_embeds.append(normed.cpu())

    out_feats = torch.cat(all_embeds).numpy()
    normed    = normalize(out_feats, norm='l2')

    sil  = float(silhouette_score(normed, labels_cont, metric='cosine'))
    fdr  = fisher_discriminant_ratio(out_feats, labels_cont)
    mn_intra, mn_inter = mean_intra_inter(normed, labels_cont)

    return {
        'seed': seed,
        'silhouette': sil,
        'fdr': fdr,
        'mean_intra_dist': mn_intra,
        'mean_inter_dist': mn_inter,
        'inter_intra_ratio': mn_inter / (mn_intra + 1e-9)
    }


def load_data(ddir):
    feats_raw  = np.load(os.path.join(ddir, "features.npy"))
    labels_raw = np.load(os.path.join(ddir, "labels.npy"))
    remap_path = os.path.join(ddir, "labels_remapped.npy")
    if os.path.exists(remap_path):
        labels_raw = np.load(remap_path)
    mask = labels_raw != -1
    feats_raw  = feats_raw[mask]
    labels_raw = labels_raw[mask]
    unique_cls  = sorted(np.unique(labels_raw).tolist())
    remap       = {old: new for new, old in enumerate(unique_cls)}
    labels_cont = np.array([remap[l] for l in labels_raw])
    return feats_raw, labels_cont, len(unique_cls)


def raw_metrics(feats_raw, labels_cont):
    normed = normalize(feats_raw, norm='l2')
    sil    = float(silhouette_score(normed, labels_cont, metric='cosine'))
    fdr    = fisher_discriminant_ratio(feats_raw, labels_cont)
    mi, mx = mean_intra_inter(normed, labels_cont)
    return {'silhouette': sil, 'fdr': fdr, 'mean_intra': mi, 'mean_inter': mx}


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 5: Multi-Seed Statistical Validation")
    print("=" * 60)

    all_results = {}

    datasets = [
        (SPARK_DIR, "SPARK-2022"),
        (ROBO_DIR,  "Roboflow"),
    ]

    for ddir, dname in datasets:
        if not os.path.exists(os.path.join(ddir, "features.npy")):
            print(f"  [!] Skipping {dname}: features.npy not found")
            continue

        print(f"\n  Dataset: {dname}")
        feats_raw, labels_cont, K = load_data(ddir)
        print(f"  Samples={len(feats_raw)}, Classes={K}")

        # Raw backbone baseline
        raw = raw_metrics(feats_raw, labels_cont)
        print(f"  Raw backbone: sil={raw['silhouette']:.4f}, FDR={raw['fdr']:.4f}")

        # Multi-seed QGAFF
        seed_results = []
        for seed in SEEDS:
            print(f"    Seed {seed}...", end=" ", flush=True)
            res = run_one_seed(feats_raw, labels_cont, K, seed=seed)
            seed_results.append(res)
            print(f"sil={res['silhouette']:.4f}, FDR={res['fdr']:.4f}")

        sils = [r['silhouette'] for r in seed_results]
        fdrs = [r['fdr'] for r in seed_results]
        intras = [r['mean_intra_dist'] for r in seed_results]
        inters = [r['mean_inter_dist'] for r in seed_results]
        ratios = [r['inter_intra_ratio'] for r in seed_results]

        print(f"\n  QGAFF Summary ({dname}):")
        print(f"    Silhouette : {np.mean(sils):.4f} ± {np.std(sils):.4f}  "
              f"[{min(sils):.4f}, {max(sils):.4f}]")
        print(f"    FDR        : {np.mean(fdrs):.4f} ± {np.std(fdrs):.4f}")
        print(f"    Intra-dist : {np.mean(intras):.4f} ± {np.std(intras):.4f}")
        print(f"    Inter-dist : {np.mean(inters):.4f} ± {np.std(inters):.4f}")
        print(f"    I/I ratio  : {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

        all_results[dname] = {
            'raw': raw,
            'qgaff_seeds': seed_results,
            'qgaff_summary': {
                'silhouette_mean': float(np.mean(sils)),
                'silhouette_std':  float(np.std(sils)),
                'silhouette_min':  float(min(sils)),
                'silhouette_max':  float(max(sils)),
                'fdr_mean':        float(np.mean(fdrs)),
                'fdr_std':         float(np.std(fdrs)),
                'intra_mean':      float(np.mean(intras)),
                'intra_std':       float(np.std(intras)),
                'inter_mean':      float(np.mean(inters)),
                'inter_std':       float(np.std(inters)),
                'ratio_mean':      float(np.mean(ratios)),
                'ratio_std':       float(np.std(ratios)),
            }
        }

    # Save JSON
    results_path = os.path.join(OUT_DIR, "multiseed_stats.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Plot — one subplot per dataset
    n_datasets = len(all_results)
    if n_datasets == 0:
        print("  No results to plot.")
        exit(0)

    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, (dname, dres) in zip(axes, all_results.items()):
        seeds_  = [r['seed'] for r in dres['qgaff_seeds']]
        sils_   = [r['silhouette'] for r in dres['qgaff_seeds']]
        raw_sil = dres['raw']['silhouette']

        ax.axhline(raw_sil, color='#b0bec5', linestyle='--', linewidth=1.5,
                   label=f'Raw backbone ({raw_sil:.4f})')
        ax.axhline(0, color='red', linestyle=':', linewidth=1.2, alpha=0.6)

        ax.bar(range(len(seeds_)), sils_, color='#42a5f5', edgecolor='black', linewidth=0.7)
        ax.errorbar(
            range(len(seeds_)), sils_,
            yerr=dres['qgaff_summary']['silhouette_std'],
            fmt='none', color='black', capsize=5, linewidth=1.5
        )

        for i, (seed, sil) in enumerate(zip(seeds_, sils_)):
            ax.text(i, sil + 0.005, f'{sil:.3f}', ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(seeds_)))
        ax.set_xticklabels([f'Seed {s}' for s in seeds_], fontsize=9)
        ax.set_ylabel('Silhouette Score (cosine)', fontsize=11)
        ax.set_title(f'{dname}\nMean={dres["qgaff_summary"]["silhouette_mean"]:.4f} '
                     f'±{dres["qgaff_summary"]["silhouette_std"]:.4f}', fontsize=11)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Multi-Seed Statistical Validation of QGAFF\n(5 independent runs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "multiseed_stats.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_path}")

    print("\n" + "="*60)
    print("  Multi-Seed Validation Complete.")
    print("="*60)
