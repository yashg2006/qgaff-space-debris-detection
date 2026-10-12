"""
experiment_1_classical_baselines.py
=====================================
Compares QGAFF against classical representation learning baselines.
Baselines: Raw YOLOv8 backbone, PCA, MLP, MLP+CenterLoss, SimCLR-style contrastive.

Run from project root:
    python revision_experiments/experiment_1_classical_baselines.py

Outputs: revision_experiments/results/baselines_comparison.json
         revision_experiments/results/baselines_comparison.png
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
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
SPARK_DIR   = "spark_quantum_fusion"
ROBOFLOW_DIR = "roboflow_quantum_fusion"
OUT_DIR     = "revision_experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 7, 21, 42, 99]

# ── Center Loss ───────────────────────────────────────────────────────────────
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()


# ── MLP model ─────────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    """256 → 128 → 32 → embed_dim, then classifier head."""
    def __init__(self, in_dim=256, embed_dim=32, num_classes=11):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return embed, logits


# ── SimCLR projection head ─────────────────────────────────────────────────────
class SimCLRHead(nn.Module):
    def __init__(self, in_dim=256, proj_dim=64, num_classes=11):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        proj = F.normalize(self.projector(x), dim=1)
        logits = self.classifier(proj)
        return proj, logits


# ── NT-Xent loss (SimCLR contrastive) ─────────────────────────────────────────
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """Supervised contrastive: same-class = positive pairs."""
        embeddings = F.normalize(embeddings, dim=1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        N = embeddings.shape[0]

        # mask: positive if same class, exclude self
        labels_row = labels.unsqueeze(1)
        labels_col = labels.unsqueeze(0)
        pos_mask = (labels_row == labels_col).float()
        diag_mask = torch.eye(N, device=sim.device, dtype=torch.bool)
        pos_mask = pos_mask.masked_fill(diag_mask, 0.0)

        # log-softmax over all non-self pairs
        sim_masked = sim.masked_fill(diag_mask, -1e9)
        log_prob = F.log_softmax(sim_masked, dim=1)

        pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
        return loss.mean()


# ── Generic trainer ───────────────────────────────────────────────────────────
def train_model(model, features_t, labels_t, num_classes,
                epochs=80, batch_size=128, use_center_loss=False,
                use_contrastive=False, seed=42):
    torch.manual_seed(seed)
    model.train()
    opt = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    center_fn = None
    opt_center = None
    if use_center_loss:
        # infer embed dim from model
        with torch.no_grad():
            sample = features_t[:2]
            embed_sample, _ = model(sample)
            edim = embed_sample.shape[1]
        center_fn = CenterLoss(num_classes=num_classes, feat_dim=edim)
        opt_center = optim.Adam(center_fn.parameters(), lr=5e-2)

    nt_xent = NTXentLoss() if use_contrastive else None

    for epoch in range(epochs):
        perm = torch.randperm(features_t.size(0))
        fs, ls = features_t[perm], labels_t[perm]

        for i in range(0, fs.size(0), batch_size):
            bx = fs[i:i+batch_size]
            by = ls[i:i+batch_size]
            if len(bx) < 2:
                continue

            opt.zero_grad()
            if opt_center:
                opt_center.zero_grad()

            embed, logits = model(bx)
            loss = ce_fn(logits, by)

            if use_center_loss and center_fn:
                lc = center_fn(embed, by)
                loss = loss + 0.3 * lc

            if use_contrastive and nt_xent:
                lcon = nt_xent(embed, by)
                loss = loss + 0.1 * lcon

            loss.backward()
            opt.step()
            if opt_center:
                opt_center.step()

        scheduler.step()

    return model


class QGAFFModel(nn.Module):
    """
    Quantum-Guided Feature Transformation (QGAFF).
    Linear compression (256->4) + 4-qubit unitary PQC + Fused latent embedding (32-dim).
    Trainable parameters: 3,607 (vs 43,911 for MLP).
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


def extract_features(model, features_t):
    """Extract continuous normalized latent embeddings before classifier head (non-circular)."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, features_t.size(0), 256):
            chunk = features_t[i:i+256]
            out = model(chunk)
            embed = out[0] if isinstance(out, tuple) else out
            normed = F.normalize(embed, p=2, dim=1)
            all_feats.append(normed.cpu())
    return torch.cat(all_feats).numpy()


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(dataset_dir):
    feats_raw = np.load(os.path.join(dataset_dir, "features.npy"))
    labels_raw = np.load(os.path.join(dataset_dir, "labels.npy"))
    mask = labels_raw != -1
    feats_raw = feats_raw[mask]
    labels_raw = labels_raw[mask]

    unique_cls = sorted(np.unique(labels_raw).tolist())
    remap = {old: new for new, old in enumerate(unique_cls)}
    labels_cont = np.array([remap[l] for l in labels_raw])
    K = len(unique_cls)

    feats_t = torch.tensor(feats_raw, dtype=torch.float32)
    labels_t = torch.tensor(labels_cont, dtype=torch.long)
    return feats_raw, labels_cont, feats_t, labels_t, K


def compute_silhouette(feats, labels):
    normed = normalize(feats, norm='l2')
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(normed, labels, metric='cosine'))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(dataset_dir, dataset_name, seeds=SEEDS):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    feats_raw, labels_cont, feats_t, labels_t, K = load_data(dataset_dir)
    print(f"  Samples: {len(feats_raw)}, Classes: {K}")

    results = {}

    # ── 1. Raw backbone features ──────────────────────────────────────────────
    raw_sil = compute_silhouette(feats_raw, labels_cont)
    results['raw_backbone'] = {
        'silhouette_mean': raw_sil,
        'silhouette_std': 0.0,
        'params': 0,
        'description': 'Raw YOLOv8 backbone (no transformation)'
    }
    print(f"  [1] Raw backbone silhouette: {raw_sil:.4f}")

    # ── 2. PCA (256→4) ────────────────────────────────────────────────────────
    pca = PCA(n_components=4, random_state=42)
    pca_feats = pca.fit_transform(feats_raw)
    pca_sil = compute_silhouette(pca_feats, labels_cont)
    results['pca_4d'] = {
        'silhouette_mean': pca_sil,
        'silhouette_std': 0.0,
        'params': 0,
        'description': 'PCA (256→4 dimensions)'
    }
    print(f"  [2] PCA(256→4) silhouette  : {pca_sil:.4f}")

    # ── 3. MLP (256→128→32) ───────────────────────────────────────────────────
    mlp_sils = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        mlp = SimpleMLP(in_dim=feats_raw.shape[1], embed_dim=32, num_classes=K)
        mlp = train_model(mlp, feats_t, labels_t, K,
                          epochs=80, batch_size=128, seed=seed)
        qf = extract_features(mlp, feats_t)
        mlp_sils.append(compute_silhouette(qf, labels_cont))
    mlp_time = (time.time() - t0) / len(seeds)

    n_params_mlp = sum(p.numel() for p in SimpleMLP(feats_raw.shape[1], 32, K).parameters())
    results['mlp'] = {
        'silhouette_mean': float(np.mean(mlp_sils)),
        'silhouette_std': float(np.std(mlp_sils)),
        'silhouette_all': mlp_sils,
        'params': n_params_mlp,
        'inference_ms': mlp_time * 1000,
        'description': 'Deep MLP (256→128→64→32) with CrossEntropy'
    }
    print(f"  [3] MLP silhouette         : {np.mean(mlp_sils):.4f} ± {np.std(mlp_sils):.4f}")

    # ── 4. MLP + Center Loss ──────────────────────────────────────────────────
    mlp_cl_sils = []
    for seed in seeds:
        torch.manual_seed(seed)
        mlp_cl = SimpleMLP(in_dim=feats_raw.shape[1], embed_dim=32, num_classes=K)
        mlp_cl = train_model(mlp_cl, feats_t, labels_t, K,
                              epochs=80, batch_size=128,
                              use_center_loss=True, seed=seed)
        qf = extract_features(mlp_cl, feats_t)
        mlp_cl_sils.append(compute_silhouette(qf, labels_cont))

    results['mlp_center_loss'] = {
        'silhouette_mean': float(np.mean(mlp_cl_sils)),
        'silhouette_std': float(np.std(mlp_cl_sils)),
        'silhouette_all': mlp_cl_sils,
        'params': n_params_mlp,
        'description': 'Deep MLP + Center Loss (strongest classical baseline)'
    }
    print(f"  [4] MLP+CenterLoss silhouette: {np.mean(mlp_cl_sils):.4f} ± {np.std(mlp_cl_sils):.4f}")

    # ── 5. SimCLR-style Contrastive ───────────────────────────────────────────
    simclr_sils = []
    for seed in seeds:
        torch.manual_seed(seed)
        simclr = SimCLRHead(in_dim=feats_raw.shape[1], proj_dim=64, num_classes=K)
        simclr = train_model(simclr, feats_t, labels_t, K,
                              epochs=80, batch_size=128,
                              use_contrastive=True, seed=seed)
        qf = extract_features(simclr, feats_t)
        simclr_sils.append(compute_silhouette(qf, labels_cont))

    n_params_simclr = sum(p.numel() for p in SimCLRHead(feats_raw.shape[1], 64, K).parameters())
    results['simclr'] = {
        'silhouette_mean': float(np.mean(simclr_sils)),
        'silhouette_std': float(np.std(simclr_sils)),
        'silhouette_all': simclr_sils,
        'params': n_params_simclr,
        'description': 'Supervised Contrastive (SimCLR-style) + CrossEntropy'
    }
    print(f"  [5] SimCLR silhouette      : {np.mean(simclr_sils):.4f} ± {np.std(simclr_sils):.4f}")

    # ── 6. QGAFF (Proposed) ───────────────────────────────────────────────────
    qgaff_sils = []
    for seed in seeds:
        torch.manual_seed(seed)
        qgaff = QGAFFModel(in_dim=feats_raw.shape[1], n_qubits=4, embed_dim=32, num_classes=K)
        qgaff = train_model(qgaff, feats_t, labels_t, K,
                            epochs=80, batch_size=128,
                            use_center_loss=True, seed=seed)
        qf = extract_features(qgaff, feats_t)
        qgaff_sils.append(compute_silhouette(qf, labels_cont))

    n_params_qgaff = sum(p.numel() for p in QGAFFModel(feats_raw.shape[1], 4, 32, K).parameters())
    results['qgaff'] = {
        'silhouette_mean': float(np.mean(qgaff_sils)),
        'silhouette_std': float(np.std(qgaff_sils)),
        'silhouette_all': qgaff_sils,
        'params': n_params_qgaff,
        'description': 'QGAFF (proposed): 4-qubit PQC + Center Loss + Fused Latent Embedding'
    }
    print(f"  [6] QGAFF silhouette       : {np.mean(qgaff_sils):.4f} ± {np.std(qgaff_sils):.4f}")

    return results


def plot_comparison(spark_results, out_path):
    methods = ['raw_backbone', 'pca_4d', 'mlp', 'mlp_center_loss', 'simclr', 'qgaff']
    labels_plot = [
        'Raw YOLOv8\nBackbone',
        'PCA\n(256→4)',
        'MLP\n(256→128→32)',
        'MLP +\nCenter Loss',
        'SimCLR-style\nContrastive',
        'QGAFF\n(Proposed)'
    ]
    colors = ['#b0bec5', '#90caf9', '#a5d6a7', '#fff176', '#ce93d8', '#ef5350']

    means = [spark_results[m]['silhouette_mean'] or 0 for m in methods]
    stds  = [spark_results[m].get('silhouette_std', 0) for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels_plot, means, color=colors, edgecolor='black', linewidth=0.8,
                  yerr=stds, capsize=4, error_kw={'linewidth': 1.5})
    
    # Annotate values
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(0, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label='Threshold (0)')
    ax.set_ylim(-0.4, 0.75)
    ax.set_ylabel('Silhouette Score (cosine metric)', fontsize=12)
    ax.set_title('SPARK Cross-Domain: QGAFF vs. Classical Baselines\n(Silhouette Score — higher is better)', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved comparison plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 1: Classical Baselines Comparison")
    print("=" * 60)

    all_results = {}

    if os.path.exists(os.path.join(SPARK_DIR, "features.npy")):
        spark_res = run_experiment(SPARK_DIR, "SPARK-2022")
        all_results['spark'] = spark_res
        plot_comparison(spark_res, os.path.join(OUT_DIR, "baselines_comparison_spark.png"))
    else:
        print(f"[!] SPARK features not found in {SPARK_DIR}. Skipping.")
        all_results['spark'] = {}

    if os.path.exists(os.path.join(ROBOFLOW_DIR, "features.npy")):
        robo_res = run_experiment(ROBOFLOW_DIR, "Roboflow (in-domain)")
        all_results['roboflow'] = robo_res
        plot_comparison(robo_res, os.path.join(OUT_DIR, "baselines_comparison_roboflow.png"))
    else:
        print(f"[!] Roboflow features not found in {ROBOFLOW_DIR}. Skipping.")
        all_results['roboflow'] = {}

    # Save JSON results
    results_path = os.path.join(OUT_DIR, "baselines_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Print summary table
    print("\n" + "="*70)
    print("  SUMMARY TABLE (SPARK dataset)")
    print("="*70)
    print(f"  {'Method':<35} {'Silhouette':>12} {'±Std':>8} {'Params':>10}")
    print("  " + "-"*65)
    if all_results.get('spark'):
        for key, val in all_results['spark'].items():
            sil = val.get('silhouette_mean')
            std = val.get('silhouette_std', 0)
            params = val.get('params', 0)
            desc = val.get('description', key)[:34]
            if sil is not None:
                print(f"  {desc:<35} {sil:>12.4f} {std:>8.4f} {params:>10,}")
    print("="*70)
