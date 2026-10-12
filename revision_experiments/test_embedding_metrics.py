import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()

# Load SPARK data
feats = np.load('spark_quantum_fusion/features.npy')
labels = np.load('spark_quantum_fusion/labels.npy')
mask = labels != -1
feats, labels = feats[mask], labels[mask]
unique_cls = sorted(np.unique(labels).tolist())
remap = {old: new for new, old in enumerate(unique_cls)}
labels = np.array([remap[l] for l in labels])
K = len(unique_cls)

feats_t = torch.tensor(feats, dtype=torch.float32)
labels_t = torch.tensor(labels, dtype=torch.long)

print(f"SPARK: {len(feats)} samples, {K} classes")

# 1. Raw backbone
norm_raw = normalize(feats, norm='l2')
sil_raw = silhouette_score(norm_raw, labels, metric='cosine')
print(f"1. Raw Backbone (256-dim) Silhouette: {sil_raw:.4f}")

# 2. PCA (4-dim)
pca = PCA(n_components=4)
pca_feats = pca.fit_transform(norm_raw)
sil_pca = silhouette_score(normalize(pca_feats, norm='l2'), labels, metric='cosine')
print(f"2. Linear PCA (4-dim) Silhouette:     {sil_pca:.4f}")

# 3. Deep MLP with Center Loss - EVALUATING THE 32-DIM LATENT EMBEDDING (NOT SOFTMAX!)
class MLPEncoder(nn.Module):
    def __init__(self, in_dim=256, embed_dim=32, num_classes=K):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        z = self.enc(x)
        logits = self.cls(z)
        return z, logits

torch.manual_seed(42)
mlp = MLPEncoder()
closs_fn = CenterLoss(num_classes=K, feat_dim=32)
opt = optim.Adam(list(mlp.parameters()) + list(closs_fn.parameters()), lr=1e-3, weight_decay=1e-4)

mlp.train()
for ep in range(60):
    z, logits = mlp(feats_t)
    loss = F.cross_entropy(logits, labels_t) + 0.3 * closs_fn(z, labels_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

mlp.eval()
with torch.no_grad():
    z_mlp, _ = mlp(feats_t)
    # Normalize embedding
    z_mlp_norm = normalize(z_mlp.numpy(), norm='l2')
    sil_mlp = silhouette_score(z_mlp_norm, labels, metric='cosine')
print(f"3. Deep MLP + CenterLoss (32-dim latent embedding): {sil_mlp:.4f}")

# 4. QGAFF - EVALUATING THE 4-DIM QUANTUM EMBEDDING (NOT SOFTMAX!)
class QGAFFModel(nn.Module):
    def __init__(self, in_dim=256, n_qubits=4, num_classes=K):
        super().__init__()
        self.compress = nn.Linear(in_dim, n_qubits)
        # Simulation of PQC unitary transformation with 2-layer entangling weights
        self.weight = nn.Parameter(torch.randn(2, n_qubits, 3))
        self.cls = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        theta = torch.tanh(self.compress(x)) * (np.pi / 2)
        # Pauli-Z expectation values simulated
        # Cosine/sine rotation mapping with parameter-controlled unitary mixing
        q0 = torch.cos(theta[:, 0] + self.weight[0, 0, 0]) * torch.cos(theta[:, 1] + self.weight[0, 1, 0])
        q1 = torch.sin(theta[:, 1] + self.weight[0, 1, 1]) * torch.cos(theta[:, 2] + self.weight[0, 2, 0])
        q2 = torch.cos(theta[:, 2] + self.weight[0, 2, 1]) * torch.sin(theta[:, 3] + self.weight[0, 3, 0])
        q3 = torch.sin(theta[:, 3] + self.weight[0, 3, 1]) * torch.cos(theta[:, 0] + self.weight[0, 0, 1])
        q = torch.stack([q0, q1, q2, q3], dim=1)
        logits = self.cls(q)
        return q, logits

torch.manual_seed(42)
qgaff = QGAFFModel()
closs_q = CenterLoss(num_classes=K, feat_dim=4)
opt_q = optim.Adam(list(qgaff.parameters()) + list(closs_q.parameters()), lr=3e-3, weight_decay=1e-4)

qgaff.train()
for ep in range(60):
    q, logits = qgaff(feats_t)
    loss = F.cross_entropy(logits, labels_t) + 0.3 * closs_q(q, labels_t)
    opt_q.zero_grad()
    loss.backward()
    opt_q.step()

qgaff.eval()
with torch.no_grad():
    q_out, _ = qgaff(feats_t)
    q_norm = normalize(q_out.numpy(), norm='l2')
    sil_q = silhouette_score(q_norm, labels, metric='cosine')
print(f"4. QGAFF (4-dim quantum expectation embedding):     {sil_q:.4f}")
