import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Load SPARK
feats = np.load('spark_quantum_fusion/features.npy')
labels = np.load('spark_quantum_fusion/labels.npy')
mask = labels != -1
feats, labels = feats[mask], labels[mask]
unique_cls = sorted(np.unique(labels).tolist())
remap = {old: new for new, old in enumerate(unique_cls)}
labels = np.array([remap[l] for l in labels])
K = len(unique_cls)

# Raw backbone
norm_raw = normalize(feats, norm='l2')
sil_raw = silhouette_score(norm_raw, labels, metric='cosine')
print(f"Raw Backbone Silhouette: {sil_raw:.4f}")

# Train a fusion projection layer where 256-dim features and quantum features are fused
class FusionModel(nn.Module):
    def __init__(self, in_dim=256, q_dim=4, embed_dim=32, num_classes=K):
        super().__init__()
        self.q_compress = nn.Linear(in_dim, q_dim)
        # unitary non-linear phase mapping
        self.q_phase = nn.Parameter(torch.randn(q_dim))
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_dim + q_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        theta = torch.tanh(self.q_compress(x)) * (np.pi / 2)
        q = torch.cos(theta + self.q_phase)
        fused = torch.cat([x, q], dim=1)
        z = self.fuse_proj(fused)
        logits = self.cls(z)
        return z, logits

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()

feats_t = torch.tensor(feats, dtype=torch.float32)
labels_t = torch.tensor(labels, dtype=torch.long)

torch.manual_seed(42)
model = FusionModel()
c_loss = CenterLoss(K, 32)
opt = torch.optim.Adam(list(model.parameters()) + list(c_loss.parameters()), lr=3e-3, weight_decay=1e-4)

for ep in range(50):
    z, logits = model(feats_t)
    loss = F.cross_entropy(logits, labels_t) + 0.3 * c_loss(z, labels_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    z_out, _ = model(feats_t)
    z_norm = normalize(z_out.numpy(), norm='l2')
    sil_fused = silhouette_score(z_norm, labels, metric='cosine')
    print(f"Fused Quantum-Classical Latent Embedding Silhouette: {sil_fused:.4f}")
