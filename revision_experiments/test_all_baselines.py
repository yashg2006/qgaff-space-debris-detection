import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()

# Model definitions
class MLP(nn.Module):
    def __init__(self, in_dim=256, embed_dim=32, num_classes=K, act=nn.ReLU):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            act(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act(),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.cls = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        z = self.enc(x)
        return z, self.cls(z)

class QGAFF(nn.Module):
    def __init__(self, in_dim=256, n_qubits=4, embed_dim=32, num_classes=K):
        super().__init__()
        self.compress = nn.Linear(in_dim, n_qubits)
        self.pqc_w = nn.Parameter(torch.randn(2, n_qubits, 3) * 0.1)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim + n_qubits, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.cls = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        theta = torch.tanh(self.compress(x)) * (np.pi / 2)
        q0 = torch.cos(theta[:, 0] + self.pqc_w[0, 0, 0]) * torch.cos(theta[:, 1] + self.pqc_w[0, 1, 0])
        q1 = torch.sin(theta[:, 1] + self.pqc_w[0, 1, 1]) * torch.cos(theta[:, 2] + self.pqc_w[0, 2, 0])
        q2 = torch.cos(theta[:, 2] + self.pqc_w[0, 2, 1]) * torch.sin(theta[:, 3] + self.pqc_w[0, 3, 0])
        q3 = torch.sin(theta[:, 3] + self.pqc_w[0, 3, 1]) * torch.cos(theta[:, 0] + self.pqc_w[0, 0, 1])
        q = torch.stack([q0, q1, q2, q3], dim=1)
        z = self.fuse(torch.cat([x, q], dim=1))
        return z, self.cls(z)

def train_and_eval(model_fn, use_closs=True, seeds=[0, 7, 21, 42, 99], lr=3e-3, epochs=60):
    sils = []
    for s in seeds:
        torch.manual_seed(s)
        m = model_fn()
        cl = CenterLoss(K, 32) if use_closs else None
        params = list(m.parameters()) + (list(cl.parameters()) if cl else [])
        opt = optim.Adam(params, lr=lr, weight_decay=1e-4)
        m.train()
        for ep in range(epochs):
            z, logits = m(feats_t)
            loss = F.cross_entropy(logits, labels_t)
            if cl:
                loss = loss + 0.3 * cl(z, labels_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            z_out, _ = m(feats_t)
            sils.append(float(silhouette_score(normalize(z_out.numpy()), labels, metric='cosine')))
    return np.mean(sils), np.std(sils)

# 1. Raw
sil_raw = silhouette_score(normalize(feats), labels, metric='cosine')
print(f"Raw Backbone:      {sil_raw:.4f}")

# 2. PCA
pca = PCA(n_components=4)
sil_pca = silhouette_score(normalize(pca.fit_transform(feats)), labels, metric='cosine')
print(f"Linear PCA (4d):   {sil_pca:.4f}")

# 3. MLP (GELU)
m_gelu, s_gelu = train_and_eval(lambda: MLP(act=nn.GELU), use_closs=False)
print(f"MLP (GELU):        {m_gelu:.4f} ± {s_gelu:.4f}")

# 4. MLP + CenterLoss
m_cl, s_cl = train_and_eval(lambda: MLP(act=nn.ReLU), use_closs=True)
print(f"MLP + CenterLoss:  {m_cl:.4f} ± {s_cl:.4f}")

# 5. QGAFF
m_q, s_q = train_and_eval(lambda: QGAFF(), use_closs=True)
print(f"QGAFF (Proposed):  {m_q:.4f} ± {s_q:.4f}")
