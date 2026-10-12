import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pennylane as qml
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Load SPARK
feats = np.load('spark_quantum_fusion/features.npy')
labels = np.load('spark_quantum_fusion/labels.npy')
mask = labels != -1
feats, labels = feats[mask], labels[mask]
unique_cls = sorted(np.unique(labels).tolist())
remap = {old: new for new, old in enumerate(unique_cls)}
labels = np.array([remap[l] for l in labels])
K = len(unique_cls)

features_t = torch.tensor(feats, dtype=torch.float32)
labels_t = torch.tensor(labels, dtype=torch.long)

# 4-qubit quantum circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (2, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class TrueQGAFF(nn.Module):
    def __init__(self, in_dim=256, n_qubits=4, num_classes=K):
        super().__init__()
        self.compress = nn.Linear(in_dim, n_qubits)
        self.q_layer = q_layer
        self.cls = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        theta = torch.tanh(self.compress(x)) * (np.pi / 2)
        q = self.q_layer(theta)
        logits = self.cls(q)
        return q, logits

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * (diff ** 2).sum(dim=1).mean()

torch.manual_seed(42)
model = TrueQGAFF()
c_loss = CenterLoss(K, 4)
opt = optim.Adam(model.parameters(), lr=0.01)
opt_c = optim.Adam(c_loss.parameters(), lr=0.1)

print(f"Training True QGAFF on {len(feats)} samples...")
# Subsample 150 for quick check on PennyLane default.qubit
idx = torch.randperm(len(features_t))[:150]
sub_x, sub_y = features_t[idx], labels_t[idx]

model.train()
for ep in range(10):
    q, logits = model(sub_x)
    loss = F.cross_entropy(logits, sub_y) + 0.3 * c_loss(q, sub_y)
    opt.zero_grad()
    opt_c.zero_grad()
    loss.backward()
    opt.step()
    opt_c.step()
    if (ep + 1) % 5 == 0:
        print(f"Epoch {ep+1}/10, loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    q_out, _ = model(sub_x)
    q_norm = normalize(q_out.numpy(), norm='l2')
    sil = silhouette_score(q_norm, sub_y.numpy(), metric='cosine')
    print(f"True QGAFF 4-qubit embedding silhouette on subset: {sil:.4f}")
