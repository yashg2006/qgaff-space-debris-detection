import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy import stats

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# We will evaluate TRUE latent representations:
# In QGAFF: the 4-qubit quantum expectation vector q in [-1, 1]^4
# In MLP: the 32-dim latent embedding z before the classification head
# In SimCLR: the projection embedding z
# In Raw Backbone: the 256-dim GAP vector

def load_dataset(dataset_dir):
    feats = np.load(os.path.join(dataset_dir, "features.npy"))
    labels = np.load(os.path.join(dataset_dir, "labels.npy"))
    mask = labels != -1
    feats, labels = feats[mask], labels[mask]
    unique_cls = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique_cls)}
    labels = np.array([remap[l] for l in labels])
    return feats, labels, len(unique_cls)

# Let's inspect raw backbone silhouette for all 3 datasets
for name, d in [("SPARK-2022", "spark_quantum_fusion"), 
                ("Zenodo Debris", "Quantum_fusion"), 
                ("Roboflow", "roboflow_quantum_fusion")]:
    f, l, K = load_dataset(d)
    norm = normalize(f, norm='l2')
    sil = silhouette_score(norm, l, metric='cosine')
    print(f"{name:15} | N={len(f):4} | K={K:2} | Raw Backbone Silhouette: {sil:+.4f}")
