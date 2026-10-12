import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Test on SPARK
feats = np.load('spark_quantum_fusion/features.npy')
labels = np.load('spark_quantum_fusion/labels.npy')
mask = labels != -1
feats, labels = feats[mask], labels[mask]
unique_cls = sorted(np.unique(labels).tolist())
remap = {old: new for new, old in enumerate(unique_cls)}
labels = np.array([remap[l] for l in labels])

# Train / Test split 70 / 30
X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.3, random_state=42, stratify=labels)

# Raw backbone on test set:
raw_norm = normalize(X_test, norm='l2')
sil_raw = silhouette_score(raw_norm, y_test, metric='cosine')
print(f"SPARK Test Set - Raw backbone silhouette: {sil_raw:.4f}")

# Check Roboflow
r_feats = np.load('roboflow_quantum_fusion/features.npy')
r_labels = np.load('roboflow_quantum_fusion/labels.npy')
r_mask = r_labels != -1
r_feats, r_labels = r_feats[r_mask], r_labels[r_mask]
r_u = sorted(np.unique(r_labels).tolist())
r_remap = {old: new for new, old in enumerate(r_u)}
r_labels = np.array([r_remap[l] for l in r_labels])

r_X_tr, r_X_te, r_y_tr, r_y_te = train_test_split(r_feats, r_labels, test_size=0.3, random_state=42, stratify=r_labels)
r_raw_norm = normalize(r_X_te, norm='l2')
r_sil_raw = silhouette_score(r_raw_norm, r_y_te, metric='cosine')
print(f"Roboflow Test Set - Raw backbone silhouette: {r_sil_raw:.4f}")

# Check Zenodo
z_feats = np.load('Quantum_fusion/features.npy')
z_labels = np.load('Quantum_fusion/labels.npy')
z_mask = z_labels != -1
z_feats, z_labels = z_feats[z_mask], z_labels[z_mask]
z_u = sorted(np.unique(z_labels).tolist())
z_remap = {old: new for new, old in enumerate(z_u)}
z_labels = np.array([z_remap[l] for l in z_labels])

z_X_tr, z_X_te, z_y_tr, z_y_te = train_test_split(z_feats, z_labels, test_size=0.3, random_state=42, stratify=z_labels)
z_raw_norm = normalize(z_X_te, norm='l2')
z_sil_raw = silhouette_score(z_raw_norm, z_y_te, metric='cosine')
print(f"Zenodo Test Set - Raw backbone silhouette: {z_sil_raw:.4f}")
