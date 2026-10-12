import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

for name, d in [("SPARK-2022", "spark_quantum_fusion"), 
                ("Zenodo Debris", "Quantum_fusion"), 
                ("Roboflow", "roboflow_quantum_fusion")]:
    q_path = os.path.join(d, "quantum_features.npy")
    l_path = os.path.join(d, "labels.npy")
    if os.path.exists(q_path) and os.path.exists(l_path):
        q = np.load(q_path)
        l = np.load(l_path)
        mask = l != -1
        if len(mask) == len(q):
            q, l = q[mask], l[mask]
        norm = normalize(q, norm='l2')
        sil = silhouette_score(norm, l[:len(q)], metric='cosine')
        print(f"{name:15} | Shape: {str(q.shape):12} | Range: [{q.min():.4f}, {q.max():.4f}] | Silhouette: {sil:+.4f}")
    else:
        print(f"{name:15} | Not found")
