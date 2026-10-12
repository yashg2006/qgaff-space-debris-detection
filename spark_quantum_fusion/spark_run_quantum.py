import torch
import numpy as np
import os
from spark_quantum_model import QuantumFusionModel, NUM_CLASSES, SPARK_CLASSES

OUT_DIR = "spark_quantum_fusion"

# ── Load raw features ────────────────────────────────────────────────────────
features = torch.tensor(np.load(os.path.join(OUT_DIR, "features.npy")), dtype=torch.float32)
labels   = torch.tensor(np.load(os.path.join(OUT_DIR, "labels.npy")),   dtype=torch.long)

print(f"[spark_run_quantum] Loaded features : {features.shape}")
print(f"[spark_run_quantum] Loaded labels   : {labels.shape}")
print(f"[spark_run_quantum] Classes present : {torch.unique(labels).tolist()}")

# ── Quantum fusion forward pass ───────────────────────────────────────────────
model = QuantumFusionModel(num_classes=NUM_CLASSES)
print(f"\n[spark_run_quantum] Running quantum fusion ({NUM_CLASSES} classes) …")
outputs = model(features)

print(f"[spark_run_quantum] Output shape : {outputs.shape}")
print(f"[spark_run_quantum] First 5 outputs:\n{outputs[:5].detach()}")

preds = torch.argmax(outputs, dim=1)
print(f"\n[spark_run_quantum] Predicted classes (first 10): {preds[:10].tolist()}")
print(f"[spark_run_quantum] Class names: {[SPARK_CLASSES[p] for p in preds[:10].tolist()]}")

# ── Save quantum features ─────────────────────────────────────────────────────
save_path = os.path.join(OUT_DIR, "quantum_features.npy")
np.save(save_path, outputs.detach().cpu().numpy())
print(f"\n[spark_run_quantum] Saved quantum_features.npy → shape {outputs.shape}")
