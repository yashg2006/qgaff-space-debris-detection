import torch
import numpy as np
from file1 import QuantumFusionModel

features = torch.tensor(np.load("features.npy"), dtype=torch.float32)
labels = torch.tensor(np.load("labels.npy"), dtype=torch.long)

num_classes = len(torch.unique(labels))
model = QuantumFusionModel(num_classes)

outputs = model(features)

print("Output shape:", outputs.shape)
print("First 5 outputs:\n", outputs[:5])

preds = torch.argmax(outputs, dim=1)
np.save("quantum_features.npy", outputs.detach().cpu().numpy())
print("Saved quantum_features.npy")