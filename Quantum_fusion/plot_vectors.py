import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load vectors
before = np.load("features.npy")
after = np.load("quantum_features.npy")

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
before_2d = tsne.fit_transform(before)
after_2d = tsne.fit_transform(after)

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(before_2d[:,0], before_2d[:,1], s=5)
plt.title("Before Quantum Fusion")

plt.subplot(1,2,2)
plt.scatter(after_2d[:,0], after_2d[:,1], s=5)
plt.title("After Quantum Fusion")

plt.savefig("vector_comparison.png", dpi=300)
plt.show()
