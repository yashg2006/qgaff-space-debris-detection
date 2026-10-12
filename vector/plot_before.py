import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = np.load("features_before.npy")
y = np.load("labels.npy")

X_2d = TSNE(n_components=2, perplexity=30).fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', s=10)
plt.colorbar()
plt.title("Vector Distribution BEFORE Quantum Fusion")
plt.savefig("vector_before.png")
plt.show()
