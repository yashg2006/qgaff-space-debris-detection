import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import os

OUT_DIR = "spark_quantum_fusion"

SPARK_CLASSES = [
    "debris", "lisa_pathfinder", "soho", "xmm_newton", "cheops",
    "smart_1", "proba_2", "proba_3_csc", "proba_3_ocs",
    "earth_observation_sat_1", "double_star"
]

# ── Load data ────────────────────────────────────────────────────────────────
before  = np.load(os.path.join(OUT_DIR, "features.npy"))        # (N, 256)
after   = np.load(os.path.join(OUT_DIR, "quantum_features.npy"))# (M, 11) or (N, 11)
labels  = np.load(os.path.join(OUT_DIR, "labels.npy"))          # (N,)

# Filter out invalid labels (-1)
# Note: labels.npy and labels_remapped.npy are already filtered in this environment
remapped_path = os.path.join(OUT_DIR, "labels_remapped.npy")
if os.path.exists(remapped_path):
    labels = np.load(remapped_path)
    print("[spark_plot_vectors] Using labels_remapped.npy")
else:
    labels = np.load(os.path.join(OUT_DIR, "labels.npy"))
    print("[spark_plot_vectors] Using labels.npy")

N = len(labels)
before = before[:N]
after = after[:N]

print(f"[spark_plot_vectors] before  shape : {before.shape}")
print(f"[spark_plot_vectors] after   shape : {after.shape}")
print(f"[spark_plot_vectors] labels  shape : {labels.shape}")
print(f"[spark_plot_vectors] Running t-SNE …")

# ── t-SNE ────────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(before)-1))
before_2d = tsne.fit_transform(before)
after_2d  = tsne.fit_transform(after)

print("[spark_plot_vectors] t-SNE done.")

# ── Colour map ───────────────────────────────────────────────────────────────
n_classes = len(SPARK_CLASSES)
cmap      = cm.get_cmap("tab20", n_classes)
colors    = [cmap(i) for i in range(n_classes)]

def scatter_with_legend(ax, xy, labels, title):
    for cls_idx, cls_name in enumerate(SPARK_CLASSES):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=8, color=colors[cls_idx],
            label=cls_name, alpha=0.75
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

# ── Plot before ──────────────────────────────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(8, 7))
scatter_with_legend(ax_b, before_2d, labels, "Before Quantum Fusion  (Spark)")
ax_b.legend(loc="lower right", fontsize=7, markerscale=2,
            title="Classes", title_fontsize=8)
fig_b.tight_layout()
path_before = os.path.join(OUT_DIR, "vector_before.png")
fig_b.savefig(path_before, dpi=300)
print(f"[spark_plot_vectors] Saved {path_before}")

# ── Plot before + after side by side ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
scatter_with_legend(axes[0], before_2d, labels, "Before Quantum Fusion  (Spark)")
scatter_with_legend(axes[1], after_2d,  labels, "After Quantum Fusion   (Spark)")

# Shared legend
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=6, fontsize=8,
           markerscale=2, title="Classes", title_fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Quantum Feature Fusion – Spark Space Debris Dataset",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 1])

path_comparison = os.path.join(OUT_DIR, "vector_comparison.png")
fig.savefig(path_comparison, dpi=300, bbox_inches="tight")
print(f"[spark_plot_vectors] Saved {path_comparison}")
plt.show()
