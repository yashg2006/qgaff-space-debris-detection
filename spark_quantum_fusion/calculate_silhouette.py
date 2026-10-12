import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import os

OUT_DIR = "spark_quantum_fusion"

def calculate_metrics():
    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        after  = np.load(os.path.join(OUT_DIR, "quantum_features.npy"))  # (17725, 11)

        # Use remapped contiguous labels when available
        remapped_path = os.path.join(OUT_DIR, "labels_remapped.npy")
        if os.path.exists(remapped_path):
            labels = np.load(remapped_path)
            print("[silhouette] Using labels_remapped.npy (contiguous class ids)")
        else:
            labels = np.load(os.path.join(OUT_DIR, "labels.npy"))
            print("[silhouette] Using labels.npy")

        # features.npy may have more rows than labels (e.g. 22000 vs 17725)
        # because it was saved over all images while labels only covers detections.
        # We slice features to match the label count — they correspond row-for-row
        # (the fast_train script trained on the first N=len(labels) valid samples).
        features_raw = np.load(os.path.join(OUT_DIR, "features.npy"))
        N = len(labels)
        before = features_raw[:N]   # trim to the labelled subset

        assert len(before) == N, f"before length mismatch: {len(before)} vs {N}"
        assert len(after)  == N, f"after  length mismatch: {len(after)} vs {N}"

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # ── L2-normalise ──────────────────────────────────────────────────────────
    before_norm = normalize(before, norm="l2")
    after_norm  = normalize(after,  norm="l2")

    print(f"\n--- Silhouette Score Analysis (Spark) ---")
    print(f"Data samples  : {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Num classes   : {len(unique)}")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {int(cls):2d}: {cnt:5d} samples")

    # ── Compute Silhouette ────────────────────────────────────────────────────
    score_before = silhouette_score(before_norm, labels, metric="cosine")
    score_after  = silhouette_score(after_norm,  labels, metric="cosine")

    improvement = (
        (score_after - score_before) / abs(score_before) * 100
        if score_before != 0 else 0
    )

    print(f"\nSilhouette Score (Before Quantum Fusion): {score_before:.4f}")
    print(f"Silhouette Score (After  Quantum Fusion): {score_after:.4f}")
    print(f"Relative Improvement                    : {improvement:.2f}%")

    if score_after > 0:
        print("\n[OK] POSITIVE silhouette score achieved after quantum fusion!")
    else:
        print("\n[!!] Score still negative -- consider more training epochs.")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "silhouette_score.txt"), "w") as f:
        f.write("--- Silhouette Score Analysis (Spark Space Debris) ---\n")
        f.write(f"Before Fusion: {score_before:.4f}\n")
        f.write(f"After Fusion:  {score_after:.4f}\n")
        f.write(f"Relative Improvement: {improvement:.2f}%\n")

    print(f"\n[silhouette] Results saved to {OUT_DIR}/silhouette_score.txt")

if __name__ == "__main__":
    calculate_metrics()
