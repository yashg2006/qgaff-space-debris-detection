import numpy as np
from sklearn.metrics import silhouette_score
import os

OUT_DIR = "Quantum_fusion"

def calculate_metrics():
    # Load data
    try:
        before  = np.load(os.path.join(OUT_DIR, "features.npy"))        
        after   = np.load(os.path.join(OUT_DIR, "quantum_features.npy"))
        labels  = np.load(os.path.join(OUT_DIR, "labels.npy"))
        
        # Filter out invalid labels (-1)
        mask = labels != -1
        if len(after) != len(before):
            # Assume 'after' is already filtered from 239 -> ~200
            before = before[mask]
            labels = labels[mask]
        else:
            before = before[mask]
            after = after[mask]
            labels = labels[mask]

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"--- Silhouette Score Analysis (Quantum Fusion) ---")
    print(f"Data samples: {len(labels)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Calculate Silhouette Score for 'Before'
    score_before = silhouette_score(before, labels)
    print(f"Silhouette Score (Before Quantum Fusion): {score_before:.4f}")
    
    # Calculate Silhouette Score for 'After'
    score_after = silhouette_score(after, labels)
    print(f"Silhouette Score (After Quantum Fusion):  {score_after:.4f}")
    
    improvement = (score_after - score_before) / abs(score_before) * 100 if score_before != 0 else 0
    print(f"Relative Improvement: {improvement:.2f}%")

    # Save results to text file
    with open(os.path.join(OUT_DIR, "silhouette_score.txt"), "w") as f:
        f.write("--- Silhouette Score Analysis (Quantum Fusion) ---\n")
        f.write(f"Before Fusion: {score_before:.4f}\n")
        f.write(f"After Fusion:  {score_after:.4f}\n")
        f.write(f"Relative Improvement: {improvement:.2f}%\n")

if __name__ == "__main__":
    calculate_metrics()
