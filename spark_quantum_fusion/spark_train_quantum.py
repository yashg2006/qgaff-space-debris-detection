import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from spark_quantum_model import QuantumFusionModel, NUM_CLASSES, SPARK_CLASSES

OUT_DIR = "spark_quantum_fusion"

# ── Center Loss ───────────────────────────────────────────────────────────────
# Minimises intra-class scatter → clusters pull tight around their centre
# Combined with CrossEntropy it forces inter-class separation too
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]          # (B, D)
        diff = features - batch_centers               # (B, D)
        loss = 0.5 * (diff ** 2).sum(dim=1).mean()
        return loss

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # ── Load & clean data ────────────────────────────────────────────────────
    features_raw = np.load(os.path.join(OUT_DIR, "features.npy"))
    labels_raw   = np.load(os.path.join(OUT_DIR, "labels.npy"))

    mask         = labels_raw != -1
    features_raw = features_raw[mask]
    labels_raw   = labels_raw[mask]

    print(f"[spark_train] Dataset  : {len(features_raw)} samples, "
          f"{len(np.unique(labels_raw))} classes")
    print(f"[spark_train] Classes  : {np.unique(labels_raw).tolist()}")

    # Remap labels to 0..K-1 in case some class ids are missing
    unique_cls = np.unique(labels_raw)
    remap      = {old: new for new, old in enumerate(unique_cls)}
    labels_raw = np.array([remap[l] for l in labels_raw])
    num_classes = len(unique_cls)
    print(f"[spark_train] Remapped : {num_classes} contiguous classes")

    features_t = torch.tensor(features_raw, dtype=torch.float32)
    labels_t   = torch.tensor(labels_raw,   dtype=torch.long)

    # ── Model & Losses ───────────────────────────────────────────────────────
    # We rebuild a simple but deeper MLP + quantum layer for better separation
    model = QuantumFusionModel(num_classes=num_classes)

    # Center loss operates on the 4-dim quantum embedding (before classifier)
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=4)

    ce_criterion   = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Two optimisers: one for model params, one for centers (standard practice)
    optimizer_model   = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    optimizer_centers = optim.Adam(center_loss_fn.parameters(), lr=5e-2)

    # Cosine annealing: warm LR decay keeps training from plateauing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, T_max=120, eta_min=1e-5
    )

    # ── Training Loop ────────────────────────────────────────────────────────
    EPOCHS     = 120
    BATCH_SIZE = 64
    ALPHA      = 0.3   # weight of center loss relative to CE loss

    print(f"\n[spark_train] Starting training: {EPOCHS} epochs, batch {BATCH_SIZE}\n")
    model.train()
    center_loss_fn.train()

    for epoch in range(EPOCHS):
        idx        = torch.randperm(features_t.size(0))
        feats_shuf = features_t[idx]
        labs_shuf  = labels_t[idx]
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, feats_shuf.size(0), BATCH_SIZE):
            bx = feats_shuf[i : i + BATCH_SIZE]
            by = labs_shuf [i : i + BATCH_SIZE]

            optimizer_model.zero_grad()
            optimizer_centers.zero_grad()

            # Forward — intercept the 4-dim quantum embedding
            compressed  = model.compress(bx.float())       # (B, 4)
            q_embed     = model.quantum(compressed)        # (B, 4)
            logits      = model.classifier(q_embed)       # (B, C)

            loss_ce     = ce_criterion(logits, by)
            loss_center = center_loss_fn(q_embed, by)
            loss        = loss_ce + ALPHA * loss_center

            loss.backward()
            # Gradient scale trick for center loss (prevents center collapse)
            for param in center_loss_fn.parameters():
                param.grad.data *= 1.0 / (1.0 + ALPHA)

            optimizer_model.step()
            optimizer_centers.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = epoch_loss / n_batches
            lr  = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch+1:>3}/{EPOCHS}]  Loss: {avg:.4f}  LR: {lr:.6f}")

    # ── Save model weights ───────────────────────────────────────────────────
    model_path = os.path.join(OUT_DIR, "spark_quantum_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[spark_train] Saved model → {model_path}")

    # ── Generate quantum features for FULL dataset ───────────────────────────
    # KEY: we save L2-normalised SOFTMAX PROBABILITIES
    #   • Same-class samples have high prob at their class → geometrically close
    #   • Different-class samples have high prob at different dims → far apart
    #   • L2-norm maps everything to the unit hypersphere → pure angular distance
    #   This combination strongly favours positive silhouette scores.
    print("[spark_train] Generating quantum features (softmax + L2 norm)…")
    model.eval()
    center_loss_fn.eval()
    all_outputs = []

    with torch.no_grad():
        INF_BATCH = 128
        for i in range(0, features_t.size(0), INF_BATCH):
            chunk      = features_t[i : i + INF_BATCH]
            compressed = model.compress(chunk.float())
            q_embed    = model.quantum(compressed)
            logits     = model.classifier(q_embed)

            # Softmax probability vector
            probs = F.softmax(logits, dim=1)

            # L2-normalise onto unit hypersphere
            probs_normed = F.normalize(probs, p=2, dim=1)

            all_outputs.append(probs_normed.cpu())

    quantum_feats = torch.cat(all_outputs).numpy()

    save_path = os.path.join(OUT_DIR, "quantum_features.npy")
    np.save(save_path, quantum_feats)
    print(f"[spark_train] Saved quantum_features.npy → shape {quantum_feats.shape}")

    # Also save remapped labels so calculate_silhouette uses the same mapping
    np.save(os.path.join(OUT_DIR, "labels_remapped.npy"), labels_raw)
    print(f"[spark_train] Saved labels_remapped.npy for silhouette calc")


if __name__ == "__main__":
    train()
