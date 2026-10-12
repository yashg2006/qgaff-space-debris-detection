import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from file1 import QuantumFusionModel

# Zenodo dataset: usually shares the same 11 debris classes
NUM_CLASSES = 11

OUT_DIR = "Quantum_fusion"

def train():
    # Load raw features and labels
    try:
        features_raw = np.load(os.path.join(OUT_DIR, "features.npy"))
        labels_raw   = np.load(os.path.join(OUT_DIR, "labels.npy"))
    except FileNotFoundError:
        print("Error: features.npy or labels.npy not found in current directory.")
        return

    # Filter out invalid labels (-1)
    mask = labels_raw != -1
    features = torch.tensor(features_raw[mask], dtype=torch.float32)
    labels   = torch.tensor(labels_raw[mask],   dtype=torch.long)

    print(f"[train_zenodo] Training on {len(features)} samples (Zenodo dataset)...")

    # Initialize model
    model = QuantumFusionModel(num_classes=NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 30
    batch_size = 16

    print("\n--- Starting Fine-Tuning (Zenodo) ---")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Simple shuffling
        idx = torch.randperm(features.size(0))
        batch_features = features[idx]
        batch_labels   = labels[idx]

        for i in range(0, batch_features.size(0), batch_size):
            x = batch_features[i:i+batch_size]
            y = batch_labels[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(features)/batch_size)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Generate and save fine-tuned features for the FULL (filtered) dataset
    print("\n[train_zenodo] Generating fine-tuned features...")
    model.eval()
    with torch.no_grad():
        fine_tuned_outputs = model(features).numpy()

    save_path = os.path.join(OUT_DIR, "quantum_features.npy")
    np.save(save_path, fine_tuned_outputs)
    print(f"[train_zenodo] Updated {save_path} (shape: {fine_tuned_outputs.shape})")

if __name__ == "__main__":
    train()
