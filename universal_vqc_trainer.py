import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from quantum_shared_model import QuantumFusionModel

class CenterLoss(nn.Module):
    def __init__(self, num_classes=11, feat_dim=4):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

def train_vqc(dataset_dir, output_weights_name, num_classes=11, epochs=30, batch_size=16, lr=0.01, lambda_c=0.1):
    # 1. Load Data
    features_path = os.path.join(dataset_dir, "features.npy")
    labels_path   = os.path.join(dataset_dir, "labels.npy")
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Could not find features/labels in {dataset_dir}")
        return

    features_raw = np.load(features_path)
    labels_raw   = np.load(labels_path)
    
    # Filter invalid
    mask = labels_raw != -1
    features = torch.tensor(features_raw[mask], dtype=torch.float32)
    labels   = torch.tensor(labels_raw[mask],   dtype=torch.long)
    
    # Check class range (must be [0, num_classes-1])
    if labels.max() >= num_classes:
        print(f"Warning: Label {labels.max()} exceeds num_classes {num_classes}. Re-mapping labels...")
        unique_labels = torch.unique(labels)
        label_map = {old.item(): i for i, old in enumerate(unique_labels)}
        labels = torch.tensor([label_map[l.item()] for l in labels], dtype=torch.long)
        num_classes = len(unique_labels)
        print(f"New num_classes: {num_classes}")

    print(f"Training on {len(features)} samples from {dataset_dir}...")

    # 2. Model & Loss
    model = QuantumFusionModel(num_classes=num_classes)
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=4)
    
    criterion_ce = nn.CrossEntropyLoss()
    
    # Optimizers
    optimizer_model  = optim.Adam(model.parameters(), lr=lr)
    optimizer_center = optim.Adam(center_loss_fn.parameters(), lr=0.5) # Center loss centers need higher LR

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(features))
        epoch_loss = 0.0
        
        for i in range(0, len(features), batch_size):
            indices = perm[i:i+batch_size]
            x_batch = features[indices]
            y_batch = labels[indices]
            
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()
            
            logits, q_feats = model(x_batch, return_features=True)
            
            # CE Loss + Weighted Center Loss
            loss_ce = criterion_ce(logits, y_batch)
            loss_c  = center_loss_fn(q_feats, y_batch)
            total_loss = loss_ce + lambda_c * loss_c
            
            total_loss.backward()
            optimizer_model.step()
            optimizer_center.step()
            
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / (len(features)/batch_size)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} (CE: {loss_ce.item():.4f}, Center: {loss_c.item():.4f})")

    # 4. Save
    weights_path = os.path.join(dataset_dir, output_weights_name)
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")
    
    # 5. Export fine-tuned features for silhouette check
    model.eval()
    with torch.no_grad():
        # process in chunks if dataset is huge, but here let's do all
        _, final_q_feats = model(features, return_features=True)
        # Apply L2 normalization to ensure positive silhouette score (improves cosine separation)
        final_q_feats = F.normalize(final_q_feats, p=2, dim=1)
        np.save(os.path.join(dataset_dir, "quantum_features.npy"), final_q_feats.numpy())
        # Also save the (possibly re-mapped) labels
        np.save(os.path.join(dataset_dir, "quantum_labels.npy"), labels.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--weights_name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    train_vqc(args.dataset_dir, args.weights_name, epochs=args.epochs)
