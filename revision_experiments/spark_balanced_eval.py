"""
Task B, step 2: raw features vs MLP vs QGAFF on the balanced SPARK subset.

ONE config for every number:
  data   : revision_experiments/spark_balanced (770 images, 70/class, GT labels)
  split  : one stratified 70/30 train/test split (random_state=42), shared by all
           methods and seeds -> 539 train / 231 test (21 per class)
  seeds  : 0, 7, 42 (model init + batch order)
  models : QGAFFModel (experiment_5_multiseed_stats.py) and SimpleMLP
           (experiment_1_classical_baselines.py), both with a 32-d embedding
  train  : CE + 0.3 * CenterLoss, Adam lr 3e-3 wd 1e-4 (centers: Adam lr 5e-2),
           cosine LR to 1e-5, 60 epochs, batch 32  (= experiment_5 settings)
  raw    : no training; accuracy from LogisticRegression on standardized raw
           features (deterministic, so std is 0)
  metrics: computed on the TEST split only, with ground-truth labels.
           silhouette = cosine silhouette of L2-normalized test vectors
           (raw features or 32-d embedding); accuracy = test accuracy.

Run from project root:
    python revision_experiments/spark_balanced_eval.py
Output: revision_experiments/results/spark_balanced_eval.json
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_5_multiseed_stats import QGAFFModel, CenterLoss   # noqa: E402
from experiment_1_classical_baselines import SimpleMLP           # noqa: E402

DATA = "revision_experiments/spark_balanced"
OUT = "revision_experiments/results/spark_balanced_eval.json"
SEEDS = [0, 7, 42]
SPLIT_SEED = 42
TEST_SIZE = 0.3
EPOCHS, BATCH, LR, WD, CENTER_LR, CENTER_W, EMBED = 60, 32, 3e-3, 1e-4, 5e-2, 0.3, 32


def train(model, X, y, K, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    center = CenterLoss(num_classes=K, feat_dim=EMBED)
    opt_m = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    opt_c = optim.Adam(center.parameters(), lr=CENTER_LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=EPOCHS, eta_min=1e-5)
    X, y = torch.tensor(X), torch.tensor(y)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt_m.zero_grad()
            opt_c.zero_grad()
            emb, logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx]) + CENTER_W * center(emb, y[idx])
            loss.backward()
            opt_m.step()
            opt_c.step()
        sched.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        emb, logits = model(torch.tensor(X))
    emb = normalize(emb.numpy(), norm="l2")
    return {"silhouette": float(silhouette_score(emb, y, metric="cosine")),
            "accuracy": float(accuracy_score(y, logits.argmax(1).numpy()))}


def summary(runs, key):
    v = np.array([r[key] for r in runs])
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
            "values": [round(float(x), 4) for x in v]}


X = np.load(os.path.join(DATA, "features.npy")).astype(np.float32)
y = np.load(os.path.join(DATA, "labels.npy")).astype(np.int64)
K = len(np.unique(y))
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y,
                                  random_state=SPLIT_SEED)
Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
print(f"n={len(y)} K={K} train={len(ytr)} test={len(yte)} (per-class test: {np.bincount(yte).tolist()})")

results = {"config": {"data": DATA, "n": int(len(y)), "K": int(K), "train": int(len(ytr)),
                      "test": int(len(yte)), "split_seed": SPLIT_SEED, "test_size": TEST_SIZE,
                      "seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                      "weight_decay": WD, "center_lr": CENTER_LR, "center_weight": CENTER_W,
                      "embed_dim": EMBED, "metrics_on": "test split, ground-truth labels",
                      "silhouette": "cosine, L2-normalized vectors"}}

# Raw backbone features (no training)
scaler = StandardScaler().fit(Xtr)
raw_runs = []
for s in SEEDS:
    clf = LogisticRegression(max_iter=5000, random_state=s).fit(scaler.transform(Xtr), ytr)
    raw_runs.append({"seed": s,
                     "silhouette": float(silhouette_score(normalize(Xte), yte, metric="cosine")),
                     "accuracy": float(accuracy_score(yte, clf.predict(scaler.transform(Xte))))})
results["raw_features"] = {"params": 0, "runs": raw_runs,
                           "silhouette": summary(raw_runs, "silhouette"),
                           "accuracy": summary(raw_runs, "accuracy")}

for name, make in [("MLP", lambda: SimpleMLP(in_dim=X.shape[1], embed_dim=EMBED, num_classes=K)),
                   ("QGAFF", lambda: QGAFFModel(in_dim=X.shape[1], n_qubits=4, embed_dim=EMBED, num_classes=K))]:
    runs = []
    for s in SEEDS:
        torch.manual_seed(s)            # same init seed for the model as for training
        m = train(make(), Xtr, ytr, K, s)
        r = evaluate(m, Xte, yte)
        r["seed"] = s
        runs.append(r)
        print(f"  {name:5s} seed {s:2d}: silhouette {r['silhouette']:.4f}  accuracy {r['accuracy']:.4f}")
    results[name] = {"params": int(sum(p.numel() for p in make().parameters())), "runs": runs,
                     "silhouette": summary(runs, "silhouette"), "accuracy": summary(runs, "accuracy")}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n{'method':13s} {'params':>7s}  silhouette (mean±std)   accuracy (mean±std)")
for name in ["raw_features", "MLP", "QGAFF"]:
    r = results[name]
    print(f"{name:13s} {r['params']:>7d}  {r['silhouette']['mean']:+.4f} ± {r['silhouette']['std']:.4f}"
          f"      {r['accuracy']['mean']:.4f} ± {r['accuracy']['std']:.4f}")
print("Saved", OUT)
