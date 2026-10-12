# CHANGES_REPORT — SPARK labels, SPARK detection, SPARK feature separability

Date: 2026-10-13. Scope: Task 1 (label fix), A (detection with correct boxes),
B (balanced SPARK subset: raw vs MLP vs QGAFF). Earlier Tasks 3–7 were skipped on request.
The manuscript text and the Zenodo DOI were **not** edited.

All numbers below are copied from result files produced in this session:
`revision_experiments/results/spark_detection_rowcol.json` and
`revision_experiments/results/spark_balanced_eval.json`.

Interpreter for every command: system CPython 3.14.2 on Windows
(system Python 3.14.2; `.venv` lacks the packages). Exact package versions: `requirements_used.txt`.
Commands are run from the project root; `python` below means that interpreter.

---

## 1. SPARK label conversion (Task 1)

**Finding.** The SPARK CSV `bbox` is `[rmin, cmin, rmax, cmax]` = `[ymin, xmin, ymax, xmax]`
(rows first). This was decided by drawing both readings on 30 random val images and opening
all 30 (`spark_label_verify/rowcol_vs_xyxy/`): the rows-first reading enclosed the object
in all 30; the `[xmin, ymin, xmax, ymax]` reading did so only in 2 (objects near the diagonal).

Every label set that existed before was wrong:

| Label set | Written by | Problem |
|---|---|---|
| `spark-2022-stream-1/{train,val}/<split>/*.txt` (next to images) | old `spark_yolo/csv_to_yolo.py` | read bbox as `[x, y, w, h]`; coords clipped to 1.0; every line written twice |
| `spark-2022-stream-1/labels/{train,val}/*.txt` and `val_subset/labels/` | `fix_labels.py` (also older `csv_to_txt.py`) | read bbox as `[xmin, ymin, xmax, ymax]` → x and y swapped |
| `spark_yolo/labels_xyxy/` (made earlier this session, from your formula) | — | x and y swapped; **deleted** after the new labels passed the check |

**Fix.** `spark_yolo/csv_to_yolo.py` → `spark_yolo/labels_rowcol/{train,val}` (66,000 + 22,000
files, 0 skipped, 0 clipped by the converter).

**Check of the new labels** (`python verify_spark_labels.py --labels spark_yolo/labels_rowcol/val --out spark_label_verify/labels_rowcol_val`,
30 random val images, seed 42). I opened all 30 overlays, plus brightened full-resolution crops of 5 faint ones.
- Box on the object: **30/30**. Script counts: 0 empty, 0 outside, 5 touching the image border (objects cut off by the frame).
- Faint but inside the box (seen only in crops): img022817 (debris), img045562, img038191, img014211 (edge of cheops cut by the frame).
- **img082784 (smart_1, box x 421–620, y 162–1024): extent not confirmed.** The lit panel lines sit in the lower-middle and span the box width; the top ~60% of the box shows only stars even when brightened. It may be unlit body; I cannot tell from the pixels.

**Wiring.** `spark.yaml` now uses image lists `spark_yolo/rowcol_{train,val}.txt` built by
`python spark_yolo/make_rowcol_dataset.py` (junctions in `%USERPROFILE%\spark_rowcol_links\`,
outside OneDrive). Pointing a yaml at the image folders would silently load the old wrong
`.txt` files next to the images. Loading 20 val images through Ultralytics gave the
`labels_rowcol` boxes for 20/20.

**Other setup commits.** `544412cf` baseline; `58816af8` requirements_used.txt; `bd2a27e4`
untracked six `.npy`/`.png` files (still on disk); `0d57eed8` Task 1.

---

## 2. Where the old SPARK numbers came from

- `spark_quantum_fusion/labels.npy` / `labels_remapped.npy` (n=763, K=7) are **not ground truth**.
  `spark_extract_features_subset.py` stores the class **predicted** by `runs/detect/train3`
  (the Roboflow model) for the first 1,000 sorted images of `spark_yolo/images/val`, dropping
  images with no detection (1,000 → 763). Per the CSV, **all 1,000 of those images are proba_2**.
  The 7 "classes" are detector guesses on one true class. No box label file was used.
  These labels feed experiments 1, 1b, 2, 3, 4, 4b, 5, 7b and Table 5.
- `latest_metrics.txt` / `research_paper_spark_results.txt` (P 0.2934, R 0.0326): `save_metrics.py`,
  train3 on `val_subset` (300 random val images, made by `make_subset.py`, seed 42) with
  `fix_labels.py` labels (x/y swapped).
- `runs/detect/runs/spark/opt_train{2,3}`: YOLOv8n trained on 3% of SPARK with the old wrong labels
  (final mAP@0.5 0 and 0.003). No feature or metric script loads these weights; every SPARK
  result uses train3, which was trained on Roboflow only.

---

## 3. A — SPARK detection with correct boxes

Command: `python revision_experiments/spark_detection_rowcol.py`
Model `runs/detect/train3/weights/best.pt` (Roboflow-trained; no SPARK training), the same 300
`val_subset` images as before, imgsz 640, CPU. `labels_rowcol` classes (debris=0) were
remapped by name to the model's ids (alphabetical, debris=1; model spells `double_start`).

| Run | Labels | conf | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| old reproduced | val_subset (x/y swapped) | 0.25 | 0.2934 | 0.0326 | 0.0314 | 0.0126 |
| **new** | labels_rowcol | 0.25 | **0.4739** | **0.2883** | **0.3274** | **0.1387** |
| **new** | labels_rowcol | 0.001 (Ultralytics default) | **0.4739** | **0.2883** | **0.2854** | **0.1069** |

The old run reproduces `latest_metrics.txt` exactly, so the change comes from the labels alone.

Per class (n = images of that class among the 300):

| Class | n | old P | old R | old mAP50 | new P | new R | new mAP50 (0.25) | new mAP50-95 (0.25) | new mAP50 (0.001) | new mAP50-95 (0.001) |
|---|---|---|---|---|---|---|---|---|---|---|
| cheops | 35 | 0.164 | 0.086 | 0.086 | 0.579 | 0.429 | 0.548 | 0.289 | 0.525 | 0.252 |
| debris | 23 | 0.000 | 0.000 | 0.000 | 0.899 | 0.304 | 0.525 | 0.110 | 0.454 | 0.091 |
| double_star | 35 | 0.000 | 0.000 | 0.000 | 0.088 | 0.086 | 0.065 | 0.015 | 0.051 | 0.015 |
| earth_observation_sat_1 | 29 | 0.000 | 0.000 | 0.000 | 0.083 | 0.035 | 0.045 | 0.032 | 0.022 | 0.010 |
| lisa_pathfinder | 20 | 0.186 | 0.050 | 0.059 | 0.474 | 0.550 | 0.514 | 0.194 | 0.471 | 0.161 |
| proba_2 | 27 | 0.657 | 0.072 | 0.100 | 0.611 | 0.296 | 0.454 | 0.252 | 0.431 | 0.214 |
| proba_3_csc | 28 | 0.028 | 0.071 | 0.028 | 0.100 | 0.464 | 0.155 | 0.062 | 0.143 | 0.055 |
| proba_3_ocs | 25 | 0.000 | 0.000 | 0.023 | 0.570 | 0.425 | 0.488 | 0.161 | 0.449 | 0.135 |
| smart_1 | 25 | 0.194 | 0.080 | 0.049 | 0.355 | 0.320 | 0.240 | 0.100 | 0.212 | 0.073 |
| soho | 29 | 1.000 | 0.000 | 0.000 | 0.477 | 0.095 | 0.123 | 0.057 | 0.124 | 0.056 |
| xmm_newton | 24 | 1.000 | 0.000 | 0.000 | 0.977 | 0.167 | 0.444 | 0.255 | 0.257 | 0.114 |

(Per-class P and R are the same at both conf values, so they are listed once.)

**Open point:** mAP is lower at conf 0.001 than at 0.25 (largest drop: xmm_newton). A likely cause,
not verified: Ultralytics matches predictions to ground truth by IoU, not confidence, so extra
low-confidence boxes can take matches from higher-confidence ones. I have not decided which conf
the paper should report. 0.001 is the Ultralytics default for mAP; 0.25 is what the old number used.

---

## 4. B — Balanced SPARK subset: raw vs MLP vs QGAFF

Commands:
`python revision_experiments/spark_balanced_extract.py` then `python revision_experiments/spark_balanced_eval.py`

- Data: 70 val images per class × 11 = 770, sampled from `val.csv` with seed 42
  (`revision_experiments/spark_balanced/subset.csv`); labels = **ground truth** from the CSV.
  Features: train3 layer-8 output, global-average-pooled (same hook as the old extractor);
  no image dropped. Boxes are not used.
- One stratified 70/30 split (random_state 42) shared by all methods and seeds:
  539 train / 231 test (21 per class). Seeds 0, 7, 42.
- One config for MLP and QGAFF (experiment_5 settings): CE + 0.3·CenterLoss, Adam lr 3e-3,
  wd 1e-4 (centers: Adam lr 5e-2), cosine LR to 1e-5, 60 epochs, batch 32, 32-d embedding.
  QGAFF = `QGAFFModel` imported from `experiment_5_multiseed_stats.py`; MLP = `SimpleMLP`
  imported from `experiment_1_classical_baselines.py`.
- Raw baseline: no training. Accuracy from logistic regression on standardized raw features,
  which is deterministic, so std is 0.
- Metrics are computed on the **test split only**: cosine silhouette of L2-normalized vectors,
  and test accuracy. Values are mean ± sample std over 3 seeds.

| Method | Trainable params | Silhouette (test) | Accuracy (test) |
|---|---|---|---|
| Raw layer-8 features | 0 | −0.1893 ± 0.0000 | 0.7056 ± 0.0000 |
| MLP + center loss | 44,043 | +0.2001 ± 0.0056 | 0.7128 ± 0.0109 |
| QGAFF + center loss | 9,831 | +0.1218 ± 0.0052 | 0.7201 ± 0.0066 |

Per seed (0, 7, 42): MLP silhouette 0.2021 / 0.2043 / 0.1938, accuracy 0.7229 / 0.7143 / 0.7013;
QGAFF silhouette 0.1158 / 0.1253 / 0.1243, accuracy 0.7186 / 0.7273 / 0.7143.

Reading: on ground-truth labels and held-out images, QGAFF's accuracy is close to the MLP's and
to logistic regression on raw features. Its silhouette is lower than the MLP's.

---

## 5. Old value → new value

"UNSOURCED" = no file or script in the repository produces this number.

| Where | Quantity | Old value | Source of old value | New value (this session) |
|---|---|---|---|---|
| Manuscript Table 4 | SPARK Precision | 0.85 | **UNSOURCED** | 0.4739 |
| Manuscript Table 4 | SPARK Recall | 0.81 | **UNSOURCED** | 0.2883 |
| Manuscript Table 4 | SPARK mAP@0.5 | 0.79 | **UNSOURCED** | 0.3274 (conf 0.25) / 0.2854 (conf 0.001) |
| Manuscript Table 4 | SPARK mAP@0.5:0.95 | 0.510 | **UNSOURCED** | 0.1387 (conf 0.25) / 0.1069 (conf 0.001) |
| latest_metrics.txt | SPARK P / R / mAP50 / mAP50-95 | 0.2934 / 0.0326 / 0.0314 / 0.0126 | save_metrics.py, x/y-swapped labels | 0.4739 / 0.2883 / 0.3274 / 0.1387 (same conf 0.25) |
| Manuscript Table 5 + abstract | SPARK silhouette, original features | −0.2967 | **UNSOURCED** | −0.1893 |
| Manuscript Table 5 + abstract | SPARK silhouette, QGAFF | +0.6060 | **UNSOURCED** | +0.1218 ± 0.0052 |
| Manuscript Table 5 | Absolute improvement | +0.9027 | **UNSOURCED** | +0.3111 (from the two values above) |
| spark_quantum_fusion/silhouette_score.txt | before / after | −0.2729 / 0.7187 | predicted labels, one true class | −0.1893 / +0.1218 |
| multiseed_stats.json (exp 5) | SPARK raw / QGAFF silhouette | −0.2729 / 0.3855 ± 0.0319 (5 seeds) | predicted labels, one true class, train = eval | −0.1893 / +0.1218 ± 0.0052 (3 seeds) |
| baselines_comparison_extended.json (exp 1b) | SPARK QGAFF silhouette | 0.1036 ± 0.025 | same predicted labels | +0.1218 ± 0.0052 |
| baselines_comparison_extended.json | SPARK MLP silhouette | 0.2004 ± 0.0252 (no center loss); 0.5221 ± 0.0278 (center loss) | same predicted labels | +0.2001 ± 0.0056 (center loss) |
| baselines / multiseed | SPARK n, K | 763, 7 | predicted labels on 1,000 proba_2 images | 770, 11 (70 per class, ground truth) |
| QGAFF parameter count | — | 3,607 (code comments, response docs); 9,699 (K=7, json) | — | 9,831 for K=11 (counted from the model) |

The old and new rows use different images, labels and protocols (old: one true class, predicted
labels, scored on training data; new: ground truth, 11 classes, held-out test). They are not
like-for-like; the new rows replace the old ones.

---

## 6. Unresolved — needs your decision

1. **Table 4 and Table 5 / abstract SPARK numbers are UNSOURCED.** The new values above are the only reproducible ones.
2. **Which conf to report for SPARK mAP** (0.25 vs 0.001), and why mAP falls at 0.001 (cause not verified).
3. **QGAFF is not a quantum simulation in any revision script.** `QGAFFModel` computes four fixed
   cos/sin products and uses only part of `pqc_weights`. The PennyLane circuit (`quantum_shared_model.py`,
   4 qubits, StronglyEntanglingLayers, 24 circuit weights, which matches "Trainable parameters 24"
   in the manuscript) is not used by the revision experiments. All of section 4 uses `QGAFFModel`.
4. **Parameter count**: 3,607 appears in code comments and response documents; the model has
   9,831 parameters (K=11) or 9,699 (K=7).
5. **img082784** box extent not confirmed by eye (see section 1).
6. Every other SPARK result built on `spark_quantum_fusion/labels*.npy` (experiments 1, 1b, 2, 3, 4, 4b, 5, 7b
   and their figures) is invalid as a class-separability result and was not re-run (out of scope).
7. Dataset citations: Parisot’s StreaksYoloDataset is cited as a dataset (Zenodo record 14047944) and is not this repository’s DOI.
