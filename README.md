# Quantum-Guided Feature Transformation for Cross-Domain Space-Debris Detection

Code for QGAFF, a feature transformation applied to YOLOv8 backbone features (layer-8 C2f
output, global-average-pooled to 256-d). The detector is trained on the Roboflow
space-debris dataset only and is then evaluated, unchanged, on two other datasets:
SPARK-2022 and StreaksYoloDataset. The code covers:

- conversion of SPARK-2022 CSV labels to YOLO format,
- cross-domain detection metrics on SPARK-2022,
- layer-8 feature extraction, and silhouette / accuracy evaluation of raw features vs an MLP vs QGAFF,
- an image-level robustness study (noise, brightness, blur) on the Roboflow test split.

Datasets and model weights are **not** included (see [Data](#data)).

> **Important: which SPARK results are valid.** An audit during revision found that earlier
> SPARK results used detector-predicted labels or box labels with x and y swapped. Read
> [`CHANGES_REPORT.md`](CHANGES_REPORT.md) before using any SPARK number. The valid SPARK
> results come only from the scripts listed under [Valid scripts](#valid-scripts).

## QGAFF implementations: surrogate vs PennyLane

- `QGAFFModel` in the revision experiments (`revision_experiments/experiment_*.py`, imported by
  `revision_experiments/spark_balanced_eval.py`) is a **classical cos/sin surrogate** of the
  quantum circuit. It is **not** a PennyLane simulation: it computes four fixed products of
  cosines and sines of the compressed inputs and uses only part of its `pqc_weights`. All QGAFF
  numbers in `CHANGES_REPORT.md` come from this surrogate. It has 9,831 trainable parameters for
  11 classes.
- The PennyLane circuit (4 qubits, `AngleEmbedding` + 2 `StronglyEntanglingLayers`, 24 circuit
  weights) is in **`quantum_shared_model.py`** (`QuantumFusionModel`). Related PennyLane code is in
  `Quantum_fusion/file1.py` and `spark_quantum_fusion/spark_quantum_model.py`.

## Requirements

- Python **3.14.2** (CPython, Windows). The exact package versions used are in
  [`requirements_used.txt`](requirements_used.txt) (a full `pip freeze`). Key packages: torch 2.10.0
  (CPU), ultralytics 8.4.14, pennylane 0.43.1, scikit-learn 1.8.0, opencv-python 4.13.0.92, numpy 2.4.0,
  pandas 2.3.3.
- `pip install -r requirements_used.txt`
- No GPU is needed for the commands below.
- `revision_experiments/spark_detection_rowcol.py` and `spark_yolo/make_rowcol_dataset.py` create
  Windows directory junctions (`mklink /J`), so they run on Windows only as written.

## Data

| Dataset | Role | Source |
|---|---|---|
| Roboflow space-debris dataset (v2, CC BY 4.0) | training (source domain) | https://universe.roboflow.com/woah-noah/space-debris-mugw2/dataset/2 |
| SPARK-2022 (Stream 1) | cross-domain evaluation | https://doi.org/10.5281/zenodo.6599762 |
| StreaksYoloDataset (Parisot) | cross-domain evaluation | O. Parisot, "StreaksYoloDataset: Labeled raw astronomical images for streaks detection," 2024, Zenodo dataset, https://doi.org/10.5281/zenodo.14047944 |

Expected local layout (all of it is ignored by git):

```
Roboflow Dataset/                        # Roboflow export: train/ valid/ test/ (images + labels)
spark-2022-stream-1/
    labels/train.csv, labels/val.csv     # SPARK CSV labels
    train/train/*.jpg, val/val/*.jpg     # SPARK images
runs/detect/train3/weights/best.pt       # YOLOv8n trained on the Roboflow dataset (not included)
```

`runs/detect/train3/weights/best.pt` is a YOLOv8n detector trained on the Roboflow dataset only
(training arguments: `yolov8n.pt`, 15 epochs, imgsz 640). It never saw SPARK images. It is not
included, so you need to train your own before running the commands below.

**SPARK bounding boxes.** The `bbox` column of the SPARK CSVs is
**`[row_min, col_min, row_max, col_max]`**, i.e. `[ymin, xmin, ymax, xmax]`, **not**
`[xmin, ymin, xmax, ymax]`. For example, `"[375, 273, 610, 496]"` means x 273–496 and y 375–610.
This was confirmed by drawing both readings on 30 random validation images (see `CHANGES_REPORT.md`, section 1).

## Reproducing the valid results

Run all commands from the repository root.

1. **SPARK labels → YOLO format** (writes `spark_yolo/labels_rowcol/{val,train}/`, 22,000 + 66,000 files):

   ```
   python spark_yolo/csv_to_yolo.py
   python verify_spark_labels.py --labels spark_yolo/labels_rowcol/val --out spark_label_verify/labels_rowcol_val
   ```

   The second command draws the boxes on 30 random validation images (seed 42) and reports empty,
   clipped or outside-image boxes. Open the overlays to check them.
   Optional: `python spark_yolo/make_rowcol_dataset.py` makes `spark.yaml` usable for Ultralytics
   training or validation on the corrected labels.

2. **SPARK cross-domain detection** (300-image `val_subset`, imgsz 640, CPU):

   ```
   python fix_labels.py        # old x/y-swapped labels; needed only for the "old labels" comparison run
   python make_subset.py       # copies 300 random val images (seed 42) to spark-2022-stream-1/val_subset/
   python revision_experiments/spark_detection_rowcol.py
   ```

   Output: `revision_experiments/results/spark_detection_rowcol.json`.
   - With corrected labels at conf 0.25: P 0.4739, R 0.2883, mAP@0.5 0.3274, mAP@0.5:0.95 0.1387.
   - At conf 0.001 (Ultralytics default): mAP@0.5 0.2854, mAP@0.5:0.95 0.1069.
   - `make_subset.py` samples from `os.listdir` order. The same 300 images are selected only if the
     listing order matches (alphabetical on NTFS).

3. **Balanced SPARK subset: raw features vs MLP vs QGAFF** (770 val images, 70 per class,
   ground-truth labels, one stratified 70/30 split, seeds 0, 7, 42, metrics on the test split only):

   ```
   python revision_experiments/spark_balanced_extract.py
   python revision_experiments/spark_balanced_eval.py
   ```

   Output: `revision_experiments/results/spark_balanced_eval.json`. The images used are listed in
   `revision_experiments/spark_balanced/subset.csv`.

   | Method | Params | Silhouette (test) | Accuracy (test) |
   |---|---|---|---|
   | Raw layer-8 features | 0 | −0.1893 ± 0.0000 | 0.7056 ± 0.0000 |
   | MLP + center loss | 44,043 | +0.2001 ± 0.0056 | 0.7128 ± 0.0109 |
   | QGAFF (surrogate) + center loss | 9,831 | +0.1218 ± 0.0052 | 0.7201 ± 0.0066 |

4. **Image-level robustness study (Roboflow test split, 123 images):**

   ```
   python Revision_Image_Outputs/code/run_study.py runs/detect/train3/weights/best.pt "Roboflow Dataset/test" Revision_Image_Outputs/figures/image_perturbation_results.json
   ```

   `Revision_Image_Outputs/figures/robustness_curves.png` plots values from
   `image_perturbation_results.json`. The script that drew this figure is not in the repository.

## Valid scripts

- `spark_yolo/csv_to_yolo.py`, `spark_yolo/make_rowcol_dataset.py`, `verify_spark_labels.py`
- `revision_experiments/spark_detection_rowcol.py`
- `revision_experiments/spark_balanced_extract.py`, `revision_experiments/spark_balanced_eval.py`
- `Revision_Image_Outputs/code/*` (`run_study.py`, `perturb.py`, `yolo_np.py`, `make_figs.py`)

## Legacy scripts (do not use for SPARK results)

These produce SPARK numbers from `spark_quantum_fusion/labels*.npy`. Those labels are classes
**predicted** by the detector on 1,000 SPARK images that are all one true class (proba_2), and the
scores were computed on the training data. They are kept for transparency only.

- `spark_quantum_fusion/*` and `spark_extract_features_subset.py`
- `revision_experiments/experiment_1_classical_baselines.py`, `experiment_1b_param_matched_control.py`,
  `experiment_2_representation_analysis.py`, `experiment_3_noise_robustness.py`,
  `experiment_4_ablation_study.py`, `experiment_4b_missing_ablation_controls.py`,
  `experiment_5_multiseed_stats.py`, and the helpers `check_quantum_feats.py`, `check_raw.py`,
  `test_*.py`, plus `run_all_revision_experiments.py`
- `revision_experiments/experiment_6_yolo_scalability.py` (also uses detector-predicted SPARK labels)
- `fix_labels.py`, `make_subset.py` and `save_metrics.py` / `run_cross_domain.py`: these made or used
  the old x/y-swapped SPARK box labels. `make_subset.py` is still used to select the 300 images.

Their outputs are in [`legacy_invalid_spark_results/`](legacy_invalid_spark_results/). The Roboflow
and Streaks parts of the legacy scripts were not re-audited.

## Folder map

```
README.md, LICENSE, CITATION.cff, CHANGES_REPORT.md, requirements_used.txt
spark.yaml, data_cross.yaml            Ultralytics dataset configs (SPARK)
quantum_shared_model.py                PennyLane QGAFF circuit (QuantumFusionModel)
spark_yolo/                            SPARK CSV -> YOLO label conversion (valid)
verify_spark_labels.py                 label overlay check (valid)
revision_experiments/
    spark_detection_rowcol.py          SPARK detection, corrected labels (valid)
    spark_balanced_extract.py          balanced SPARK subset features (valid)
    spark_balanced_eval.py             raw vs MLP vs QGAFF (valid)
    spark_balanced/subset.csv          the 770 images used
    results/                           result JSONs
    experiment_*.py, test_*.py, check_*.py   legacy revision experiments
Revision_Image_Outputs/
    code/                              image-level robustness study (valid)
    figures/                           robustness results JSON and figure
legacy_invalid_spark_results/          old SPARK result files (do not cite)
spark_quantum_fusion/                  legacy SPARK QGAFF pipeline
Quantum_fusion/, vector/, yolo/        early feature-extraction / QGAFF scripts
flowchart/flowchart.png                pipeline flowchart
*.py (root)                            feature extraction, subset creation, training and metric helpers
```

## License and citation

MIT License (see [`LICENSE`](LICENSE)). Citation metadata is in [`CITATION.cff`](CITATION.cff).
