"""
run_all_revision_experiments.py
=================================
Master runner: executes all 6 revision experiments in sequence,
collects results, then generates a combined summary report.

Run from project root:
    python run_all_revision_experiments.py

Individual experiments can also be run standalone:
    python revision_experiments/experiment_1_classical_baselines.py
    python revision_experiments/experiment_2_representation_analysis.py
    ...etc.
"""

import subprocess
import sys
import os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import json
import time
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "revision_experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENTS = [
    {
        "id": 1,
        "name": "Classical Baselines Comparison",
        "script": "revision_experiments/experiment_1_classical_baselines.py",
        "outputs": ["baselines_comparison.json", "baselines_comparison_spark.png"],
        "required_data": ["spark_quantum_fusion/features.npy"],
        "estimated_time": "~10–25 min (multiple model training passes)",
    },
    {
        "id": 2,
        "name": "Deep Representation Analysis",
        "script": "revision_experiments/experiment_2_representation_analysis.py",
        "outputs": ["representation_analysis.json", "fdr_comparison.png"],
        "required_data": ["spark_quantum_fusion/features.npy",
                          "spark_quantum_fusion/quantum_features.npy"],
        "estimated_time": "~1–2 min",
    },
    {
        "id": 3,
        "name": "Noise / Perturbation Robustness",
        "script": "revision_experiments/experiment_3_noise_robustness.py",
        "outputs": ["noise_robustness.json", "noise_robustness.png"],
        "required_data": ["spark_quantum_fusion/features.npy"],
        "estimated_time": "~15–30 min (multiple training passes)",
    },
    {
        "id": 4,
        "name": "Ablation Study",
        "script": "revision_experiments/experiment_4_ablation_study.py",
        "outputs": ["ablation_study.json", "ablation_study.png"],
        "required_data": ["spark_quantum_fusion/features.npy"],
        "estimated_time": "~10–20 min",
    },
    {
        "id": 5,
        "name": "Multi-Seed Statistical Validation",
        "script": "revision_experiments/experiment_5_multiseed_stats.py",
        "outputs": ["multiseed_stats.json", "multiseed_stats.png"],
        "required_data": ["spark_quantum_fusion/features.npy"],
        "estimated_time": "~15–25 min (5 seeds × 2 datasets)",
    },
    {
        "id": 6,
        "name": "YOLOv8 Scalability Study",
        "script": "revision_experiments/experiment_6_yolo_scalability.py",
        "outputs": ["yolo_scalability.json", "yolo_scalability.png"],
        "required_data": ["spark_yolo/images/val"],
        "estimated_time": "~20–60 min (feature extraction from 3 backbones)",
    },
]


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def check_required_data(experiment):
    missing = []
    for path in experiment["required_data"]:
        full_path = os.path.join(ROOT, path)
        if not os.path.exists(full_path):
            missing.append(path)
    return missing


def run_experiment(exp, python_exe=sys.executable):
    script_path = os.path.join(ROOT, exp["script"])

    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"

    missing = check_required_data(exp)
    if missing:
        msg = f"Required data not found:\n  " + "\n  ".join(missing)
        return False, msg

    print(f"\n  Running: {exp['script']}")
    print(f"  Estimated time: {exp['estimated_time']}")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")
    print("  " + "─" * 50)

    t_start = time.time()
    try:
        result = subprocess.run(
            [python_exe, script_path],
            cwd=ROOT,
            capture_output=False,
            timeout=7200  # 2-hour timeout
        )
        elapsed = time.time() - t_start
        if result.returncode == 0:
            return True, f"Completed in {elapsed:.0f}s"
        else:
            return False, f"Exit code {result.returncode} after {elapsed:.0f}s"
    except subprocess.TimeoutExpired:
        return False, "Timeout (2 hours)"
    except Exception as e:
        return False, str(e)


def collect_results_summary():
    """Collect key metrics from all JSON result files."""
    summary = {}

    # Experiment 1: Baselines
    path = os.path.join(RESULTS_DIR, "baselines_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if 'spark' in data:
            spark = data['spark']
            summary['baselines'] = {
                'raw_backbone': spark.get('raw_backbone', {}).get('silhouette_mean'),
                'mlp':          spark.get('mlp',          {}).get('silhouette_mean'),
                'mlp_cl':       spark.get('mlp_center_loss', {}).get('silhouette_mean'),
                'simclr':       spark.get('simclr',        {}).get('silhouette_mean'),
                'qgaff':        spark.get('qgaff',         {}).get('silhouette_mean'),
            }

    # Experiment 2: Representation analysis
    path = os.path.join(RESULTS_DIR, "representation_analysis.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            if 'SPARK' in entry.get('dataset', ''):
                summary['representation'] = {
                    'fdr_before': entry['before']['fdr'],
                    'fdr_after':  entry['after']['fdr'] if entry.get('after') else None,
                    'intra_before': entry['before']['mean_intra_dist'],
                    'intra_after':  entry['after']['mean_intra_dist'] if entry.get('after') else None,
                    'inter_before': entry['before']['mean_inter_dist'],
                    'inter_after':  entry['after']['mean_inter_dist'] if entry.get('after') else None,
                }

    # Experiment 3: Noise robustness
    path = os.path.join(RESULTS_DIR, "noise_robustness.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        summary['noise'] = {r['key']: r['silhouette_mean'] for r in data}

    # Experiment 4: Ablation
    path = os.path.join(RESULTS_DIR, "ablation_study.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        summary['ablation'] = {c['name']: c['silhouette'] for c in data}

    # Experiment 5: Multi-seed
    path = os.path.join(RESULTS_DIR, "multiseed_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for dname, dres in data.items():
            if 'SPARK' in dname:
                s = dres.get('qgaff_summary', {})
                summary['multiseed'] = {
                    'mean': s.get('silhouette_mean'),
                    'std':  s.get('silhouette_std'),
                    'min':  s.get('silhouette_min'),
                    'max':  s.get('silhouette_max'),
                }

    # Experiment 6: Scalability
    path = os.path.join(RESULTS_DIR, "yolo_scalability.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        summary['scalability'] = {r['model']: {
            'before': r['sil_before'],
            'after':  r['sil_after'],
            'improvement': r.get('improvement')
        } for r in data if r.get('status') == 'OK'}

    return summary


def print_results_summary(summary):
    print_header("RESULTS SUMMARY")

    if 'baselines' in summary:
        print("\n  [Exp 1] Classical Baselines (SPARK Silhouette Score):")
        b = summary['baselines']
        print(f"    Raw backbone  : {b.get('raw_backbone', 'N/A')}")
        print(f"    MLP           : {b.get('mlp', 'N/A')}")
        print(f"    MLP+CenterLoss: {b.get('mlp_cl', 'N/A')}")
        print(f"    SimCLR        : {b.get('simclr', 'N/A')}")
        print(f"    QGAFF (prop.) : {b.get('qgaff', 'N/A')}")

    if 'representation' in summary:
        print("\n  [Exp 2] Fisher Discriminant Ratio (SPARK):")
        r = summary['representation']
        print(f"    FDR Before    : {r.get('fdr_before', 'N/A'):.4f}" if r.get('fdr_before') else "    FDR Before: N/A")
        print(f"    FDR After     : {r.get('fdr_after', 'N/A'):.4f}"  if r.get('fdr_after')  else "    FDR After: N/A")

    if 'noise' in summary:
        print("\n  [Exp 3] Noise Robustness (Silhouette Scores):")
        for k, v in summary['noise'].items():
            print(f"    {k:<22}: {v:.4f}" if v else f"    {k}: N/A")

    if 'multiseed' in summary:
        print("\n  [Exp 5] Multi-Seed Stats (SPARK):")
        m = summary['multiseed']
        print(f"    Mean ± Std    : {m.get('mean', 'N/A'):.4f} ± {m.get('std', 'N/A'):.4f}"
              if m.get('mean') else "    N/A")

    if 'scalability' in summary:
        print("\n  [Exp 6] Scalability:")
        for model, val in summary['scalability'].items():
            print(f"    {model:<30}: {val.get('before', 'N/A'):.4f} → {val.get('after', 'N/A'):.4f}"
                  if val.get('before') else f"    {model}: N/A")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_header("REVISION EXPERIMENTS — MASTER RUNNER")
    print(f"\n  Running from: {ROOT}")
    print(f"  Results will be saved to: {RESULTS_DIR}")
    print(f"\n  Total experiments: {len(EXPERIMENTS)}")

    # Pre-flight check
    print_header("PRE-FLIGHT: Data Availability Check")
    all_available = True
    for exp in EXPERIMENTS:
        missing = check_required_data(exp)
        status = "✓ READY" if not missing else f"⚠ MISSING: {', '.join(missing)}"
        print(f"  Exp {exp['id']}: {exp['name'][:40]:<40} {status}")
        if missing:
            all_available = False

    if not all_available:
        print("\n  [!] Some experiments have missing prerequisites.")
        print("  [!] Please ensure SPARK feature extraction has been run first:")
        print("      python spark_extract_features_subset.py")
        print("      python spark_quantum_fusion/fast_train_silhouette.py")
        print("\n  Continuing with available experiments...\n")

    # Parse CLI args for selective run
    run_ids = None
    if len(sys.argv) > 1:
        try:
            run_ids = [int(a) for a in sys.argv[1:]]
            print(f"\n  Running only experiments: {run_ids}")
        except ValueError:
            print(f"  [!] Invalid experiment IDs: {sys.argv[1:]}")

    # Run experiments
    print_header("RUNNING EXPERIMENTS")
    run_log = []

    for exp in EXPERIMENTS:
        if run_ids and exp['id'] not in run_ids:
            print(f"\n  Exp {exp['id']}: SKIPPED (not in requested IDs)")
            continue

        print(f"\n  ══ Experiment {exp['id']}: {exp['name']} ══")
        success, msg = run_experiment(exp)

        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"\n  Status: {status} — {msg}")
        run_log.append({'id': exp['id'], 'name': exp['name'],
                        'success': success, 'message': msg})

    # Collect and print summary
    print_header("COLLECTING RESULTS")
    summary = collect_results_summary()
    print_results_summary(summary)

    # Save master log
    log_path = os.path.join(RESULTS_DIR, "run_log.json")
    with open(log_path, "w") as f:
        json.dump({'run_log': run_log, 'summary': summary}, f, indent=2, default=str)
    print(f"\n  Full log saved to: {log_path}")

    # Final status
    passed = sum(1 for r in run_log if r['success'])
    total  = len(run_log)
    print_header(f"COMPLETE — {passed}/{total} experiments passed")
    print()
