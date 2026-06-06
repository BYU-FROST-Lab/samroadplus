"""
Benchmark Evaluation Script — APLS & TOPO for samroadplus

Self-contained benchmarking using:
  - TOPO: Pure Python (metrics/topo.py) — no external conda env needed
  - APLS: Go (metrics/apls/main.go) — requires Go installed

Usage:
    python benchmark_eval.py --graph_dir save/output_cbam_e92/graph
    python benchmark_eval.py --graph_dir save/output_cbam_e92/graph --num_tiles 3
    python benchmark_eval.py --graph_dir save/output_cbam_e92/graph --topo_only
"""

import os
import sys
import re
import subprocess
import json
import numpy as np
from argparse import ArgumentParser

# Local metrics
from metrics.topo import evaluate_topo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APLS_DIR = os.path.join(BASE_DIR, "metrics", "apls")

# Python interpreter for APLS convert.py (uses this env's python)
PYTHON_EXE = sys.executable

GT_DIR = os.path.join(BASE_DIR, "cityscale", "20cities")
DEFAULT_GRAPH_DIR = os.path.join(BASE_DIR, "save", "infer__cityscale100", "graph")


def cityscale_data_partition():
    """Same partition used by all models."""
    indrange_train, indrange_test, indrange_validation = [], [], []
    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 18:
            indrange_validation.append(x)
        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


# ---------------------------------------------------------------------------
# APLS Evaluation (Go — kept as subprocess)
# ---------------------------------------------------------------------------
def run_apls(gt_graph_path, pred_graph_path, output_path):
    """
    Run the APLS metric:
      1. Convert pickle graphs to JSON via convert.py
      2. Run main.go
    Returns APLS score or None on failure.
    """
    gt_graph_path = os.path.abspath(gt_graph_path)
    pred_graph_path = os.path.abspath(pred_graph_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gt_json = output_path.replace(".txt", "_gt.json")
    pred_json = output_path.replace(".txt", "_prop.json")

    # Step 1: Convert pickle to JSON
    for pkl_path, json_path in [(gt_graph_path, gt_json), (pred_graph_path, pred_json)]:
        try:
            result = subprocess.run(
                [PYTHON_EXE, "convert.py", pkl_path, json_path],
                cwd=APLS_DIR, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  APLS convert error:\n{result.stderr}")
                return None
        except Exception as e:
            print(f"  APLS convert exception: {e}")
            return None

    # Step 2: Run Go APLS
    try:
        result = subprocess.run(
            ["go", "run", "main.go", gt_json, pred_json, output_path],
            cwd=APLS_DIR, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            print(f"  APLS error:\n{result.stderr}")
            return None

        return parse_apls_output(output_path)

    except FileNotFoundError:
        print("  ERROR: 'go' not found. Install Go to run APLS.")
        return None
    except subprocess.TimeoutExpired:
        print(f"  APLS timed out")
        return None
    except Exception as e:
        print(f"  APLS exception: {e}")
        return None


def parse_apls_output(output_path):
    """Parse the APLS output file for the score."""
    if not os.path.exists(output_path):
        return None
    with open(output_path, "r") as f:
        content = f.read().strip()
    numbers = re.findall(r"[\d.]+", content)
    if numbers:
        return float(numbers[-1])
    return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(graph_dir, test_indices, results_dir,
             run_topo_flag=True, run_apls_flag=True):
    """Evaluate graphs across all test tiles."""
    os.makedirs(results_dir, exist_ok=True)

    topo_scores = []
    apls_scores = []

    for tile_id in test_indices:
        pred_path = os.path.join(graph_dir, f"{tile_id}.p")
        gt_path = os.path.join(GT_DIR, f"region_{tile_id}_refine_gt_graph.p")

        if not os.path.exists(pred_path):
            print(f"  Tile {tile_id}: predicted graph not found at {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  Tile {tile_id}: GT graph not found at {gt_path}")
            continue

        print(f"  Tile {tile_id} ...")

        # TOPO — direct Python call, no subprocess
        if run_topo_flag:
            topo_out = os.path.join(results_dir, f"topo_{tile_id}.txt")
            try:
                topo_result = evaluate_topo(gt_path, pred_path, topo_out)
                if topo_result:
                    prec, rec = topo_result
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    topo_scores.append({"tile": tile_id, "precision": prec,
                                        "recall": rec, "f1": f1})
                    print(f"    TOPO: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
                else:
                    print(f"    TOPO: FAILED")
            except Exception as e:
                print(f"    TOPO exception: {e}")

        # APLS — Go subprocess
        if run_apls_flag:
            apls_out = os.path.join(results_dir, f"apls_{tile_id}.txt")
            apls_result = run_apls(gt_path, pred_path, apls_out)
            if apls_result is not None:
                apls_scores.append({"tile": tile_id, "apls": apls_result})
                print(f"    APLS: {apls_result:.4f}")
            else:
                print(f"    APLS: FAILED")

    return topo_scores, apls_scores


def print_summary(topo_scores, apls_scores, label="samroadplus"):
    """Print a markdown comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS — Cityscale Dataset")
    print("=" * 80)

    header = f"| {'Model':<15} | {'TOPO P':>8} | {'TOPO R':>8} | {'TOPO F1':>8} | {'APLS':>8} | {'N_topo':>6} | {'N_apls':>6} |"
    separator = f"|{'-'*17}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*8}|"
    print(header)
    print(separator)

    if topo_scores:
        avg_p = np.mean([s["precision"] for s in topo_scores])
        avg_r = np.mean([s["recall"] for s in topo_scores])
        avg_f1 = np.mean([s["f1"] for s in topo_scores])
        topo_p_str, topo_r_str, topo_f1_str = f"{avg_p:.4f}", f"{avg_r:.4f}", f"{avg_f1:.4f}"
    else:
        topo_p_str = topo_r_str = topo_f1_str = "N/A"

    if apls_scores:
        avg_apls = np.mean([s["apls"] for s in apls_scores])
        apls_str = f"{avg_apls:.4f}"
    else:
        apls_str = "N/A"

    n_topo, n_apls = len(topo_scores), len(apls_scores)
    row = f"| {label:<15} | {topo_p_str:>8} | {topo_r_str:>8} | {topo_f1_str:>8} | {apls_str:>8} | {n_topo:>6} | {n_apls:>6} |"
    print(row)
    print(separator)
    print()


def save_results_json(topo_scores, apls_scores, output_path):
    """Save detailed results as JSON."""
    serializable = {"topo": topo_scores, "apls": apls_scores}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Detailed results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark evaluation with APLS & TOPO")
    parser.add_argument("--graph_dir", default=DEFAULT_GRAPH_DIR,
                        help="Directory containing predicted graph .p files")
    parser.add_argument("--label", default="samroadplus",
                        help="Label for this run in the results table")
    parser.add_argument("--num_tiles", type=int, default=-1,
                        help="Limit number of test tiles (-1 = all)")
    parser.add_argument("--results_dir", default="benchmark_results",
                        help="Directory for metric output files")
    parser.add_argument("--topo_only", action="store_true",
                        help="Run only TOPO, skip APLS")
    parser.add_argument("--apls_only", action="store_true",
                        help="Run only APLS, skip TOPO")
    args = parser.parse_args()

    _, _, test_indices = cityscale_data_partition()
    if args.num_tiles > 0:
        test_indices = test_indices[:args.num_tiles]
    print(f"Evaluating {len(test_indices)} test tiles: {test_indices}")

    run_topo_flag = not args.apls_only
    run_apls_flag = not args.topo_only

    if run_apls_flag and not os.path.exists(os.path.join(APLS_DIR, "main.go")):
        print(f"WARNING: APLS metric not found at {APLS_DIR}/main.go")
        run_apls_flag = False

    os.makedirs(args.results_dir, exist_ok=True)

    graph_dir = args.graph_dir
    print(f"\n{'='*60}")
    print(f"Evaluating: {args.label}")
    print(f"Graph dir:  {graph_dir}")
    print(f"{'='*60}")

    if not os.path.exists(graph_dir):
        print(f"  ERROR: Graph directory does not exist: {graph_dir}")
        sys.exit(1)

    topo_scores, apls_scores = evaluate(
        graph_dir, test_indices, args.results_dir,
        run_topo_flag=run_topo_flag, run_apls_flag=run_apls_flag)

    print_summary(topo_scores, apls_scores, label=args.label)

    json_path = os.path.join(args.results_dir, "benchmark_results.json")
    save_results_json(topo_scores, apls_scores, json_path)
