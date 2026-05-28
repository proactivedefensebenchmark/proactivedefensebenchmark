#!/usr/bin/env python3
"""Robustness evaluation launcher + ROB score aggregator.

For each result_<wrapper>_<attack> produced by robustness.py:
  1) evaluation/evaluate_robustness.py  → robustness_evaluation.{csv,json,summary.txt}
                                          (mse / alex_lpips / brisque per preprocessing)
  2) evaluation/evaluate_id_robust.py   → eval_id.json
                                          (relative_id_<preprocessing>_mean)

Then aggregates a final ROB score per (dataset, attack, wrapper):

    ROB(method, wrapper) = mean over preprocessings of (
        mean( mse_ratio, lpips_ratio, brisque_ratio, id_retention )
    )

  - *_ratio  = metric / baseline_metric (clipped to 1.0)
  - id_retention = relative_id_<prep>_mean / relative_id_original_mean (clipped to 1.0)

The final summary is printed to stdout and written to
<output_root>/final_robustness_{analysis.json,summary.csv}.

Examples
--------
    python robustness_evaluation.py
    python robustness_evaluation.py --attack pgd --dataset celeba --wrappers simswap
    python robustness_evaluation.py --results_dir robustness_results/ffhq/result_diffae_pgd
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from configs.paths_config import DATASETS
from white_box_attack import ATTACKS, WRAPPERS

ROOT = Path(__file__).resolve().parent
METRIC_EVALUATOR = ROOT / "evaluation" / "evaluate_robustness.py"
ID_EVALUATOR = ROOT / "evaluation" / "evaluate_id_robust.py"
DEFAULT_RESULTS_ROOT = ROOT / "robustness_results"

PREPROCESSING_TYPES = [
    "blur1", "blur3", "jpeg70", "jpeg90",
    "noise001", "noise003", "salt_pepper001", "salt_pepper003",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evaluate_robustness.py + evaluate_id_robust.py "
                    "and emit ROB score summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--attack", nargs="+", choices=sorted(ATTACKS.keys()),
                   default=sorted(ATTACKS.keys()))
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()),
                   default=sorted(DATASETS.keys()))
    p.add_argument("--wrappers", nargs="+", choices=WRAPPERS, default=WRAPPERS)
    p.add_argument("--results_dir", type=str, default=None,
                   help="Direct override (single result_* folder or parent dir). "
                        "Bypasses attack/dataset/wrappers filters.")
    p.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--output_root", type=str, default=None,
                   help="Where evaluation outputs go (default: <results_root>_eval).")
    p.add_argument("--use_pt", action="store_true",
                   help="Use .pt tensor files instead of .jpg images.", default='true')
    p.add_argument("--gpu", default="0")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip individual evaluators if their output JSON exists. "
                        "The ROB summary is always re-aggregated from current files.")
    p.add_argument("--skip_id", action="store_true",
                   help="Skip evaluate_id_robust.py (ROB will be quality-only, 3 metrics).")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def discover_targets(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Return [(result_dir, dataset_tag), ...]."""
    results_root = Path(args.results_root).resolve()

    if args.results_dir:
        path = Path(args.results_dir).resolve()
        if not path.is_dir():
            raise SystemExit(f"❌ Directory not found: {path}")
        if path.name.startswith("result_"):
            tag = path.parent.name if path.parent != results_root else "custom"
            return [(path, tag)]
        found = sorted([d for d in path.glob("result_*") if d.is_dir()])
        if found:
            return [(d, path.name) for d in found]
        found = sorted([d for d in path.glob("*/result_*") if d.is_dir()])
        if found:
            return [(d, d.parent.name) for d in found]
        raise SystemExit(f"❌ No result_* directories under {path}")

    if not results_root.is_dir():
        return []
    targets: list[tuple[Path, str]] = []
    for dataset in args.dataset:
        ds_root = results_root / dataset
        if not ds_root.is_dir():
            continue
        for attack in args.attack:
            for wrapper in args.wrappers:
                folder = ds_root / f"result_{wrapper}_{attack}"
                if folder.is_dir():
                    targets.append((folder, dataset))
    return targets


def compute_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root).resolve()
    results_root = Path(args.results_root).resolve()
    return results_root.parent / (results_root.name + "_eval")


def run_subprocess(cmd: list[str], env: dict, dry_run: bool) -> int:
    print("    " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


# ---------------------------------------------------------------------------
# ROB aggregation (mirrors robust_measure.py with the 4-metric formula)
# ---------------------------------------------------------------------------

def _parse_metric_csv(path: Path) -> dict | None:
    """Return {prep: {mse_ratio, lpips_ratio, brisque_ratio}} or None.

    Computes ratios against the 'baseline' row (mse_<prep> / mse_baseline),
    clipped to 1.0.
    """
    if not path.is_file():
        return None
    rows: dict[str, dict] = {}
    baseline: dict | None = None
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["preprocessing"]
            metrics = {
                "mse": float(row["mse"]),
                "lpips": float(row["alex_lpips"]),
                "brisque": float(row["brisque"]),
            }
            if name == "baseline":
                baseline = metrics
            else:
                rows[name] = metrics
    if baseline is None:
        return None
    ratios = {}
    for name, m in rows.items():
        if name in PREPROCESSING_TYPES:
            ratios[name] = {
                "mse_ratio":     min(1.0, m["mse"]     / (baseline["mse"]     + 1e-10)),
                "lpips_ratio":   min(1.0, m["lpips"]   / (baseline["lpips"]   + 1e-10)),
                "brisque_ratio": min(1.0, m["brisque"] / (baseline["brisque"] + 1e-10)),
            }
    return ratios


def _parse_id_json(path: Path) -> dict | None:
    """Return {prep: id_retention} or None.

    id_retention = relative_id_<prep>_mean / relative_id_original_mean
                   clipped to 1.0.
    """
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f).get("averages", {})
    base = data.get("relative_id_original_mean", 0)
    if not base:
        return None
    out = {}
    for prep in PREPROCESSING_TYPES:
        val = data.get(f"relative_id_{prep}_mean", 0)
        out[prep] = min(1.0, val / base)
    return out


def _load_baseline_cidr(eval_id_path: Path) -> float | None:
    """Return relative_id_original_mean (= baseline CIDR before any
    preprocessing perturbation) from eval_id.json, or None if missing."""
    if not eval_id_path.is_file():
        return None
    with open(eval_id_path) as f:
        data = json.load(f).get("averages", {})
    return data.get("relative_id_original_mean")


def aggregate_rob(targets_with_outputs: list[tuple[Path, str, Path]],
                  with_id: bool) -> dict:
    """For each result, compute combined ROB and the baseline CIDR.
    Returns nested dict {dataset: {attack: {wrapper: {rob, cidr}}}}."""
    report: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))

    for target, dataset_tag, out_dir in targets_with_outputs:
        csv_ratios = _parse_metric_csv(out_dir / "robustness_evaluation.csv")
        id_ratios = _parse_id_json(out_dir / "eval_id.json") if with_id else None
        cidr = _load_baseline_cidr(out_dir / "eval_id.json")

        if csv_ratios is None:
            print(f"⚠️  {dataset_tag}/{target.name}: missing robustness_evaluation.csv")
            continue
        if with_id and id_ratios is None:
            print(f"⚠️  {dataset_tag}/{target.name}: missing eval_id.json — "
                  f"falling back to 3-metric ROB for this entry")

        per_prep = []
        for prep in PREPROCESSING_TYPES:
            if prep not in csv_ratios:
                continue
            r = csv_ratios[prep]
            parts = [r["mse_ratio"], r["lpips_ratio"], r["brisque_ratio"]]
            if id_ratios and prep in id_ratios:
                parts.append(id_ratios[prep])
            per_prep.append(sum(parts) / len(parts))

        if not per_prep:
            continue
        score = sum(per_prep) / len(per_prep)

        # Extract attack + wrapper from "result_<wrapper>_<attack>".
        name = target.name[len("result_"):]
        wrapper, _, attack = name.rpartition("_")
        report[dataset_tag][attack][wrapper] = {"rob": score, "cidr": cidr}

    return report


def emit_rob_summary(report: dict, output_root: Path) -> None:
    """Print ROB + CIDR summary table to stdout and write CSV + JSON."""
    rows: list[dict] = []
    for dataset, by_attack in report.items():
        for attack, by_wrapper in by_attack.items():
            for wrapper, entry in by_wrapper.items():
                rows.append({
                    "dataset": dataset,
                    "attack": attack,
                    "wrapper": wrapper,
                    "rob": entry["rob"],
                    "cidr": entry["cidr"],
                })
    rows.sort(key=lambda r: (r["dataset"], r["attack"], r["wrapper"]))

    print("\n" + "=" * 72)
    print(" Final Robustness (ROB) Summary    (CIDR = relative_id_original_mean)")
    print("=" * 72)
    print(f"{'Dataset':<12} {'Attack':<12} {'Wrapper':<14} {'ROB':>8} {'CIDR':>8}")
    print("-" * 72)
    for r in rows:
        cidr_s = f"{r['cidr']:.4f}" if r["cidr"] is not None else "  n/a "
        print(f"{r['dataset']:<12} {r['attack']:<12} {r['wrapper']:<14} "
              f"{r['rob']:>8.4f} {cidr_s:>8}")
    print("=" * 72)
    print(f"Total entries: {len(rows)}\n")

    if not rows:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_robustness_analysis.json"
    csv_path = output_root / "final_robustness_summary.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "attack", "wrapper", "rob", "cidr"])
        w.writeheader()
        for r in rows:
            w.writerow({
                **r,
                "rob": f"{r['rob']:.4f}",
                "cidr": f"{r['cidr']:.4f}" if r["cidr"] is not None else "",
            })

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> int:
    args = parse_args()
    targets = discover_targets(args)

    print("=" * 60)
    if args.results_dir:
        print(f"Mode:        explicit (--results_dir {args.results_dir})")
    else:
        print(f"Results root: {args.results_root}")
        print(f"Attacks:      {args.attack}")
        print(f"Datasets:     {args.dataset}")
        print(f"Wrappers:     {args.wrappers}")
    print(f"Found:        {len(targets)} result_* directories")
    print(f"Use .pt:      {args.use_pt}")
    print(f"Skip ID:      {args.skip_id}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    if not targets:
        print("Nothing to evaluate.")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_root = compute_output_root(args)
    print(f"Output root:  {output_root}")
    print("=" * 60)

    success = failed = skipped = 0
    targets_with_outputs: list[tuple[Path, str, Path]] = []

    for target, dataset_tag in targets:
        out_dir = output_root / dataset_tag / target.name
        targets_with_outputs.append((target, dataset_tag, out_dir))
        print(f">>> [{dataset_tag}] {target.name}")

        # --- 1) metric CSV --------------------------------------------------
        metric_done = (out_dir / "robustness_evaluation.json").is_file()
        if args.skip_existing and metric_done:
            print(f"    ⏭️  metric eval already done")
        else:
            cmd = [sys.executable, str(METRIC_EVALUATOR),
                   "--result_dir", str(target),
                   "--output_dir", str(out_dir)]
            if args.use_pt:
                cmd += ["--use_pt"]
            rc = run_subprocess(cmd, env, args.dry_run)
            if rc != 0 and not args.dry_run:
                failed += 1
                print(f"    ❌ metric eval failed (exit {rc})\n")
                continue

        # --- 2) ID JSON -----------------------------------------------------
        if not args.skip_id:
            id_done = (out_dir / "eval_id.json").is_file()
            if args.skip_existing and id_done:
                print(f"    ⏭️  id eval already done")
            else:
                cmd = [sys.executable, str(ID_EVALUATOR),
                       "--result_dir", str(target),
                       "--output_dir", str(out_dir)]
                rc = run_subprocess(cmd, env, args.dry_run)
                if rc != 0 and not args.dry_run:
                    print(f"    ⚠️  id eval failed (exit {rc}) — continuing without it\n")
                    # Don't bump `failed`; ROB falls back to 3-metric automatically.

        if not args.dry_run:
            success += 1
            print(f"    ✅ → {out_dir}\n")

    print("=" * 60)
    print(f"Done. success={success} failed={failed} skipped={skipped}")
    print("=" * 60)

    # ROB aggregation (always runs, even after dry_run, so the user can see
    # what would be summarized from existing on-disk outputs).
    report = aggregate_rob(targets_with_outputs, with_id=not args.skip_id)
    emit_rob_summary(report, output_root)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
