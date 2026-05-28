#!/usr/bin/env python3
"""White-box attack evaluation launcher.

Replaces run_batch_pgd_evaluation.sh (and the equivalent for the other
attacks) with one entry point. For each (attack, dataset, wrapper) tuple,
locates the corresponding result_<wrapper>_<attack> directory produced by
white_box_attack.py and runs evaluation/evaluate_disruption.py against it.

By default it evaluates every attack × dataset × wrapper combination it
can find. Use --attack / --dataset / --wrappers to narrow the matrix, or
--results_dir to point at a specific folder.

Example
-------
    # Evaluate everything available on disk:
    python white_box_evaluation.py

    # Only PGD results on FFHQ for two wrappers:
    python white_box_evaluation.py --attack pgd --dataset ffhq \\
        --wrappers diffae simswap

    # Evaluate a single folder directly:
    python white_box_evaluation.py \\
        --results_dir batch_results_pgd/ffhq/result_diffae_pgd
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
METRIC_EVALUATOR = ROOT / "evaluation" / "evaluate_disruption.py"
ID_EVALUATOR = ROOT / "evaluation" / "evaluate_id.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run evaluate_disruption.py over white-box attack results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--attack", nargs="+", choices=sorted(ATTACKS.keys()),
                   default=sorted(ATTACKS.keys()),
                   help="Attacks to evaluate. Default: all.")
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()),
                   default=sorted(DATASETS.keys()),
                   help="Datasets to evaluate. Default: all.")
    p.add_argument("--wrappers", nargs="+", choices=WRAPPERS,
                   default=WRAPPERS,
                   help="Wrappers to evaluate. Default: all 8.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Direct override. Either a single result_* folder, "
                        "or a parent directory containing result_* (one or "
                        "two levels). Bypasses attack/dataset/wrappers filters.")
    p.add_argument("--source_image_dir", type=str, default=None,
                   help="Optional separate directory holding original source "
                        "images (passed through to evaluate_disruption.py).")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a step (metric / id) if its output already exists.")
    p.add_argument("--skip_id", action="store_true",
                   help="Skip evaluate_id.py (no eval_id.json produced).")
    p.add_argument("--output_root", type=str, default=None,
                   help="Where to write final_white_box_{summary.csv,analysis.json} "
                        "(default: ROOT).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be evaluated and exit.")
    return p.parse_args()


def discover_targets(args: argparse.Namespace) -> list[Path]:
    """Return the list of result_* folders to evaluate, in stable order."""
    if args.results_dir:
        path = Path(args.results_dir).resolve()
        if not path.is_dir():
            raise SystemExit(f"❌ Directory not found: {path}")
        if path.name.startswith("result_"):
            return [path]
        # Parent: look one then two levels down for result_*.
        found = sorted([d for d in path.glob("result_*") if d.is_dir()])
        if not found:
            found = sorted([d for d in path.glob("*/result_*") if d.is_dir()])
        if not found:
            raise SystemExit(f"❌ No result_* directories under {path}")
        return found

    # Auto-discovery: walk the (attack × dataset × wrapper) cube.
    targets: list[Path] = []
    for attack in args.attack:
        _, default_base = ATTACKS[attack]
        attack_root = ROOT / default_base
        if not attack_root.is_dir():
            continue
        for dataset in args.dataset:
            ds_root = attack_root / dataset
            if not ds_root.is_dir():
                continue
            for wrapper in args.wrappers:
                folder = ds_root / f"result_{wrapper}_{attack}"
                if folder.is_dir():
                    targets.append(folder)
    return targets


def run_metric_eval(target: Path, args: argparse.Namespace, env: dict) -> int:
    cmd = [
        sys.executable, str(METRIC_EVALUATOR),
        "--batch_subdir_mode",
        "--x_src_dir", str(target),
        "--output", str(target / "evaluation_results.json"),
        "--image_size", str(args.image_size),
    ]
    if args.source_image_dir:
        cmd += ["--source_image_dir", args.source_image_dir]

    print("    " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


def run_id_eval(target: Path, args: argparse.Namespace, env: dict) -> int:
    cmd = [
        sys.executable, str(ID_EVALUATOR),
        "--result_dir", str(target),
        "--output_dir", str(target),
    ]
    print("    " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


def write_summary(target: Path) -> None:
    """Render evaluation_results.json → evaluation_summary.txt (matches the
    original shell-script output format)."""
    json_path = target / "evaluation_results.json"
    if not json_path.is_file():
        return

    with open(json_path) as f:
        data = json.load(f)

    lines = [
        "========================================",
        f"Evaluation Summary: {target.name}",
        "========================================",
        "",
        "=== X Metrics (x_src vs x_adv) ===",
    ]
    for k, v in sorted(data.get("x_metric_averages", {}).items()):
        lines.append(f"{k}: {v:.6f}")
    lines += ["", "=== Decoded Metrics (decoded_src vs decoded_adv) ==="]
    for k, v in sorted(data.get("decoded_metric_averages", {}).items()):
        lines.append(f"{k}: {v:.6f}")
    lines += ["", f"Total evaluated: {data.get('total_evaluated', 'n/a')}"]

    summary_path = target / "evaluation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"    summary → {summary_path.name}")


# ---------------------------------------------------------------------------
# Final summary aggregation (CIDR per result_*)
# ---------------------------------------------------------------------------

def _parse_target_path(target: Path) -> tuple[str, str, str] | None:
    """From .../batch_results_<attack>/<dataset>/result_<wrapper>_<attack>/
    extract (dataset, attack, wrapper). Returns None if the layout doesn't match."""
    if not target.name.startswith("result_"):
        return None
    body = target.name[len("result_"):]
    wrapper, _, attack = body.rpartition("_")
    if not wrapper or not attack:
        return None
    dataset = target.parent.name  # e.g. "ffhq"
    return dataset, attack, wrapper


def _load_eval_metrics(target: Path) -> dict:
    """Return dict with cidr + decoded metric averages from per-folder outputs."""
    out: dict = {"cidr": None, "l2_mse": None, "lpips_alex": None, "brisque_adv": None}

    eval_id = target / "eval_id.json"
    if eval_id.is_file():
        with open(eval_id) as f:
            avgs = json.load(f).get("averages", {})
        out["cidr"] = avgs.get("relative_id_mean")

    eval_json = target / "evaluation_results.json"
    if eval_json.is_file():
        with open(eval_json) as f:
            d_avgs = json.load(f).get("decoded_metric_averages", {})
        for k in ("l2_mse", "lpips_alex", "brisque_adv"):
            out[k] = d_avgs.get(k)

    return out


def aggregate_white_box(targets: list[Path]) -> dict:
    """Return {dataset: {attack: {wrapper: metrics_dict}}}."""
    report: dict = defaultdict(lambda: defaultdict(dict))
    for target in targets:
        parsed = _parse_target_path(target)
        if parsed is None:
            continue
        dataset, attack, wrapper = parsed
        report[dataset][attack][wrapper] = _load_eval_metrics(target)
    return report


def emit_white_box_summary(report: dict, output_root: Path) -> None:
    rows: list[dict] = []
    for dataset, by_attack in report.items():
        for attack, by_wrapper in by_attack.items():
            for wrapper, m in by_wrapper.items():
                rows.append({
                    "dataset": dataset,
                    "attack": attack,
                    "wrapper": wrapper,
                    **m,
                })
    rows.sort(key=lambda r: (r["dataset"], r["attack"], r["wrapper"]))

    def _fmt(v, prec=4):
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "  n/a "

    print("\n" + "=" * 88)
    print(" Final White-Box Summary    (CIDR = relative_id_mean from eval_id.json)")
    print("=" * 88)
    print(f"{'Dataset':<12} {'Attack':<12} {'Wrapper':<14} "
          f"{'CIDR':>8} {'L2_MSE':>10} {'LPIPS':>8} {'BRISQUE':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['dataset']:<12} {r['attack']:<12} {r['wrapper']:<14} "
              f"{_fmt(r['cidr']):>8} {_fmt(r['l2_mse'], 6):>10} "
              f"{_fmt(r['lpips_alex']):>8} {_fmt(r['brisque_adv'], 2):>9}")
    print("=" * 88)
    print(f"Total entries: {len(rows)}\n")

    if not rows:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_white_box_analysis.json"
    csv_path = output_root / "final_white_box_summary.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        fields = ["dataset", "attack", "wrapper", "cidr", "l2_mse", "lpips_alex", "brisque_adv"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "attack": r["attack"], "wrapper": r["wrapper"],
                "cidr":        f"{r['cidr']:.4f}"        if r["cidr"]        is not None else "",
                "l2_mse":      f"{r['l2_mse']:.6f}"      if r["l2_mse"]      is not None else "",
                "lpips_alex":  f"{r['lpips_alex']:.4f}"  if r["lpips_alex"]  is not None else "",
                "brisque_adv": f"{r['brisque_adv']:.2f}" if r["brisque_adv"] is not None else "",
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
        print(f"Attacks:     {args.attack}")
        print(f"Datasets:    {args.dataset}")
        print(f"Wrappers:    {args.wrappers}")
    print(f"Found:       {len(targets)} result_* directories")
    print(f"Skip ID:      {args.skip_id}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    if not targets:
        print("Nothing to evaluate.")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    success = failed = 0
    for target in targets:
        print(f">>> {target.name}")

        # --- 1) metric evaluation -----------------------------------------
        metric_done = (target / "evaluation_results.json").is_file()
        if args.skip_existing and metric_done:
            print(f"    ⏭️  metric eval already done")
        else:
            rc = run_metric_eval(target, args, env)
            if rc != 0 and not args.dry_run:
                failed += 1
                print(f"    ❌ metric eval failed (exit {rc})\n")
                continue
            if rc == 0 and not args.dry_run:
                write_summary(target)

        # --- 2) ID evaluation ---------------------------------------------
        if not args.skip_id:
            id_done = (target / "eval_id.json").is_file()
            if args.skip_existing and id_done:
                print(f"    ⏭️  id eval already done")
            else:
                rc = run_id_eval(target, args, env)
                if rc != 0 and not args.dry_run:
                    print(f"    ⚠️  id eval failed (exit {rc}) — continuing\n")
                    # Don't bump failed; metric eval already succeeded.

        if not args.dry_run:
            success += 1
            print(f"    ✅ done\n")

    print("=" * 60)
    print(f"Per-folder eval: success={success} failed={failed}")
    print("=" * 60)

    # Final CIDR / metric summary (always runs, even after dry_run, so the user
    # can re-aggregate from existing on-disk outputs).
    report = aggregate_white_box(targets)
    output_root = Path(args.output_root).resolve() if args.output_root else ROOT
    emit_white_box_summary(report, output_root)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
