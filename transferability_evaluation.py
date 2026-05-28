#!/usr/bin/env python3
"""Black-box transferability evaluation launcher + TE aggregator.

For each black_box_<source>_<target>/ folder produced by transferability.py:
  1) evaluation/evaluate_disruption.py → evaluation_results.json + summary
  2) evaluation/evaluate_id.py         → eval_id.json (relative_id_mean)

Then aggregates a final Transferability Evaluation (TE) per
(dataset, attack, source), normalized against each TARGET's white-box
performance:

    TE(method, source) = mean over targets T (T != source) of:
        mean(
            l2_mse(BB) / l2_mse(WB),
            lpips_alex(BB) / lpips_alex(WB),
            brisque_adv(BB) / brisque_adv(WB),
            relative_id(BB) / relative_id(WB),
        )   # each term clipped to 1.0

  - "WB" = the target's own white-box result:
      batch_results_<attack>/<dataset>/result_<target>_<attack>/
  - "BB" = the cross-model transfer result:
      batch_results_<attack>/<dataset>/black_box_results/black_box_<src>_<tgt>/

The final summary is printed to stdout and written to
<output_root>/final_transferability_{analysis.json,summary.csv}.

Examples
--------
    python transferability_evaluation.py
    python transferability_evaluation.py --attack pgd --dataset celeba --wrappers simswap
    python transferability_evaluation.py --results_dir batch_results_pgd/ffhq
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
        description="Run evaluate_disruption.py + evaluate_id.py over "
                    "transferability.py outputs, and emit final TE scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--attack", nargs="+", choices=sorted(ATTACKS.keys()),
                   default=sorted(ATTACKS.keys()))
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()),
                   default=sorted(DATASETS.keys()))
    p.add_argument("--wrappers", nargs="+", choices=WRAPPERS, default=WRAPPERS,
                   help="Filter sources & targets to this subset.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Direct override: a path like batch_results_<attack>/<dataset> "
                        "(must contain a black_box_results/ subdir). "
                        "Bypasses attack/dataset filters.")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--gpu", default="0")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip per-folder steps if their output already exists. "
                        "TE aggregation always re-runs from current files.")
    p.add_argument("--skip_id", action="store_true",
                   help="Skip evaluate_id.py (TE will be quality-only, 3 metrics).")
    p.add_argument("--output_root", type=str, default=None,
                   help="Where to write final_transferability_{summary.csv,analysis.json}. "
                        "Defaults to ROOT.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery: locate the (dataset, attack) "buckets" we'll work on
# ---------------------------------------------------------------------------

def discover_buckets(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    """Return [(dataset_tag, attack, dataset_root), ...] where dataset_root is
    .../batch_results_<attack>/<dataset>/ and is expected to contain both
    result_<wrapper>_<attack>/ (WB) and black_box_results/black_box_*_*/ (BB)."""
    if args.results_dir:
        path = Path(args.results_dir).resolve()
        if not path.is_dir():
            raise SystemExit(f"❌ Directory not found: {path}")
        if not (path / "black_box_results").is_dir():
            raise SystemExit(f"❌ No black_box_results/ under {path}")
        # Best-effort: try to extract attack from grandparent (batch_results_<attack>).
        parent = path.parent.name  # e.g. batch_results_pgd
        attack = parent.replace("batch_results_", "") if parent.startswith("batch_results_") else "custom"
        return [(path.name, attack, path)]

    buckets: list[tuple[str, str, Path]] = []
    for attack in args.attack:
        _, default_base = ATTACKS[attack]
        attack_root = ROOT / default_base
        if not attack_root.is_dir():
            continue
        for dataset in args.dataset:
            ds_root = attack_root / dataset
            if not ds_root.is_dir():
                continue
            if not (ds_root / "black_box_results").is_dir():
                continue
            buckets.append((dataset, attack, ds_root))
    return buckets


# ---------------------------------------------------------------------------
# Per-folder evaluation (mirrors white_box_evaluation.py)
# ---------------------------------------------------------------------------

def run_metric_eval(folder: Path, args: argparse.Namespace, env: dict) -> int:
    cmd = [
        sys.executable, str(METRIC_EVALUATOR),
        "--batch_subdir_mode",
        "--x_src_dir", str(folder),
        "--output", str(folder / "evaluation_results.json"),
        "--image_size", str(args.image_size),
    ]
    print("    " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


def run_id_eval(folder: Path, args: argparse.Namespace, env: dict) -> int:
    cmd = [
        sys.executable, str(ID_EVALUATOR),
        "--result_dir", str(folder),
        "--output_dir", str(folder),
    ]
    print("    " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, env=env).returncode


def write_summary(folder: Path) -> None:
    """Render evaluation_results.json → evaluation_summary.txt (same format
    as run_batch_black_box_evaluation.sh produced)."""
    json_path = folder / "evaluation_results.json"
    if not json_path.is_file():
        return
    with open(json_path) as f:
        data = json.load(f)

    lines = [
        "========================================",
        f"Evaluation Summary: {folder.name}",
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
    (folder / "evaluation_summary.txt").write_text("\n".join(lines) + "\n")


def evaluate_folder(folder: Path, args: argparse.Namespace, env: dict) -> bool:
    """Run metric + ID eval on a folder. Returns True if metric eval succeeded."""
    print(f">>> {folder.name}")
    metric_done = (folder / "evaluation_results.json").is_file()
    if args.skip_existing and metric_done:
        print(f"    ⏭️  metric eval already done")
    else:
        rc = run_metric_eval(folder, args, env)
        if rc != 0 and not args.dry_run:
            print(f"    ❌ metric eval failed (exit {rc})\n")
            return False
        if rc == 0 and not args.dry_run:
            write_summary(folder)

    if not args.skip_id:
        id_done = (folder / "eval_id.json").is_file()
        if args.skip_existing and id_done:
            print(f"    ⏭️  id eval already done")
        else:
            rc = run_id_eval(folder, args, env)
            if rc != 0 and not args.dry_run:
                print(f"    ⚠️  id eval failed (exit {rc}) — continuing\n")

    if not args.dry_run:
        print(f"    ✅ done\n")
    return True


# ---------------------------------------------------------------------------
# TE aggregation (mirrors transferability_measure.py with the 4-metric formula)
# ---------------------------------------------------------------------------

def _load_metric_avgs(folder: Path) -> dict | None:
    """Read decoded_metric_averages from evaluation_results.json.
    Returns {l2_mse, lpips_alex, brisque_adv} or None."""
    p = folder / "evaluation_results.json"
    if not p.is_file():
        return None
    with open(p) as f:
        data = json.load(f)
    avgs = data.get("decoded_metric_averages", {})
    out = {k: avgs.get(k) for k in ("l2_mse", "lpips_alex", "brisque_adv")}
    return out if all(v is not None for v in out.values()) else None


def _load_relative_id(folder: Path) -> float | None:
    p = folder / "eval_id.json"
    if not p.is_file():
        return None
    with open(p) as f:
        data = json.load(f)
    return data.get("averages", {}).get("relative_id_mean")


def _parse_bb_folder(name: str, valid_wrappers: set[str]) -> tuple[str, str] | None:
    """black_box_<source>_<target> → (source, target). Returns the pair if
    EITHER source or target is in valid_wrappers (the other side just has to
    be a recognised wrapper). Handles psp_mix (underscored wrapper name).
    """
    if not name.startswith("black_box_"):
        return None
    body = name[len("black_box_"):]
    all_wrappers = set(WRAPPERS)
    # Try longest-first match on the suffix as target, rest as source.
    for w in sorted(all_wrappers, key=len, reverse=True):
        if body.endswith("_" + w):
            source = body[:-(len(w) + 1)]
            if source in all_wrappers and source != w:
                # Accept if at least one side is in the user-supplied filter.
                if source in valid_wrappers or w in valid_wrappers:
                    return source, w
                return None
    return None


def aggregate_te(buckets: list[tuple[str, str, Path]],
                 wrappers: list[str], with_id: bool) -> dict:
    """Return nested {dataset: {attack: {source: {te, cidr}}}}.

    CIDR per source = mean of BB relative_id_mean across all targets that
    received this source's adversarial perturbations (raw transfer
    identity-disruption strength, NOT normalized)."""
    report: dict = defaultdict(lambda: defaultdict(dict))
    valid = set(wrappers)

    for dataset_tag, attack, ds_root in buckets:
        bb_root = ds_root / "black_box_results"

        # Cache white-box metrics per target.
        wb_cache: dict[str, dict] = {}
        for target in wrappers:
            wb_folder = ds_root / f"result_{target}_{attack}"
            if not wb_folder.is_dir():
                continue
            wb_m = _load_metric_avgs(wb_folder)
            wb_i = _load_relative_id(wb_folder) if with_id else None
            if wb_m and (wb_i is not None or not with_id):
                wb_cache[target] = {"metrics": wb_m, "id": wb_i}

        # Walk BB folders, compute per-pair normalized scores when target's
        # WB is available, and per-pair raw CIDR (always, when bb_i exists).
        source_pair_scores: dict[str, list[float]] = defaultdict(list)
        source_pair_cidr: dict[str, list[float]] = defaultdict(list)
        skipped_no_target_wb = 0

        for bb in sorted(bb_root.iterdir()):
            if not bb.is_dir():
                continue
            parsed = _parse_bb_folder(bb.name, valid)
            if parsed is None:
                continue
            source, target = parsed

            bb_m = _load_metric_avgs(bb)
            bb_i = _load_relative_id(bb) if with_id else None

            # Raw CIDR contribution (no normalization needed).
            if bb_i is not None:
                source_pair_cidr[source].append(bb_i)

            # Normalized TE only when target's WB is on disk.
            wb = wb_cache.get(target)
            if wb is None or bb_m is None:
                if wb is None:
                    skipped_no_target_wb += 1
                continue

            parts = []
            for k in ("l2_mse", "lpips_alex", "brisque_adv"):
                wb_v = wb["metrics"][k]
                if wb_v > 0:
                    parts.append(min(1.0, bb_m[k] / wb_v))
                else:
                    parts.append(0.0)
            if with_id and bb_i is not None and wb["id"] and wb["id"] > 0:
                parts.append(min(1.0, bb_i / wb["id"]))

            if parts:
                source_pair_scores[source].append(sum(parts) / len(parts))

        if skipped_no_target_wb:
            print(f"⚠️  [{dataset_tag}/{attack}] {skipped_no_target_wb} BB pair(s) had "
                  f"no target WB folder → TE skipped (CIDR still reported).")

        # Emit a row for any source with at least raw CIDR or TE data.
        all_sources = set(source_pair_scores) | set(source_pair_cidr)
        for source in sorted(all_sources):
            scores = source_pair_scores.get(source, [])
            cidr_list = source_pair_cidr.get(source, [])
            report[dataset_tag][attack][source] = {
                "te": sum(scores) / len(scores) if scores else None,
                "cidr": sum(cidr_list) / len(cidr_list) if cidr_list else None,
                "n_targets_te": len(scores),
                "n_targets_cidr": len(cidr_list),
            }

    return report


def emit_te_summary(report: dict, output_root: Path) -> None:
    rows: list[dict] = []
    for dataset, by_attack in report.items():
        for attack, by_source in by_attack.items():
            for source, entry in by_source.items():
                rows.append({
                    "dataset": dataset,
                    "attack": attack,
                    "source": source,
                    "te": entry["te"],
                    "cidr": entry["cidr"],
                    "n_te": entry.get("n_targets_te", 0),
                    "n_cidr": entry.get("n_targets_cidr", 0),
                })
    rows.sort(key=lambda r: (r["dataset"], r["attack"], r["source"]))

    def _fmt(v, prec=4):
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "  n/a "

    print("\n" + "=" * 88)
    print(" Final Transferability (TE) Summary    (CIDR = mean BB relative_id)")
    print("=" * 88)
    print(f"{'Dataset':<12} {'Attack':<12} {'Source':<14} "
          f"{'TE':>8} {'CIDR':>8} {'#TE':>5} {'#CIDR':>6}")
    print("-" * 88)
    for r in rows:
        print(f"{r['dataset']:<12} {r['attack']:<12} {r['source']:<14} "
              f"{_fmt(r['te']):>8} {_fmt(r['cidr']):>8} "
              f"{r['n_te']:>5} {r['n_cidr']:>6}")
    print("=" * 88)
    print(f"Total entries: {len(rows)}  "
          f"(TE shows n/a when a source has no target with a white-box result)\n")

    if not rows:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_transferability_analysis.json"
    csv_path = output_root / "final_transferability_summary.csv"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "attack", "source",
                                          "te", "cidr", "n_te", "n_cidr"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "attack": r["attack"], "source": r["source"],
                "te":   f"{r['te']:.4f}"   if r["te"]   is not None else "",
                "cidr": f"{r['cidr']:.4f}" if r["cidr"] is not None else "",
                "n_te": r["n_te"], "n_cidr": r["n_cidr"],
            })

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    buckets = discover_buckets(args)

    print("=" * 60)
    if args.results_dir:
        print(f"Mode:        explicit (--results_dir {args.results_dir})")
    else:
        print(f"Attacks:     {args.attack}")
        print(f"Datasets:    {args.dataset}")
        print(f"Wrappers:    {args.wrappers}")
    print(f"Buckets:     {len(buckets)}")
    print(f"Skip ID:     {args.skip_id}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    if not buckets:
        print("Nothing to evaluate.")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Per-folder evaluation for every black_box_*_* AND every result_<w>_<a>.
    # (We need both: BB folders for transfer measurements, WB folders to
    # provide the denominator. We re-run the WB evaluators with the SAME
    # tools so the metric/ID files are guaranteed compatible.)
    valid = set(args.wrappers)
    success = failed = 0
    for dataset_tag, attack, ds_root in buckets:
        print(f"\n--- [{dataset_tag} / {attack}] {ds_root} ---")
        # WB folders (denominators)
        for wrapper in args.wrappers:
            wb = ds_root / f"result_{wrapper}_{attack}"
            if wb.is_dir():
                if evaluate_folder(wb, args, env):
                    success += 1
                else:
                    failed += 1
        # BB folders (numerators)
        bb_root = ds_root / "black_box_results"
        for bb in sorted(bb_root.iterdir()):
            if not bb.is_dir():
                continue
            parsed = _parse_bb_folder(bb.name, valid)
            if parsed is None:
                continue
            if evaluate_folder(bb, args, env):
                success += 1
            else:
                failed += 1

    print("=" * 60)
    print(f"Per-folder eval: success={success} failed={failed}")
    print("=" * 60)

    # TE aggregation (always runs so the user sees what's currently on-disk).
    report = aggregate_te(buckets, args.wrappers, with_id=not args.skip_id)
    output_root = Path(args.output_root).resolve() if args.output_root else ROOT
    emit_te_summary(report, output_root)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
