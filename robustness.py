#!/usr/bin/env python3
"""Robustness evaluation launcher for adversarial attack results.

Replaces run_batch_robustness.sh. Operates on the output produced by
white_box_attack.py: it auto-locates batch_results_<attack>/<dataset>/result_*
directories and runs attacks/batch_robustness_test.py against each one.

Example
-------
    # Evaluate PGD results on FFHQ for two wrappers:
    python robustness.py --attack pgd --dataset ffhq --wrappers diffae simswap --gpu 0

    python robustness.py --attack pgd --dataset celeba --wrappers simswap --gpu 0

    # Evaluate every wrapper found in an arbitrary batch directory:
    python robustness.py --results_dir batch_results_df_rap/celeba --gpu 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from configs.paths_config import DATASETS
from white_box_attack import ATTACKS, WRAPPERS

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "attacks" / "batch_robustness_test.py"

# Default preprocessing methods (matches the original shell script).
DEFAULT_PREPROCESSING = [
    "jpeg70", "jpeg90",
    "blur1", "blur3",
    "noise001", "noise003",
    "salt_pepper001", "salt_pepper003",
]

# Order priority used by the original shell script.
PRIORITY_WRAPPERS = ("simswap", "styleclip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run robustness tests against adversarial attack results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Either point straight at a results directory, or recreate the path from
    # --attack + --dataset (mirrors white_box_attack.py's conventions).
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results_dir", type=str,
                     help="Explicit batch_results_<attack>/<dataset> directory.")
    src.add_argument("--attack", choices=sorted(ATTACKS.keys()),
                     help="Attack whose results to evaluate (used with --dataset).")

    p.add_argument("--dataset", choices=sorted(DATASETS.keys()),
                   help="Dataset subdirectory under the attack results "
                        "(required when --attack is given, unless "
                        "--results_subdir overrides it).")
    p.add_argument("--results_subdir", type=str, default=None,
                   help="Override the dataset subdirectory name.")

    p.add_argument("--wrappers", nargs="+", choices=WRAPPERS, default=None,
                   help="Subset of wrappers to evaluate. Default: every "
                        "result_<wrapper>_<attack> directory found.")

    p.add_argument("--target_image", type=str, default=None,
                   help="Target identity image to render the swap toward. "
                        "MUST match what the attack used. If unset, falls back "
                        "to the --dataset preset target (same as white_box_attack.py).")

    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--preprocessing", nargs="+", default=DEFAULT_PREPROCESSING,
                   help="Preprocessing perturbations to test against.")
    p.add_argument("--output_base", type=str, default=None,
                   help="Override the robustness_results root "
                        "(default: ./robustness_results).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would be executed and exit.")

    args = p.parse_args()
    if args.attack and not (args.dataset or args.results_subdir):
        p.error("--attack requires --dataset (or --results_subdir).")
    return args


def resolve_results_dir(args: argparse.Namespace) -> tuple[Path, str]:
    """Return (results_dir, dataset_tag). dataset_tag is used to namespace
    the robustness output so different datasets don't collide."""
    if args.results_dir:
        path = Path(args.results_dir).resolve()
        # Best-effort: dataset is the immediate parent's basename
        # (e.g. .../batch_results_pgd/ffhq -> "ffhq"). Fall back to "custom".
        tag = path.name if path.is_dir() else "custom"
        return path, tag

    _, default_base = ATTACKS[args.attack]
    subdir = args.results_subdir if args.results_subdir is not None else args.dataset
    return (ROOT / default_base / subdir).resolve(), subdir


def discover_result_dirs(results_dir: Path,
                         wanted: list[str] | None) -> list[tuple[Path, str]]:
    """Return [(result_dir, wrapper_name), ...] in priority order."""
    if not results_dir.is_dir():
        raise SystemExit(f"❌ Directory not found: {results_dir}")

    matches: list[tuple[Path, str]] = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("result_"):
            continue
        wrapper = _wrapper_from_dirname(entry.name)
        if wrapper is None:
            print(f"⚠️  Skipping (unknown wrapper): {entry.name}")
            continue
        if wanted is not None and wrapper not in wanted:
            continue
        matches.append((entry, wrapper))

    # Sort priority wrappers first, others alphabetically (stable).
    matches.sort(key=lambda x: (
        PRIORITY_WRAPPERS.index(x[1]) if x[1] in PRIORITY_WRAPPERS else len(PRIORITY_WRAPPERS),
        x[0].name,
    ))
    return matches


def _wrapper_from_dirname(name: str) -> str | None:
    # WRAPPERS is checked longest-first so e.g. 'psp_mix' wins over 'psp'.
    for w in sorted(WRAPPERS, key=len, reverse=True):
        if w in name:
            return w
    return None


def main() -> int:
    args = parse_args()
    results_dir, dataset_tag = resolve_results_dir(args)
    output_base = Path(args.output_base).resolve() if args.output_base else ROOT / "robustness_results"

    print("=" * 60)
    print(f"Results dir:   {results_dir}")
    print(f"Output base:   {output_base / dataset_tag}")
    print(f"GPU:           {args.gpu}")
    print(f"N images:      {args.n_images}")
    print(f"Preprocessing: {args.preprocessing}")
    if args.wrappers:
        print(f"Wrappers:      {args.wrappers}")
    print("=" * 60)

    targets = discover_result_dirs(results_dir, args.wrappers)
    if not targets:
        print(f"❌ No matching result_* directories in {results_dir}")
        return 1

    print(f"Found {len(targets)} result directories\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    # Propagate the SAME target identity the attack used. white_box_attack.py
    # sets DEEPFAKE_TARGET_IMAGE from the dataset preset; mirror that here so
    # target-requiring wrappers (simswap/blendface/...) re-render the swap
    # toward the identity x_adv was actually optimized against.
    target_image = args.target_image
    if target_image is None and args.dataset and args.dataset in DATASETS:
        target_image = DATASETS[args.dataset].get("target")
    if target_image and os.path.isfile(target_image):
        env["DEEPFAKE_TARGET_IMAGE"] = target_image
    else:
        env.pop("DEEPFAKE_TARGET_IMAGE", None)
        if target_image:
            print(f"⚠️  target image not found: {target_image} "
                  "— falling back to wrapper default_target_path")
    print(f"Target image:  {env.get('DEEPFAKE_TARGET_IMAGE', '(wrapper default)')}")

    success = failed = 0
    for result_dir, wrapper in targets:
        out_dir = output_base / dataset_tag / result_dir.name
        cmd = [
            sys.executable, str(RUNNER),
            "--result_dir", str(result_dir),
            "--wrapper", wrapper,
            "--n_images", str(args.n_images),
            "--preprocessing", *args.preprocessing,
            "--output_dir", str(out_dir),
        ]

        print(f">>> {result_dir.name} (wrapper={wrapper})")
        print("    " + " ".join(cmd))
        if args.dry_run:
            continue

        rc = subprocess.run(cmd, env=env).returncode
        if rc == 0:
            print(f"✅ {wrapper} OK\n")
            success += 1
        else:
            print(f"❌ {wrapper} failed (exit {rc})\n")
            failed += 1

    print("=" * 60)
    print(f"Done. success={success} failed={failed}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
