#!/usr/bin/env python3
"""Black-box transferability evaluation launcher.

For each (source, target) wrapper pair (source != target), feed the
adversarial perturbation crafted against `source` into `target`, and save
the resulting decoded output. This measures how well a white-box attack
transfers to an unseen victim model.

Operates on the output produced by white_box_attack.py:
    batch_results_<attack>/<dataset>/result_<wrapper>_<attack>/img_*/x_adv.pt

Example
-------
    # Run transfer over all 8 wrappers using FFHQ PGD results
    python transferability.py --attack pgd --dataset ffhq --gpu 0
    python transferability.py --attack pgd --dataset celeba --gpu 0 --n_images 10

    # Only test attacks transferring TO diffae
    python transferability.py --attack pgd --dataset ffhq \\
        --target_only diffae --gpu 0
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
RUNNER = ROOT / "attacks" / "run_blackbox.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run black-box transferability evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Either point straight at the results dir, or recreate it from --attack + --dataset.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results_dir", type=str,
                     help="Explicit batch_results_<attack>/<dataset> directory.")
    src.add_argument("--attack", choices=sorted(ATTACKS.keys()),
                     help="Attack whose results to transfer (used with --dataset).")

    p.add_argument("--dataset", choices=sorted(DATASETS.keys()),
                   help="Dataset subdirectory under the attack results "
                        "(required when --attack is given).")
    p.add_argument("--results_subdir", type=str, default=None,
                   help="Override the dataset subdirectory name.")

    p.add_argument("--wrappers", nargs="+", choices=WRAPPERS, default=WRAPPERS,
                   help="Wrappers participating in the transfer matrix.")
    p.add_argument("--target_only", choices=WRAPPERS, default=None,
                   help="Restrict to a single victim model (sources = the other wrappers).")

    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--output_base", type=str, default=None,
                   help="Override output directory (default: <results_dir>/black_box_results).")
    p.add_argument("--target_image", type=str, default=None,
                   help="Override default_target_path for wrappers that require one. "
                        "If unset and --dataset has a known target, that one is used.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the command that would be executed and exit.")

    args = p.parse_args()
    if args.attack and not (args.dataset or args.results_subdir):
        p.error("--attack requires --dataset (or --results_subdir).")
    return args


def resolve_results_dir_and_target(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.results_dir:
        return Path(args.results_dir).resolve(), args.target_image

    _, default_base = ATTACKS[args.attack]
    subdir = args.results_subdir if args.results_subdir is not None else args.dataset
    results_dir = (ROOT / default_base / subdir).resolve()

    target = args.target_image
    if target is None and args.dataset:
        target = DATASETS[args.dataset]["target"]
    return results_dir, target


def main() -> int:
    args = parse_args()
    results_dir, target_image = resolve_results_dir_and_target(args)

    if not results_dir.is_dir():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    output_base = args.output_base or str(results_dir / "black_box_results")

    print("=" * 60)
    print(f"Results dir:   {results_dir}")
    print(f"Output base:   {output_base}")
    print(f"Wrappers:      {args.wrappers}")
    if args.target_only:
        print(f"Target only:   {args.target_only}")
    print(f"Target image:  {target_image or '(use wrapper defaults)'}")
    print(f"GPU:           {args.gpu}")
    print(f"N images:      {args.n_images}")
    print("=" * 60)

    cmd = [
        sys.executable, str(RUNNER),
        "--default_path", str(results_dir),
        "--output_base", output_base,
        "--wrappers", *args.wrappers,
        "--n_images", str(args.n_images),
    ]
    if args.target_only:
        cmd += ["--target_only", args.target_only]
    if target_image:
        if not os.path.isfile(target_image):
            print(f"⚠️  target_image not found: {target_image} (skipping --target_image)")
        else:
            cmd += ["--target_image", target_image]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(">>> " + " ".join(cmd))
    if args.dry_run:
        return 0

    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
