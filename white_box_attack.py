#!/usr/bin/env python3
"""Unified white-box adversarial attack launcher.

Replaces the per-attack shell scripts (run_batch_pgd.sh, run_batch_disrupting.sh,
run_batch_df_rap.sh, run_batch_anti.sh, run_batch_leat.sh) with one entry point.

Pick an attack, one or more deepfake wrappers, and a dataset, and the script
runs each wrapper sequentially through the appropriate batch_*_attack.py.

Example
-------
    python white_box_attack.py --attack pgd --dataset ffhq --wrappers diffae simswap --gpu 0 --n_images 100
    python white_box_attack.py --attack pgd --dataset celeba --wrappers simswap --gpu 0 --n_images 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from configs.paths_config import DATASETS

ROOT = Path(__file__).resolve().parent

# Attack name -> (underlying script (relative to ROOT), default output_base directory)
ATTACKS = {
    "pgd":        ("attacks/batch_pgd_attack.py",                 "batch_results_pgd"),
    "disrupting": ("attacks/batch_disrupting_deepfake_attack.py", "batch_results_disrupting"),
    "df_rap":     ("attacks/batch_df_rap_attack.py",              "batch_results_df_rap"),
    "anti":       ("attacks/batch_anti_attack.py",                "batch_results_anti"),
    "leat":       ("attacks/batch_leat_attack.py",                "batch_results_leat"),
}

WRAPPERS = [
    "blendface", "stargan", "simswap", "psp_mix",
    "diffae", "styleclip", "diffface", "diffswap",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a white-box adversarial attack against deepfake wrappers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--attack", required=True, choices=sorted(ATTACKS.keys()),
                   help="Attack method to run.")
    p.add_argument("--wrappers", nargs="+", required=True, choices=WRAPPERS,
                   help="Deepfake wrappers to attack (one or more).")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=sorted(DATASETS.keys()),
                     help="Dataset preset (resolved via configs/paths_config.py).")
    src.add_argument("--image_dir", type=str,
                     help="Explicit source image directory (overrides --dataset).")

    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--n_images", type=int, default=100,
                   help="Number of images to process per wrapper.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_base", type=str, default=None,
                   help="Override base output directory (default: ./<attack default>).")
    p.add_argument("--output_subdir", type=str, default=None,
                   help="Subdirectory under output_base. "
                        "Defaults to the --dataset name when --dataset is used.")
    p.add_argument("--target_image", type=str, default=None,
                   help="Override target identity image (sets DEEPFAKE_TARGET_IMAGE). "
                        "If unset and --dataset has a known target, that one is used.")
    p.add_argument("--comg_checkpoint", type=str, default=None,
                   help="(df_rap only) Path to a pretrained ComG checkpoint.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would be executed and exit.")
    return p.parse_args()


def resolve_image_dir_and_target(args: argparse.Namespace) -> tuple[str, str | None, str | None]:
    """Return (image_dir, target_image, dataset_tag)."""
    if args.image_dir is not None:
        return args.image_dir, args.target_image, args.output_subdir

    preset = DATASETS[args.dataset]
    target = args.target_image if args.target_image is not None else preset["target"]
    subdir = args.output_subdir if args.output_subdir is not None else args.dataset
    return preset["image_dir"], target, subdir


def build_command(attack: str, wrapper: str, image_dir: str,
                  output_base: str, output_subdir: str | None,
                  n_images: int, seed: int,
                  comg_checkpoint: str | None) -> list[str]:
    script, _ = ATTACKS[attack]
    cmd = [
        sys.executable, str(ROOT / script),
        "--image_dir", image_dir,
        "--n_images", str(n_images),
        "--wrappers", wrapper,
        "--seed", str(seed),
        "--output_base", output_base,
    ]
    if output_subdir:
        cmd += ["--output_subdir", output_subdir]
    if attack == "df_rap" and comg_checkpoint:
        cmd += ["--comg_checkpoint", comg_checkpoint]
    return cmd


def main() -> int:
    args = parse_args()
    image_dir, target_image, dataset_tag = resolve_image_dir_and_target(args)

    output_base = args.output_base or str(ROOT / ATTACKS[args.attack][1])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if target_image and os.path.isfile(target_image):
        env["DEEPFAKE_TARGET_IMAGE"] = target_image
    else:
        env.pop("DEEPFAKE_TARGET_IMAGE", None)
        if target_image:
            print(f"⚠️  target image not found: {target_image} — DEEPFAKE_TARGET_IMAGE unset")

    print("=" * 60)
    print(f"Attack:        {args.attack}")
    print(f"Wrappers:      {args.wrappers}")
    print(f"Image dir:     {image_dir}")
    print(f"Output base:   {output_base}")
    if dataset_tag:
        print(f"Output subdir: {dataset_tag}")
    print(f"GPU:           {args.gpu}")
    print(f"N images:      {args.n_images}")
    print(f"Target image:  {env.get('DEEPFAKE_TARGET_IMAGE', '(none)')}")
    print("=" * 60)

    if args.attack == "df_rap" and args.comg_checkpoint is None:
        print("ℹ️  df_rap: no --comg_checkpoint provided; using random init.")

    overall_rc = 0
    for wrapper in args.wrappers:
        cmd = build_command(
            attack=args.attack,
            wrapper=wrapper,
            image_dir=image_dir,
            output_base=output_base,
            output_subdir=dataset_tag,
            n_images=args.n_images,
            seed=args.seed,
            comg_checkpoint=args.comg_checkpoint,
        )
        print(f"\n>>> [{args.attack}] wrapper={wrapper}")
        print("    " + " ".join(cmd))
        if args.dry_run:
            continue

        rc = subprocess.run(cmd, env=env).returncode
        if rc == 0:
            print(f"✅ {wrapper} completed")
        else:
            print(f"❌ {wrapper} failed with exit code {rc}")
            overall_rc = rc

    print("\n" + "=" * 60)
    print(f"Done. Exit code: {overall_rc}")
    print("=" * 60)
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
