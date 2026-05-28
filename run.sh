#!/bin/bash

set -e

source ~/miniconda3/etc/profile.d/conda.sh

CONDA_ENV="${CONDA_ENV:-deepfake-benchmark}"
conda activate "$CONDA_ENV"

echo "===== 1. white_box_attack: bldra wrappers ====="
python white_box_attack.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1 \
  --wrappers "blendface" "stargan" "simswap" "psp_mix" "diffae" "styleclip"

echo "===== 2. white_box_attack: ldm wrappers ====="
python white_box_attack.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1 \
  --wrappers "diffface" "diffswap"

echo "===== 3. Robustness: bldra wrappers ====="
python robustness.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1 \
  --wrappers "blendface" "stargan" "simswap" "psp_mix" "diffae" "styleclip"

echo "===== 4. Robustness: ldm wrappers ====="
python robustness.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1 \
  --wrappers "diffface" "diffswap"

echo "===== 5. transferability: bldra all/default ====="
python transferability.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1

echo "===== 6. transferability: ldm all/default ====="
python transferability.py \
  --attack pgd \
  --dataset celeba \
  --gpu 0 \
  --n_images 1

echo "===== 7. evaluations: bldra ====="
python white_box_evaluation.py

python robustness_evaluation.py

python transferability_evaluation.py

echo "===== All jobs finished successfully ====="