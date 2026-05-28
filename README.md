# Proactive Defense Benchmark against Deepfake Generation

**Project page:** https://proactivedefensebenchmark.github.io/

Reference implementation for *"Proactive Defense Benchmark against Deepfake Generation"* (ICML 2026).
The benchmark unifies the evaluation of proactive defenses against deepfake generation along
three axes — **Disruption**, **Robustness**, and **Transferability** — across pixel-level
fidelity, perceptual fidelity, identity disruption (**CIDR**), and visual quality (BRISQUE).

The pipeline has three stages, each with a paired launcher and evaluator:

| Stage | Attack launcher | Evaluator | Purpose |
|---|---|---|---|
| White-box | `white_box_attack.py` | `white_box_evaluation.py` | Run PGD / Disrupting / DF-RAP / Anti-Forgery / LEAT against each deepfake generator and score the protected outputs. |
| Robustness | `robustness.py` | `robustness_evaluation.py` | Apply JPEG / blur / noise / salt-and-pepper to the protected images, then re-render and aggregate **ROB**. |
| Transferability | `transferability.py` | `transferability_evaluation.py` | Cross-feed perturbations crafted on a source generator into every other target generator and aggregate **TE**. |

`run.sh` chains all six scripts end-to-end with a minimal example configuration.

---

## 1. Setup

### 1.1 Conda environment

```bash
conda env create -f environment.deepfake-benchmark.yml
conda activate deepfake-benchmark
```

The environment file pins Python 3.9.7, PyTorch + CUDA 11.8, and every extra
package (LPIPS, kornia, mtcnn, face_alignment, taming-transformers,
pytorch-lightning 1.9.5, etc.) needed by the wrappers.

### 1.2 Datasets

Datasets are **not** shipped in the repo. Download the three packed archives
from Google Drive and unpack them under `datasets/` so the layout matches:

```
datasets/
├── CelebAHQ/
│   ├── source/     # 100 source images (CelebA-HQ 256×256)
│   └── target/29550.jpg
├── FFHQ/
│   ├── source/
│   └── target/69999.png
└── VGGface2/
    ├── source/
    └── target/0066_01.jpg
```

| Dataset | Google Drive |
|---|---|
| CelebA-HQ | https://drive.google.com/file/d/1WbDoqu7HU5TjPFI3AhHI3kfuDblsfLTZ/view?usp=share_link |
| FFHQ | https://drive.google.com/file/d/1KXBhkjX8X_gMTW25iQSJWv58KXr6SJ13/view?usp=share_link |
| VGGFace2-HQ | https://drive.google.com/file/d/1BuboHLC_hRul0ASbDLxSZA0ljk_WAfcs/view?usp=share_link |

`configs/paths_config.py` resolves these locations via the `DATASETS` table
(`celeba` / `ffhq` / `vggface2hq`); you do not need to touch it unless you
relocate the data outside the repo.

### 1.3 Deepfake generators

Each generator lives in its own folder under `deepfake_generators/`, alongside
`face_idloss/` (the shared ArcFace identity backbone, see 1.4):

```
deepfake_generators/
├── BlendFace/
├── diffae/
├── DiffFace/
├── DiffSwap/
├── SimSwap/
├── StarGAN/
├── styleclip/
├── pixel2style2pixel/
└── face_idloss/
```

Each generator is fetched from its upstream repository. Clone them into the
expected directory names:

```bash
cd deepfake_generators

StarGAN: git clone https://github.com/yunjey/StarGAN.git
StyleCLIP: git clone https://github.com/orpatashnik/StyleCLIP.git
DiffAE: git clone https://github.com/konpatp/diffae.git
SimSwap: git clone https://github.com/neuralchen/SimSwap.git             
pixel2style2pixel: git clone https://github.com/eladrich/pixel2style2pixel.git
BlendFace: git clone https://github.com/mapooon/BlendFace.git           
DiffSwap: git clone https://github.com/wl-zhao/DiffSwap.git            
DiffFace: git clone https://github.com/hxngiee/DiffFace.git
```

Place each generator's pretrained weights inside its `checkpoints/` directory
following the upstream instructions. Wrappers in `wrappers/` adapt every
generator to a common `(x_src, x_tgt) → decoded` interface used by the attack
and evaluation scripts.

### 1.4 ArcFace identity checkpoint

The identity-disruption metric (**CIDR** / ID loss) relies on an ArcFace
backbone. Download `model_ir_se50.pth` from Google Drive and place it under
`deepfake_generators/face_idloss/` so the path matches `ARCFACE_CKPT` in
`configs/paths_config.py`:

```
deepfake_generators/
└── face_idloss/
    └── model_ir_se50.pth
```

| Checkpoint | Google Drive |
|---|---|
| ArcFace (`model_ir_se50.pth`) | https://drive.google.com/file/d/12jmxzEELXcMmFSSrSr98wM-Q51Zl11nc/view?usp=share_link |

`configs/paths_config.py` resolves this via `ARCFACE_CKPT`; you do not need to
touch it unless you relocate the checkpoint.

---

## 2. Pipeline

All scripts share a common CLI:
`--attack`, `--dataset` (`celeba` / `ffhq` / `vggface2hq`),
`--wrappers` (subset of `blendface stargan simswap psp_mix diffae styleclip diffface diffswap`),
`--gpu`, `--n_images`. Add `--dry_run` to any launcher to print the commands
without executing them.

### 2.1 White-box attack — `white_box_attack.py`

Optimizes the protected image `x_adv` against each chosen generator and dumps
`x_adv.pt`, `decoded_src.pt`, `decoded_adv.pt` per image under
`batch_results_<attack>/<dataset>/result_<wrapper>_<attack>/img_XXX/`.

```bash
# PGD on CelebA-HQ for the GAN/AE-based wrappers
python white_box_attack.py \
  --attack pgd --dataset celeba --gpu 0 --n_images 100 \
  --wrappers blendface stargan simswap psp_mix diffae styleclip

# Same attack on the diffusion-based wrappers
python white_box_attack.py \
  --attack pgd --dataset celeba --gpu 0 --n_images 100 \
  --wrappers diffface diffswap
```

Supported attacks: `pgd`, `disrupting`, `df_rap`, `anti`, `leat`.

### 2.2 White-box evaluation — `white_box_evaluation.py`

Walks the `batch_results_*/` cube, runs
`evaluation/evaluate_disruption.py` (MSE / LPIPS / BRISQUE) and
`evaluation/evaluate_id.py` (CIDR) per `result_*`, and writes
`final_white_box_summary.csv` + `final_white_box_analysis.json` at the repo
root.

```bash
# Evaluate every (attack × dataset × wrapper) combination found on disk
python white_box_evaluation.py

# Or narrow it down
python white_box_evaluation.py --attack pgd --dataset celeba --wrappers diffae simswap
```

### 2.3 Robustness — `robustness.py`

For every `result_*` directory produced by the white-box attack, applies the
post-processing transforms below to `x_adv`, re-renders the generator, and
saves `robustness_results/<dataset>/result_<wrapper>_<attack>/<prep>/...`.

Default transforms (mirror those in the paper): `jpeg70`, `jpeg90`, `blur1`,
`blur3`, `noise001`, `noise003`, `salt_pepper001`, `salt_pepper003`.

```bash
python robustness.py \
  --attack pgd --dataset celeba --gpu 0 --n_images 100 \
  --wrappers blendface stargan simswap psp_mix diffae styleclip

python robustness.py \
  --attack pgd --dataset celeba --gpu 0 --n_images 100 \
  --wrappers diffface diffswap
```

### 2.4 Robustness evaluation — `robustness_evaluation.py`

Runs `evaluate_robustness.py` (quality metrics under each perturbation) and
`evaluate_id_robust.py` (CIDR retention), then aggregates the final **ROB**
score per `(dataset, attack, wrapper)`:

```
ROB = mean over preprocessings of mean(
    mse_ratio, lpips_ratio, brisque_ratio, id_retention
)   # each ratio clipped to 1.0
```

```bash
python robustness_evaluation.py                                   # everything found
python robustness_evaluation.py --attack pgd --dataset celeba     # subset
```

Outputs `final_robustness_summary.csv` + `final_robustness_analysis.json`.

### 2.5 Transferability — `transferability.py`

For every `(source, target)` wrapper pair, feeds the `x_adv.pt` crafted
against `source` into `target` and saves the cross-rendered outputs to
`batch_results_<attack>/<dataset>/black_box_results/black_box_<src>_<tgt>/`.

```bash
python transferability.py --attack pgd --dataset celeba --gpu 0 --n_images 100

# Only test attacks transferring TO diffae:
python transferability.py --attack pgd --dataset celeba --target_only diffae
```

### 2.6 Transferability evaluation — `transferability_evaluation.py`

Evaluates each `black_box_<src>_<tgt>/` folder and computes **TE**
normalized against the target's own white-box performance:

```
TE(source) = mean over targets T (T != source) of mean(
    l2_mse(BB)/l2_mse(WB), lpips(BB)/lpips(WB),
    brisque(BB)/brisque(WB), relative_id(BB)/relative_id(WB)
)   # clipped to 1.0
```

```bash
python transferability_evaluation.py
python transferability_evaluation.py --attack pgd --dataset celeba
```

Outputs `final_transferability_summary.csv` + `final_transferability_analysis.json`.

---

## 3. One-shot reproduction — `run.sh`

`run.sh` activates the conda environment and chains stages 2.1 → 2.6 with a
minimal `--n_images 1` smoke-test configuration on CelebA-HQ:

```bash
CONDA_ENV=deepfake-benchmark bash run.sh
```

To run the full paper configuration, bump `--n_images` to `100` inside
`run.sh` (or copy it and parameterize). The script aborts on the first
failing stage (`set -e`) and prints a banner before each step so you can
match the on-disk artifacts to the corresponding stage.

---

## 4. Citation
