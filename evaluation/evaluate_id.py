#!/usr/bin/env python3
"""
ID Loss 계산 스크립트
- x_src와 decoded_src 간의 ID (org_id)
- x_src와 decoded_adv 간의 ID (adv_id)
- org_id - adv_id 계산
- 각 result 폴더별 100장 평균

결과: <PROJECT_ROOT>/id_result/방법론_생성모델.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Path setup — make project root importable from anywhere.
current_dir = os.path.dirname(os.path.abspath(__file__))      # evaluation/
project_root = os.path.dirname(current_dir)                   # Project Root
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from evaluation.face_idloss import IDLoss
from utils.utils import denorm
from configs.paths_config import ARCFACE_CKPT

# Lazy load — only instantiate when compute_id_loss is actually called.
arcface = None


def _arcface() -> IDLoss:
    global arcface
    if arcface is None:
        arcface = IDLoss(ARCFACE_CKPT).to(DEVICE).eval()
    return arcface


def load_image_as_tensor(image_path, size=256):
    """Load image and convert to tensor [-1, 1]"""
    try:
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        return tensor.detach()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_tensor_from_pt(file_path):
    """Load tensor from .pt file"""
    try:
        tensor = torch.load(file_path, map_location=DEVICE)
        if isinstance(tensor, dict):
            for key in ['tensor', 'image', 'data', 'img']:
                if key in tensor:
                    tensor = tensor[key]
                    break
            if isinstance(tensor, dict):
                tensor = list(tensor.values())[0]
        
        tensor = tensor.to(DEVICE)
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.detach()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_id_loss(img1, img2):
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    img1 = F.interpolate(denorm(img1), size=(112, 112), mode='bilinear')
    img2 = F.interpolate(denorm(img2), size=(112, 112), mode='bilinear')
    loss = _arcface()(img1, img2)
    return float(loss)


def process_result_folder(result_dir):
    """
    단일 result 폴더 처리
    Returns: dict with per-image results and averages
    """
    results = {
        "per_image": {},
        "averages": {}
    }
    
    org_ids = []
    adv_ids = []
    diffs = []
    relative_ids = []
    
    # img_001 ~ img_100 폴더 처리
    img_folders = sorted([d for d in os.listdir(result_dir) if d.startswith('img_')])
    
    for img_folder in tqdm(img_folders, desc=f"Processing {os.path.basename(result_dir)}"):
        img_path = os.path.join(result_dir, img_folder)
        
        if not os.path.isdir(img_path):
            continue
        
        # 파일 경로 설정
        x_src_jpg = os.path.join(img_path, "x_src.jpg")
        x_src_pt = os.path.join(img_path, "x_src.pt")
        decoded_src_pt = os.path.join(img_path, "decoded_src.pt")
        decoded_adv_pt = os.path.join(img_path, "decoded_adv.pt")
        
        # x_src 로드 (pt가 있으면 pt, 없으면 jpg)
        if os.path.exists(x_src_pt):
            x_src = load_tensor_from_pt(x_src_pt)
        elif os.path.exists(x_src_jpg):
            x_src = load_image_as_tensor(x_src_jpg)
        else:
            print(f"Warning: x_src not found in {img_path}")
            continue

        # x_src = load_tensor_from_pt(x_src_pt)
        
        # decoded_src, decoded_adv 로드
        if not os.path.exists(decoded_src_pt) or not os.path.exists(decoded_adv_pt):
            print(f"Warning: decoded files not found in {img_path}")
            continue
        
        decoded_src = load_tensor_from_pt(decoded_src_pt)
        decoded_adv = load_tensor_from_pt(decoded_adv_pt)
        
        if x_src is None or decoded_src is None or decoded_adv is None:
            print(f"Warning: Failed to load tensors in {img_path}")
            continue
        
        # ID loss 계산
        try:
            org_id = compute_id_loss(x_src, decoded_src)
            adv_id = compute_id_loss(x_src, decoded_adv)
            diff = org_id - adv_id
            # relative_id: max(0, 1 - org_id/adv_id)
            relative_id = max(0, 1 - org_id / adv_id) if adv_id != 0 else 0
            
            results["per_image"][img_folder] = {
                "org_id": org_id,
                "adv_id": adv_id,
                "diff": diff,
                "relative_id": relative_id
            }
            
            org_ids.append(org_id)
            adv_ids.append(adv_id)
            diffs.append(diff)
            relative_ids.append(relative_id)
            
        except Exception as e:
            print(f"Error computing ID loss for {img_folder}: {e}")
            continue
    
    # 평균 계산
    if len(org_ids) > 0:
        results["averages"] = {
            "org_id_mean": sum(org_ids) / len(org_ids),
            "adv_id_mean": sum(adv_ids) / len(adv_ids),
            "diff_mean": sum(diffs) / len(diffs),
            "relative_id_mean": sum(relative_ids) / len(relative_ids),
            "total_images": len(org_ids)
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute relative_id ArcFace metrics for a folder of img_* "
                    "subdirectories (works for both white-box result_* and "
                    "black-box black_box_*_* layouts)."
    )
    parser.add_argument("--result_dir", required=True,
                        help="Directory containing img_* subdirectories with "
                             "x_src, decoded_src, decoded_adv files.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write eval_id.json. Defaults to --result_dir.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.is_dir():
        print(f"❌ Directory not found: {result_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {result_dir.name}")
    results = process_result_folder(str(result_dir))

    if not results.get("averages"):
        print(f"⚠️  No valid results for {result_dir.name}")
        sys.exit(1)

    output_path = output_dir / "eval_id.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {output_path}")
    avgs = results["averages"]
    print(f"  org_id_mean:      {avgs['org_id_mean']:.6f}")
    print(f"  adv_id_mean:      {avgs['adv_id_mean']:.6f}")
    print(f"  diff_mean:        {avgs['diff_mean']:.6f}")
    print(f"  relative_id_mean: {avgs['relative_id_mean']:.4f}")
    print(f"  total_images:     {avgs['total_images']}")


if __name__ == "__main__":
    main()
