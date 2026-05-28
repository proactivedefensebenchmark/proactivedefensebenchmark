#!/usr/bin/env python3
"""
Robustness ID Loss 계산 스크립트
- x_src_clean과 decoded_clean 간의 ID (clean_id)
- x_src_clean과 각 decoded_adv_* 간의 ID (adv_id)
- clean_id - adv_id 계산 (diff)
- relative_id: max(0, 1 - clean_id/adv_id)

결과: 각 result_* 폴더 내에 eval_id.json으로 저장
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

# Path setup — make project root importable from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))      # evaluation/
project_root = os.path.dirname(current_dir)                   # Project Root
for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from evaluation.face_idloss import IDLoss
from utils.utils import denorm
from configs.paths_config import ARCFACE_CKPT

# Lazy load: only instantiate when compute_id_loss is actually called.
arcface = None


def _arcface() -> IDLoss:
    global arcface
    if arcface is None:
        arcface = IDLoss(ARCFACE_CKPT).to(DEVICE).eval()
    return arcface

# Robustness 테스트 종류
ROBUSTNESS_TYPES = [
    "original",
    "blur1",
    "blur3", 
    "jpeg70",
    "jpeg90",
    "noise001",
    "noise003",
    "salt_pepper001",
    "salt_pepper003"
]


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
        
        # 메모리 연속성 보장 (view 연산 호환성)
        return tensor.detach().contiguous()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_id_loss(img1, img2):
    """Compute ID loss between two images using ArcFace"""
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    img1 = F.interpolate(denorm(img1), size=(112, 112), mode='bilinear')
    img2 = F.interpolate(denorm(img2), size=(112, 112), mode='bilinear')
    loss = _arcface()(img1, img2)
    return float(loss)


def process_robustness_folder(result_dir):
    """
    단일 result_* 폴더 처리
    Returns: dict with per-image results and averages
    """
    results = {
        "per_image": {},
        "averages": {}
    }
    
    # 각 robustness 타입별 수집 리스트
    clean_ids = []
    adv_ids_by_type = {rt: [] for rt in ROBUSTNESS_TYPES}
    diffs_by_type = {rt: [] for rt in ROBUSTNESS_TYPES}
    relative_ids_by_type = {rt: [] for rt in ROBUSTNESS_TYPES}
    
    # samples_img_001 ~ samples_img_100 폴더 처리
    img_folders = sorted([d for d in os.listdir(result_dir) if d.startswith('samples_img_')])
    
    for img_folder in tqdm(img_folders, desc=f"Processing {os.path.basename(result_dir)}"):
        img_path = os.path.join(result_dir, img_folder)
        
        if not os.path.isdir(img_path):
            continue
        
        # x_src_clean 로드
        x_src_clean_pt = os.path.join(img_path, "x_src_clean.pt")
        x_src_clean_jpg = os.path.join(img_path, "x_src_clean.jpg")
        
        if os.path.exists(x_src_clean_pt):
            x_src_clean = load_tensor_from_pt(x_src_clean_pt)
        elif os.path.exists(x_src_clean_jpg):
            x_src_clean = load_image_as_tensor(x_src_clean_jpg)
        else:
            print(f"Warning: x_src_clean not found in {img_path}")
            continue
        
        # decoded_clean 로드
        decoded_clean_pt = os.path.join(img_path, "decoded_clean.pt")
        decoded_clean_jpg = os.path.join(img_path, "decoded_clean.jpg")
        
        if os.path.exists(decoded_clean_pt):
            decoded_clean = load_tensor_from_pt(decoded_clean_pt)
        elif os.path.exists(decoded_clean_jpg):
            decoded_clean = load_image_as_tensor(decoded_clean_jpg)
        else:
            print(f"Warning: decoded_clean not found in {img_path}")
            continue
        
        if x_src_clean is None or decoded_clean is None:
            print(f"Warning: Failed to load tensors in {img_path}")
            continue
        
        # clean_id 계산
        try:
            clean_id = compute_id_loss(x_src_clean, decoded_clean)
            clean_ids.append(clean_id)
        except Exception as e:
            print(f"Error computing clean ID loss for {img_folder}: {e}")
            continue
        
        # 이미지별 결과 초기화
        results["per_image"][img_folder] = {
            "clean_id": clean_id
        }
        
        # 각 robustness 타입별 ID loss 계산
        for rt in ROBUSTNESS_TYPES:
            decoded_adv_pt = os.path.join(img_path, f"decoded_adv_{rt}.pt")
            decoded_adv_jpg = os.path.join(img_path, f"decoded_adv_{rt}.jpg")
            
            if os.path.exists(decoded_adv_pt):
                decoded_adv = load_tensor_from_pt(decoded_adv_pt)
            elif os.path.exists(decoded_adv_jpg):
                decoded_adv = load_image_as_tensor(decoded_adv_jpg)
            else:
                continue
            
            if decoded_adv is None:
                continue
            
            try:
                adv_id = compute_id_loss(x_src_clean, decoded_adv)
                diff = clean_id - adv_id
                relative_id = max(0, 1 - clean_id / adv_id) if adv_id != 0 else 0
                
                results["per_image"][img_folder][f"adv_id_{rt}"] = adv_id
                results["per_image"][img_folder][f"diff_{rt}"] = diff
                results["per_image"][img_folder][f"relative_id_{rt}"] = relative_id
                
                adv_ids_by_type[rt].append(adv_id)
                diffs_by_type[rt].append(diff)
                relative_ids_by_type[rt].append(relative_id)
                
            except Exception as e:
                print(f"Error computing ID loss for {img_folder} ({rt}): {e}")
                continue
    
    # 평균 계산
    if len(clean_ids) > 0:
        results["averages"]["clean_id_mean"] = sum(clean_ids) / len(clean_ids)
        results["averages"]["total_images"] = len(clean_ids)
        
        for rt in ROBUSTNESS_TYPES:
            if len(adv_ids_by_type[rt]) > 0:
                results["averages"][f"adv_id_{rt}_mean"] = sum(adv_ids_by_type[rt]) / len(adv_ids_by_type[rt])
                results["averages"][f"diff_{rt}_mean"] = sum(diffs_by_type[rt]) / len(diffs_by_type[rt])
                results["averages"][f"relative_id_{rt}_mean"] = sum(relative_ids_by_type[rt]) / len(relative_ids_by_type[rt])
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute relative_id per preprocessing for a robustness result folder."
    )
    parser.add_argument("--result_dir", required=True,
                        help="Folder containing samples_img_* subdirectories "
                             "(produced by robustness.py / batch_robustness_test.py).")
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
    results = process_robustness_folder(str(result_dir))

    if not results.get("averages"):
        print(f"⚠️  No valid results for {result_dir.name}")
        sys.exit(1)

    output_path = output_dir / "eval_id.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {output_path}")
    print(f"  clean_id_mean: {results['averages']['clean_id_mean']:.6f}")
    print(f"  total_images:  {results['averages']['total_images']}")
    for rt in ROBUSTNESS_TYPES:
        key = f"relative_id_{rt}_mean"
        if key in results["averages"]:
            print(f"  [{rt}] relative_id: {results['averages'][key]:.4f}")


if __name__ == "__main__":
    main()
