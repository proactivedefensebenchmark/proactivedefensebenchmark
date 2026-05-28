import torch
import argparse
import os
import sys
import gc
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import importlib

# ==============================================================================
# Path Setup (Global)
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))      # attacks/
project_root = os.path.dirname(current_dir)                   # Project Root

# 1. Root 경로 확보 (utils를 위해)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Generator 경로들 미리 계산
DEEPFAKE_DIR = os.path.join(project_root, 'deepfake_generators')
PATH_MAP = {
    'simswap':   os.path.join(DEEPFAKE_DIR, 'SimSwap'),
    'psp_mix':   os.path.join(DEEPFAKE_DIR, 'pixel2style2pixel'),
    'diffae':    os.path.join(DEEPFAKE_DIR, 'diffae'),
    'styleclip': os.path.join(DEEPFAKE_DIR, 'styleclip'),
    'stargan':   os.path.join(DEEPFAKE_DIR, 'StarGAN'),
    'blendface': os.path.join(DEEPFAKE_DIR, 'BlendFace'),
    'diffface':  os.path.join(DEEPFAKE_DIR, 'DiffFace'),
    'diffswap':  os.path.join(DEEPFAKE_DIR, 'DiffSwap'),
}

# ==============================================================================
# 💀 Nuclear Option: Dependency Context Switcher
# ==============================================================================
def switch_model_context(model_name):

    target_path = PATH_MAP.get(model_name)
    
    # 1. 해당 모델 경로가 유효한지 확인
    if not target_path or not os.path.exists(target_path):
        return

    # 2. sys.path 정리 (다른 모델 경로가 맨 앞에 있으면 제거)
    #    우선순위: [Target Model Path] -> [Project Root] -> ...
    if target_path in sys.path:
        sys.path.remove(target_path)
    sys.path.insert(0, target_path)

    # 3. [핵심] 이름 충돌나는 모듈 강제 삭제
    #    SimSwap/pSp -> 'models'
    #    DiffAE/StarGAN -> 'model' (이게 충돌 원인이었음)
    #    DiffAE vs pSp -> 'configs' (pSp의 configs가 DiffAE의 configs를 덮어씀)
    keys_to_purge = [
        'models',           # SimSwap, pSp 등
        'model',            # ★ DiffAE vs StarGAN 충돌 해결을 위해 추가
        'options', 
        'criteria', 
        'models.psp', 
        'models.fs_networks',
        'solver',           # StarGAN 등에서 사용하는 Solver
        'data',             # 일반적인 데이터로더 모듈 이름
        'utils',            # 프로젝트 루트 utils와 꼬일 수 있음
        'configs',          # ★ DiffAE vs pSp configs 충돌 해결
        'diffae',           # DiffAE 모듈 자체도 정리
        'styleclip',        # StyleCLIP 모듈
        'stylegan2',        # StyleGAN2 모듈
    ]
    
    for key in list(sys.modules.keys()):
        for keyword in keys_to_purge:
            # 해당 키워드와 정확히 일치하거나, 해당 패키지의 하위 모듈인 경우 삭제
            if key == keyword or key.startswith(keyword + '.'):
                del sys.modules[key]
                # print(f"   [Context Switch] Purged module: {key}")
                break
    
    # 4. 다른 모델 경로들도 sys.path에서 제거 (우선순위 충돌 방지)
    for other_model, other_path in PATH_MAP.items():
        if other_model != model_name and other_path in sys.path:
            sys.path.remove(other_path)

# Import utils
try:
    from utils.utils import WRAPPER_REGISTRY, load_wrapper, load_image
except ImportError:
    # utils가 없으면 심각한 문제
    sys.path.append(project_root)
    from utils.utils import WRAPPER_REGISTRY, load_wrapper, load_image

# ==============================================================================
# Helper Functions
# ==============================================================================
def save_tensor_as_image(tensor, path):
    if tensor.ndim == 4: tensor = tensor[0]
    tensor = tensor.detach().cpu()
    tensor = torch.clamp((tensor + 1.0) / 2.0, 0, 1)
    img = (tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(img).save(path)

def get_target_image_for_wrapper(wrapper_name, device):
    if wrapper_name not in WRAPPER_REGISTRY: return None
    config = WRAPPER_REGISTRY[wrapper_name]
    if not config.get('requires_target', False): return None
    target_path = config.get('default_target_path')
    if target_path and os.path.exists(target_path):
        return load_image(target_path, device)[0]
    return None

def find_result_folder(base_path, model_name):
    search_pattern = os.path.join(base_path, f"result_{model_name}*")
    candidates = glob.glob(search_pattern)
    return candidates[0] if candidates else None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_path", type=str, required=True)
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--wrappers", nargs="+",
                        default=["blendface", "stargan", "simswap", "psp_mix",
                                 "diffae", "styleclip", "diffface", "diffswap"])
    parser.add_argument("--target_only", type=str, default=None, help="Run only this target model (source will be all other wrappers)")
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_image", type=str, default=None, help="Override default_target_path for all wrappers that require a target image")
    return parser.parse_args()

def main(args):
    device = args.device
    print(f"Device: {device}")
    
    if args.output_base is None:
        args.output_base = os.path.join(args.default_path, "black_box_results")
    os.makedirs(args.output_base, exist_ok=True)
    
    # Sources must have a result_<wrapper>_*/ folder (where x_adv.pt lives).
    # Targets only need to be a valid wrapper that can be loaded as a model;
    # their decoded_src baseline is taken from their own WB folder if
    # available, otherwise computed on-the-fly from the source's x_src.
    valid_wrappers = [w for w in args.wrappers if w in WRAPPER_REGISTRY]
    source_folders = {}
    for w in valid_wrappers:
        folder = find_result_folder(args.default_path, w)
        if folder: source_folders[w] = folder

    if not source_folders:
        print(f"❌ No result_*/ folders found under {args.default_path}. "
              f"Run white_box_attack.py for at least one wrapper first.")
        return

    print(f"Sources (have WB results): {sorted(source_folders.keys())}")

    if args.target_only:
        if args.target_only not in valid_wrappers:
            print(f"❌ --target_only '{args.target_only}' not in --wrappers list.")
            return
        target_wrappers = [args.target_only]
    else:
        target_wrappers = list(valid_wrappers)
    print(f"Targets (will be loaded):  {target_wrappers}")

    # ==========================================================================
    # Main Loop
    # ==========================================================================
    for target_name in target_wrappers:
        print(f"\n{'='*60}")
        print(f"🎯 Target Model (Victim): {target_name.upper()}")
        print(f"{'='*60}")
        
        # ★★★ [Context Switching] 모델 로드 전 경로 납치 및 모듈 세탁 ★★★
        switch_model_context(target_name)
        
        try:
            # 래퍼 로드 (이제 안전함)
            target_wrapper, target_config = load_wrapper(target_name, device=device)
        except Exception as e:
            print(f"❌ Failed to load target wrapper {target_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Target Reference 준비
        target_ref_img = None
        if target_config['requires_target']:
            if args.target_image and os.path.exists(args.target_image):
                target_ref_img = load_image(args.target_image, device)[0]
            else:
                target_ref_img = get_target_image_for_wrapper(target_name, device)
        target_ref_attr = target_config.get('default_attr') if target_config['requires_attr'] else None

        # Where to source the target's clean baseline (decoded_src).
        # If target also has a WB folder we copy from there (fast, exact match
        # with what white_box_attack produced). Otherwise we compute it
        # on-the-fly using target_wrapper on the source's x_src.
        tgt_wb_base = source_folders.get(target_name)

        def _run_target(x: "torch.Tensor") -> "torch.Tensor":
            if target_config['requires_target'] and target_ref_img is not None:
                return target_wrapper(x, ref=target_ref_img, preprocess=False)
            if target_config['requires_attr'] and target_ref_attr is not None:
                return target_wrapper(x, target_attr=target_ref_attr)
            return target_wrapper(x)

        # Source Model Loop
        for source_name in source_folders:
            if source_name == target_name: continue

            dst_folder = os.path.join(args.output_base, f"black_box_{source_name}_{target_name}")
            os.makedirs(dst_folder, exist_ok=True)

            src_base = source_folders[source_name]

            for i in tqdm(range(1, args.n_images + 1), desc=f"{source_name}->{target_name}", leave=False):
                img_id = f"img_{i:03d}"
                src_dir = os.path.join(src_base, img_id)
                out_dir = os.path.join(dst_folder, img_id)

                x_adv_path = os.path.join(src_dir, "x_adv.pt")
                if not os.path.exists(x_adv_path): continue

                os.makedirs(out_dir, exist_ok=True)

                try:
                    # 1. Source side: copy x_adv (and visualisation) over.
                    shutil.copy(x_adv_path, os.path.join(out_dir, "x_adv.pt"))
                    if os.path.exists(os.path.join(src_dir, "x_adv.jpg")):
                        shutil.copy(os.path.join(src_dir, "x_adv.jpg"), os.path.join(out_dir, "x_adv.jpg"))

                    # 2. Target's clean baseline (decoded_src + x_src).
                    if tgt_wb_base is not None:
                        tgt_dir = os.path.join(tgt_wb_base, img_id)
                        for f in ["x_src.jpg", "decoded_src.jpg", "decoded_src.pt"]:
                            tgt_f = os.path.join(tgt_dir, f)
                            if os.path.exists(tgt_f):
                                shutil.copy(tgt_f, os.path.join(out_dir, f))
                    else:
                        # Compute decoded_src = target_wrapper(source's x_src).
                        # x_src is shared input across wrappers; reuse from src.
                        # batch_*_attack.py only saves x_src.jpg (no .pt), so
                        # prefer .pt if present, otherwise load_image from .jpg.
                        for f in ["x_src.jpg", "x_src.pt"]:
                            src_f = os.path.join(src_dir, f)
                            if os.path.exists(src_f):
                                shutil.copy(src_f, os.path.join(out_dir, f))

                        x_src = None
                        x_src_pt = os.path.join(src_dir, "x_src.pt")
                        x_src_jpg = os.path.join(src_dir, "x_src.jpg")
                        if os.path.exists(x_src_pt):
                            x_src = torch.load(x_src_pt, map_location=device)
                        elif os.path.exists(x_src_jpg):
                            x_src, _ = load_image(x_src_jpg, device)

                        if x_src is not None:
                            with torch.no_grad():
                                decoded_src = _run_target(x_src)
                            save_tensor_as_image(decoded_src, os.path.join(out_dir, "decoded_src.jpg"))
                            torch.save(decoded_src, os.path.join(out_dir, "decoded_src.pt"))

                    # 3. Transfer: target_wrapper(source's x_adv).
                    x_adv = torch.load(x_adv_path, map_location=device)
                    with torch.no_grad():
                        decoded_adv = _run_target(x_adv)

                    save_tensor_as_image(decoded_adv, os.path.join(out_dir, "decoded_adv.jpg"))
                    torch.save(decoded_adv, os.path.join(out_dir, "decoded_adv.pt"))

                except Exception as e:
                    pass

        # Cleanup
        del target_wrapper
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print("✅ Experiment All Done.")

if __name__ == "__main__":
    main(parse_arguments())