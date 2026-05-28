"""
Robustness Test for Adversarial Perturbations
Tests if perturbation effects survive preprocessing (JPEG, blur, noise, etc.)
"""
import torch
import argparse
import os
import sys
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.utils import (
    WRAPPER_REGISTRY,
    load_wrapper,
    load_image,
)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed fixed to {seed} for reproducibility")


# Set seed at module level for global reproducibility
set_seed(42)


def save_tensor_as_image(tensor, path):
    """Save tensor as image file"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    tensor = torch.clamp((tensor + 1.0) / 2.0, 0, 1)
    img = (tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(img).save(path)


def jpeg_compression(tensor, quality=75):
    """Apply JPEG compression"""
    from io import BytesIO
    
    # Convert to PIL
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor_01 = torch.clamp((tensor + 1.0) / 2.0, 0, 1)
    img_np = (tensor_01 * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    # JPEG compress
    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    
    # Back to tensor
    img_np = np.array(img_compressed).astype(np.float32) / 255.0
    tensor_compressed = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    tensor_compressed = tensor_compressed * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    return tensor_compressed.to(tensor.device)


def gaussian_blur(tensor, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur"""
    from torchvision.transforms import GaussianBlur
    
    # [-1, 1] -> [0, 1]
    tensor_01 = (tensor + 1.0) / 2.0
    
    # Apply blur
    blur_fn = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred = blur_fn(tensor_01)
    
    # [0, 1] -> [-1, 1]
    return (blurred * 2.0 - 1.0).detach()


def gaussian_noise(tensor, std=0.01):
    """Add Gaussian noise"""
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, -1.0, 1.0).detach()


def salt_and_pepper_noise(tensor, prob=0.01):
    """Add salt and pepper noise
    Args:
        tensor: [B, C, H, W] tensor in [-1, 1] range
        prob: probability of noise (half salt, half pepper)
    Returns:
        Noisy tensor in [-1, 1] range
    """
    noisy = tensor.clone().detach()
    # Salt (white pixels -> 1.0 in [-1,1] range)
    salt_mask = torch.rand_like(tensor) < (prob / 2)
    noisy[salt_mask] = 1.0
    # Pepper (black pixels -> -1.0 in [-1,1] range)
    pepper_mask = torch.rand_like(tensor) < (prob / 2)
    noisy[pepper_mask] = -1.0
    return noisy.detach()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test robustness of adversarial perturbations"
    )
    
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing adversarial results (e.g., batch_results_df_rap/result_styleclip_df_rap)"
    )
    
    parser.add_argument(
        "--wrapper",
        type=str,
        required=True,
        help="Wrapper name (simswap, diffae, styleclip, etc.)"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to original dataset (optional, for loading clean images if x_src.jpg not available)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for robustness test results"
    )
    
    parser.add_argument(
        "--n_images",
        type=int,
        default=100,
        help="Number of images to test"
    )
    
    parser.add_argument(
        "--preprocessing",
        nargs="+",
        default=["jpeg75", "jpeg90", "blur3", "blur5", "noise001", "noise005", "salt_pepper001", "salt_pepper010"],
        help="Preprocessing methods to test (jpeg75, jpeg90, blur3, blur5, noise001, noise005, salt_pepper001, salt_pepper010)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


def apply_preprocessing(tensor, method):
    """Apply preprocessing based on method string"""
    if method.startswith("jpeg"):
        quality = int(method[4:])
        return jpeg_compression(tensor, quality=quality).detach()
    elif method.startswith("blur"):
        kernel_size = int(method[4:])
        return gaussian_blur(tensor, kernel_size=kernel_size, sigma=1.0).detach()
    elif method.startswith("noise"):
        std = float(method[5:]) / 1000.0  # noise001 -> 0.001
        return gaussian_noise(tensor, std=std).detach()
    elif method.startswith("salt_pepper"):
        prob = float(method[11:]) / 1000.0  # salt_pepper001 -> 0.001, salt_pepper010 -> 0.01
        return salt_and_pepper_noise(tensor, prob=prob).detach()
    else:
        return tensor.detach()


def main(args):
    """Main execution function"""
    # Re-confirm seed for reproducibility
    set_seed(42)
    
    device = args.device
    print(f"Device: {device}")
    
    # Set default output_dir
    if args.output_dir is None:
        result_basename = os.path.basename(args.result_dir.rstrip('/'))
        args.output_dir = os.path.join(project_root, "robustness_results", result_basename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load wrapper
    wrapper_name = args.wrapper
    if wrapper_name not in WRAPPER_REGISTRY:
        print(f"❌ Wrapper '{wrapper_name}' not found")
        return
    
    print(f"\nLoading wrapper: {wrapper_name}")
    try:
        wrapper, config = load_wrapper(wrapper_name, device=device)
    except Exception as e:
        print(f"❌ Failed to load wrapper: {e}")
        return
    
    # Get target image if needed
    target_tensor = None
    if config['requires_target']:
        # Resolve the target exactly like the attack did (batch_*_attack.py):
        # honor DEEPFAKE_TARGET_IMAGE first, fall back to the registry default.
        # Otherwise robustness re-renders the swap toward a different identity
        # than x_adv was optimized against -> meaningless ROB numbers.
        target_path = os.environ.get('DEEPFAKE_TARGET_IMAGE',
                                     config.get('default_target_path'))
        if target_path and os.path.exists(target_path):
            target_tensor, _ = load_image(target_path, device)
            print(f"   Target image loaded: {target_path}")
            if hasattr(wrapper, 'set_target'):
                wrapper.set_target(target_tensor)
    
    # Determine ref
    if config['requires_target']:
        wrapper_ref = target_tensor
    elif config['requires_attr']:
        wrapper_ref = config['default_attr']
    else:
        wrapper_ref = None
    
    # Find all x_adv.pt files
    img_folders = []
    for folder_name in sorted(os.listdir(args.result_dir)):
        if folder_name.startswith("img_"):
            folder_path = os.path.join(args.result_dir, folder_name)
            x_adv_path = os.path.join(folder_path, "x_adv.pt")
            if os.path.exists(x_adv_path):
                img_folders.append((folder_name, folder_path))
    
    img_folders = img_folders[:args.n_images]
    print(f"Found {len(img_folders)} adversarial images to test")
    
    # Image-only output: this script generates samples_img_*/decoded_*.jpg/.pt
    # for downstream evaluation. Metric / CSV / heatmap aggregation lives in
    # robustness_evaluation.py (which dispatches to evaluation/evaluate_*.py).
    for folder_name, folder_path in tqdm(img_folders, desc="Generating samples"):
        img_id = folder_name.replace("img_", "")
        
        try:
            # Load x_adv
            x_adv_path = os.path.join(folder_path, "x_adv.pt")
            x_adv = torch.load(x_adv_path, map_location=device)
            if x_adv.ndim == 3:
                x_adv = x_adv.unsqueeze(0)
            
            # Load x_src (clean image)
            x_src_jpg_path = os.path.join(folder_path, "x_src.jpg")
            if os.path.exists(x_src_jpg_path):
                x_src, _ = load_image(x_src_jpg_path, device)
            else:
                print(f"\n⚠️  x_src.jpg not found for {folder_name}, skipping...")
                continue
            
            # Load or generate decoded outputs
            decoded_adv_path = os.path.join(folder_path, "decoded_adv.pt")
            decoded_src_path = os.path.join(folder_path, "decoded_src.pt")
            
            if os.path.exists(decoded_adv_path):
                decoded_adv_original = torch.load(decoded_adv_path, map_location=device).detach()
            else:
                # Generate decoded_adv
                with torch.no_grad():
                    if config['requires_target']:
                        decoded_adv_original = wrapper(x_adv, ref=target_tensor, preprocess=False).detach()
                    elif config['requires_attr']:
                        decoded_adv_original = wrapper(x_adv, target_attr=wrapper_ref).detach()
                    else:
                        decoded_adv_original = wrapper(x_adv).detach()
            
            if os.path.exists(decoded_src_path):
                decoded_src_original = torch.load(decoded_src_path, map_location=device).detach()
            else:
                # Generate decoded_src (clean output)
                with torch.no_grad():
                    if config['requires_target']:
                        decoded_src_original = wrapper(x_src, ref=target_tensor, preprocess=False).detach()
                    elif config['requires_attr']:
                        decoded_src_original = wrapper(x_src, target_attr=wrapper_ref).detach()
                    else:
                        decoded_src_original = wrapper(x_src).detach()
            
            # Generate outputs for each preprocessing method.
            for preproc_method in args.preprocessing:
                # Apply preprocessing to x_adv
                x_adv_preprocessed = apply_preprocessing(x_adv, preproc_method)
                
                # Generate output with preprocessed adversarial input
                with torch.no_grad():
                    if config['requires_target']:
                        decoded_adv_preprocessed = wrapper(x_adv_preprocessed, ref=target_tensor, preprocess=False).detach()
                    elif config['requires_attr']:
                        decoded_adv_preprocessed = wrapper(x_adv_preprocessed, target_attr=wrapper_ref).detach()
                    else:
                        decoded_adv_preprocessed = wrapper(x_adv_preprocessed).detach()
                
                # Save sample images (first 100)
                if int(img_id) <= 100:
                    sample_dir = os.path.join(args.output_dir, f"samples_img_{img_id}")
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Save once per image (not per preprocessing)
                    if preproc_method == args.preprocessing[0]:
                        save_tensor_as_image(x_src, os.path.join(sample_dir, f"x_src_clean.jpg"))
                        torch.save(x_src.detach().cpu(), os.path.join(sample_dir, f"x_src_clean.pt"))
                        save_tensor_as_image(x_adv, os.path.join(sample_dir, f"x_adv.jpg"))
                        torch.save(x_adv.detach().cpu(), os.path.join(sample_dir, f"x_adv.pt"))
                        save_tensor_as_image(decoded_src_original, os.path.join(sample_dir, f"decoded_clean.jpg"))
                        torch.save(decoded_src_original.detach().cpu(), os.path.join(sample_dir, f"decoded_clean.pt"))
                        save_tensor_as_image(decoded_adv_original, os.path.join(sample_dir, f"decoded_adv_original.jpg"))
                        torch.save(decoded_adv_original.detach().cpu(), os.path.join(sample_dir, f"decoded_adv_original.pt"))
                    
                    save_tensor_as_image(x_adv_preprocessed, os.path.join(sample_dir, f"x_adv_{preproc_method}.jpg"))
                    torch.save(x_adv_preprocessed.detach().cpu(), os.path.join(sample_dir, f"x_adv_{preproc_method}.pt"))
                    save_tensor_as_image(decoded_adv_preprocessed, os.path.join(sample_dir, f"decoded_adv_{preproc_method}.jpg"))
                    torch.save(decoded_adv_preprocessed.detach().cpu(), os.path.join(sample_dir, f"decoded_adv_{preproc_method}.pt"))
        
        except Exception as e:
            print(f"\n⚠️  Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Robustness samples generated")
    print(f"   Output: {args.output_dir}/samples_img_*")
    print(f"   Run robustness_evaluation.py on this directory for metrics + ROB.")



if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
