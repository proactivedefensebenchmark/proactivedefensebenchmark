import os
import torch
import torch.nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import warnings

# Import metric functions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation_metrics import (
    mae, mse, rmse, ssim, psnr, id_loss, 
    vgg_lpips, alex_lpips, brisque_metric, DEVICE
)

warnings.filterwarnings("ignore")

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

# def load_image_as_tensor(image_path, size=256):
#     """Load image and convert to tensor [-1, 1]"""
#     try:
#         img = Image.open(image_path).convert('RGB')
#         transform = T.Compose([
#             T.Resize((size, size)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         tensor = transform(img).unsqueeze(0).to(DEVICE)
        
#         # [수정됨] 여기도 .detach() 추가
#         return tensor.detach()
#     except Exception as e:
#         print(f"Error loading image {image_path}: {e}")
#         return None
        
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
        
        # Ensure tensor is on device
        tensor = tensor.to(DEVICE)
        
        # Add batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.detach()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
# def load_tensor_from_pt(file_path):
#     """Load tensor from .pt file"""
#     try:
#         tensor = torch.load(file_path, map_location=DEVICE)
#         if isinstance(tensor, dict):
#             for key in ['tensor', 'image', 'data', 'img']:
#                 if key in tensor:
#                     tensor = tensor[key]
#                     break
#             if isinstance(tensor, dict):
#                 tensor = list(tensor.values())[0]
        
#         # Ensure tensor is on device
#         tensor = tensor.to(DEVICE)
        
#         # Add batch dimension if needed
#         if tensor.dim() == 3:
#             tensor = tensor.unsqueeze(0)
        
#         # [수정됨] 족쇄 끊기: 미분 추적 비활성화
#         return tensor.detach() 
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None

def compute_x_metrics(x_src_tensor, x_adv_tensor):
    """
    Compute metrics for x_src vs x_adv (only SSIM, PSNR)
    """
    results = {}
    
    try:
        results['ssim'] = float(ssim(x_src_tensor, x_adv_tensor))
        results['psnr'] = float(psnr(x_src_tensor, x_adv_tensor))
    except Exception as e:
        print(f"Error computing x metrics: {e}")
        return None
    
    return results

def compute_decoded_metrics(decoded_src_tensor, decoded_adv_tensor):
    """
    Compute metrics for decoded_src vs decoded_adv
    (SSIM, PSNR, L2, L1, LPIPS, BRISQUE, SR, ID loss)
    """
    results = {}
    
    try:
        # SSIM, PSNR
        results['ssim'] = float(ssim(decoded_src_tensor, decoded_adv_tensor))
        results['psnr'] = float(psnr(decoded_src_tensor, decoded_adv_tensor))
        
        # L2 (MSE), L1 (MAE)
        results['l2_mse'] = float(mse(decoded_src_tensor, decoded_adv_tensor).item())
        results['l1_mae'] = float(mae(decoded_src_tensor, decoded_adv_tensor).item())
        
        # LPIPS
        results['lpips_alex'] = float(alex_lpips(decoded_src_tensor, decoded_adv_tensor).item())
        results['lpips_vgg'] = float(vgg_lpips(decoded_src_tensor, decoded_adv_tensor).item())
        
        # BRISQUE
        results['brisque_src'] = float(brisque_metric(decoded_src_tensor))
        results['brisque_adv'] = float(brisque_metric(decoded_adv_tensor))
        
        # SR (Success Rate): 1 if MSE >= 0.05, else 0
        results['sr'] = 1 if results['l2_mse'] >= 0.05 else 0
        
        # ID loss
        results['id_loss'] = float(id_loss(decoded_src_tensor, decoded_adv_tensor))
        
    except Exception as e:
        print(f"Error computing decoded metrics: {e}")
        return None
    
    return results

def evaluate_disruption_batch(x_src_dir, x_adv_dir, decoded_src_dir, decoded_adv_dir, output_json, image_size=256, subdir_mode=False, source_image_dir=None):
    """
    Batch evaluate disruption results
    
    Args:
        x_src_dir: Directory containing source images (jpg, png, etc.) or parent dir with subdirs
        x_adv_dir: Directory containing adversarial .pt files (or None if subdir_mode)
        decoded_src_dir: Directory containing decoded source .pt files (or None if subdir_mode)
        decoded_adv_dir: Directory containing decoded adversarial .pt files (or None if subdir_mode)
        output_json: Path to save results
        image_size: Image size for loading
        subdir_mode: If True, expect img_*/x_src.jpg structure
        source_image_dir: Optional separate directory for source images (if different from x_src_dir)
    """
    x_src_dir = Path(x_src_dir)
    
    all_results = []
    x_metric_sums = {}
    x_metric_counts = {}
    decoded_metric_sums = {}
    decoded_metric_counts = {}
    
    if subdir_mode:
        # Process img_* subdirectories
        img_dirs = sorted([d for d in x_src_dir.iterdir() if d.is_dir() and d.name.startswith('img_')])
        
        if len(img_dirs) == 0:
            print(f"No img_* subdirectories found in {x_src_dir}")
            return
        
        print(f"Found {len(img_dirs)} image subdirectories to evaluate")
        
        # If source_image_dir is provided, use it for x_src files
        if source_image_dir:
            source_image_dir = Path(source_image_dir)
            print(f"Loading source images from: {source_image_dir}")
        
        for img_dir in tqdm(img_dirs, desc="Evaluating disruption results"):
            base_name = img_dir.name
            
            # File paths within subdirectory
            if source_image_dir:
                # Look for source image in separate directory
                # Try different extensions
                x_src_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_file = source_image_dir / f"{base_name}{ext}"
                    if potential_file.exists():
                        x_src_file = potential_file
                        break
                if x_src_file is None:
                    x_src_file = img_dir / "x_src.jpg"  # fallback to subdirectory
            else:
                x_src_file = img_dir / "x_src.jpg"
            
            x_adv_file = img_dir / "x_adv.pt"
            
            # Find decoded_src and decoded_adv files (support both exact and pattern matching)
            # Pattern: decoded_src.pt OR decoded_src_*.pt
            decoded_src_file = None
            decoded_adv_file = None
            
            # First try exact match
            if (img_dir / "decoded_src.pt").exists():
                decoded_src_file = img_dir / "decoded_src.pt"
            else:
                # Try pattern: decoded_src_*.pt
                candidates = list(img_dir.glob("decoded_src_*.pt"))
                if candidates:
                    decoded_src_file = candidates[0]
            
            if (img_dir / "decoded_adv.pt").exists():
                decoded_adv_file = img_dir / "decoded_adv.pt"
            else:
                # Try pattern: decoded_adv_*.pt
                candidates = list(img_dir.glob("decoded_adv_*.pt"))
                if candidates:
                    decoded_adv_file = candidates[0]
            
            result = {
                'filename': base_name,
                'x_metrics': None,
                'decoded_metrics': None
            }
            
            # Compute x metrics - support both image and pt files for x_src
            x_src_tensor = None
            x_adv_tensor = None
            
            # Try to load x_src (pt file first, then image)
            x_src_pt = img_dir / "x_src.pt"
            if x_src_pt.exists():
                x_src_tensor = load_tensor_from_pt(x_src_pt)
            elif x_src_file.exists():
                x_src_tensor = load_image_as_tensor(x_src_file, size=image_size)
            else:
                # Try png
                x_src_png = img_dir / "x_src.png"
                if x_src_png.exists():
                    x_src_tensor = load_image_as_tensor(x_src_png, size=image_size)
            
            # Load x_adv
            if x_adv_file.exists():
                x_adv_tensor = load_tensor_from_pt(x_adv_file)
            
            if x_src_tensor is not None and x_adv_tensor is not None:
                x_metrics = compute_x_metrics(x_src_tensor, x_adv_tensor)
                if x_metrics:
                    result['x_metrics'] = x_metrics
                    for key, value in x_metrics.items():
                        x_metric_sums[key] = x_metric_sums.get(key, 0) + value
                        x_metric_counts[key] = x_metric_counts.get(key, 0) + 1
            
            # Compute decoded metrics
            if decoded_src_file is not None and decoded_adv_file is not None:
                decoded_src_tensor = load_tensor_from_pt(decoded_src_file)
                decoded_adv_tensor = load_tensor_from_pt(decoded_adv_file)
                
                if decoded_src_tensor is not None and decoded_adv_tensor is not None:
                    decoded_metrics = compute_decoded_metrics(decoded_src_tensor, decoded_adv_tensor)
                    if decoded_metrics:
                        result['decoded_metrics'] = decoded_metrics
                        for key, value in decoded_metrics.items():
                            decoded_metric_sums[key] = decoded_metric_sums.get(key, 0) + value
                            decoded_metric_counts[key] = decoded_metric_counts.get(key, 0) + 1
            
            if result['x_metrics'] or result['decoded_metrics']:
                all_results.append(result)
    
    else:
        # Original flat directory mode
        x_adv_dir = Path(x_adv_dir)
        decoded_src_dir = Path(decoded_src_dir)
        decoded_adv_dir = Path(decoded_adv_dir)
    
        # Get all source images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        source_files = []
        for ext in image_extensions:
            source_files.extend(list(x_src_dir.glob(ext)))
        source_files = sorted(source_files)
        
        if len(source_files) == 0:
            print(f"No image files found in {x_src_dir}")
            return
        
        print(f"Found {len(source_files)} source images to evaluate")
    
        for source_file in tqdm(source_files, desc="Evaluating disruption results"):
            base_name = source_file.stem
            
            # Find corresponding files
            x_adv_file = x_adv_dir / f"{base_name}.pt"
            decoded_src_file = decoded_src_dir / f"{base_name}.pt"
            decoded_adv_file = decoded_adv_dir / f"{base_name}.pt"
            
            result = {
                'filename': base_name,
                'x_metrics': None,
                'decoded_metrics': None
            }
            
            # Compute x metrics (x_src vs x_adv)
            if x_adv_file.exists():
                x_src_tensor = load_image_as_tensor(source_file, size=image_size)
                x_adv_tensor = load_tensor_from_pt(x_adv_file)
                
                if x_src_tensor is not None and x_adv_tensor is not None:
                    x_metrics = compute_x_metrics(x_src_tensor, x_adv_tensor)
                    if x_metrics:
                        result['x_metrics'] = x_metrics
                        for key, value in x_metrics.items():
                            x_metric_sums[key] = x_metric_sums.get(key, 0) + value
                            x_metric_counts[key] = x_metric_counts.get(key, 0) + 1
            else:
                print(f"Warning: No x_adv file for {base_name}")
            
            # Compute decoded metrics (decoded_src vs decoded_adv)
            if decoded_src_file.exists() and decoded_adv_file.exists():
                decoded_src_tensor = load_tensor_from_pt(decoded_src_file)
                decoded_adv_tensor = load_tensor_from_pt(decoded_adv_file)
                
                if decoded_src_tensor is not None and decoded_adv_tensor is not None:
                    decoded_metrics = compute_decoded_metrics(decoded_src_tensor, decoded_adv_tensor)
                    if decoded_metrics:
                        result['decoded_metrics'] = decoded_metrics
                        for key, value in decoded_metrics.items():
                            decoded_metric_sums[key] = decoded_metric_sums.get(key, 0) + value
                            decoded_metric_counts[key] = decoded_metric_counts.get(key, 0) + 1
            else:
                if not decoded_src_file.exists():
                    print(f"Warning: No decoded_src file for {base_name}")
                if not decoded_adv_file.exists():
                    print(f"Warning: No decoded_adv file for {base_name}")
            
            if result['x_metrics'] or result['decoded_metrics']:
                all_results.append(result)
    
    # Calculate averages
    x_metric_averages = {}
    for key in x_metric_sums:
        x_metric_averages[key] = x_metric_sums[key] / x_metric_counts[key]
    
    decoded_metric_averages = {}
    for key in decoded_metric_sums:
        decoded_metric_averages[key] = decoded_metric_sums[key] / decoded_metric_counts[key]
    
    # Save results
    output_data = {
        'x_src_directory': str(x_src_dir),
        'x_adv_directory': str(x_adv_dir) if x_adv_dir else 'N/A (subdir mode)',
        'decoded_src_directory': str(decoded_src_dir) if decoded_src_dir else 'N/A (subdir mode)',
        'decoded_adv_directory': str(decoded_adv_dir) if decoded_adv_dir else 'N/A (subdir mode)',
        'total_evaluated': len(all_results),
        'x_metric_averages': x_metric_averages,
        'decoded_metric_averages': decoded_metric_averages,
        'individual_results': all_results
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_json}")
    print("\n=== X Metrics (x_src vs x_adv) ===")
    for key, value in sorted(x_metric_averages.items()):
        print(f"{key}: {value:.6f}")
    
    print("\n=== Decoded Metrics (decoded_src vs decoded_adv) ===")
    for key, value in sorted(decoded_metric_averages.items()):
        print(f"{key}: {value:.6f}")
    
    return output_data

def evaluate_single_disruption(x_src_path, x_adv_path, decoded_src_path, decoded_adv_path, image_size=256):
    """
    Evaluate a single disruption result
    
    Args:
        x_src_path: Path to source image
        x_adv_path: Path to adversarial .pt file
        decoded_src_path: Path to decoded source .pt file
        decoded_adv_path: Path to decoded adversarial .pt file
        image_size: Image size for loading
    """
    print("=== Evaluating Single Disruption Result ===\n")
    
    # Load tensors
    x_src_tensor = load_image_as_tensor(x_src_path, size=image_size)
    x_adv_tensor = load_tensor_from_pt(x_adv_path)
    decoded_src_tensor = load_tensor_from_pt(decoded_src_path)
    decoded_adv_tensor = load_tensor_from_pt(decoded_adv_path)
    
    results = {}
    
    # Compute x metrics
    if x_src_tensor is not None and x_adv_tensor is not None:
        x_metrics = compute_x_metrics(x_src_tensor, x_adv_tensor)
        if x_metrics:
            results['x_metrics'] = x_metrics
            print("--- X Metrics (x_src vs x_adv) ---")
            for key, value in sorted(x_metrics.items()):
                print(f"{key}: {value:.6f}")
    
    # Compute decoded metrics
    if decoded_src_tensor is not None and decoded_adv_tensor is not None:
        decoded_metrics = compute_decoded_metrics(decoded_src_tensor, decoded_adv_tensor)
        if decoded_metrics:
            results['decoded_metrics'] = decoded_metrics
            print("\n--- Decoded Metrics (decoded_src vs decoded_adv) ---")
            for key, value in sorted(decoded_metrics.items()):
                print(f"{key}: {value:.6f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate disruption results')
    parser.add_argument('--x_src_dir', type=str, help='Directory containing source images')
    parser.add_argument('--x_adv_dir', type=str, help='Directory containing adversarial .pt files')
    parser.add_argument('--decoded_src_dir', type=str, help='Directory containing decoded source .pt files')
    parser.add_argument('--decoded_adv_dir', type=str, help='Directory containing decoded adversarial .pt files')
    parser.add_argument('--x_src_file', type=str, help='Single source image file')
    parser.add_argument('--x_adv_file', type=str, help='Single adversarial .pt file')
    parser.add_argument('--decoded_src_file', type=str, help='Single decoded source .pt file')
    parser.add_argument('--decoded_adv_file', type=str, help='Single decoded adversarial .pt file')
    parser.add_argument('--batch_subdir_mode', action='store_true', 
                       help='Enable subdirectory mode (expect img_*/x_src.jpg structure)')
    parser.add_argument('--source_image_dir', type=str, 
                       help='Optional separate directory for source images')
    parser.add_argument('--output', type=str, default='disruption_evaluation_results.json', 
                       help='Output JSON file for results')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for loading')
    
    args = parser.parse_args()
    
    if args.batch_subdir_mode and args.x_src_dir:
        # Subdirectory batch mode
        evaluate_disruption_batch(
            args.x_src_dir,
            None,
            None,
            None,
            args.output,
            args.image_size,
            subdir_mode=True,
            source_image_dir=args.source_image_dir
        )
    elif args.x_src_dir and args.x_adv_dir and args.decoded_src_dir and args.decoded_adv_dir:
        # Batch evaluation
        evaluate_disruption_batch(
            args.x_src_dir,
            args.x_adv_dir,
            args.decoded_src_dir,
            args.decoded_adv_dir,
            args.output,
            args.image_size
        )
    elif args.x_src_file and args.x_adv_file and args.decoded_src_file and args.decoded_adv_file:
        # Single file evaluation
        evaluate_single_disruption(
            args.x_src_file,
            args.x_adv_file,
            args.decoded_src_file,
            args.decoded_adv_file,
            args.image_size
        )
    else:
        print("For batch evaluation, provide:")
        print("  --x_src_dir --x_adv_dir --decoded_src_dir --decoded_adv_dir")
        print("\nFor single file evaluation, provide:")
        print("  --x_src_file --x_adv_file --decoded_src_file --decoded_adv_file")
