import os
import torch
import torch.nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
import lpips
import piq
import warnings
from pathlib import Path
import json
from tqdm import tqdm

# Import metric functions from the original file
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluation_metrics import (
    mae, mse, rmse, ssim, psnr, id_loss, 
    vgg_lpips, alex_lpips, brisque_metric, 
    fsim, leat_cdsr, l2_mask, sr_mask_single, 
    vgg_loss, scol_pds, DEVICE
)

warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", module="torchvision.models._utils")

def load_tensor_from_pt(file_path):
    """Load tensor from .pt file"""
    try:
        tensor = torch.load(file_path, map_location=DEVICE)
        if isinstance(tensor, dict):
            # If it's a dictionary, try to get the tensor from common keys
            for key in ['tensor', 'image', 'data', 'img']:
                if key in tensor:
                    tensor = tensor[key]
                    break
            # If still a dict, get the first value
            if isinstance(tensor, dict):
                tensor = list(tensor.values())[0]
        return tensor
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def ensure_tensor_format(tensor):
    """Ensure tensor is in the correct format [B, C, H, W] with values in [-1, 1]"""
    if tensor is None:
        return None
    
    # Move to device
    tensor = tensor.to(DEVICE)
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Check value range and normalize if needed
    if tensor.min() >= 0 and tensor.max() <= 1:
        # Convert [0, 1] to [-1, 1]
        tensor = tensor * 2 - 1
    elif tensor.min() >= 0 and tensor.max() > 1:
        # Assume [0, 255] range
        tensor = tensor / 127.5 - 1
    
    return tensor

def compute_metrics_for_tensor_pair(tensor1, tensor2, tensor3=None, tensor4=None, metrics_to_compute=None):
    """
    Compute metrics for tensors
    
    Args:
        tensor1: Original/source tensor
        tensor2: Perturbed/target tensor
        tensor3: Source deepfake result (optional, for scol_pds, sr_mask)
        tensor4: Perturbed deepfake result (optional, for scol_pds, sr_mask)
        metrics_to_compute: List of metric names to compute. If None, compute all.
    
    Returns:
        Dictionary of metric results
    """
    if metrics_to_compute is None:
        metrics_to_compute = [
            'mae', 'mse', 'rmse', 'psnr',
            'lpips_alex', 'lpips_vgg', 'id_loss',
            'brisque', 'fsim', 'l2_mask', 'vgg_loss'
        ]
    
    results = {}
    
    # Ensure tensors are in correct format
    tensor1 = ensure_tensor_format(tensor1)
    tensor2 = ensure_tensor_format(tensor2)
    
    if tensor1 is None or tensor2 is None:
        return None
    
    # For metrics that need deepfake results
    if tensor3 is not None:
        tensor3 = ensure_tensor_format(tensor3)
    if tensor4 is not None:
        tensor4 = ensure_tensor_format(tensor4)
    
    try:
        if 'mae' in metrics_to_compute:
            results['mae'] = float(mae(tensor1, tensor2).item())
        
        if 'mse' in metrics_to_compute:
            results['mse'] = float(mse(tensor1, tensor2).item())
        
        if 'rmse' in metrics_to_compute:
            results['rmse'] = float(rmse(tensor1, tensor2).item())
        
        if 'ssim' in metrics_to_compute:
            results['ssim'] = float(ssim(tensor1, tensor2))
        
        if 'psnr' in metrics_to_compute:
            results['psnr'] = float(psnr(tensor1, tensor2))
        
        if 'lpips_alex' in metrics_to_compute:
            results['lpips_alex'] = float(alex_lpips(tensor1, tensor2).item())
        
        if 'lpips_vgg' in metrics_to_compute:
            results['lpips_vgg'] = float(vgg_lpips(tensor1, tensor2).item())
        
        if 'id_loss' in metrics_to_compute:
            results['id_loss'] = float(id_loss(tensor1, tensor2))
        
        if 'brisque' in metrics_to_compute:
            results['brisque_tensor1'] = float(brisque_metric(tensor1))
            results['brisque_tensor2'] = float(brisque_metric(tensor2))
        
        if 'fsim' in metrics_to_compute:
            results['fsim'] = float(fsim(tensor1, tensor2))
        
        if 'l2_mask' in metrics_to_compute:
            results['l2_mask'] = float(l2_mask(tensor1, tensor2))
        
        if 'vgg_loss' in metrics_to_compute:
            results['vgg_loss'] = float(vgg_loss(tensor1, tensor2).item())
        
        # LEAT CDSR (returns dict)
        if 'leat_cdsr' in metrics_to_compute:
            cdsr_result = leat_cdsr(tensor1, tensor2)
            results['leat_cdsr_success'] = cdsr_result['success']
        
        # SR_mask (returns bool)
        if 'sr_mask' in metrics_to_compute and tensor3 is not None and tensor4 is not None:
            results['sr_mask'] = int(sr_mask_single(tensor3, tensor4))
        
        # SCOL PDS (needs 4 images)
        if 'scol_pds' in metrics_to_compute and tensor3 is not None and tensor4 is not None:
            results['scol_pds'] = float(scol_pds(tensor1, tensor3, tensor2, tensor4))
            
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None
    
    return results

def evaluate_tensor_directory(source_dir, perturbed_dir, output_json, 
                            source_df_dir=None, perturbed_df_dir=None, 
                            metrics_to_compute=None):
    """
    Evaluate all tensor pairs/groups from directories
    
    Args:
        source_dir: Directory containing original tensors
        perturbed_dir: Directory containing perturbed tensors
        source_df_dir: Directory containing source deepfake results (optional)
        perturbed_df_dir: Directory containing perturbed deepfake results (optional)
        output_json: Path to save results
        metrics_to_compute: List of metrics to compute
    """
    source_dir = Path(source_dir)
    perturbed_dir = Path(perturbed_dir)
    source_df_dir = Path(source_df_dir) if source_df_dir else None
    perturbed_df_dir = Path(perturbed_df_dir) if perturbed_df_dir else None
    
    # Get all .pt files
    source_files = sorted(list(source_dir.glob('*.pt')))
    
    if len(source_files) == 0:
        print(f"No .pt files found in {source_dir}")
        return
    
    print(f"Found {len(source_files)} tensor files to evaluate")
    
    all_results = []
    metric_sums = {}
    metric_counts = {}
    
    for source_file in tqdm(source_files, desc="Evaluating tensors"):
        # Find corresponding perturbed file
        perturbed_file = perturbed_dir / source_file.name
        
        if not perturbed_file.exists():
            print(f"Warning: No matching file for {source_file.name}")
            continue
        
        # Load tensors
        source_tensor = load_tensor_from_pt(source_file)
        perturbed_tensor = load_tensor_from_pt(perturbed_file)
        
        if source_tensor is None or perturbed_tensor is None:
            continue
        
        # Load deepfake results if directories provided
        source_df_tensor = None
        perturbed_df_tensor = None
        
        if source_df_dir is not None:
            source_df_file = source_df_dir / source_file.name
            if source_df_file.exists():
                source_df_tensor = load_tensor_from_pt(source_df_file)
        
        if perturbed_df_dir is not None:
            perturbed_df_file = perturbed_df_dir / source_file.name
            if perturbed_df_file.exists():
                perturbed_df_tensor = load_tensor_from_pt(perturbed_df_file)
        
        # Compute metrics
        metrics = compute_metrics_for_tensor_pair(
            source_tensor, 
            perturbed_tensor,
            source_df_tensor,
            perturbed_df_tensor,
            metrics_to_compute
        )
        
        if metrics is None:
            continue
        
        result = {
            'filename': source_file.name,
            'metrics': metrics
        }
        all_results.append(result)
        
        # Accumulate for average
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
    
    # Calculate averages
    metric_averages = {}
    for key in metric_sums:
        metric_averages[key] = metric_sums[key] / metric_counts[key]
    
    # Save results
    output_data = {
        'source_directory': str(source_dir),
        'perturbed_directory': str(perturbed_dir),
        'source_deepfake_directory': str(source_df_dir) if source_df_dir else None,
        'perturbed_deepfake_directory': str(perturbed_df_dir) if perturbed_df_dir else None,
        'total_evaluated': len(all_results),
        'metric_averages': metric_averages,
        'individual_results': all_results
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_json}")
    print("\n=== Average Metrics ===")
    for key, value in sorted(metric_averages.items()):
        print(f"{key}: {value:.6f}")
    
    return output_data

def evaluate_single_tensor_pair(source_pt, perturbed_pt, source_df_pt=None, perturbed_df_pt=None, metrics_to_compute=None):
    """
    Evaluate a single pair/group of tensor files
    
    Args:
        source_pt: Path to source .pt file
        perturbed_pt: Path to perturbed .pt file
        source_df_pt: Path to source deepfake .pt file (optional)
        perturbed_df_pt: Path to perturbed deepfake .pt file (optional)
        metrics_to_compute: List of metrics to compute
    
    Returns:
        Dictionary of metric results
    """
    source_tensor = load_tensor_from_pt(source_pt)
    perturbed_tensor = load_tensor_from_pt(perturbed_pt)
    
    if source_tensor is None or perturbed_tensor is None:
        print("Failed to load tensors")
        return None
    
    source_df_tensor = None
    perturbed_df_tensor = None
    
    if source_df_pt:
        source_df_tensor = load_tensor_from_pt(source_df_pt)
    if perturbed_df_pt:
        perturbed_df_tensor = load_tensor_from_pt(perturbed_df_pt)
    
    metrics = compute_metrics_for_tensor_pair(
        source_tensor, 
        perturbed_tensor,
        source_df_tensor,
        perturbed_df_tensor,
        metrics_to_compute
    )
    
    if metrics:
        print("\n=== Metrics ===")
        for key, value in sorted(metrics.items()):
            print(f"{key}: {value:.6f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate metrics for tensor (.pt) files')
    parser.add_argument('--source_dir', type=str, help='Directory containing source tensors')
    parser.add_argument('--perturbed_dir', type=str, help='Directory containing perturbed tensors')
    parser.add_argument('--source_df_dir', type=str, help='Directory containing source deepfake results (optional)')
    parser.add_argument('--perturbed_df_dir', type=str, help='Directory containing perturbed deepfake results (optional)')
    parser.add_argument('--source_file', type=str, help='Single source tensor file')
    parser.add_argument('--perturbed_file', type=str, help='Single perturbed tensor file')
    parser.add_argument('--source_df_file', type=str, help='Single source deepfake tensor file (optional)')
    parser.add_argument('--perturbed_df_file', type=str, help='Single perturbed deepfake tensor file (optional)')
    parser.add_argument('--output', type=str, default='tensor_evaluation_results.json', 
                       help='Output JSON file for results')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       help='Specific metrics to compute (default: all)')
    
    args = parser.parse_args()
    
    if args.source_dir and args.perturbed_dir:
        # Batch evaluation
        evaluate_tensor_directory(
            args.source_dir, 
            args.perturbed_dir,
            args.output,
            args.source_df_dir,
            args.perturbed_df_dir,
            args.metrics
        )
    elif args.source_file and args.perturbed_file:
        # Single file evaluation
        evaluate_single_tensor_pair(
            args.source_file,
            args.perturbed_file,
            args.source_df_file,
            args.perturbed_df_file,
            args.metrics
        )
    else:
        print("Please provide either --source_dir and --perturbed_dir for batch evaluation,")
        print("or --source_file and --perturbed_file for single file evaluation")
