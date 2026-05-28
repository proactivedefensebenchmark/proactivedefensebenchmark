"""
Evaluation script for ensemble decoded results (diffae + simswap)
Evaluates:
  - diffae_decoded_src.pt vs diffae_decoded_adv.pt
  - simswap_decoded_src.pt vs simswap_decoded_adv.pt

Usage:
    python evaluate_ensemble_decoded.py --result_dir /path/to/result_diffae_simswap_ensemble
"""

import os
import torch
import json
import argparse
from pathlib import Path
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


def evaluate_model_decoded(result_dir, model_name, output_dir=None):
    """
    Evaluate decoded_src vs decoded_adv for a specific model (diffae or simswap)
    
    Args:
        result_dir: Directory containing img_* subdirectories
        model_name: 'diffae' or 'simswap'
        output_dir: Directory to save results (default: result_dir)
    
    Returns:
        dict: Evaluation results
    """
    result_dir = Path(result_dir)
    if output_dir is None:
        output_dir = result_dir
    else:
        output_dir = Path(output_dir)
    
    # Find all img_* subdirectories
    img_dirs = sorted([d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith('img_')])
    
    if len(img_dirs) == 0:
        print(f"No img_* subdirectories found in {result_dir}")
        return None
    
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name.upper()} decoded results")
    print(f"{'='*50}")
    print(f"Found {len(img_dirs)} image subdirectories")
    
    all_results = []
    metric_sums = {}
    metric_counts = {}
    
    for img_dir in tqdm(img_dirs, desc=f"Evaluating {model_name}"):
        base_name = img_dir.name
        
        # File paths for this model
        decoded_src_file = img_dir / f"decoded_src_{model_name}.pt"
        decoded_adv_file = img_dir / f"decoded_adv_{model_name}.pt"
        
        # Check if files exist
        if not decoded_src_file.exists():
            print(f"Warning: {decoded_src_file} not found, skipping...")
            continue
        if not decoded_adv_file.exists():
            print(f"Warning: {decoded_adv_file} not found, skipping...")
            continue
        
        # Load tensors
        decoded_src_tensor = load_tensor_from_pt(decoded_src_file)
        decoded_adv_tensor = load_tensor_from_pt(decoded_adv_file)
        
        if decoded_src_tensor is None or decoded_adv_tensor is None:
            print(f"Warning: Failed to load tensors for {base_name}, skipping...")
            continue
        
        # Compute metrics
        metrics = compute_decoded_metrics(decoded_src_tensor, decoded_adv_tensor)
        
        if metrics:
            result = {
                'filename': base_name,
                'metrics': metrics
            }
            all_results.append(result)
            
            # Accumulate for averages
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1
    
    # Calculate averages
    metric_averages = {}
    for key in metric_sums:
        if metric_counts[key] > 0:
            metric_averages[key] = metric_sums[key] / metric_counts[key]
    
    # Prepare final results
    final_results = {
        'model': model_name,
        'total_evaluated': len(all_results),
        'metric_averages': metric_averages,
        'individual_results': all_results
    }
    
    # Save results
    output_json = output_dir / f"{model_name}_evaluation_results.json"
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary text
    output_txt = output_dir / f"{model_name}_evaluation_summary.txt"
    with open(output_txt, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation Summary: {model_name.upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nTotal evaluated: {len(all_results)}\n\n")
        f.write("=== Decoded Metrics (decoded_src vs decoded_adv) ===\n")
        for key, value in sorted(metric_averages.items()):
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"\nResults saved to:")
    print(f"  - JSON: {output_json}")
    print(f"  - Summary: {output_txt}")
    
    # Print summary
    print(f"\n=== {model_name.upper()} Summary ===")
    print(f"Total evaluated: {len(all_results)}")
    for key, value in sorted(metric_averages.items()):
        print(f"  {key}: {value:.6f}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble decoded results (diffae + simswap)')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing img_* subdirectories with decoded .pt files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as result_dir)')
    parser.add_argument('--models', type=str, nargs='+', default=['diffae', 'simswap'],
                        help='Models to evaluate (default: diffae simswap)')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    
    if not result_dir.exists():
        print(f"ERROR: Result directory not found: {result_dir}")
        return
    
    # Evaluate each model
    all_model_results = {}
    for model_name in args.models:
        results = evaluate_model_decoded(result_dir, model_name, output_dir)
        if results:
            all_model_results[model_name] = results
    
    # Create total result file
    total_result_file = output_dir / "ensemble_total_result.txt"
    with open(total_result_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ENSEMBLE EVALUATION TOTAL RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Result Directory: {result_dir}\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, results in all_model_results.items():
            f.write("-" * 60 + "\n")
            f.write(f"Model: {model_name.upper()}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total evaluated: {results['total_evaluated']}\n\n")
            f.write("Metric Averages:\n")
            for key, value in sorted(results['metric_averages'].items()):
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\n")
        
        # Comparison table
        f.write("=" * 60 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Metric':<15}")
        for model_name in all_model_results:
            f.write(f"{model_name.upper():<15}")
        f.write("\n")
        f.write("-" * (15 + 15 * len(all_model_results)) + "\n")
        
        # Get all metrics
        all_metrics = set()
        for results in all_model_results.values():
            all_metrics.update(results['metric_averages'].keys())
        
        for metric in sorted(all_metrics):
            f.write(f"{metric:<15}")
            for model_name, results in all_model_results.items():
                value = results['metric_averages'].get(metric, float('nan'))
                f.write(f"{value:<15.6f}")
            f.write("\n")
    
    print(f"\n{'='*60}")
    print("All evaluations completed!")
    print(f"{'='*60}")
    print(f"Total result file: {total_result_file}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':<15}", end="")
    for model_name in all_model_results:
        print(f"{model_name.upper():<15}", end="")
    print()
    print("-" * (15 + 15 * len(all_model_results)))
    
    all_metrics = set()
    for results in all_model_results.values():
        all_metrics.update(results['metric_averages'].keys())
    
    for metric in sorted(all_metrics):
        print(f"{metric:<15}", end="")
        for model_name, results in all_model_results.items():
            value = results['metric_averages'].get(metric, float('nan'))
            print(f"{value:<15.6f}", end="")
        print()


if __name__ == "__main__":
    main()
