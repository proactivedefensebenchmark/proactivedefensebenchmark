"""
Evaluate Robustness Test Results
각 전처리 방법별로 decoded 결과를 원본과 비교하여 메트릭 측정
Based on evaluate_disruption.py logic
"""
import torch
import argparse
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv
import json
from collections import defaultdict
from pathlib import Path

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))      # evaluation/
project_root = os.path.dirname(current_dir)                   # Project Root

for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.evaluation_metrics import (
    mae, mse, rmse, ssim, psnr, id_loss,
    vgg_lpips, alex_lpips, brisque_metric, DEVICE
)


def load_image_as_tensor(image_path, device='cuda'):
    """
    Load image and convert to tensor [-1, 1]
    Based on evaluate_disruption.py's load_image_as_tensor
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        tensor = tensor * 2.0 - 1.0  # [-1, 1]
        return tensor.to(device).detach()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_tensor_from_pt(file_path, device='cuda'):
    """
    Load tensor from .pt file
    Based on evaluate_disruption.py's load_tensor_from_pt
    """
    try:
        tensor = torch.load(file_path, map_location=device)
        if isinstance(tensor, dict):
            for key in ['tensor', 'image', 'data', 'img']:
                if key in tensor:
                    tensor = tensor[key]
                    break
            if isinstance(tensor, dict):
                tensor = list(tensor.values())[0]
        
        # Ensure tensor is on device
        tensor = tensor.to(device)
        
        # Add batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Make contiguous to avoid view/stride errors
        return tensor.detach().contiguous()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_metrics(img1_tensor, img2_tensor):
    """
    Calculate all metrics between two tensors
    Based on evaluate_disruption.py's compute_decoded_metrics
    """
    try:
        metrics = {
            'mae': float(mae(img1_tensor, img2_tensor)),
            'mse': float(mse(img1_tensor, img2_tensor)),
            'rmse': float(rmse(img1_tensor, img2_tensor)),
            'ssim': float(ssim(img1_tensor, img2_tensor)),
            'psnr': float(psnr(img1_tensor, img2_tensor)),
            'id_loss': float(id_loss(img1_tensor, img2_tensor)),
            'vgg_lpips': float(vgg_lpips(img1_tensor, img2_tensor)),
            'alex_lpips': float(alex_lpips(img1_tensor, img2_tensor)),
            'brisque': float(brisque_metric(img2_tensor))
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def evaluate_sample_folder(sample_dir, device='cuda', use_pt=False):
    """
    하나의 sample_img_* 폴더를 평가
    decoded_clean을 기준으로 각 전처리별 decoded_adv_* 비교
    
    Args:
        sample_dir: samples_img_* 폴더 경로
        device: cuda or cpu
        use_pt: True면 .pt 파일 사용, False면 .jpg 사용
    """
    # Determine file extension
    ext = '.pt' if use_pt else '.jpg'
    
    # Load reference (decoded_clean)
    decoded_clean_path = os.path.join(sample_dir, f"decoded_clean{ext}")
    
    if not os.path.exists(decoded_clean_path):
        return None
    
    # Load reference image/tensor
    try:
        if use_pt:
            decoded_clean = load_tensor_from_pt(decoded_clean_path, device)
        else:
            decoded_clean = load_image_as_tensor(decoded_clean_path, device)
            
        if decoded_clean is None:
            return None
    except Exception as e:
        print(f"Error loading {decoded_clean_path}: {e}")
        return None
    
    # Find all decoded_adv_* files (including original as baseline)
    results = {}
    
    for filename in os.listdir(sample_dir):
        if filename.startswith("decoded_adv_") and filename.endswith(ext):
            # Extract preprocessing method name
            preproc_method = filename.replace("decoded_adv_", "").replace(ext, "")
            
            # Include original as baseline
            if preproc_method == "original":
                preproc_method = "baseline"
            
            file_path = os.path.join(sample_dir, filename)
            
            try:
                if use_pt:
                    decoded_adv = load_tensor_from_pt(file_path, device)
                else:
                    decoded_adv = load_image_as_tensor(file_path, device)
                
                if decoded_adv is None:
                    continue
                    
                metrics = calculate_metrics(decoded_clean, decoded_adv)
                if metrics:
                    results[preproc_method] = metrics
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return results


def evaluate_result_folder(result_dir, device='cuda', use_pt=False):
    """
    하나의 result_* 폴더 전체를 평가
    Based on evaluate_disruption.py's evaluate_disruption_batch logic
    
    Args:
        result_dir: result_* 폴더 경로
        device: cuda or cpu
        use_pt: True면 .pt 파일 사용, False면 .jpg 사용
    """
    result_dir = Path(result_dir)
    
    print(f"\n{'='*50}")
    print(f"Evaluating: {result_dir.name}")
    print(f"{'='*50}")
    
    # Find all samples_img_* folders
    sample_folders = sorted([
        d for d in result_dir.iterdir()
        if d.name.startswith("samples_img_") and d.is_dir()
    ])
    
    if not sample_folders:
        print("No sample folders found!")
        return None
    
    print(f"Found {len(sample_folders)} sample folders")
    print(f"Using {'PT tensors' if use_pt else 'JPG images'}")
    
    # Collect all results - similar to evaluate_disruption.py's structure
    all_results = []
    metric_sums = defaultdict(lambda: defaultdict(float))  # {preproc_method: {metric: sum}}
    metric_counts = defaultdict(lambda: defaultdict(int))  # {preproc_method: {metric: count}}
    
    for sample_folder in tqdm(sample_folders, desc="Processing samples"):
        sample_results = evaluate_sample_folder(sample_folder, device, use_pt)
        
        if sample_results:
            # Store per-sample results
            result_entry = {
                'filename': sample_folder.name,
                'preprocessing_metrics': sample_results
            }
            all_results.append(result_entry)
            
            # Accumulate for averages
            for preproc_method, metrics in sample_results.items():
                for metric_name, value in metrics.items():
                    metric_sums[preproc_method][metric_name] += value
                    metric_counts[preproc_method][metric_name] += 1
    
    # Calculate averages - matching evaluate_disruption.py's output format
    preprocessing_averages = {}
    for preproc_method in metric_sums.keys():
        avg_metrics = {}
        for metric_name in metric_sums[preproc_method].keys():
            total = metric_sums[preproc_method][metric_name]
            count = metric_counts[preproc_method][metric_name]
            avg_metrics[metric_name] = float(total / count) if count > 0 else 0.0
        
        preprocessing_averages[preproc_method] = {
            'average_metrics': avg_metrics,
            'num_samples': metric_counts[preproc_method].get('mae', 0)  # Use any metric's count
        }
    
    # Return structure similar to evaluate_disruption.py
    return {
        'preprocessing_averages': preprocessing_averages,
        'total_evaluated': len(all_results),
        'all_results': all_results
    }


def save_results_to_csv(results, output_path):
    """
    Save results to CSV file
    Based on evaluate_disruption.py's result format
    """
    if not results or 'preprocessing_averages' not in results:
        print("No results to save!")
        return
    
    preprocessing_averages = results['preprocessing_averages']
    
    # Prepare CSV data
    rows = []
    for preproc_method, data in sorted(preprocessing_averages.items()):
        metrics = data['average_metrics']
        row = {
            'preprocessing': preproc_method,
            'num_samples': data['num_samples'],
            'mae': metrics.get('mae', 0.0),
            'mse': metrics.get('mse', 0.0),
            'rmse': metrics.get('rmse', 0.0),
            'ssim': metrics.get('ssim', 0.0),
            'psnr': metrics.get('psnr', 0.0),
            'id_loss': metrics.get('id_loss', 0.0),
            'vgg_lpips': metrics.get('vgg_lpips', 0.0),
            'alex_lpips': metrics.get('alex_lpips', 0.0),
            'brisque': metrics.get('brisque', 0.0)
        }
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['preprocessing', 'num_samples', 'mae', 'mse', 'rmse', 'ssim', 'psnr', 'id_loss', 'vgg_lpips', 'alex_lpips', 'brisque']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results):
    """
    Print summary of results
    Based on evaluate_disruption.py's summary format
    """
    if not results or 'preprocessing_averages' not in results:
        print("No results to display!")
        return
        
    preprocessing_averages = results['preprocessing_averages']
    
    print(f"\n{'='*110}")
    print(f"{'Preprocessing':<15} {'Samples':<8} {'MAE':<10} {'SSIM':<8} {'PSNR':<8} {'ID Loss':<10} {'VGG-LPIPS':<11} {'Alex-LPIPS':<12} {'BRISQUE':<10}")
    print(f"{'='*110}")
    
    for preproc_method, data in sorted(preprocessing_averages.items()):
        metrics = data['average_metrics']
        print(f"{preproc_method:<15} {data['num_samples']:<8} "
              f"{metrics.get('mae', 0.0):<10.6f} {metrics.get('ssim', 0.0):<8.4f} "
              f"{metrics.get('psnr', 0.0):<8.2f} {metrics.get('id_loss', 0.0):<10.6f} "
              f"{metrics.get('vgg_lpips', 0.0):<11.6f} {metrics.get('alex_lpips', 0.0):<12.6f} "
              f"{metrics.get('brisque', 0.0):<10.4f}")
    
    print(f"{'='*110}")
    print(f"Total samples evaluated: {results['total_evaluated']}")
    print(f"{'='*110}")


def main():
    """
    Main function - structure based on evaluate_disruption.py
    """
    parser = argparse.ArgumentParser(description='Evaluate Robustness Test Results')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Path to result_* folder (e.g., robustness_results/result_blendface_pgd)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as result_dir)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all result_* folders in the given directory')
    parser.add_argument('--use_pt', action='store_true',
                        help='Use .pt tensor files instead of .jpg images')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Determine folders to process - similar to run_batch_pgd_evaluation.sh logic
    if args.batch_mode:
        # Process all result_* folders (like the shell script does)
        parent_dir = Path(args.result_dir)
        result_folders = sorted([
            d for d in parent_dir.iterdir()
            if d.name.startswith("result_") and d.is_dir()
        ])
        print(f"\nBatch mode: Found {len(result_folders)} result folders")
    else:
        # Process single folder
        result_folders = [Path(args.result_dir)]
    
    # Process each folder - matching shell script pattern
    for result_dir in result_folders:
        if not result_dir.exists():
            print(f"Directory not found: {result_dir}")
            continue
        
        # Evaluate - similar to evaluate_disruption.py call
        results = evaluate_result_folder(result_dir, args.device, args.use_pt)
        
        if results and results['total_evaluated'] > 0:
            # Set output directory - create separate evaluation results folder
            if args.output_dir:
                # User specified output directory
                if args.batch_mode:
                    # Create subdirectory for each result folder
                    output_dir = Path(args.output_dir) / result_dir.name
                else:
                    output_dir = Path(args.output_dir)
            else:
                # Default: create robustness_evaluation subfolder inside result_dir
                output_dir = result_dir / "robustness_evaluation"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results - matching evaluate_disruption.py output pattern
            csv_path = output_dir / "robustness_evaluation.csv"
            json_path = output_dir / "robustness_evaluation.json"
            summary_path = output_dir / "robustness_evaluation_summary.txt"
            
            save_results_to_csv(results, csv_path)
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"JSON saved to: {json_path}")
            
            # Save summary to text file (like shell script does)
            with open(summary_path, 'w') as f:
                f.write("="*50 + "\n")
                f.write(f"Robustness Evaluation Summary: {result_dir.name}\n")
                f.write("="*50 + "\n\n")
                
                preprocessing_averages = results['preprocessing_averages']
                for preproc_method, data in sorted(preprocessing_averages.items()):
                    f.write(f"\n=== {preproc_method} ===\n")
                    metrics = data['average_metrics']
                    for metric_name, value in sorted(metrics.items()):
                        f.write(f"{metric_name}: {value:.6f}\n")
                
                f.write(f"\nTotal evaluated: {results['total_evaluated']}\n")
            
            print(f"Summary saved to: {summary_path}")
            
            # Print summary to console
            print_summary(results)
        else:
            print(f"No valid results for {result_dir}")
    
    print("\n All evaluations completed!")


if __name__ == "__main__":
    main()
