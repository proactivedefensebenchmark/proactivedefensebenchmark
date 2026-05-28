from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from datetime import datetime


def tensor_to_temp_image(tensor, temp_path):
    """Convert tensor to temporary image file for DeepFace processing"""
    # Detach tensor if it requires grad
    tensor = tensor.detach()
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from [-1, 1] or [0, 1] to [0, 255]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy [H, W, C]
    if tensor.shape[0] == 3:  # [C, H, W] -> [H, W, C]
        np_img = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        np_img = tensor.cpu().numpy()
    
    np_img = (np_img * 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    img.save(temp_path)
    return temp_path


def compute_face_embedding(img_path):
    """Extract face embedding vector of given image"""
    try:
        resps = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="retinaface",
            align=True
        )
        if isinstance(resps, list) and len(resps) > 0:
            if len(resps) == 1:
                return np.array(resps[0]["embedding"])
            else:
                # Multiple faces: choose the biggest one
                resps.sort(key=lambda resp: resp["facial_area"]["h"] * resp["facial_area"]["w"], reverse=True)
                return np.array(resps[0]["embedding"])
        return None
    except Exception as e:
        return None


def compute_face_embedding_from_tensor(tensor, temp_dir):
    """Extract face embedding from tensor"""
    temp_path = os.path.join(temp_dir, "temp_face.jpg")
    tensor_to_temp_image(tensor, temp_path)
    embedding = compute_face_embedding(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return embedding


def compute_ism_fdfr_for_folder(result_folder, src_folder, temp_dir):
    """
    Compute ISM and FDFR for a single result folder (e.g., result_blendface_anti)
    
    ISM: Identity Score Matching between decoded_adv and source images
    FDFR: Face Detection Failure Rate
    """
    img_folders = sorted([f for f in os.listdir(result_folder) if f.startswith("img_")])
    
    total_count = 0
    fail_detection_count = 0
    total_ism = 0.0
    
    results_per_image = []
    
    for img_folder in img_folders:
        img_path = os.path.join(result_folder, img_folder)
        decoded_adv_path = os.path.join(img_path, "decoded_adv.pt")
        decoded_src_path = os.path.join(img_path, "decoded_src.pt")
        
        # Also check for source image in src_folder if provided
        img_num = img_folder.replace("img_", "")
        
        if not os.path.exists(decoded_adv_path):
            print(f"  [SKIP] {img_folder}: decoded_adv.pt not found")
            continue
            
        total_count += 1
        
        try:
            # Load tensors
            decoded_adv = torch.load(decoded_adv_path, map_location='cpu')
            
            # Get embedding for decoded_adv
            adv_embedding = compute_face_embedding_from_tensor(decoded_adv, temp_dir)
            
            if adv_embedding is None:
                fail_detection_count += 1
                results_per_image.append({
                    'img': img_folder,
                    'ism': None,
                    'detected': False
                })
                print(f"  [FAIL] {img_folder}: Face not detected in decoded_adv")
                continue
            
            # Get source embedding
            src_embedding = None
            
            # Try to get source from decoded_src.pt first
            if os.path.exists(decoded_src_path):
                decoded_src = torch.load(decoded_src_path, map_location='cpu')
                src_embedding = compute_face_embedding_from_tensor(decoded_src, temp_dir)
            
            # If src embedding failed, try from source folder
            if src_embedding is None and src_folder:
                possible_src_paths = [
                    os.path.join(src_folder, f"{int(img_num):05d}.jpg"),
                    os.path.join(src_folder, f"{int(img_num):05d}.png"),
                    os.path.join(src_folder, f"img_{img_num}.jpg"),
                    os.path.join(src_folder, f"img_{img_num}.png"),
                ]
                for src_path in possible_src_paths:
                    if os.path.exists(src_path):
                        src_embedding = compute_face_embedding(src_path)
                        if src_embedding is not None:
                            break
            
            if src_embedding is None:
                fail_detection_count += 1
                results_per_image.append({
                    'img': img_folder,
                    'ism': None,
                    'detected': False
                })
                print(f"  [FAIL] {img_folder}: Source face not detected")
                continue
            
            # Compute ISM (cosine similarity)
            adv_tensor = torch.Tensor(adv_embedding)
            src_tensor = torch.Tensor(src_embedding)
            ism = F.cosine_similarity(adv_tensor, src_tensor, dim=0).item()
            
            total_ism += ism
            results_per_image.append({
                'img': img_folder,
                'ism': ism,
                'detected': True
            })
            print(f"  [OK] {img_folder}: ISM = {ism:.4f}")
            
        except Exception as e:
            fail_detection_count += 1
            results_per_image.append({
                'img': img_folder,
                'ism': None,
                'detected': False,
                'error': str(e)
            })
            print(f"  [ERROR] {img_folder}: {str(e)}")
    
    # Calculate final metrics
    successful_count = total_count - fail_detection_count
    avg_ism = total_ism / successful_count if successful_count > 0 else 0.0
    fdfr = fail_detection_count / total_count if total_count > 0 else 1.0
    
    return {
        'total': total_count,
        'successful': successful_count,
        'failed': fail_detection_count,
        'avg_ism': avg_ism,
        'fdfr': fdfr,
        'details': results_per_image
    }


def process_batch_results(batch_dir, src_folder=None, output_file=None):
    """
    Process all result folders in batch_dir (excluding black_box_results)
    """
    # Create temp directory for image conversion
    temp_dir = os.path.join(batch_dir, "temp_ism_fdfr")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Find all result folders (exclude black_box_results)
    result_folders = sorted([
        f for f in os.listdir(batch_dir) 
        if os.path.isdir(os.path.join(batch_dir, f)) 
        and f.startswith("result_") 
        and "black_box" not in f
    ])
    
    if not result_folders:
        print(f"No result folders found in {batch_dir}")
        return
    
    print(f"Found {len(result_folders)} result folders: {result_folders}")
    
    all_results = {}
    
    for folder_name in result_folders:
        folder_path = os.path.join(batch_dir, folder_name)
        print(f"\n{'='*60}")
        print(f"Processing: {folder_name}")
        print(f"{'='*60}")
        
        results = compute_ism_fdfr_for_folder(folder_path, src_folder, temp_dir)
        all_results[folder_name] = results
        
        print(f"\n[Summary] {folder_name}:")
        print(f"  Total images: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Average ISM: {results['avg_ism']:.4f}")
        print(f"  FDFR: {results['fdfr']:.4f} ({results['fdfr']*100:.2f}%)")
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    # Generate output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(batch_dir, f"ism_fdfr_results_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ISM (Identity Score Matching) & FDFR (Face Detection Failure Rate) Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch Directory: {batch_dir}\n")
        if src_folder:
            f.write(f"Source Folder: {src_folder}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<30} {'Total':>8} {'Success':>8} {'Failed':>8} {'ISM':>10} {'FDFR':>10}\n")
        f.write("-" * 80 + "\n")
        
        for folder_name, results in all_results.items():
            model_name = folder_name.replace("result_", "").replace("_anti", "").replace("_", " ")
            f.write(f"{model_name:<30} {results['total']:>8} {results['successful']:>8} "
                   f"{results['failed']:>8} {results['avg_ism']:>10.4f} {results['fdfr']:>10.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Overall average
        if all_results:
            avg_ism_all = np.mean([r['avg_ism'] for r in all_results.values()])
            avg_fdfr_all = np.mean([r['fdfr'] for r in all_results.values()])
            f.write(f"Overall Average ISM: {avg_ism_all:.4f}\n")
            f.write(f"Overall Average FDFR: {avg_fdfr_all:.4f} ({avg_fdfr_all*100:.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Results per Model\n")
        f.write("=" * 80 + "\n")
        
        for folder_name, results in all_results.items():
            f.write(f"\n[{folder_name}]\n")
            f.write(f"  Total: {results['total']}, Success: {results['successful']}, Failed: {results['failed']}\n")
            f.write(f"  Avg ISM: {results['avg_ism']:.4f}, FDFR: {results['fdfr']:.4f}\n")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description='Batch ISM and FDFR evaluation for decoded tensors')
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Path to batch results directory (e.g., batch_results_anti)')
    parser.add_argument('--src_folder', type=str, default=None,
                        help='Path to source images folder (optional, for better ISM calculation)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated in batch_dir)')
    return parser.parse_args()


def main():
    args = parse_args()
    process_batch_results(args.batch_dir, args.src_folder, args.output)


if __name__ == '__main__':
    main()
