"""
Batch DF_RAP Attack Runner
Based on original DF_RAP implementation with wrapper interface
"""
import torch
import argparse
import os
import sys
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.paths_config import CELEBA_HQ_TEST_DIR

from utils.utils import (
    WRAPPER_REGISTRY,
    load_wrapper,
    load_image,
)
from disrupting_methods.df_rap.df_rap import df_rap_attack
# Import ComGenerator from original net folder
sys.path.insert(0, os.path.join(project_root, 'disrupting_methods/df_rap'))
from net.ComGAN import ComGenerator


def save_tensor_as_image(tensor, path):
    """Save tensor as image file"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    tensor = torch.clamp((tensor + 1.0) / 2.0, 0, 1)
    img = (tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(img).save(path)


def get_target_image_for_wrapper(wrapper_name, device):
    """Get target image for a specific wrapper using its config"""
    if wrapper_name not in WRAPPER_REGISTRY:
        return None
    
    config = WRAPPER_REGISTRY[wrapper_name]
    
    if not config.get('requires_target', False):
        return None
    
    target_path = os.environ.get('DEEPFAKE_TARGET_IMAGE', config.get('default_target_path'))
    if target_path is None:
        return None
    
    if os.path.exists(target_path):
        try:
            return load_image(target_path, device)[0]
        except Exception as e:
            print(f"⚠️  Failed to load target image from {target_path}: {e}")
            return None
    else:
        print(f"⚠️  Target image not found: {target_path}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch DF_RAP Attack on Multiple Wrappers"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default=CELEBA_HQ_TEST_DIR,
        help="Directory containing test images"
    )
    
    parser.add_argument(
        "--n_images",
        type=int,
        default=100,
        help="Number of images to process per wrapper"
    )
    
    parser.add_argument(
        "--output_base",
        type=str,
        default=None,
        help="Base output directory (default: current_dir/batch_results_df_rap)"
    )

    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Optional subdirectory under output_base (e.g., ffhq)"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Perturbation budget"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Step size for PGD"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of PGD steps"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    parser.add_argument(
        "--wrappers",
        nargs="+",
        default=["simswap", "psp_mix", "diffae", "styleclip", "diffswap"],
        help="Wrappers to use"
    )
    
    parser.add_argument(
        "--comg_checkpoint",
        type=str,
        default=None,
        help="Path to ComG checkpoint (.pth file, optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main(args):
    """Main execution function"""
    device = args.device
    print(f"Device: {device}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Set default output_base if not provided
    if args.output_base is None:
        args.output_base = os.path.join(project_root, "batch_results_df_rap")

    # Construct output root with subdir if provided
    output_root = args.output_base
    if args.output_subdir:
        output_root = os.path.join(args.output_base, args.output_subdir)

    print(f"Output base directory: {args.output_base}")
    if args.output_subdir:
        print(f"Output subdirectory: {args.output_subdir}")
    print(f"Final output root: {output_root}")
    
    # Load ComG (required for DF_RAP)
    print("\nLoading ComG...")
    if args.comg_checkpoint:
        try:
            print(f"Loading ComG checkpoint: {args.comg_checkpoint}")
            checkpoint = torch.load(args.comg_checkpoint, map_location=device, weights_only=False)
            
            # Official format: checkpoint['ComG']
            if isinstance(checkpoint, dict) and 'ComG' in checkpoint:
                ComG = checkpoint['ComG'].to(device)
                print("✅ ComG loaded successfully (official format)")
            else:
                # Try to load as state dict
                ComG = ComGenerator(dim_in=3, dim_out=32, isJPEG=False).to(device)
                ComG.load_state_dict(checkpoint)
                print("✅ ComG loaded successfully (state dict)")
        except Exception as e:
            print(f"⚠️  Failed to load ComG checkpoint: {e}")
            print("   Using randomly initialized ComG instead.")
            ComG = ComGenerator(dim_in=3, dim_out=32, isJPEG=False).to(device)
    else:
        print("⚠️  No ComG checkpoint provided. Using randomly initialized ComG.")
        ComG = ComGenerator(dim_in=3, dim_out=32, isJPEG=False).to(device)
    ComG.eval()
    
    # Load image paths
    img_dir = args.image_dir
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    img_paths = img_paths[:args.n_images]
    
    print(f"Found {len(img_paths)} images to process")
    
    # Process each wrapper
    for wrapper_name in args.wrappers:
        if wrapper_name not in WRAPPER_REGISTRY:
            print(f"⚠️  Wrapper '{wrapper_name}' not found, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {wrapper_name.upper()} - DF_RAP Attack")
        print(f"{'='*60}")
        
        config = WRAPPER_REGISTRY[wrapper_name]
        
        # Load target image if needed
        target_tensor = None
        if config['requires_target']:
            target_tensor = get_target_image_for_wrapper(wrapper_name, device)
            if target_tensor is None:
                print(f"⚠️  {wrapper_name} requires target image but none available, skipping")
                continue
            else:
                effective_target_path = os.environ.get('DEEPFAKE_TARGET_IMAGE', config.get('default_target_path', 'unknown'))
                print(f"   Target image loaded from: {effective_target_path}")
        
        # Create output directory
        output_dir = os.path.join(output_root, f"result_{wrapper_name}_df_rap")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load wrapper
        try:
            wrapper, config = load_wrapper(wrapper_name, device=device)
        except Exception as e:
            print(f"❌ Failed to load wrapper: {e}")
            continue
        
        # Determine ref
        if config['requires_target']:
            wrapper_ref = target_tensor
            if hasattr(wrapper, 'set_target'):
                wrapper.set_target(target_tensor)
            print(f"   Using target image as reference")
        elif config['requires_attr']:
            wrapper_ref = config['default_attr']
            print(f"   Using attribute as reference: {wrapper_ref}")
        else:
            wrapper_ref = None
            print(f"   No reference needed")
        
        # Process images
        for idx, img_path in enumerate(tqdm(img_paths, desc=wrapper_name)):
            try:
                # Create image-specific folder checksum
                folder_path = os.path.join(output_dir, f"img_{idx+1:03d}")

                # Skip if already processed
                if os.path.exists(os.path.join(folder_path, "decoded_adv.pt")):
                    continue

                # Load source image
                source_tensor, _ = load_image(img_path, device)
                
                # Run DF_RAP attack
                adv_tensor = df_rap_attack(
                    wrapper=wrapper,
                    X_nat=source_tensor,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    steps=args.steps,
                    ref=wrapper_ref,
                    ComG=ComG
                )
                
                # Generate outputs
                with torch.no_grad():
                    if config['requires_target'] and target_tensor is not None:
                        decoded_src = wrapper(source_tensor, ref=target_tensor, preprocess=False)
                        decoded_adv = wrapper(adv_tensor, ref=target_tensor, preprocess=False)
                    elif config['requires_attr'] and wrapper_ref is not None:
                        decoded_src = wrapper(source_tensor, target_attr=wrapper_ref)
                        decoded_adv = wrapper(adv_tensor, target_attr=wrapper_ref)
                    else:
                        decoded_src = wrapper(source_tensor)
                        decoded_adv = wrapper(adv_tensor)

                # Create image-specific folder
                os.makedirs(folder_path, exist_ok=True)
                
                # Save images
                save_tensor_as_image(source_tensor, os.path.join(folder_path, "x_src.jpg"))
                save_tensor_as_image(adv_tensor, os.path.join(folder_path, "x_adv.jpg"))
                save_tensor_as_image(decoded_src, os.path.join(folder_path, "decoded_src.jpg"))
                save_tensor_as_image(decoded_adv, os.path.join(folder_path, "decoded_adv.jpg"))
                
                # Save tensors
                torch.save(adv_tensor, os.path.join(folder_path, "x_adv.pt"))
                torch.save(decoded_src, os.path.join(folder_path, "decoded_src.pt"))
                torch.save(decoded_adv, os.path.join(folder_path, "decoded_adv.pt"))
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n⚠️  Error processing image {idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        print(f"✅ {wrapper_name} completed!")
        print(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    print(f"\n{'='*60}")
    print("✅ All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
