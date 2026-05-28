import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import numpy as np

from configs.paths_config import FFHQ_TARGET_69999, SIMSWAP_TARGET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Image Processing Utilities
# ============================================================================
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    return x


def Image2tensor(imagepath, process=False, resize=256):
    img = Image.open(imagepath).convert("RGB")
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
        transform.append(T.Normalize(mean=0.5, std=0.5))
    if process:
        transform.append(T.Resize([resize, resize]))
    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img), dim=0).to(device)
    return img


def load_image(path, device='cuda'):
    """
    Load image from file and convert to tensor
    Returns: (tensor [-1,1], PIL image)
    device: 'cuda' or 'cpu' (string)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img_pil = Image.open(path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    return img_tensor, img_pil


def tensor2img(t):
    """
    Convert tensor [-1, 1] to numpy image [0, 1]
    Input: (B, C, H, W) or (C, H, W)
    Output: (H, W, C) numpy array in [0, 1]
    """
    t = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    t = (t + 1) / 2
    return t.clamp(0, 1).numpy()


def save_image_tensor(tensor, path):
    """
    Save tensor as image file
    Input: tensor in [-1, 1] range
    """
    pil_image = TF.to_pil_image(tensor.detach().cpu().squeeze(0))
    pil_image.save(path)


# ============================================================================
# Wrapper Registry - All available generative models
# ============================================================================
WRAPPER_REGISTRY = {
    'diffae': {
        'class': 'DiffAEWrapper',
        'module': 'diffae_wrapper',
        'requires_target': False,
        'requires_attr': True,
        'default_attr': 'Smiling',
        'description': 'DiffAE (Diffusion Autoencoder) for face attribute manipulation',
    },
    'styleclip': {
        'class': 'StyleCLIPWrapper',
        'module': 'styleclip_wrapper',
        'requires_target': False,
        'requires_attr': True,
        'default_attr': 'purple hair',
        'description': 'StyleCLIP for text-guided face manipulation',
    },
    'stargan': {
        'class': 'StarGANWrapper',
        'module': 'stargan_wrapper',
        'requires_target': False,
        'requires_attr': True,
        'default_attr': 'Blond_Hair',
        'description': 'StarGAN for multi-domain face attribute transfer',
    },
    'aggan': {
        'class': 'AGGANWrapper',
        'module': 'aggan_wrapper',
        'requires_target': False,
        'requires_attr': True,
        'default_attr': 'Blond_Hair',
        'description': 'AGGAN (Attention-Guided GAN) for face attribute manipulation',
    },
    'diffusionclip': {
        'class': 'DiffusionCLIPWrapper',
        'module': 'diffusionclip_wrapper',
        'requires_target': False,
        'requires_attr': False,
        'default_attr': None,
        'description': 'DiffusionCLIP for diffusion-based face manipulation (blond_hair)',
    },
    'psp_self': {
        'class': 'PSPSelfWrapper',
        'module': 'psp_self_wrapper',
        'requires_target': False,
        'requires_attr': False,
        'default_attr': None,
        'description': 'pSp Self-Reconstruction (encode -> decode)',
    },
    'simswap': {
        'class': 'SimSwapWrapper',
        'module': 'simswap_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        # 'default_target_path': '/home/df25/dftest/data/simswap/simswap_target_white/29550.jpg',
        'default_target_path': FFHQ_TARGET_69999,
        # 'default_target_path': '/mnt/SafeAILab/datasets/VGGFace2-HQ/original/VGGface2_None_norm_512_true_bygfpgan/n009279/0066_01.jpg',
        'description': 'SimSwap for AE-based face swapping',
        'extra_args': {
            'crop_size': 224,
            'gradient_mode': True,
        },
    },
    'blendface': {
        'class': 'BlendFaceWrapper',
        'module': 'blendface_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        # 'default_target_path': '/home/df25/dftest/data/simswap/simswap_target_white/29550.jpg',
        'default_target_path': FFHQ_TARGET_69999,
        # 'default_target_path': '/mnt/SafeAILab/datasets/VGGFace2-HQ/original/VGGface2_None_norm_512_true_bygfpgan/n009279/0066_01.jpg',
        'description': 'BlendFace for AE-based face swapping',
        'extra_args': {
            'gradient_mode': True,
        },
    },
    'psp_mix': {
        'class': 'PSPMixWrapper',
        'module': 'psp_mix_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        # 'default_target_path': '/home/df25/dftest/data/simswap/simswap_target_white/29550.jpg',
        'default_target_path': FFHQ_TARGET_69999,
        # 'default_target_path': '/mnt/SafeAILab/datasets/VGGFace2-HQ/original/VGGface2_None_norm_512_true_bygfpgan/n009279/0066_01.jpg',
        'description': 'pSp Style Mixing (source identity + target style)',
    },
    'diffface': {
        'class': 'DiffFaceWrapper',
        'module': 'diffface_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        # 'default_target_path': '/mnt/SafeAILab/datasets/VGGFace2-HQ/original/VGGface2_None_norm_512_true_bygfpgan/n009279/0066_01.jpg',
        'default_target_path': SIMSWAP_TARGET,
        'description': 'DiffFace for diffusion-based face swapping',
        'extra_args': {
            'iterations_num': 1,
            'gradient_mode': True,
        },
    },
    'diffswap': {
        'class': 'DiffSwapWrapper',
        'module': 'diffswap_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        # 'default_target_path': '/mnt/SafeAILab/datasets/VGGFace2-HQ/original/VGGface2_None_norm_512_true_bygfpgan/n009279/0066_01.jpg',
        'default_target_path': FFHQ_TARGET_69999,
        'description': 'DiffSwap for diffusion-based face swapping (DDIM inpainting)',
        'extra_args': {
            'ddim_steps': 200,
            'ddim_eta': 0.0,
            'tgt_scale': 0.01,
            'gradient_mode': True,
        },
    },
    'reface': {
        'class': 'REFaceWrapper',
        'module': 'reface_wrapper',
        'requires_target': True,
        'requires_attr': False,
        'default_attr': None,
        'description': 'REFace for diffusion-based face swapping (DDIM inpainting)',
        'extra_args': {
            'ddim_steps': 50,
            'ddim_eta': 0.0,
            'scale': 3.5,
            'gradient_mode': True,
        },
    },
}


# ============================================================================
# Wrapper Loading
# ============================================================================
def get_available_wrappers():
    """Return list of available generative models"""
    return list(WRAPPER_REGISTRY.keys())


def load_wrapper(wrapper_name, device='cuda', **kwargs):
    """
    Load generative model wrapper
    Args:
        wrapper_name: Name of wrapper to load
        device: torch device or device string
        **kwargs: Additional arguments for wrapper
    Returns:
        (wrapper instance, config dict)
    """
    if wrapper_name not in WRAPPER_REGISTRY:
        raise ValueError(f"Unknown wrapper: {wrapper_name}. Available: {list(WRAPPER_REGISTRY.keys())}")
    
    if isinstance(device, str):
        device = torch.device(device)
    
    config = WRAPPER_REGISTRY[wrapper_name]
    
    # Get wrappers directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wrappers_dir = os.path.join(current_dir, "wrappers")
    
    # Ensure wrappers_dir is at the front of sys.path
    if wrappers_dir in sys.path:
        sys.path.remove(wrappers_dir)
    sys.path.insert(0, wrappers_dir)
    
    # Remove cached module if exists to ensure fresh import from wrappers_dir
    module_name = config['module']
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Dynamic import
    module = __import__(module_name, fromlist=[config['class']])
    wrapper_class = getattr(module, config['class'])
    
    # Prepare init arguments
    init_kwargs = {'device': device}
    if 'extra_args' in config:
        init_kwargs.update(config['extra_args'])
    init_kwargs.update(kwargs)
    
    print(f"Loading {wrapper_name}: {config['description']}")
    wrapper = wrapper_class(**init_kwargs)
    
    return wrapper, config


# ============================================================================
# Disrupting Methods
# ============================================================================
def get_available_methods():
    """Return list of available disrupting methods"""
    return ["pgd"]  # Can extend with FGSM, C&W, etc.


# ============================================================================
# Statistics and Metrics
# ============================================================================
def compute_attack_stats(source, adv_source, clean_output, adv_output):
    """
    Compute attack statistics
    Returns: dict with metrics
    """
    delta = (adv_source - source).abs()
    diff = (clean_output - adv_output).abs()
    
    stats = {
        'perturbation_max': delta.max().item(),
        'perturbation_mean': delta.mean().item(),
        'perturbation_std': delta.std().item(),
        'output_diff_max': diff.max().item(),
        'output_diff_mean': diff.mean().item(),
        'output_diff_std': diff.std().item(),
    }
    
    return stats


def print_attack_stats(stats):
    """Print attack statistics in formatted way"""
    print("\nAttack Statistics:")
    print(f"  Perturbation Max: {stats['perturbation_max']:.4f}")
    print(f"  Perturbation Mean: {stats['perturbation_mean']:.4f}")
    print(f"  Perturbation Std: {stats['perturbation_std']:.4f}")
    print(f"  Output Difference Max: {stats['output_diff_max']:.4f}")
    print(f"  Output Difference Mean: {stats['output_diff_mean']:.4f}")
    print(f"  Output Difference Std: {stats['output_diff_std']:.4f}")


def print_stats(source_tensor, adv_tensor, clean_output, adv_output):
    """Print attack statistics (wrapper_test.py compatible)"""
    delta = (adv_tensor - source_tensor).abs()
    diff = (clean_output - adv_output).abs()
    
    print(f"\nAttack Statistics:")
    print(f"   Perturbation Max: {delta.max().item():.4f}")
    print(f"   Perturbation Mean: {delta.mean().item():.4f}")
    print(f"   Output Difference Max: {diff.max().item():.4f}")
    print(f"   Output Difference Mean: {diff.mean().item():.4f}")


def run_pgd_attack(wrapper, source_tensor, target_tensor=None, target_attr=None,
                   epsilon=0.05, alpha=0.01, steps=30, config=None):
    """
    Run PGD attack (wrapper_test.py compatible)
    """
    # Import here to avoid circular dependency
    from pathlib import Path
    import importlib.util
    
    # Find and import pgd_attack
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pgd_module_path = os.path.join(project_root, 'disrupting_methods', 'PGD', 'pgd.py')
    
    spec = importlib.util.spec_from_file_location("pgd_module", pgd_module_path)
    pgd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pgd_module)
    pgd_attack_fn = pgd_module.pgd_attack
    
    # Determine ref based on wrapper type
    if config['requires_target'] and target_tensor is not None:
        if hasattr(wrapper, 'set_target'):
            wrapper.set_target(target_tensor)
        ref = target_tensor  # Pass target_tensor as ref for PGD
    elif config['requires_attr'] and target_attr is not None:
        ref = target_attr
    else:
        ref = None
    
    print(f"\nRunning PGD Attack...")
    print(f"   Epsilon: {epsilon}")
    print(f"   Alpha: {alpha}")
    print(f"   Steps: {steps}")
    if config['requires_target']:
        print(f"   Ref: <target_tensor>")
    else:
        print(f"   Ref: {ref}")
    
    adv_tensor = pgd_attack_fn(
        wrapper=wrapper,
        x_source=source_tensor,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        ref=ref
    )
    
    return adv_tensor, ref

def run_fgsm_attack(wrapper, source_tensor, target_tensor=None, target_attr=None,
                   epsilon=0.05, config=None):
    """
    Run FGSM attack (wrapper_test.py compatible)
    """
    # Import here to avoid circular dependency
    from pathlib import Path
    import importlib.util
    
    # Find and import pgd_attack
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    fgsm_module_path = os.path.join(project_root, 'disrupting_methods', 'FGSM', 'fgsm.py')
    
    spec = importlib.util.spec_from_file_location("fgsm_module", fgsm_module_path)
    fgsm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fgsm_module)
    fgsm_attack_fn = fgsm_module.fgsm_attack
    
    # Determine ref based on wrapper type
    if config['requires_target'] and target_tensor is not None:
        if hasattr(wrapper, 'set_target'):
            wrapper.set_target(target_tensor)
        ref = target_tensor  # Pass target_tensor as ref for PGD
    elif config['requires_attr'] and target_attr is not None:
        ref = target_attr
    else:
        ref = None
    
    print(f"\nRunning FGSM Attack...")
    print(f"   Epsilon: {epsilon}")
    if config['requires_target']:
        print(f"   Ref: <target_tensor>")
    else:
        print(f"   Ref: {ref}")
    
    adv_tensor = fgsm_attack_fn(
        wrapper=wrapper,
        x_source=source_tensor,
        epsilon=epsilon,
        ref=ref
    )
    
    return adv_tensor, ref


def run_anti_attack(wrapper, source_tensor, target_tensor=None, target_attr=None,
                    epsilon=0.05, lr=1e-4, steps=500, config=None):
    """
    Run Anti-Forgery Lab attack
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    disrupting_methods_path = os.path.join(project_root, 'disrupting_methods')
    if disrupting_methods_path not in sys.path:
        sys.path.insert(0, disrupting_methods_path)
    
    from anti_forgery.anti_forgery import lab_attack
    anti_attack_fn = lab_attack
    
    # Determine ref based on wrapper type
    if config['requires_target'] and target_tensor is not None:
        if hasattr(wrapper, 'set_target'):
            wrapper.set_target(target_tensor)
        ref = target_tensor
    elif config['requires_attr'] and target_attr is not None:
        ref = target_attr
    else:
        ref = None
    
    print(f"\nRunning Anti-Forgery Lab Attack...")
    print(f"   Epsilon: {epsilon}")
    print(f"   Learning Rate: {lr}")
    print(f"   Steps: {steps}")
    
    adv_tensor = anti_attack_fn(
        wrapper=wrapper,
        X_nat=source_tensor,
        epsilon=epsilon,
        lr=lr,
        steps=steps,
        ref=ref
    )
    
    return adv_tensor, ref


def run_disrupting_attack(wrapper, source_tensor, target_tensor=None, target_attr=None,
                          epsilon=0.05, alpha=0.01, steps=30, config=None):
    """
    Run Disrupting Deepfake attack
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    disrupting_methods_path = os.path.join(project_root, 'disrupting_methods')
    if disrupting_methods_path not in sys.path:
        sys.path.insert(0, disrupting_methods_path)
    
    from disrupting_deepfake.disrupting_deepfake import disrupting_attack
    disrupting_attack_fn = disrupting_attack
    
    # Determine ref based on wrapper type
    if config['requires_target'] and target_tensor is not None:
        if hasattr(wrapper, 'set_target'):
            wrapper.set_target(target_tensor)
        ref = target_tensor
    elif config['requires_attr'] and target_attr is not None:
        ref = target_attr
    else:
        ref = None
    
    print(f"\nRunning Disrupting Deepfake Attack...")
    print(f"   Epsilon: {epsilon}")
    print(f"   Alpha: {alpha}")
    print(f"   Steps: {steps}")
    
    adv_tensor = disrupting_attack_fn(
        wrapper=wrapper,
        X_nat=source_tensor,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        ref=ref
    )
    
    return adv_tensor, ref


def run_df_rap_attack(wrapper, source_tensor, target_tensor=None, target_attr=None,
                      epsilon=0.05, alpha=0.01, steps=10, config=None,
                      ComG=None, ComG_woj=None, balance=1.0, use_comg=True):
    """
    Run DF_RAP attack with Complementary Generators
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    disrupting_methods_path = os.path.join(project_root, 'disrupting_methods')
    if disrupting_methods_path not in sys.path:
        sys.path.insert(0, disrupting_methods_path)
    
    from df_rap.df_rap import df_rap_attack
    df_rap_attack_fn = df_rap_attack
    
    # Determine ref based on wrapper type
    if config['requires_target'] and target_tensor is not None:
        if hasattr(wrapper, 'set_target'):
            wrapper.set_target(target_tensor)
        ref = target_tensor
    elif config['requires_attr'] and target_attr is not None:
        ref = target_attr
    else:
        ref = None
    
    print(f"\nRunning DF_RAP Attack...")
    print(f"   Epsilon: {epsilon}")
    print(f"   Alpha: {alpha}")
    print(f"   Steps: {steps}")
    print(f"   Use ComG: {use_comg}")
    if use_comg:
        print(f"   ComG: {'Loaded' if ComG is not None else 'None'}")
        print(f"   ComG_woj: {'Loaded' if ComG_woj is not None else 'None'}")
        if ComG is not None and ComG_woj is not None:
            print(f"   Balance: {balance}")
    
    adv_tensor = df_rap_attack_fn(
        wrapper=wrapper,
        X_nat=source_tensor,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        ref=ref,
        ComG=ComG,
        ComG_woj=ComG_woj,
        balance=balance,
        use_comg=use_comg
    )
    
    return adv_tensor, ref



# ============================================================================
# Validation
# ============================================================================
def validate_inputs(source_path, target_path, model_name, epsilon, alpha, steps):
    """
    Validate input arguments
    Raises: ValueError or FileNotFoundError
    """
    # Check if source image exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")
    
    # Check if target image is required
    config = WRAPPER_REGISTRY[model_name]
    if config['requires_target']:
        if target_path is None:
            raise ValueError(f"Model '{model_name}' requires target image")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target image not found: {target_path}")
    
    # Validate attack parameters
    if not (0 <= epsilon <= 1):
        raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")
    
    if not (0 <= alpha <= epsilon):
        raise ValueError(f"Alpha must be in [0, epsilon], got {alpha}")
    
    if steps < 1:
        raise ValueError(f"Steps must be >= 1, got {steps}")