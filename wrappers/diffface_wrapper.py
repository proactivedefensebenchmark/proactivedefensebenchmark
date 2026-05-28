import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
diffface_root = os.path.join(project_root, "deepfake_generators", "DiffFace")

for p in [project_root, diffface_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from deepfake_generators.DiffFace.optimization.image_editor import ImageEditor
except ImportError as e:
    raise ImportError(
        f"Failed to import DiffFace modules from {diffface_root}. "
        f"Make sure the DiffFace code exists. Original error: {e}"
    )

class DiffFaceWrapper(nn.Module):
    def __init__(
        self,
        device='cuda',
        gpu_id=0,
        skip_timesteps=30,
        timestep_respacing="50",
        iterations_num=1,
        masking_threshold=30,
        ddim=False,
        aug_num=8,
        gradient_mode=True,
    ):
        super().__init__()
        self.device = device
        self.gradient_mode = gradient_mode
        
        args = argparse.Namespace(
            batch_size=1,
            skip_timesteps=skip_timesteps,
            ddim=ddim,
            timestep_respacing=timestep_respacing,
            enforce_background=True,
            aug_num=aug_num,
            seed=404,
            gpu_id=gpu_id,
            output_path="output",
            output_file="output.png",
            iterations_num=iterations_num,
            masking_threshold=masking_threshold,
        )
        
        original_cwd = os.getcwd()
        os.chdir(diffface_root)
        
        try:
            print(f"Loading DiffFace models...")
            self.image_editor = ImageEditor(args)
            print("DiffFace model loaded successfully")
            print(f"   Skip timesteps: {skip_timesteps}")
            print(f"   Timestep respacing: {timestep_respacing}")
            print(f"   Iterations: {iterations_num}")
        finally:
            os.chdir(original_cwd)
        
        self._cached_target_data = None
        self._cached_source_data = None
        self._cached_id_embedding = None
        
        self.landmark_indices = {
            'l_eye': list(range(36, 42)),
            'r_eye': list(range(42, 48)),
            'nose': list(range(27, 36)),
            'mouth': list(range(48, 68)),
        }
    
    def _tensor_to_numpy(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return img
    
    def set_target(self, target_img):
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(0)
        self._cached_target_data = target_img.clone().detach()
        print(f"Target image cached")
    
    def preprocess(self, x):
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2.0 - 1.0
        
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        return x.clamp(-1, 1)
    
    def encode(self, x, ref=None):
        if ref is not None:
            target = ref
            self._cached_target_data = ref.clone().detach()
        elif self._cached_target_data is not None:
            target = self._cached_target_data
        else:
            raise ValueError("No target image provided. Use ref argument or set_target() first.")
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        
        self._cached_source_data = x
        self._cached_target_data = target
        
        img = (x + 1) / 2
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = F.interpolate(img, (112, 112))
        
        if self.gradient_mode:
            id_embedding = self.image_editor.netArc(img)
            id_embedding = F.normalize(id_embedding, p=2, dim=1)
        else:
            with torch.no_grad():
                id_embedding = self.image_editor.netArc(img)
                id_embedding = F.normalize(id_embedding, p=2, dim=1)
        
        self._cached_id_embedding = id_embedding
        
        return id_embedding
    
    def decode(self, latent, ref=None):
        if self._cached_source_data is None or self._cached_target_data is None:
            raise ValueError("Must call encode() first")
        
        source = self._cached_source_data
        target = self._cached_target_data
        
        with torch.no_grad():
            result = self.image_editor.decode_one(source.detach(), target.detach())
        
        if self.gradient_mode and latent.requires_grad:
            source_resized = source
            if source.shape[2:] != result.shape[2:]:
                source_resized = F.interpolate(source, size=result.shape[2:], mode='bilinear', align_corners=False)

            source_proxy = source_resized.mean(dim=[2, 3], keepdim=True).expand_as(result)
            result = result.detach() + (source_proxy - source_proxy.detach())
        
        result = torch.clamp(result, min=-1.0, max=1.0)
        
        return result
    
    def forward(self, x, ref=None, preprocess=True):
        if preprocess:
            x = self.preprocess(x)
            if ref is not None:
                ref = self.preprocess(ref)
        
        latent = self.encode(x, ref=ref)
        return self.decode(latent)
    
    def get_id_embedding(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        img = (x + 1) / 2
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = F.interpolate(img, (112, 112))
        
        with torch.no_grad():
            id_embedding = self.image_editor.netArc(img)
            id_embedding = F.normalize(id_embedding, p=2, dim=1)
        
        return id_embedding
    
    def compute_id_loss(self, x1, x2):
        id1 = self.get_id_embedding(x1)
        id2 = self.get_id_embedding(x2)
        
        cosine_sim = F.cosine_similarity(id1, id2, dim=1)
        return 1 - cosine_sim.mean()


def test_diffface_wrapper():
    print("=" * 50)
    print("Testing DiffFace Wrapper")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    wrapper = DiffFaceWrapper(device=device, gradient_mode=True)   
    source = torch.randn(1, 3, 256, 256).to(device).clamp(-1, 1)
    target = torch.randn(1, 3, 256, 256).to(device).clamp(-1, 1)
    
    print("\n--- Test 1: Forward Pass ---")
    with torch.no_grad():
        output = wrapper(source, ref=target, preprocess=False)
    print(f"Input shape: {source.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    print("\n--- Test 2: Encode/Decode ---")
    wrapper.set_target(target)
    latent = wrapper.encode(source, ref=target)
    print(f"Latent shape: {latent.shape}")
    
    output = wrapper.decode(latent)
    print(f"Output shape: {output.shape}")
    
    print("\n--- Test 3: Gradient Flow ---")
    source_grad = source.clone().detach().requires_grad_(True)
    latent = wrapper.encode(source_grad, ref=target)
    output = wrapper.decode(latent)
    
    loss = output.sum()
    loss.backward()
    
    if source_grad.grad is not None:
        grad_mean = source_grad.grad.abs().mean().item()
        print(f"Gradient computed! Mean abs gradient: {grad_mean:.6f}")
    else:
        print("No gradient computed")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_diffface_wrapper()
