import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
styleclip_root = os.path.join(project_root, 'deepfake_generators')

for p in [styleclip_root, os.path.join(styleclip_root, 'styleclip')]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from styleclip import StyleClip
except ImportError as e:
    raise ImportError(
        f"Failed to import StyleClip from {styleclip_root}. "
        f"Make sure the StyleCLIP code and checkpoints exist. Original error: {e}"
    )

class StyleCLIPWrapper(nn.Module):
    SUPPORTED_ATTRS = [
        'angry', 'Beyonce', 'bobcut', 'curly hair',
        'Hilary Clinton', 'mohawk', 'purple hair',
        'surprised', 'trump', 'Mark Zuckerberg'
    ]

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        self.net = StyleClip(attributes=self.SUPPORTED_ATTRS, device=self.device)
        self.net.to(device)
        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad = False
            
        self.inverted_image_cache = None

    def encode(self, x: torch.Tensor):
        inverted_img, z = self.net.get_latent(x)
        self.inverted_image_cache = inverted_img.detach()
        
        return z

    def decode(self, z: torch.Tensor, target_attr: str = None, ref: str = None):
        # Handle ref parameter (for compatibility with pgd.py)
        if ref is not None and target_attr is None:
            target_attr = ref
        
        if target_attr is None:
            raise ValueError("target_attr or ref must be provided")
        
        if target_attr not in self.SUPPORTED_ATTRS:
            raise ValueError(f"Unsupported attribute: {target_attr}")
            
        return self.net.get_output(z, target_attr)
    
    def forward(self, x, target_attr: str):
        z = self.encode(x)
        return self.decode(z, target_attr)

    def get_inverted_image(self):
        return self.inverted_image_cache