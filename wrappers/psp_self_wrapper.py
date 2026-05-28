import torch
import torch.nn as nn
import sys
import os
from argparse import Namespace

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
psp_root = os.path.join(project_root, "deepfake_generators", "pixel2style2pixel")

for p in [psp_root, os.path.join(psp_root, "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.psp import pSp
except ImportError as e:
    raise ImportError(
        f"Failed to import pSp from {psp_root}. "
        f"Make sure the pSp code exists. Original error: {e}"
    )

class PSPSelfWrapper(nn.Module):
    def __init__(
        self,
        device='cuda',
        checkpoint_path=None,
        output_size=1024,
        resize_outputs=True,
    ):

        super().__init__()
        self.device = device
        self.resize_outputs = resize_outputs
        self.output_size = output_size
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                psp_root, "checkpoints", "psp_ffhq_encode.pt"
            )
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"pSp checkpoint not found: {checkpoint_path}")
        
        print(f"Loading pSp from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        
        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = device
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        if 'output_size' not in opts:
            opts['output_size'] = output_size
        
        self.opts = Namespace(**opts)
        
        self.net = pSp(self.opts)
        self.net.to(device)
        self.net.eval()
        
        for param in self.net.parameters():
            param.requires_grad = False
        
        self._cached_latent = None
        self._cached_input = None
        
        print(f"pSp model loaded successfully")
        print(f"   Output size: {self.opts.output_size}")
        print(f"   Encoder type: {self.opts.encoder_type}")
        print(f"   N styles: {self.opts.n_styles}")
    
    def preprocess(self, x):
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2.0 - 1.0
        
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = torch.nn.functional.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=False
            )
        
        return x.clamp(-1, 1)
    
    def postprocess(self, x):
        return (x + 1.0) * 0.5
    
    def encode(self, x, ref=None):
        self._cached_input = x.clone().detach()
        codes = self.net.encoder(x)
        
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
        
        self._cached_latent = codes.clone().detach()
        return codes
    
    def decode(self, codes, ref=None):
        images, _ = self.net.decoder(
            [codes],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True
        )
        
        if self.resize_outputs:
            images = self.net.face_pool(images)
        
        return images
    
    def forward(self, x, ref=None, preprocess=True):
        if preprocess:
            x = self.preprocess(x)
        
        codes = self.encode(x)
        return self.decode(codes)
    
    def get_cached_latent(self):
        return self._cached_latent
    
    def get_cached_input(self):
        return self._cached_input