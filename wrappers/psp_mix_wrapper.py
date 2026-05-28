import torch
import torch.nn as nn
import sys
import os
from argparse import Namespace

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
psp_root = os.path.join(project_root, "deepfake_generators", "pixel2style2pixel")

# Add pSp paths
for p in [psp_root, os.path.join(psp_root, "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# pSp's `models/psp.py` does `from configs.paths_config import model_paths`,
# but the benchmark root also ships a `configs` package (with DATASETS etc.,
# but no `model_paths`). Whichever is imported first wins the `configs` name
# in sys.modules, and the main one is imported at startup by
# attacks/batch_pgd_attack.py — so pSp's import would fail with
# "cannot import name 'model_paths'".
#
# Temporarily evict the main `configs.*` modules and put psp_root first on
# sys.path so pSp loads its OWN configs package, then restore the main one.
# pSp only needs `model_paths` at import time (it's solely dereferenced in
# psp.py's `checkpoint_path is None` branch, which the wrapper never hits),
# so restoring afterwards is safe and keeps the rest of the benchmark intact.
_saved_configs = {
    name: mod for name, mod in list(sys.modules.items())
    if name == "configs" or name.startswith("configs.")
}
for name in _saved_configs:
    del sys.modules[name]

if psp_root in sys.path:
    sys.path.remove(psp_root)
sys.path.insert(0, psp_root)

try:
    from models.psp import pSp
except ImportError as e:
    raise ImportError(
        f"Failed to import pSp from {psp_root}. "
        f"Make sure the pSp code exists. Original error: {e}"
    )
finally:
    # Drop pSp's configs package and restore the benchmark's configs package
    # so subsequent `from configs.paths_config import ...` resolves correctly.
    for name in [n for n in list(sys.modules)
                 if n == "configs" or n.startswith("configs.")]:
        del sys.modules[name]
    sys.modules.update(_saved_configs)


class PSPMixWrapper(nn.Module):
    DEFAULT_LATENT_MASK = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    def __init__(
        self,
        device='cuda',
        checkpoint_path=None,
        output_size=1024,
        resize_outputs=True,
        latent_mask=None,
        mix_alpha=1.0,
    ):
        super().__init__()
        self.device = device
        self.resize_outputs = resize_outputs
        self.output_size = output_size
        self.mix_alpha = mix_alpha
        
        if latent_mask is None:
            self.latent_mask = self.DEFAULT_LATENT_MASK.copy()
        else:
            self.latent_mask = latent_mask
        
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
        
        self._cached_source_latent = None
        self._cached_target_latent = None
        self._cached_mixed_latent = None
        self._cached_target_img = None
        
        print(f"✅ pSp Style Mixing model loaded successfully")
        print(f"   Output size: {self.opts.output_size}")
        print(f"   Encoder type: {self.opts.encoder_type}")
        print(f"   N styles: {self.opts.n_styles}")
        print(f"   Latent mask: {self.latent_mask}")
        print(f"   Mix alpha: {self.mix_alpha}")
    
    def set_latent_mask(self, latent_mask):
        self.latent_mask = latent_mask
        print(f"Updated latent mask: {self.latent_mask}")
    
    def set_mix_alpha(self, mix_alpha):
        self.mix_alpha = mix_alpha
        print(f"Updated mix alpha: {self.mix_alpha}")
    
    def set_target(self, target_img):
        self._cached_target_img = target_img.clone().detach()
        
        with torch.no_grad():
            self._cached_target_latent = self._encode_single(target_img)
        
        print(f"✅ Target image cached (latent shape: {self._cached_target_latent.shape})")
    
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
    
    def _encode_single(self, x):
        codes = self.net.encoder(x)
        
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
        
        return codes
    
    def _mix_latents(self, source_latent, target_latent):
        mixed_latent = source_latent.clone()
        
        for layer_idx in self.latent_mask:
            if self.mix_alpha == 1.0:
                mixed_latent[:, layer_idx, :] = target_latent[:, layer_idx, :]
            else:
                mixed_latent[:, layer_idx, :] = (
                    (1 - self.mix_alpha) * source_latent[:, layer_idx, :] +
                    self.mix_alpha * target_latent[:, layer_idx, :]
                )
        
        return mixed_latent
    
    def encode(self, x, ref=None):
        source_latent = self._encode_single(x)
        self._cached_source_latent = source_latent.clone().detach()
        
        if ref is not None:
            with torch.no_grad():
                target_latent = self._encode_single(ref)
                self._cached_target_latent = target_latent.clone().detach()
                self._cached_target_img = ref.clone().detach()
        elif self._cached_target_latent is not None:
            target_latent = self._cached_target_latent
        else:
            print("⚠️ No target provided, using self-reconstruction mode")
            self._cached_mixed_latent = source_latent.clone().detach()
            return source_latent
        
        mixed_latent = self._mix_latents(source_latent, target_latent)
        self._cached_mixed_latent = mixed_latent.clone().detach()
        
        return mixed_latent
    
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
            if ref is not None:
                ref = self.preprocess(ref)
        
        codes = self.encode(x, ref=ref)
        return self.decode(codes)
    
    def get_cached_source_latent(self):
        return self._cached_source_latent
    
    def get_cached_target_latent(self):
        return self._cached_target_latent
    
    def get_cached_mixed_latent(self):
        return self._cached_mixed_latent
    
    def get_cached_target_img(self):
        return self._cached_target_img