import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
simswap_path = os.path.join(project_root, "deepfake_generators", "SimSwap")

if not os.path.exists(simswap_path):
    raise FileNotFoundError(f"SimSwap path not found: {simswap_path}")

if simswap_path not in sys.path:
    sys.path.insert(0, simswap_path)

from models.fs_networks import Generator_Adain_Upsample


class SimSwapWrapper(nn.Module):    
    def __init__(
        self,
        device='cuda',
        crop_size=224,
        simswap_checkpoint=None,
        arcface_checkpoint=None,
        gradient_mode=True,
    ):
        super().__init__()
        self.device = device
        self.crop_size = crop_size
        self.gradient_mode = gradient_mode
        
        if simswap_checkpoint is None:
            simswap_checkpoint = os.path.join(
                simswap_path, "checkpoints", "people", "latest_net_G.pth"
            )
        if arcface_checkpoint is None:
            arcface_checkpoint = os.path.join(
                simswap_path, "arcface_model", "arcface_checkpoint.tar"
            )
        
        if not os.path.exists(simswap_checkpoint):
            raise FileNotFoundError(f"SimSwap checkpoint not found: {simswap_checkpoint}")
        if not os.path.exists(arcface_checkpoint):
            raise FileNotFoundError(f"ArcFace checkpoint not found: {arcface_checkpoint}")
        
        print(f"Loading SimSwap Generator from {simswap_checkpoint}")
        self.netG = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False
        )
        checkpoint = torch.load(simswap_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.netG.load_state_dict(checkpoint["state_dict"])
        else:
            self.netG.load_state_dict(checkpoint)
        self.netG.to(device)
        self.netG.eval()
        
        print(f"Loading ArcFace from {arcface_checkpoint}")
        arcface_ckpt = torch.load(arcface_checkpoint, map_location='cpu')
        self.netArc = arcface_ckpt
        self.netArc.to(device)
        self.netArc.eval()
        
        self._cached_target = None

        print(f"SimSwap model loaded successfully")
    
    def set_target(self, target_img):
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(0)
        self._cached_target = target_img.clone().detach()

    def encode(self, img_id):
        img_id_01 = (img_id + 1) / 2
        
        img_id_downsample = F.interpolate(img_id_01, size=(112, 112))
        
        if self.gradient_mode:
            latent_id = self.netArc(img_id_downsample)
        else:
            with torch.no_grad():
                latent_id = self.netArc(img_id_downsample)
        
        latent_id = latent_id / torch.linalg.norm(latent_id, dim=1, keepdim=True)
        
        return latent_id
    
    def decode(self, latent_id, ref=None):
        if ref is not None:
            img_att = ref
        elif self._cached_target is not None:
            img_att = self._cached_target
        else:
            raise ValueError("No target image provided.")
        
        img_att_01 = (img_att + 1) / 2
        img_fake = self.netG(img_att_01, latent_id)
        img_fake = img_fake * 2 - 1
        
        return img_fake
    
    def forward(self, x, ref=None, preprocess=False):
        if ref is not None:
            target = ref
            self._cached_target = ref.clone().detach()
        elif self._cached_target is not None:
            target = self._cached_target
        else:
            raise ValueError("No target image provided. Use ref argument or set_target() first.")

        latent_id = self.encode(x)
        img_fake = self.decode(latent_id, target)
        
        return img_fake

    @staticmethod
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    @staticmethod
    def norm(x):
        return x * 2 - 1