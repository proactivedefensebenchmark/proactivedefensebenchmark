import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
blendface_root = os.path.join(project_root, "deepfake_generators", "BlendFace")
blendface_path = os.path.join(blendface_root, "swapping")

if not os.path.exists(blendface_path):
    raise FileNotFoundError(f"BlendFace path not found: {blendface_path}")

if blendface_root not in sys.path:
    sys.path.insert(0, blendface_root)

from swapping.blendswap import BlendSwap


class BlendFaceWrapper(nn.Module):
    def __init__(
        self,
        device='cuda',
        checkpoint=None,
        gradient_mode=True,
    ):
        super().__init__()
        self.device = device
        self.gradient_mode = gradient_mode
        
        if checkpoint is None:
            checkpoint = os.path.join(
                project_root, "deepfake_generators", "BlendFace", "checkpoints", "blendswap.pth"
            )
        
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"BlendFace checkpoint not found: {checkpoint}")
        
        print(f"Loading BlendFace from {checkpoint}")
        self.net = BlendSwap()
        self.net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.net.eval()
        self.net.to(device)
        
        for param in self.net.parameters():
            param.requires_grad = False
        
        self._cached_target = None
        
        print(f"BlendFace model loaded successfully")
    
    def set_target(self, target_img):
        if target_img.dim() == 3:
            target_img = target_img.unsqueeze(0)
        self._cached_target = target_img.clone().detach()

    def encode(self, img_id, ref=None):
        if ref is not None:
            self._cached_target = ref.clone().detach()
        img_01 = (img_id + 1) / 2
        img_112 = F.interpolate(img_01, size=(112, 112), mode='bilinear', align_corners=False)
        if self.gradient_mode:
            z_id = self.net.Z_e(img_112)
            z_id = F.normalize(z_id)
        else:
            with torch.no_grad():
                z_id = self.net.Z_e(img_112)
                z_id = F.normalize(z_id)
        return z_id

    def decode(self, z_id, ref=None):
        if ref is not None:
            img_att = ref
        elif self._cached_target is not None:
            img_att = self._cached_target
        else:
            raise ValueError("No target image provided.")
        
        img_att_01 = (img_att + 1) / 2
        
        E = self.net.E_ema
        G = self.net.G_ema
        mask_head = self.net.mask_head_ema
        
        feature_map_t = E(img_att_01)
        output_g = G(z_id, feature_map_t)
        mask = mask_head(feature_map_t[-1]).sigmoid()
        output = output_g * mask + img_att_01 * (1 - mask)
        
        return output
    
    def forward(self, x, ref=None, preprocess=False):
        if ref is not None:
            target = ref
            self._cached_target = ref.clone().detach()
        elif self._cached_target is not None:
            target = self._cached_target
        else:
            raise ValueError("No target image provided. Use ref argument or set_target() first.")
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        z_id = self.encode(x)
        img_fake = self.decode(z_id, target)
        img_fake = img_fake * 2 - 1 
        return img_fake

    @staticmethod
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    @staticmethod
    def norm(x):
        return x * 2 - 1