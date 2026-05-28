import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as utils
from PIL import Image
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
diffae_path = os.path.join(project_root, "deepfake_generators", "diffae")
generators_path = os.path.join(project_root, "deepfake_generators")

psp_configs_path = os.path.join(project_root, "deepfake_generators", "pixel2style2pixel", "configs")
for key in list(sys.modules.keys()):
    if key == 'configs' or key.startswith('configs.'):
        module = sys.modules.get(key)
        if module and hasattr(module, '__file__') and module.__file__:
            if 'pixel2style2pixel' in module.__file__:
                del sys.modules[key]

psp_path = os.path.join(project_root, "deepfake_generators", "pixel2style2pixel")
if psp_path in sys.path:
    sys.path.remove(psp_path)

for p in [project_root, generators_path, diffae_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from diffae.data_utils import CelebAHQDataset
from diffae.templates import ffhq256_autoenc
from diffae.templates_cls import ffhq256_autoenc_cls
from diffae.experiment import LitModel
from diffae.dataset import CelebAttrDataset


class DiffAEWrapper(torch.nn.Module):
    SUPPORTED_ATTRS = CelebAttrDataset.id_to_cls
    ATTR2IDX = CelebAttrDataset.cls_to_id
    
    def __init__(self, device='cuda', default_attr='Smiling', sampling_step=10, **kwargs):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.sampling_step = sampling_step

        conf = ffhq256_autoenc()
        cls_conf = ffhq256_autoenc_cls()
        diffae_net = LitModel(conf=conf, cls_conf=cls_conf).to(device).eval()

        if default_attr in self.ATTR2IDX:
            self.cls_id = self.ATTR2IDX[default_attr]
        else:
            self.cls_id = self.ATTR2IDX['Smiling']

        self.net = diffae_net
        self._cached_input = None
        print(f"DiffAE model loaded successfully on {device}")
        print(f"   Default attribute: {default_attr} (cls_id={self.cls_id})")
        print(f"   Sampling steps (T): {self.sampling_step}")

    def set_attribute(self, attr_name):
        """Set the target attribute for manipulation"""
        if attr_name in self.ATTR2IDX:
            self.cls_id = self.ATTR2IDX[attr_name]
        else:
            print(f"Warning: {attr_name} not in supported attributes, using Smiling")
            self.cls_id = self.ATTR2IDX['Smiling']

    def encode(self, x: torch.Tensor):
        self._cached_input = x.clone()
        cond = self.net.encode(x)
        z_src = self.net.get_latent(cond)
        return z_src

    def decode(self, z: torch.Tensor, ref=None, target_attr=None):
        if target_attr is not None and isinstance(target_attr, str):
            if target_attr in self.ATTR2IDX:
                cls_id = self.ATTR2IDX[target_attr]
            else:
                cls_id = self.cls_id
        else:
            cls_id = self.cls_id

        if ref is None:
            ref = self._cached_input
        if ref is None:
            raise ValueError("No reference image provided. Call encode first or provide ref.")

        if isinstance(ref, str):
            if ref in self.ATTR2IDX:
                cls_id = self.ATTR2IDX[ref]
            ref = self._cached_input
            if ref is None:
                raise ValueError("No cached input available for attribute-based manipulation.")

        import math
        xT = self.net.get_xt(ref, self.net.encode(ref))
        cond = z + 0.3 * math.sqrt(512) * torch.nn.functional.normalize(
            self.net.classifier.weight[cls_id][None, :], dim=1
        )
        cond = self.net.denormalize(cond)
        return self.net.render(xT, cond, T=self.sampling_step)

    def forward(self, x: torch.Tensor, target_attr=None, ref=None, preprocess=True):
        z = self.encode(x)
        return self.decode(z, ref=x, target_attr=target_attr)