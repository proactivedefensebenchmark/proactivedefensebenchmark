import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
aggan_path = os.path.join(project_root, "deepfake_generators", "AGGAN")

if not os.path.exists(aggan_path):
    raise FileNotFoundError(f"AGGAN path not found: {aggan_path}")

if aggan_path not in sys.path:
    sys.path.append(aggan_path)

from model import Generator


class AGGANWrapper(nn.Module):    
    SUPPORTED_ATTRS = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    ATTR2IDX = {attr: i for i, attr in enumerate(SUPPORTED_ATTRS)}
    
    HAIR_COLOR_ATTRS = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    
    ALL_CELEBA_ATTRS = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
        'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    
    def __init__(
        self, 
        device='cuda',
        image_size=256,
        c_dim=5,
        g_conv_dim=64,
        g_repeat_num=6,
        selected_attrs=None,
        model_save_dir=None,
        test_iters=200000,
        attr_path=None,
        default_c_org=None
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.c_dim = c_dim
        self.g_conv_dim = g_conv_dim
        self.g_repeat_num = g_repeat_num
        self.test_iters = test_iters
        
        if selected_attrs is not None:
            self.selected_attrs = selected_attrs
            self.SUPPORTED_ATTRS = selected_attrs
            self.ATTR2IDX = {attr: i for i, attr in enumerate(selected_attrs)}
            self.c_dim = len(selected_attrs)
        else:
            self.selected_attrs = self.SUPPORTED_ATTRS
        
        if model_save_dir is None:
            model_save_dir = os.path.join(aggan_path)
        if not os.path.isabs(model_save_dir):
            model_save_dir = os.path.join(project_root, model_save_dir)
        self.model_save_dir = model_save_dir
        
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        
        G_path = os.path.join(self.model_save_dir, 'checkpoints', f'{test_iters}-G.ckpt')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location='cpu'))
            print(f"Loaded AGGAN Generator from {G_path}")
        else:
            raise FileNotFoundError(f"Generator checkpoint not found: {G_path}")
        
        self.G.to(device)
        
        for param in self.G.parameters():
            param.requires_grad = False
            
        self._cached_input = None
        self._cached_attention_mask = None
        self._cached_content_mask = None
        
        if attr_path is None:
            attr_path = os.path.join(aggan_path, 'data', 'celeba', 'list_attr_celeba.txt')
        self.attr_path = attr_path
        self.attr_dict = {}
        if os.path.exists(attr_path):
            self._load_attr_file(attr_path)
            print(f"Loaded attribute file from {attr_path}")
        
        if default_c_org is not None:
            self.default_c_org = torch.FloatTensor(default_c_org)
        else:
            self.default_c_org = torch.FloatTensor([0, 0, 1, 1, 1])
    
    def _load_attr_file(self, attr_path):
        lines = [line.rstrip() for line in open(attr_path, 'r')]
        all_attr_names = lines[1].split()
        
        all_attr2idx = {attr: i for i, attr in enumerate(all_attr_names)}
        
        for line in lines[2:]:
            split = line.split()
            filename = split[0]
            values = split[1:]
            
            label = []
            for attr_name in self.selected_attrs:
                if attr_name in all_attr2idx:
                    idx = all_attr2idx[attr_name]
                    label.append(1.0 if values[idx] == '1' else 0.0)
                else:
                    label.append(0.0)
            
            self.attr_dict[filename] = label
    
    def get_attr_by_filename(self, filename):
        if '/' in filename:
            filename = os.path.basename(filename)
        
        if filename in self.attr_dict:
            return torch.FloatTensor([self.attr_dict[filename]])
        else:
            return self.default_c_org.unsqueeze(0)

    def get_hair_color_indices(self):
        hair_indices = []
        for i, attr_name in enumerate(self.selected_attrs):
            if attr_name in self.HAIR_COLOR_ATTRS:
                hair_indices.append(i)
        return hair_indices

    def create_target_label(self, c_org, target_attr):
        if isinstance(target_attr, str):
            if target_attr not in self.ATTR2IDX:
                raise ValueError(f"Unsupported attribute: {target_attr}. Supported: {self.selected_attrs}")
            target_idx = self.ATTR2IDX[target_attr]
        else:
            target_idx = target_attr
            
        c_trg = c_org.clone()
        hair_color_indices = self.get_hair_color_indices()
        
        if target_idx in hair_color_indices:
            c_trg[:, target_idx] = 1
            for j in hair_color_indices:
                if j != target_idx:
                    c_trg[:, j] = 0
        else:
            c_trg[:, target_idx] = (c_trg[:, target_idx] == 0).float()
            
        return c_trg.to(self.device)

    def create_all_target_labels(self, c_org):
        c_trg_list = []
        for i in range(self.c_dim):
            c_trg = self.create_target_label(c_org, i)
            c_trg_list.append(c_trg)
        return c_trg_list

    def encode(self, x: torch.Tensor):
        self._cached_input = x.clone()
        return x

    def decode(self, x: torch.Tensor, target_attr=None, c_org=None, ref=None, return_attention=False):
        batch_size = x.size(0)
        
        if ref is not None and target_attr is None:
            target_attr = ref
        
        if isinstance(target_attr, str):
            if target_attr not in self.ATTR2IDX:
                raise ValueError(f"Unsupported attribute: {target_attr}. Supported: {self.selected_attrs}")
            target_idx = self.ATTR2IDX[target_attr]
        else:
            target_idx = target_attr
        
        if c_org is None:
            c_org = self.default_c_org.unsqueeze(0).expand(batch_size, -1).clone()
        
        c_trg = self.create_target_label(c_org, target_attr)
        
        with torch.enable_grad():
            result, attention_mask, content_mask = self.G(x, c_trg)
            self._cached_attention_mask = attention_mask
            self._cached_content_mask = content_mask
            
            if return_attention:
                return result, attention_mask, content_mask
            return result

    def forward(self, x: torch.Tensor, target_attr, c_org=None, return_attention=False):
        self._cached_input = x.clone()
        return self.decode(x, target_attr, c_org, return_attention)

    def transform_with_label(self, x: torch.Tensor, c_trg: torch.Tensor, return_attention=False):
        c_trg = c_trg.to(self.device)
        with torch.enable_grad():
            result, attention_mask, content_mask = self.G(x, c_trg)
            self._cached_attention_mask = attention_mask
            self._cached_content_mask = content_mask
            
            if return_attention:
                return result, attention_mask, content_mask
            return result

    def get_cached_input(self):
        return self._cached_input
    
    def get_cached_attention_mask(self):
        return self._cached_attention_mask
    
    def get_cached_content_mask(self):
        return self._cached_content_mask

    @staticmethod
    def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    @staticmethod
    def norm(x):
        return x * 2 - 1
