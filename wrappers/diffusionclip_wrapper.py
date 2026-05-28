import sys
import os
import argparse
import yaml

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
diffusionclip_path = os.path.join(project_root, "deepfake_generators", "DiffusionCLIP")

if not os.path.exists(diffusionclip_path):
    raise FileNotFoundError(f"DiffusionCLIP path not found: {diffusionclip_path}")

if diffusionclip_path not in sys.path:
    sys.path.insert(0, diffusionclip_path)

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from configs.paths_config import MODEL_PATHS

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class DiffusionCLIPWrapper(nn.Module):
    SUPPORTED_ATTRS = [
        'blond_hair'
        ]
    
    def __init__(
        self,
        device="cuda",
        config_path=None,
        model_path=None,
        t_0=500,
        n_inv_step=40,
        n_test_step=40,
        sample_type='ddim',
        eta=0.0,
        model_ratio=1.0,
    ):

        super().__init__()
        self.device = device
        self.t_0 = t_0
        self.n_inv_step = n_inv_step
        self.n_test_step = n_test_step
        self.sample_type = sample_type
        self.eta = eta
        self.model_ratio = model_ratio
        
        if config_path is None:
            config_path = os.path.join(diffusionclip_path, "configs", "celeba.yml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.config.device = device
        
        self.model_var_type = self.config.model.var_type
        betas = get_beta_schedule(
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(device)
        self.num_timesteps = betas.shape[0]
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        
        self.models = []
        self.learn_sigma = False
        self.model_path = model_path
        
        self._load_models(model_path)
        
        self._cached_x_lat = None
        self._cached_x0 = None
        
    def _load_models(self, model_path=None):
        self.models = []
        
        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
            self.learn_sigma = False
            
            model_orig = DDPM(self.config)
            ckpt_orig = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            model_orig.load_state_dict(ckpt_orig)
            model_orig.to(self.device)
            model_orig = torch.nn.DataParallel(model_orig)
            model_orig.eval()
            for p in model_orig.parameters():
                p.requires_grad_(False)
            self.models.append(model_orig)
            print("Original CelebA_HQ diffusion model loaded.")
            
            if model_path is not None:
                model_ft = DDPM(self.config)
                ckpt_ft = torch.load(model_path, map_location=self.device)
                model_ft.load_state_dict(ckpt_ft)
                model_ft.to(self.device)
                model_ft = torch.nn.DataParallel(model_ft)
                model_ft.eval()
                for p in model_ft.parameters():
                    p.requires_grad_(False)
                self.models.append(model_ft)
                print(f"Fine-tuned model loaded from {model_path}")
            else:
                self.models.append(None)
                
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            self.learn_sigma = True
            
            model_orig = i_DDPM(self.config.data.dataset)
            ckpt_orig = torch.load(MODEL_PATHS[self.config.data.dataset], map_location=self.device)
            model_orig.load_state_dict(ckpt_orig)
            model_orig.to(self.device)
            model_orig = torch.nn.DataParallel(model_orig)
            model_orig.eval()
            for p in model_orig.parameters():
                p.requires_grad_(False)
            self.models.append(model_orig)
            print(f"Original {self.config.data.dataset} diffusion model loaded.")
            
            if model_path is not None:
                model_ft = i_DDPM(self.config.data.dataset)
                ckpt_ft = torch.load(model_path, map_location=self.device)
                model_ft.load_state_dict(ckpt_ft)
                model_ft.to(self.device)
                model_ft = torch.nn.DataParallel(model_ft)
                model_ft.eval()
                for p in model_ft.parameters():
                    p.requires_grad_(False)
                self.models.append(model_ft)
                print(f"Fine-tuned model loaded from {model_path}")
            else:
                self.models.append(None)
        else:
            raise ValueError(f"Dataset {self.config.data.dataset} not supported")
    
    def load_finetuned_model(self, model_path):
        self.model_path = model_path
        
        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model_ft = DDPM(self.config)
            ckpt_ft = torch.load(model_path, map_location=self.device)
            model_ft.load_state_dict(ckpt_ft)
            model_ft.to(self.device)
            model_ft = torch.nn.DataParallel(model_ft)
            model_ft.eval()
            for p in model_ft.parameters():
                p.requires_grad_(False)
        else:
            model_ft = i_DDPM(self.config.data.dataset)
            ckpt_ft = torch.load(model_path, map_location=self.device)
            model_ft.load_state_dict(ckpt_ft)
            model_ft.to(self.device)
            model_ft = torch.nn.DataParallel(model_ft)
            model_ft.eval()
            for p in model_ft.parameters():
                p.requires_grad_(False)
        
        if len(self.models) > 1:
            self.models[1] = model_ft
        else:
            self.models.append(model_ft)
        print(f"Fine-tuned model loaded from {model_path}")
    
    def preprocess(self, x):
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2.0 - 1.0
        
        target_size = self.config.data.image_size
        if x.shape[2] != target_size or x.shape[3] != target_size:
            x = torch.nn.functional.interpolate(
                x, size=(target_size, target_size), 
                mode='bicubic', align_corners=False
            )
        
        return x.clamp(-1, 1)
    
    def postprocess(self, x):

        return (x + 1.0) * 0.5
    
    def encode(self, x, require_grad=None):
        if require_grad is None:
            require_grad = x.requires_grad
            
        self._cached_x0 = x.clone().detach()
        
        seq_inv = np.linspace(0, 1, self.n_inv_step) * self.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        
        n = x.shape[0]
        x_t = x 
        
        if require_grad:
            for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                t = (torch.ones(n) * i).to(self.device)
                t_prev = (torch.ones(n) * j).to(self.device)
                
                x_t = denoising_step(
                    x_t, t=t, t_next=t_prev,
                    models=self.models,
                    logvars=self.logvar,
                    sampling_type='ddim',
                    b=self.betas,
                    eta=0,
                    learn_sigma=self.learn_sigma,
                    ratio=0 
                )
        else:
            with torch.no_grad():
                for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)
                    
                    x_t = denoising_step(
                        x_t, t=t, t_next=t_prev,
                        models=self.models,
                        logvars=self.logvar,
                        sampling_type='ddim',
                        b=self.betas,
                        eta=0,
                        learn_sigma=self.learn_sigma,
                        ratio=0
                    )
        
        self._cached_x_lat = x_t.clone().detach()
        return x_t
    
    def decode(self, x_lat, model_ratio=None):
        if model_ratio is None:
            model_ratio = self.model_ratio
        
        if self.models[1] is None and model_ratio > 0:
            raise RuntimeError("Fine-tuned model not loaded. Call load_finetuned_model() first.")
        
        seq_test = np.linspace(0, 1, self.n_test_step) * self.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        
        n = x_lat.shape[0]
        x = x_lat
        
        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
            t = (torch.ones(n) * i).to(self.device)
            t_next = (torch.ones(n) * j).to(self.device)
            
            x = denoising_step(
                x, t=t, t_next=t_next,
                models=self.models,
                logvars=self.logvar,
                sampling_type=self.sample_type,
                b=self.betas,
                eta=self.eta,
                learn_sigma=self.learn_sigma,
                ratio=model_ratio
            )
        
        return x
    
    def decode_with_progress(self, x_lat, model_ratio=None):
        if model_ratio is None:
            model_ratio = self.model_ratio
        
        if self.models[1] is None and model_ratio > 0:
            raise RuntimeError("Fine-tuned model not loaded. Call load_finetuned_model() first.")
        
        seq_test = np.linspace(0, 1, self.n_test_step) * self.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        
        n = x_lat.shape[0]
        x = x_lat.clone()
        
        with torch.no_grad():
            with tqdm(total=len(seq_test), desc="Generation") as pbar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)
                    
                    x = denoising_step(
                        x, t=t, t_next=t_next,
                        models=self.models,
                        logvars=self.logvar,
                        sampling_type=self.sample_type,
                        b=self.betas,
                        eta=self.eta,
                        learn_sigma=self.learn_sigma,
                        ratio=model_ratio
                    )
                    pbar.update(1)
        
        return x
    
    def forward(self, x, model_ratio=None, preprocess=True):
        if preprocess:
            x = self.preprocess(x)
        
        x_lat = self.encode(x)
        return self.decode(x_lat, model_ratio)
    
    def forward_with_grad(self, x, model_ratio=None, preprocess=True):
        if model_ratio is None:
            model_ratio = self.model_ratio
            
        if preprocess:
            x = self.preprocess(x)
        
        x_lat = self.encode(x, require_grad=True)
        return self.decode(x_lat, model_ratio)
    
    def reconstruct(self, x, preprocess=True):
        if preprocess:
            x = self.preprocess(x)
        
        x_lat = self.encode(x)
        return self.decode(x_lat, model_ratio=0.0)
    
    def manipulate(self, x, model_ratio=None, preprocess=True):
        if model_ratio is None:
            model_ratio = self.model_ratio
            
        if preprocess:
            x = self.preprocess(x)
        
        x_lat = self.encode(x)
        return self.decode(x_lat, model_ratio)
    
    def get_cached_latent(self):
        return self._cached_x_lat
    
    def get_cached_input(self):
        return self._cached_x0