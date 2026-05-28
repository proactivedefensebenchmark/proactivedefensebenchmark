"""
Disrupting Deepfake Attack
Based on disrupting_deepfake.py from deep_fake_benchmark_test
"""
import torch
import torch.nn.functional as F
import numpy as np
from . import smoothing


def disrupting_attack(wrapper, X_nat, epsilon=0.05, alpha=0.01, steps=30, ref=None, rand=True):
    device = X_nat.device
    
    # Expand ref if batch size mismatch
    if ref is not None and isinstance(ref, torch.Tensor):
        if ref.shape[0] == 1 and X_nat.shape[0] > 1:
            ref = ref.expand(X_nat.shape[0], -1, -1, -1).contiguous()
    
    # Get clean output
    with torch.no_grad():
        encoded_src = wrapper.encode(X_nat)
        decoded_src = wrapper.decode(encoded_src, ref=ref)
    
    # Initialize with random perturbation
    if rand:
        x_adv = X_nat.clone().detach() + torch.tensor(
            np.random.uniform(-epsilon, epsilon, X_nat.shape).astype('float32')
        ).to(device)
        x_adv = torch.clamp(x_adv, -1, 1)
    else:
        x_adv = X_nat.clone().detach()
    
    # Smoothing parameters
    ks_gauss = 11
    ks_avg = 3
    sig = 1.0
    blur_type = 1  # 1: Gaussian, 2: Average
    
    for it in range(steps):
        # Select smoothing type
        if blur_type == 1:
            preproc = smoothing.GaussianSmoothing2D(
                sigma=sig, channels=3, kernel_size=ks_gauss
            ).to(device)
        else:
            preproc = smoothing.AverageSmoothing2D(
                channels=3, kernel_size=ks_avg
            ).to(device)
        
        x_adv.requires_grad_(True)
        
        # Apply smoothing
        x_smoothed = preproc(x_adv)
        
        # Get output
        encoded_adv = wrapper.encode(x_smoothed)
        decoded_adv = wrapper.decode(encoded_adv, ref=ref)
        
        # Loss: maximize difference
        loss = F.mse_loss(decoded_adv, decoded_src)
        
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        
        loss.backward()
        grad = x_adv.grad
        
        # Gradient ascent
        x_next = x_adv + alpha * grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(x_next - X_nat, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(X_nat + eta, min=-1, max=1).detach()
        
        # Update smoothing parameters
        if blur_type == 1:
            sig += 0.5
            if sig > 3.2:
                blur_type = 2
                sig = 1.0
        elif blur_type == 2:
            ks_avg += 2
            if ks_avg >= 11:
                blur_type = 1
                ks_avg = 3
    
    return x_adv
