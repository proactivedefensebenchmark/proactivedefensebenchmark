"""
DF_RAP Attack Implementation
Based on original DF_RAP code with wrapper interface
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add net folder to path for importing ComGenerator
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class LinfPGDAttack(object):
    """
    PGD Attack with Complementary Generator (ComG)
    Based on original DF_RAP implementation
    """
    def __init__(self, wrapper=None, ComG=None, device=None, epsilon=0.05, k=10, a=0.01):
        """
        Args:
            wrapper: Deepfake model wrapper
            ComG: Complementary Generator (required)
            device: Device
            epsilon: Perturbation budget
            k: Number of iterations
            a: Step size
        """
        self.wrapper = wrapper
        self.ComG = ComG
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = F.mse_loss
        self.device = device
        self.rand = True  # PGD with random initialization
    
    def perturb(self, X_nat, ref=None):
        """
        Generate adversarial perturbation
        
        Args:
            X_nat: Natural input image
            ref: Reference (target image or attribute)
        
        Returns:
            X: Adversarial image
            perturbation: X - X_nat
        """
        # Random initialization
        if self.rand:
            X = X_nat.clone().detach() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')
            ).to(self.device)
        else:
            X = X_nat.clone().detach()
        
        # PGD iterations
        for i in range(self.k):
            X.requires_grad = True
            
            # Zero gradients
            if hasattr(self.wrapper, 'zero_grad'):
                self.wrapper.zero_grad()
            if hasattr(self.ComG, 'zero_grad'):
                self.ComG.zero_grad()
            
            # Forward through ComG and wrapper
            X_comg = self.ComG(X).contiguous()
            encoded = self.wrapper.encode(X_comg)
            output = self.wrapper.decode(encoded, ref=ref)
            
            # Get clean output for comparison
            with torch.no_grad():
                X_nat_comg = self.ComG(X_nat).contiguous()
                encoded_nat = self.wrapper.encode(X_nat_comg)
                output_nat = self.wrapper.decode(encoded_nat, ref=ref)
            
            # Loss: MSE between outputs
            loss = self.loss_fn(output, output_nat)
            loss.backward()
            
            # Update
            grad = X.grad
            X_adv = X + self.a * grad.sign()
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1.0, max=1.0).detach()
        
        return X, X - X_nat


def df_rap_attack(wrapper, X_nat, epsilon=0.05, alpha=0.01, steps=10, 
                  ref=None, ComG=None):
    """
    DF_RAP attack using Complementary Generator
    
    Args:
        wrapper: Deepfake model wrapper
        X_nat: Natural input image tensor [-1, 1]
        epsilon: Perturbation budget
        alpha: Step size (a)
        steps: Number of iterations (k)
        ref: Reference (target image or attribute)
        ComG: Complementary Generator (required)
    
    Returns:
        X_adv: Adversarial image tensor
    """
    if ComG is None:
        raise ValueError("ComG is required for DF_RAP attack")
    
    device = X_nat.device
    
    # Create attack instance
    attack = LinfPGDAttack(
        wrapper=wrapper,
        ComG=ComG,
        device=device,
        epsilon=epsilon,
        k=steps,
        a=alpha
    )
    
    # Generate adversarial example
    X_adv, _ = attack.perturb(X_nat, ref=ref)
    
    return X_adv


def df_rap_attack_legacy(wrapper, X_nat, epsilon=0.05, alpha=0.01, steps=10,
                         ref=None, faketype="simswap", model=None, ComG=None):
    """
    Legacy DF_RAP attack with original interface
    For compatibility with original DF_RAP code
    
    Args:
        wrapper: Can be None (uses model directly)
        model: Original deepfake model (StarGAN or SimSwap)
        faketype: "StarGAN" or "simswap"
        ... other args same as df_rap_attack
    """
    device = X_nat.device
    
    # Random initialization
    X = X_nat.clone().detach() + torch.tensor(
        np.random.uniform(-epsilon, epsilon, X_nat.shape).astype('float32')
    ).to(device)
    
    for i in range(steps):
        X.requires_grad = True
        
        # Forward pass based on faketype
        if faketype == "StarGAN":
            if use_comg and ComG is not None:
                if ComG_woj is not None:
                    output1, _ = model.features(ComG(X), ref)
                    output2, _ = model.features(ComG_woj(X), ref)
                    output = balance * output1 + (1.0 - balance) * output2
                else:
                    output, _ = model.features(ComG(X), ref)
            else:
                output, _ = model.features(X, ref)
                
        elif faketype == "simswap":
            if use_comg and ComG is not None:
                if ComG_woj is not None:
                    # Process with ComG
                    img_id_downsample1 = F.interpolate(ComG(X), size=(112, 112))
                    latent_id1 = model.netArc(img_id_downsample1)
                    latent_id1 = latent_id1 / torch.norm(latent_id1, p=2, dim=1, keepdim=True)
                    output1 = model(ComG(X), ref, latent_id1, latent_id1, True)
                    
                    # Process with ComG_woj
                    img_id_downsample2 = F.interpolate(ComG_woj(X), size=(112, 112))
                    latent_id2 = model.netArc(img_id_downsample2)
                    latent_id2 = latent_id2 / torch.norm(latent_id2, p=2, dim=1, keepdim=True)
                    output2 = model(ComG_woj(X), ref, latent_id2, latent_id2, True)
                    
                    output = balance * output1 + (1.0 - balance) * output2
                else:
                    img_id_downsample = F.interpolate(ComG(X), size=(112, 112))
                    latent_id = model.netArc(img_id_downsample)
                    latent_id = latent_id / torch.norm(latent_id, p=2, dim=1, keepdim=True)
                    output = model(ComG(X), ref, latent_id, latent_id, True)
            else:
                img_id_downsample = F.interpolate(X, size=(112, 112))
                latent_id = model.netArc(img_id_downsample)
                latent_id = latent_id / torch.norm(latent_id, p=2, dim=1, keepdim=True)
                output = model(X, ref, latent_id, latent_id, True)
        
        # Get clean output
        with torch.no_grad():
            if faketype == "StarGAN":
                gen_clean, _ = model.features(X_nat, ref)
            elif faketype == "simswap":
                img_id_clean = F.interpolate(X_nat, size=(112, 112))
                latent_clean = model.netArc(img_id_clean)
                latent_clean = latent_clean / torch.norm(latent_clean, p=2, dim=1, keepdim=True)
                gen_clean = model(X_nat, ref, latent_clean, latent_clean, True)
        
        # Loss
        loss = F.mse_loss(output, gen_clean)
        loss.backward()
        
        grad = X.grad
        X_adv = X + alpha * grad.sign()
        eta = torch.clamp(X_adv - X_nat, min=-epsilon, max=epsilon)
        X = torch.clamp(X_nat + eta, min=-1.0, max=1.0).detach()
    
    return X
