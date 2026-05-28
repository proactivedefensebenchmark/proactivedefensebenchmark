# import torch
# import torch.nn as nn
# from .utils import *


# def lab_attack(wrapper, X_nat, epsilon=0.05, lr=1e-4, steps=500, ref=None):
#     device = X_nat.device
#     criterion = nn.MSELoss().to(device)
    
#     wrapper.eval()
    
#     # Freeze all BatchNorm/InstanceNorm running stats (only for StarGAN)
#     if 'StarGAN' in wrapper.__class__.__name__:
#         for module in wrapper.modules():
#             if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
#                 module.eval()
#                 module.track_running_stats = False
    
#     pert_a = torch.zeros(
#         X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3],
#         device=device, requires_grad=True
#     )
#     optimizer = torch.optim.Adam([pert_a], lr=lr, betas=(0.9, 0.999))
    
#     # Convert to [0, 1] for Lab conversion
#     X = denorm(X_nat.clone())
    
#     # Get original output ONCE (not every iteration)
#     with torch.no_grad():
#         encoded_src = wrapper.encode(X_nat)
#         decoded_src = wrapper.decode(encoded_src, ref=ref)
    
#     for i in range(steps):
#         # Convert to Lab space (from ORIGINAL X each time)
#         X_lab = rgb2lab(X).to(device)
        
#         # Apply perturbation
#         pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
#         X_lab_pert = torch.cat([
#             X_lab[:, :1, :, :],
#             X_lab[:, 1:, :, :] + pert
#         ], dim=1)
        
#         # Convert back: lab2rgb then normalize (exactly like official code)
#         X_new = norm(lab2rgb(X_lab_pert))
        
#         # Get adversarial output WITH gradient (but wrapper params frozen)
#         encoded_adv = wrapper.encode(X_new)
#         decoded_adv = wrapper.decode(encoded_adv, ref=ref)
        
#         # Loss: maximize difference from original
#         loss = -criterion(decoded_adv, decoded_src)
        
#         if (i + 1) % 10 == 0 or i == 0:
#             pert_range = f"Pert: [{pert.min().item():.6f}, {pert.max().item():.6f}]"
#             print(f"Step {i+1}/{steps}, Loss: {loss.item():.6f}, {pert_range}, Adv Range: [{X_new.min().item():.4f}, {X_new.max().item():.4f}]")
        
#         optimizer.zero_grad()
#         loss.backward()
        
#         if pert_a.grad is not None:
#             torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        
#         optimizer.step()
    
#     return X_new.detach()

# import torch
# import torch.nn as nn
# from .utils import *


# def lab_attack(wrapper, X_nat, epsilon=0.05, lr=1e-4, steps=500, ref=None):
#     device = X_nat.device
#     criterion = nn.MSELoss().to(device)
    
#     wrapper.eval()
    
#     # Freeze all BatchNorm/InstanceNorm running stats (only for StarGAN)
#     if 'StarGAN' in wrapper.__class__.__name__:
#         for module in wrapper.modules():
#             if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
#                 module.eval()
#                 module.track_running_stats = False
    
#     pert_a = torch.zeros(
#         X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3],
#         device=device, requires_grad=True
#     )
#     optimizer = torch.optim.Adam([pert_a], lr=lr, betas=(0.9, 0.999))
    
#     X = denorm(X_nat.clone()).clamp(1e-6, 1.0 - 1e-6)
    
#     with torch.no_grad():
#         encoded_src = wrapper.encode(X_nat)
#         decoded_src = wrapper.decode(encoded_src, ref=ref)
    
#     for i in range(steps):
#         X_lab = rgb2lab(X).to(device)
        
#         pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
#         X_lab_perturbed = torch.cat([
#             X_lab[:, :1, :, :],
#             X_lab[:, 1:, :, :] + pert
#         ], dim=1)
        
#         X_new_01 = lab2rgb(X_lab_perturbed)
#         X_new_01 = torch.nan_to_num(X_new_01, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        
#         X_new = norm(X_new_01)
#         X_new = torch.nan_to_num(X_new, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        
#         encoded_adv = wrapper.encode(X_new)
#         decoded_adv = wrapper.decode(encoded_adv, ref=ref)
        
#         loss = -criterion(decoded_adv, decoded_src)
        
#         if (i + 1) % 10 == 0 or i == 0:
#             pert_range = f"Pert: [{pert.min().item():.6f}, {pert.max().item():.6f}]"
#             print(f"Step {i+1}/{steps}, Loss: {loss.item():.6f}, {pert_range}, Adv Range: [{X_new.min().item():.4f}, {X_new.max().item():.4f}]")
        
#         optimizer.zero_grad()
#         loss.backward()
        
#         if pert_a.grad is not None:
#             if not torch.isfinite(pert_a.grad).all():
#                 print("Non-finite gradient detected, stopping attack.")
#                 continue
#             torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        
#         optimizer.step()
#         X_new = X_new.detach()
        
#         with torch.no_grad():
#             if not torch.isfinite(pert_a.data).all():
#                 break
#             pert_a.data.clamp_(-epsilon, epsilon)
    
#     return X_new.clamp(-1, 1).detach()



import torch
import torch.nn as nn
from .utils import *
import torchvision.transforms as T

def lab_attack(wrapper, X_nat, epsilon=0.05, lr=1e-4, steps=500, ref=None):
    device = X_nat.device
    criterion = nn.MSELoss().to(device)
    
    wrapper.eval()
    
    # Freeze all BatchNorm/InstanceNorm running stats (only for StarGAN)
    if 'StarGAN' in wrapper.__class__.__name__:
        for module in wrapper.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                module.eval()
                module.track_running_stats = False
    
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=lr, betas=(0.9, 0.999))
    
    # X = denorm(X_nat.clone())
    
    # with torch.no_grad():
    #     encoded_src = wrapper.encode(X_nat)
    #     decoded_src = wrapper.decode(encoded_src, ref=ref)
    
    # for i in range(steps):
    #     X_lab = rgb2lab(X).to(device)
    #     pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        
    #     # Use torch.cat instead of in-place to preserve computation graph
    #     X_lab_pert = torch.cat([
    #         X_lab[:, :1, :, :],
    #         X_lab[:, 1:, :, :] + pert
    #     ], dim=1)
        
    #     X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab_pert))
    #     # rgb = lab2rgb(X_lab_pert)
    #     # rgb = torch.clamp(rgb, 0.0, 1.0)
    #     # X_new = (rgb - 0.5) / 0.5
        
    #     encoded_adv = wrapper.encode(X_new)
    #     decoded_adv = wrapper.decode(encoded_adv, ref=ref)
        
    #     loss = -criterion(decoded_adv, decoded_src)
        
    #     if (i + 1) % 3 == 0 or i == 0:
    #         pert_range = f"Pert: [{pert.min().item():.6f}, {pert.max().item():.6f}]"
    #         print(f"Step {i+1}/{steps}, Loss: {loss.item():.6f}, {pert_range}, Adv Range: [{X_new.min().item():.4f}, {X_new.max().item():.4f}]")
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
        
    # return X_new
    X = denorm(X_nat.clone()).detach()
    
    pert_a = torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pert_a], lr=lr, betas=(0.9, 0.999))
    
    with torch.no_grad():
        encoded_src = wrapper.encode(X_nat)
        decoded_src = wrapper.decode(encoded_src, ref=ref)
    
    for i in range(steps):
        # 1. Perturbation 생성 및 NaN 방지
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        
        X_lab = rgb2lab(X).to(device)
        # In-place 연산 대신 새로운 텐서 생성 (Autograd 안전성 확보)
        X_lab_pert = torch.cat([X_lab[:, :1], X_lab[:, 1:] + pert], dim=1)
        
        # 2. RGB 변환 후 NaN/Inf 제거 (Forward Pass 안전장치)
        X_new_raw = lab2rgb(X_lab_pert)
        if torch.isnan(X_new_raw).any() or torch.isinf(X_new_raw).any():
            X_new_raw = torch.nan_to_num(X_new_raw, nan=0.0, posinf=1.0, neginf=0.0)
            
        X_new = norm(X_new_raw)
        
        # 모델 Forward
        encoded_adv = wrapper.encode(X_new)
        decoded_adv = wrapper.decode(encoded_adv, ref=ref)
        
        loss = -criterion(decoded_adv, decoded_src)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 3. 기울기(Gradient) 안전장치 (Backward Pass 안전장치)
        if pert_a.grad is not None:
            # 기울기에 NaN이 있으면 0으로 치환 (치명적 오류 방지)
            torch.nan_to_num_(pert_a.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 기울기 클리핑 (Gradient Clipping) - 폭주 방지
            torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        
        optimizer.step()
        
        # 로그 출력 (에러 확인용)
        if (i + 1) % 10 == 0 or i == 0:
            pert_range = f"Pert: [{pert.min().item():.6f}, {pert.max().item():.6f}]"
            print(f"Step {i+1}/{steps}, Loss: {loss.item():.6f}, {pert_range}, Adv Range: [{X_new.min().item():.4f}, {X_new.max().item():.4f}]")

    return X_new.clamp(-1, 1).detach()

