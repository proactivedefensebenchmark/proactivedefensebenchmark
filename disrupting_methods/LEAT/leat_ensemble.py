import torch
import torch.nn as nn
import torch.nn.functional as F


def lpgd_ensemble_attack(
    wrapper1,
    wrapper2,
    x_source,
    epsilon=0.05,
    alpha=0.01,
    steps=30,
    ref1=None,
    ref2=None,
):

    x_source = x_source.detach()
    
    # Initialize adversarial example with random perturbation
    x_adv = torch.clamp(
        x_source + torch.empty_like(x_source).uniform_(-epsilon, epsilon),
        -1, 1
    ).detach()
    
    # Get original encoded representations from both wrappers
    with torch.no_grad():
        encoded_src1 = wrapper1.encode(x_source)
        encoded_src2 = wrapper2.encode(x_source)
    
    for i in range(steps):
        # Clone x_adv for each wrapper
        x1 = x_adv.clone().detach()
        x1.requires_grad = True
        x2 = x_adv.clone().detach()
        x2.requires_grad = True
        
        total_grad = 0.
        
        # Wrapper1 forward & backward
        encoded_adv1 = wrapper1.encode(x1)
        loss1 = F.mse_loss(encoded_src1, encoded_adv1)
        loss1.backward()
        x1.grad /= torch.norm(x1.grad)
        total_grad += x1.grad
        
        # Wrapper2 forward & backward
        encoded_adv2 = wrapper2.encode(x2)
        loss2 = F.mse_loss(encoded_src2, encoded_adv2)
        loss2.backward()
        x2.grad /= torch.norm(x2.grad)
        total_grad += x2.grad
        
        # Update adversarial example
        x_adv = x_adv + alpha * total_grad.sign()
        delta = torch.clamp(x_adv - x_source, -epsilon, epsilon)
        x_adv = torch.clamp(x_source + delta, -1, 1).detach()
    
    return x_adv