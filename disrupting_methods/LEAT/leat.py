import torch
import torch.nn as nn
import torch.nn.functional as F

def lpgd_attack(wrapper, x_source, epsilon=0.05, alpha=0.01, steps=30, ref = None):
    x_source = x_source.detach()
    
    x_adv = torch.clamp(x_source + torch.empty_like(x_source).uniform_(-epsilon, epsilon), -1, 1).detach()
    
    with torch.no_grad():
        encoded_src = wrapper.encode(x_source)
        
    for i in range(steps):
        x_adv.requires_grad_()
        
        encoded_adv = wrapper.encode(x_adv)

        loss = F.mse_loss(encoded_src, encoded_adv)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_source, -epsilon, epsilon)
        x_adv = torch.clamp(x_source + delta, -1, 1).detach()
        # print(loss.item())
        
    return x_adv