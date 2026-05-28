import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)

class SEResBottleneck_NS(nn.Module):
    def __init__(self, in_c, out_c, stride=1, reduction=16, bottleneck_ratio=4):
        super().__init__()
        mid_c = max(out_c // bottleneck_ratio, 1)

        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)

        self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)

        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_mid = max(out_c // reduction, 1)
        self.se_fc1 = nn.Linear(out_c, se_mid, bias=False)
        self.se_fc2 = nn.Linear(se_mid, out_c, bias=False)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        y = F.relu(self.se_fc1(y), inplace=True)
        y = torch.sigmoid(self.se_fc2(y)).view(b, c, 1, 1)
        out = out * y

        if self.downsample is not None:
            res = self.downsample(x)

        out = self.relu(out + res)
        return out

class IDExtraction(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        blocks = [SEResBottleneck_NS(64, 64, stride=1) for _ in range(L)]
        self.net = nn.Sequential(
            ConvBlock(3, 64, k=3, s=2, p=1),
            nn.MaxPool2d(3, 2, 1),
            *blocks
        )
    def forward(self, x): 
        return self.net(x)

class PerturbationBlock(nn.Module):
    def __init__(self, M=3):
        super().__init__()
        self.refine = ConvBlock(64, 64, k=3, s=1, p=1)
        self.blocks = nn.Sequential(*[SEResBottleneck_NS(64, 64, stride=1) for _ in range(M)])

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        self.eta   = nn.Parameter(torch.randn(1, 64, 1, 1) * 0.01)

    def forward(self, x):
        feat = self.blocks(self.refine(x))
        if self.training:
            noise = self.alpha * torch.randn_like(feat) + self.eta
            return feat + self.beta * noise
        return feat

class FeatureBlock(nn.Module):
    def __init__(self, N=5):
        super().__init__()
        blocks = [SEResBottleneck_NS(64, 64, stride=1) for _ in range(N)]
        self.net = nn.Sequential(
            ConvBlock(3, 32, k=3, s=2, p=1),
            ConvBlock(32, 64, k=3, s=2, p=1),
            ConvBlock(64, 64, k=3, s=1, p=1),
            *blocks
        )
    def forward(self, x): 
        return self.net(x)

class CloakingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_p = nn.Parameter(torch.tensor(1.0))
        self.fuse = SEResBottleneck_NS(128, 128, stride=1)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ConvBlock(64, 64, k=3, s=1, p=1),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            ConvBlock(67, 64, k=3, s=1, p=1),
            ConvBlock(64, 32, k=3, s=1, p=1),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, img_feat, pert_feat, org_img):
        fused = torch.cat([img_feat, pert_feat * self.gamma_p], dim=1)
        recon_feat = self.upsample(self.fuse(fused))

        if recon_feat.shape[-2:] != org_img.shape[-2:]:
            recon_feat = F.interpolate(recon_feat, size=org_img.shape[-2:], mode="bilinear", align_corners=False)

        out = self.final(torch.cat([recon_feat, org_img], dim=1))
        return out

class NullSwap(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = IDExtraction(L=4)
        self.pert = PerturbationBlock(M=3)
        self.feat = FeatureBlock(N=5)
        self.cloak = CloakingBlock()

    def forward(self, x):
        img_feat = self.feat(x)
        pert_feat = self.pert(self.id(x))
        return self.cloak(img_feat, pert_feat, x)