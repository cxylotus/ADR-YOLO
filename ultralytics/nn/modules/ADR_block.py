# Copyright (C) 2026 cxylotus 
# ADR-YOLO: An Adaptive Dual-Refinement Network for Object Detection in Aerial Images
######################################## blocks for ADR ########################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import PSABlock, C2PSA


__all__ = ['DNM', 'C2GSA', 'CCFM']


class DN(nn.Module):
    """Denoising."""
    def __init__(self, c1, c2):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1)
        self.norm = nn.BatchNorm2d(c1)

        self.pwconv1 = nn.Conv2d(c1, c1*2, kernel_size=1)
        self.pwconv2 = nn.Conv2d(c1, c2, kernel_size=1)

        self.dwconv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1) # v1.7 + v1.8
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = self.norm(self.dwconv(x))
        x = self.pwconv1(identity)
        y1, y2 = x.chunk(2, dim=1)
        x = y1 * self.sigmoid(self.dwconv2(y2))
        x = self.pwconv2(x)

        return x + identity

class DNM(nn.Module):
    """Implementation of DNM."""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.m = nn.Sequential(*(DN(c1, c1) for _ in range(n)))
        self.conv = Conv(c1, c2, 1) # v1

    def forward(self, x):
        return self.conv(self.m(x))
    

class GSA(nn.Module):
    "Gated Selective Attention"
    def __init__(self, dim, attn_ratio=0.5, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the GSA.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after GSA.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        
        attn = F.softplus(attn)**2
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)
        attn_out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        
        attn_w = self.gate(x)
        attn_out = attn_out * attn_w

        attn_out = self.proj(attn_out)
        return attn_out

class GSABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, num_heads, shortcut)
        self.attn = GSA(c, attn_ratio=attn_ratio, num_heads=num_heads)

class C2GSA(C2PSA):
    "Gated Selective Attention in C2"
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(GSABlock(self.c, num_heads=self.c // 32) for _ in range(n)))
 

class CCFM(nn.Module):
    """Channel-Compressed Fusion Module"""
    def __init__(self, c1_1, c1_2, c2, dimension=1):
        super().__init__()
        self.d = dimension
        c1_mid = min(c1_1, c1_2) // 2
        self.pwconv1 = Conv(c1_1, c1_mid, k=1, act=False)
        self.pwconv2 = Conv(c1_2, c1_mid, k=1, act=False)

        self.pwproj = Conv(2*c1_mid, c2, 1)
        self.grn = GRN(c2)
        
    def forward(self, x):
        x1 = self.pwconv1(x[0])
        x2 = self.pwconv2(x[1])

        out = torch.cat([x1, x2], self.d)
        return self.grn(self.pwproj(out))
    
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is from OverLoCK (https://arxiv.org/abs/2502.20087)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x
