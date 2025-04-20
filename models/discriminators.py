"""
Module: discriminators.py

Defines standalone PatchGAN discriminators (D_X and D_Y) for CycleGAN-style unpaired image translation.
Usage:
    from discriminators import define_D

    # Instantiate discriminator for domain X (e.g., 3-channel RGB)
    D_X = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                   norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])

    # Instantiate discriminator for domain Y (e.g., 3-channel RGB)
    D_Y = define_D(input_nc=3, ndf=64, netD='basic', n_layers_D=3,
                   norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
"""
import torch
import torch.nn as nn

def weights_init_normal(m, mean=0.0, std=0.02):
    """
    Initialize convolutional and normalization layers with a normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, mean, std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm2d') != -1 or classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class NLayerDiscriminator(nn.Module):
    """
    A 70×70 PatchGAN Discriminator.

    Architecture:
      1. Conv (no norm) + LeakyReLU
      2. (n_layers-1) × [Conv + Norm + LeakyReLU]
      3. Conv + Norm + LeakyReLU (stride=1)
      4. Final Conv to 1-channel output
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        kw, pad = 4, 1 #3 padding a tester
        sequence = [
            # 1. Initial conv block
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True)
        ]
        nf_prev = 1
        # 2. Hidden layers with increasing filters
        for n in range(1, n_layers):
            nf = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_prev, ndf * nf, kernel_size=kw, stride=2, padding=pad),
                norm_layer(ndf * nf),
                nn.LeakyReLU(0.2, True)
            ]
            nf_prev = nf
        # 3. Penultimate layer (stride=1)
        nf = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_prev, ndf * nf, kernel_size=kw, stride=1, padding=pad),
            norm_layer(ndf * nf),
            nn.LeakyReLU(0.2, True)
        ]
        # 4. Final 1-channel conv
        sequence += [
            nn.Conv2d(ndf * nf, 1, kernel_size=kw, stride=1, padding=pad)
        ]

        self.model = nn.Sequential(*sequence)
        # Initialisation des poids
        self.model.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)
