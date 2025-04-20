import torch.nn as nn
import functools

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, padding_type='reflect'):
        super().__init__()
        self.block = []
        if padding_type == 'reflect':
            self.block += [nn.ReflectionPad2d(1)]
            conv1 = nn.Conv2d(dim, dim, 3, padding=0, bias=norm_layer == nn.InstanceNorm2d)
        else:
            conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=norm_layer == nn.InstanceNorm2d)

        self.block += [conv1, norm_layer(dim), nn.ReLU(True)]

        if use_dropout:
            self.block += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            self.block += [nn.ReflectionPad2d(1)]
            conv2 = nn.Conv2d(dim, dim, 3, padding=0, bias=norm_layer == nn.InstanceNorm2d)
        else:
            conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=norm_layer == nn.InstanceNorm2d)

        self.block += [conv2, norm_layer(dim)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9, padding_type='reflect'):
        super().__init__()
        # initial conv
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, 7, bias=norm_layer==nn.InstanceNorm2d),
                 norm_layer(ngf), nn.ReLU(True)]
        # downsampling
        n_down = 2
        for i in range(n_down):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1, bias=norm_layer==nn.InstanceNorm2d),
                      norm_layer(ngf*mult*2), nn.ReLU(True)]
        # resblocks
        mult = 2**n_down
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult, norm_layer, use_dropout, padding_type)]
        # upsampling
        for i in range(n_down):
            mult = 2**(n_down-i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 3, stride=2, padding=1, output_padding=1,
                                         bias=norm_layer==nn.InstanceNorm2d),
                      norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
