from .generators import ResnetGenerator
from .discriminators import NLayerDiscriminator
import torch.nn as nn
import functools, torch

def get_norm_layer(norm_type='instance'):
    if norm_type=='instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type=='batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    raise

def define_G(input_nc, output_nc, ngf, netG='resnet_9blocks', norm='instance',
             use_dropout=False, init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm)
    if netG in ['resnet_9blocks','resnet_6blocks']:
        blocks = 9 if netG=='resnet_9blocks' else 6
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout, blocks)
    else:
        raise
    return _init_net(net, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD='basic', n_layers_D=3, norm='instance',
             init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm)
    if netD=='basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer)
    else:
        raise
    return _init_net(net, init_gain, gpu_ids)

def _init_net(net, init_gain, gpu_ids):
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(f'cuda:{gpu_ids[0]}'); net = nn.DataParallel(net, gpu_ids)
    return net

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode=='lsgan': self.loss = nn.MSELoss()
        else: self.loss = nn.BCEWithLogitsLoss()
    def get_target(self, pred, real):
        val = 1.0 if real else 0.0
        return torch.full_like(pred, val)
    def forward(self, pred, real):
        target = self.get_target(pred, real)
        return self.loss(pred, target)
