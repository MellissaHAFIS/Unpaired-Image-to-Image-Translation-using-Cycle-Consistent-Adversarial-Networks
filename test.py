import torch
from options.test_options import TestOptions
from data.unaligned_dataset import get_dataloader
from models.networks import define_G
from utils import save_sample

opt = TestOptions().parse()
device = torch.device('cuda' if opt.gpu_ids else 'cpu')

# Charger générateurs
netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                  opt.init_gain, opt.gpu_ids)
netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                  opt.init_gain, opt.gpu_ids)
# load weights
suffix = opt.model_suffix
netG_A.load_state_dict(torch.load(f"{opt.checkpoints_dir}/{opt.name}/{opt.epoch}_net_G_A{suffix}.pth"))
netG_B.load_state_dict(torch.load(f"{opt.checkpoints_dir}/{opt.name}/{opt.epoch}_net_G_B{suffix}.pth"))
netG_A.eval(); netG_B.eval()

loader = get_dataloader(
    opt.dataroot, 'test', opt.image_size,
    opt.batch_size, opt.num_threads
)

for i, data in enumerate(loader):
    real_A, real_B = data['A'].to(device), data['B'].to(device)
    fake_B = netG_A(real_A)
    fake_A = netG_B(real_B)
    # enregistrer
    save_sample(netG_A, netG_B, real_A, real_B, f"test_{i}")
