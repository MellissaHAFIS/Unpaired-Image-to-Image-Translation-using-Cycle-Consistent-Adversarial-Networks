import torch, itertools
from options.train_options import TrainOptions
from data.unaligned_dataset import get_dataloader
from models.networks import define_G, define_D, GANLoss
from utils import save_sample, save_model, ImagePool


opt = TrainOptions().parse()
device = torch.device('cuda' if opt.gpu_ids else 'cpu')

# Dataloader
loader = get_dataloader(
    opt.dataroot, 'train', opt.image_size,
    opt.batch_size, opt.num_threads
)

# Réseaux
netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_gain, opt.gpu_ids)
netG_B = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_gain, opt.gpu_ids)
netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_gain, opt.gpu_ids)
netD_B = define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_gain, opt.gpu_ids)

# Pertes & optimiseurs
criterionGAN   = GANLoss(opt.gan_mode).to(device)
criterionCycle = torch.nn.L1Loss().to(device)
criterionIdt   = torch.nn.L1Loss().to(device)

optG = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lr, betas=(opt.beta1,0.999))
optD = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lr, betas=(opt.beta1,0.999))

fake_A_pool, fake_B_pool = ImagePool(opt.pool_size), ImagePool(opt.pool_size)

for epoch in range(opt.epoch_count, opt.n_epochs+opt.n_epochs_decay+1):
    for i, data in enumerate(loader):
        real_A, real_B = data['A'].to(device), data['B'].to(device)

        # --- G ---
        optG.zero_grad()
        fake_B = netG_A(real_A);    rec_A = netG_B(fake_B)
        fake_A = netG_B(real_B);    rec_B = netG_A(fake_A)

        # identity
        idt_A = netG_A(real_B); loss_idt_A = criterionIdt(idt_A, real_B)*opt.lambda_B*opt.lambda_identity
        idt_B = netG_B(real_A); loss_idt_B = criterionIdt(idt_B, real_A)*opt.lambda_A*opt.lambda_identity

        # GAN
        loss_G_A = criterionGAN(netD_A(fake_B), True)
        loss_G_B = criterionGAN(netD_B(fake_A), True)
        # cycle
        loss_cycle_A = criterionCycle(rec_A, real_A)*opt.lambda_A
        loss_cycle_B = criterionCycle(rec_B, real_B)*opt.lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward(); optG.step()

        # --- D ---
        # D_A
        optD.zero_grad()
        pred_real = netD_A(real_B); loss_D_real = criterionGAN(pred_real, True)
        pred_fake = netD_A(fake_B_pool.query(fake_B.detach())); loss_D_fake = criterionGAN(pred_fake, False)
        loss_D_A = (loss_D_real + loss_D_fake)*0.5; loss_D_A.backward()
        # D_B
        pred_real = netD_B(real_A); loss_D_real = criterionGAN(pred_real, True)
        pred_fake = netD_B(fake_A_pool.query(fake_A.detach())); loss_D_fake = criterionGAN(pred_fake, False)
        loss_D_B = (loss_D_real + loss_D_fake)*0.5; loss_D_B.backward()
        optD.step()

        if i%100==0:
            print(f"[Epoch {epoch}][{i}/{len(loader)}] G: {loss_G.item():.3f}, D_A: {loss_D_A.item():.3f}, D_B: {loss_D_B.item():.3f}")

    # sauvegardes
    save_sample(netG_A, netG_B, real_A, real_B, epoch)
    save_model(netG_A, netG_B, netD_A, netD_B, epoch)
