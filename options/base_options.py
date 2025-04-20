# options/base_options.py

import argparse
import os

class BaseOptions:
    """Cette classe définit les options utilisées à la fois en entraînement et en test.
    Elle gère le parsing, la configuration des GPU, et la création des dossiers de checkpoint.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Déclare les options communes à l'entraînement et au test."""
        
        parser.add_argument('--dataroot',         type=str,   default='./datasets',       help='chemin vers le dossier des datasets')
        parser.add_argument('--name',             type=str,   default='experiment_name',  help='nom de l\'expérience (sous-dossier dans checkpoints_dir)')
        parser.add_argument('--checkpoints_dir',  type=str,   default='./checkpoints',    help='où sauvegarder les modèles')
        parser.add_argument('--model',            type=str,   default='cycle_gan',        help='choix du modèle [cycle_gan]')
        parser.add_argument('--verbose',          action='store_true',               help='pour plus de logs')

        # Matériel
        parser.add_argument('--gpu_ids',          type=str,   default='0',               help='ids de GPU, ex: "0,1"; "-1" pour CPU')
        parser.add_argument('--num_threads',      type=int,   default=4,                 help='nombre de threads pour le DataLoader')
        parser.add_argument('--batch_size',       type=int,   default=1,                 help='taille du batch')

        # Images
        parser.add_argument('--image_size',       type=int,   default=256,               help='taille (carrée) des images transformées')
        parser.add_argument('--input_nc',         type=int,   default=3,                 help='nombre de canaux en entrée')
        parser.add_argument('--output_nc',        type=int,   default=3,                 help='nombre de canaux en sortie')

        # Réseaux
        parser.add_argument('--netG',             type=str,   default='resnet_9blocks',  help='archi du générateur [resnet_6blocks|resnet_9blocks]')
        parser.add_argument('--netD',             type=str,   default='basic',            help='archi du discriminateur [basic|n_layers|pixel]')
        parser.add_argument('--ngf',              type=int,   default=64,                help='# filtres du générateur first layer')
        parser.add_argument('--ndf',              type=int,   default=64,                help='# filtres du discriminateur first layer')
        parser.add_argument('--n_layers_D',       type=int,   default=3,                 help='nombre de couches conv dans le discriminateur (pour netD n_layers)')
        parser.add_argument('--norm',             type=str,   default='instance',        help='normalisation [batch|instance|none]')
        parser.add_argument('--init_type',        type=str,   default='normal',           help='initialisation [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain',        type=float, default=0.02,              help='gain pour l\'initialisation')

        # Mode GAN
        parser.add_argument('--gan_mode',         type=str,   default='lsgan',            help='type de loss GAN [vanilla | lsgan | wgangp]')

        # Dataset
        parser.add_argument('--dataset_mode',     type=str,   default='unaligned',       help='mode dataset [unaligned|single]')
        parser.add_argument('--direction',        type=str,   default='AtoB',             help='AtoB ou BtoA')

        # Entraînement / test
        parser.add_argument('--epoch_count',      type=int,   default=1,                 help='époque de départ')
        parser.add_argument('--n_epochs',         type=int,   default=100,               help='nombre d\'époques avant decay lr')
        parser.add_argument('--n_epochs_decay',   type=int,   default=100,               help='nombre d\'époques de decay lr')
        parser.add_argument('--lr',               type=float, default=0.0002,            help='learning rate initial')
        parser.add_argument('--beta1',            type=float, default=0.5,               help='beta1 pour Adam')
        parser.add_argument('--lr_policy',        type=str,   default='linear',           help='policy lr [linear|step|plateau|cosine]')
        parser.add_argument('--lr_decay_iters',   type=int,   default=50,                help='décay lr tous les x iters si step')

        # Divers
        parser.add_argument('--pool_size',        type=int,   default=50,                help='taille du buffer d\'images pour le D')
        parser.add_argument('--no_dropout',       action='store_true',               help='pas de dropout pour G')
        parser.add_argument('--continue_train',   action='store_true',               help='continuer l\'entraînement')
        parser.add_argument('--load_iter',        type=int,   default=0,                 help='itération à charger (0 = dernier checkpoint)')

        self.initialized = True
        return parser

    def gather_options(self):
        """Parse d'abord les options communes, puis ajoute les options spécifiques au modèle."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # parse known pour récupérer opt.model
        opt, _ = parser.parse_known_args()

        # import dynamique des options spécifique au modèle
        if opt.model == 'cycle_gan':
            from models.cycle_gan_model import CycleGANModel
            parser = CycleGANModel.modify_commandline_options(parser, is_train=not opt.continue_train)
        else:
            raise ValueError(f"Modèle inconnu : {opt.model}")

        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Gère le parsing final, conversion des gpu_ids en liste, création du dossier checkpoint."""
        opt = self.gather_options()

        # transformer 'gpu_ids' string en liste d'int
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [int(i) for i in str_ids if int(i) >= 0]

        # config CUDA_VISIBLE_DEVICES
        if len(opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.gpu_ids))
        else:
            opt.gpu_ids = []

        # créer le dossier pour sauvegarder
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)

        self.opt = opt
        return self.opt
