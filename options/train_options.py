# options/train_options.py

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """Options spécifiques à l'entraînement CycleGAN."""

    def initialize(self, parser):
        parser = super().initialize(parser)
        # poids pour les 2 pertes cycle-consistency
        parser.add_argument('--lambda_A',        type=float, default=10.0,
                            help='poids perte cycle A→B→A')
        parser.add_argument('--lambda_B',        type=float, default=10.0,
                            help='poids perte cycle B→A→B')
        parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='poids perte identité (G(A)≈A, G(B)≈B)')
        return parser
