# options/test_options.py

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """Options spécifiques au test / génération CycleGAN."""

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.set_defaults(dataset_mode='single',  continue_train=False)
        parser.add_argument('--model_suffix', type=str, default='',
                            help='suffixe du netG à charger (ex: "")')
        return parser
