# models/cycle_gan_model.py

class CycleGANModel:
    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        """
        Stub minimal pour que BaseOptions puisse importer dynamiquement
        CycleGANModel sans planter. Ne modifie pas le parser.
        """
        return parser
