

from utils.utils import LabelConverter



class Trainer():
    def __init__(
        self,
        opt
    ) :
        super().__init__()
        self.data_train = opt.data_train 
        self.data_valid = opt.data_valid
        self.model      = opt.model 
        self.criterion  = opt.criterion
        self.schedule   = opt.schedule 
        self.optimizer  = opt.optimizer 
        self.alpha      = opt.alpha
        self.converter  = LabelConverter(opt.alphabet)
