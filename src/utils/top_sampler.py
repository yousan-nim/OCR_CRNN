import os
import pdb
import sys
import math
import pickle
import random
import numpy as np
import Levenshtein as lev
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random as rand
from copy import deepcopy
from tqdm import *

from src.utils.utils import Eval, LabelConverter
from src.utils.lm import LM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SamplingTop(object): 
    def __init__(self, args): 
        self.model = args.model.to(device)
        self.percent = args.model.to(device)
        self.percent = args.percent 
        self.converter = LabelConverter(args.alphabet)
        self.evaluator = Eval()
        self.batch_size = args.batch_size 
        self.data = args.data_train
        self.collate_fn = args.collate_fn 
        self.target_folder = os.path.join(args.path, args.imgdir)

    def _top_cpu(self, item): 
        return item.detach().cpu().numpy()
    
    def get_loader(self, data, sampler=None):
        return torch.utils.data.DataLoader(
                self.data,
                batch_size=32,
                collate_fn=self.collate_fn,
                shuffle=False,
                sampler=sampler
        )
    
    def get_samples(self, train_on_pred=False, combine_scoring=False):
        pass