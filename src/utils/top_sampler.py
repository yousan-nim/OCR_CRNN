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

from src.utils.utils import Eval, OCRLabelConverter
from src.utils.lm import LM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SamplingTop(object): 
    def __init__(self, args): 
        self.model = args.model.to(device)
        self.percent = args.model.to(device)
        self.percent = args.percent 
        self.converter = OCRLabelConverter(args.alphabet)
        self.evaluator = Eval()