
import os
import pdb
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from argparse import ArgumentParser

from src.options.opt import options
from src.utils.utils import gmkdir
from src.utils.top_sampler import SamplingTop
from src.data.pickle_dataset import PickleDataset
from src.data.synth_dataset import SynthDataset, SynthCollator
from src.models.model_crnn import CRNN
from src.loss.ctc_loss import CTC_LOSS



device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = ArgumentParser()
    options(parser)
    args = parser.parse_args()

    # loading data 
    args.imgdir = 'train'
    args.source_data = SynthDataset(args)
    args.collate_fn = SynthCollator()

    args.imgdir = 'test'
    target_data = SynthDataset(args)
    train_split = int(0.8*len(target_data))
    val_split = len(target_data) - train_split
    args.data_train, args.data_val = random_split(target_data, (train_split, val_split))

    args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%""" 

    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    model = model.to(device)
    args.criterion = CTC_LOSS()

    savepath = os.path.join(args.save_dir, args.name)
    gmkdir(savepath)
    gmkdir(args.log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    resume_file = savepath + '/' + 'best.ckpt'
    print('Loading model %s'%resume_file)
    checkpoint = torch.load(resume_file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])

    # Generating top samples
    args.model = model
    args.imgdir = 'target_top'
    finetunepath = args.path + '/' + args.imgdir
    gmkdir(finetunepath)

    sampler = SamplingTop(args)
    sampler.get_samples(train_on_pred=args.train_on_pred, 
        combine_scoring=args.combine_scoring)
    
    # Joining source and top samples
    # args.top_samples = SynthDataset(args)
    # args.data_train = torch.utils.data.ConcatDataset([args.source_data, args.top_samples])
    # print('Traininig Data Size:{}\nVal Data Size:{}'.format(
    #     len(args.data_train), len(args.data_val)))
    # learner = LearnerSemi(args.model, optimizer, savepath=savepath, resume=args.resume)
    # learner.fit(args)
    # shutil.rmtree(finetunepath)