import os
import torch
import pickle
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PickleDataset(Dataset): 
    def __init__(self, opt): 
        super(PickleDataset, self).__init__()
        file = os.path.join( opt.imgdir, "%s.data.pkl"%opt.language)
        with open(file, 'rb') as f :
            self.data = pickle.load(f)
        self.nSamles = len(self.data['train'])
        transform_list = [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.nSamles
    
    def __getitem__(self, index):
        img, label = self.data['train'][index]
        img = Image.fromarray(img.astype(np.unit8))
        if self.transform is not None: 
            img = self.transform(img)
        item = {'img':img, 'idx':index}
        item['label'] = label
        return item

