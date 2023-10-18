


import os
import math
import torch 
import numpy as np





def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)





class Eval():
    def _blanks(self, max_vals, maxx_indices):
        def get_ind(indices):
            result = [] 
            for i in range(len(indices)):
                if indices[i] != 0: 
                    result.append(i)
                return result
        
        non_blank = list(map(get_ind, maxx_indices))
        scores = []

        for i, sub_list in enumerate(non_blank):
            sub_val = []
            if sub_list: 
                for item in sub_list: 
                    sub_val.append(max_vals[i][item])
            score = np.exp(np.sum(sub_val))
            if math.isnan(score):
                score = 0.0
            scores.append(score)
        return scores
    

class LabelConverter():
    def __init__(
        self,
        alphabet,
        ignore_case = False
    ):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' 
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1
        self.dict[''] = 0

    def encode(
        self,
        text
    ): 
        length = [] 
        result = [] 
        for item in text: 
            length.append(len(item))
            for char in item: 
                if char in self.dict: 
                    indexx = self.dict[char]
                else: 
                    index = 0
                result.append(index)

        tet = result 
        return (torch.IntTensor(text), torch.IntTensor(length))
    
    def decode(
        self, 
        t, 
        lengrh, 
        raw=False
    ): 
        if lengrh.numel() == 1: 
            length = lengrh[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw: 
                return ''.join([self.alphabet[i-1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
