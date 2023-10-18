import pdb
import math
import torch
import torch.nn as nn


class CTC_LOSS(nn.Module):
    def __init__(
        self,
        dim=2    
    ): 
        super().__init__()
        self.dim = dim 
        self.ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(
        self,
        logits,
        labels,
        prediction_sizes,
        target_sizes    
    ):
        EPS = 1e-7
        loss = self.ctc_loss(
            logits, 
            labels, 
            prediction_sizes, 
            target_sizes
        )
        return self.debug(
                    loss, 
                    logits, 
                    labels, 
                    prediction_sizes, 
                    target_sizes
                )
    
    def sanitize(
        self, 
        loss
    ):
        EPS = 1e-7
        if abs(loss.item() - float('inf')) < EPS:
            return torch.zeros_like(loss)
        if math.isnan(loss.item()):
            return torch.zeros_like(loss)
        return loss
    
    def debug(
        self, 
        loss, 
        logits, 
        labels,
        prediction_sizes, 
        target_sizes
    ):
        if math.isnan(loss.item()):
            print("Loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained.")
        return loss