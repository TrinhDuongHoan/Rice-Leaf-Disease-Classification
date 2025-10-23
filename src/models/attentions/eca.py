import torch.nn as nn
import numpy as np

class ECA(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)                    
        y = self.conv(y.squeeze(-1).transpose(1, 2))         
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)    
        return x * y