# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:27:17 2024

@author: lshaw
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

##LogReg
class LogisticRegression(nn.Module):    
    def __init__(self, input_dim):
        super().__init__()
        act=nn.Softmax
        self.linear = nn.Sequential(
            nn.Flatten(),
           nn.Linear(input_dim, 2),
           act(dim=-1)
       )
 
    # make predictions
    def forward(self, x):
        return self.linear(x)

##BNN
class BNN(nn.Module):
    '''Default MNIST. Architecture from SGFS '''
    def __init__(self, input_dim=20, hidden_dim=30, output_dim=10):
        super().__init__()
        act=nn.Sigmoid
        self.linear = nn.Sequential(
            nn.Flatten(),
           nn.Linear(input_dim, hidden_dim),
           act(),
           nn.Linear(hidden_dim, output_dim),
       )

    def forward(self, x):
        return self.linear(x)
    