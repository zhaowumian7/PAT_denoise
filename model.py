# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: model.py
#   Author  : Youshen Xiao
#   Email   : xiaoysh2023@shanghaitech.edu.cn
#   Date    : 2024/12/6
# -----------------------------------------
import torch.nn as nn
import numpy as np
import math
import torch


# -------------------------------
# a simple MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self,depth=3,mapping_size=1024,hidden_size=256):
        super().__init__()
        layers = []
        layers.append(nn.Linear(mapping_size,hidden_size))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Linear(hidden_size,hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_size,1))
        self.layers = nn.Sequential(*layers)
    def forward(self,x):
        # return (self.layers(x))
        return torch.sigmoid(self.layers(x))



# -------------------------------
# mapping processing(2====>256)
# -------------------------------
def map_x(x,B):
    xp = torch.matmul(2*math.pi*x,B)
    return torch.cat([torch.sin(xp),torch.cos(xp)],dim=-1)