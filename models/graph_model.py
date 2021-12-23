import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import math
import numpy as np
from torch.autograd import Variable
from models.gcn_adjacency import *
from torch.autograd import Function
import warnings
import params

warnings.filterwarnings("ignore")

class Graph_Model(nn.Module):

    def __init__(self, dataset_name=None):
        super(Graph_Model, self).__init__()

        if(dataset_name=="UCF-HMDB"):
            self.graph = GCN(nfeat=1024, nhid=256, nclass=12, dropout=0.1)
        elif(dataset_name=="Jester"):
            self.graph = GCN(nfeat=1024, nhid=256, nclass=7, dropout=0.1)
        elif(dataset_name=="Epic-Kitchens"):
            self.graph = GCN(nfeat=1024, nhid=256, nclass=8, dropout=0.1)


    def forward(self, inputs):

        x = self.graph(inputs)
        x = torch.mean(x, dim=1)

        return x