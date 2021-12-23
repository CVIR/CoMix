import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.w_one = Parameter(torch.FloatTensor(in_features, in_features)) # in_features = out_features = d
        self.w_two = Parameter(torch.FloatTensor(in_features, in_features)) # in_features = out_features = d
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        w_one_stdv = 1./math.sqrt(self.w_one.size(0))
        self.w_one.data.uniform_(-w_one_stdv, w_one_stdv)
        self.w_two.data.uniform_(-w_one_stdv, w_one_stdv)                          

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        phi_one = torch.matmul(self.w_one, torch.transpose(input,1,2)) #compute phi(x)
        phi_two = torch.matmul(self.w_two, torch.transpose(input,1,2)) #compute phi'(x)
        adj_matrix = torch.matmul(torch.transpose(phi_one,1,2), phi_two) #form the similarity/affinity matrix
        adj_matrix = F.softmax(adj_matrix, dim=1) # Normalise the matrix along the columns
        output = torch.matmul(adj_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x))
        x = F.dropout(x, self.dropout)
        x = self.gc3(x)
        return x