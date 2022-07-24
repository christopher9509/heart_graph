#!/usr/bin/env python
# coding: utf-8


from doctest import OutputChecker
from numpy import indices
import torch.nn as nn
import torch.nn.functional as F
#from pygcn.layers import GraphConvolution

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

#GATED SKIP CONNECTION 
class GraphSAGELayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)          
        self.weight = Parameter(torch.zeros(size=(2*in_features, out_features)))

        #nn.init.xavier_uniform_(self.linear.weight)
        #bias = False
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        if init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        else:
            raise NotImplementedError        
        
    def forward(self, x, adj):
        """
        support : torch([4096, 34])
        
        adj :
        tensor(indices=tensor([[   0,   49,  356,  ..., 3571, 3585, 4095], [   0,    0,    0,  ..., 4095, 4095, 4095]]), values=tensor([1., 1., 1.,  ..., 1., 1., 1.]),
        device='cuda:0', size=(4096, 4096), nnz=317566, layout=torch.sparse_coo)
        """
        support = torch.matmul(adj, x)
        print(support.size())
        exit()
        pre_degree = torch.sparse.sum(adj, 1)
        degree = torch.stack([pre_degree for _ in range(self.in_features)], dim=1).t()

        support = torch.div(support, degree)
        support = torch.cat([x, support], dim=1)
        out = torch.sparse.mm(support, self.weight)

        if self.bias is not None:
            return out + self.bias, adj
        else:
            return out, adj
    
    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)    

class GCN(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, dropout, sc='gsc'):
        super(GCN, self).__init__()
        
        self.gc0 = GraphSAGELayer(
            in_features=nfeat,  # consequence of concatenation
            out_features=nhid,
            bias=True)
        
        self.gc1 = GraphSAGELayer(
            in_features=nhid,  # consequence of concatenation
            out_features=nhid,
            bias=True)
        
        self.gc2 = GraphSAGELayer(
            in_features=nhid,  # consequence of concatenation
            out_features=nclass,
            bias=True)
        
        self.fc0 = nn.Linear(nhid, nclass)

        self.dropout = dropout
        
#         self.relu = nn.ReLU()
        
        if sc=='gsc':
            self.sc = GatedSkipConnection(nfeat, nhid)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."       
     
    def forward(self, x, adj):
        residual = x
        
        out1, adj_1 = self.gc0(x, adj)
        out2 = self.sc(residual, out1)
        
        out3 = F.relu(out2)
        
        out4, adj_2 = self.gc1(out3, adj_1)
        out5 = self.sc(residual, out4)
        out6 = F.relu(out5)
        
        out7, adj_3 = self.gc2(out6, adj_2)

        out8 = self.sc(residual, out7)

        out10 = self.fc0(out8)

        return out10
    
class GatedSkipConnection(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(GatedSkipConnection, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear_coef_in = nn.Linear(out_features, out_features)
        self.linear_coef_out = nn.Linear(out_features, out_features)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_x, out_x):
        if (self.in_features != self.out_features):
            in_x = self.linear(in_x)
            
        z = self.gate_coefficient(in_x, out_x)
#         print("Gated control Z PRINT")
#         print(z)
#         print(z.size())
#         print("="*40)
        out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out
            
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)
    
class MyEnsemble(nn.Module):
    def __init__(self, gcn0, gcn1, gcn2):
        super(MyEnsemble, self).__init__()
        self.gcn0 = gcn0
        self.gcn1 = gcn1
        self.gcn2 = gcn2
#         self.node_features0 = node_features0
#         self.node_features1 = node_features1
#         self.node_features2 = node_features2
        
        self.classifier = nn.Linear(48, 1)
        
    def forward(self, node_features0, node_features1, node_features2, adj0, adj1, adj2):
        head1 = self.gcn0(node_features0, adj0)
        head2 = self.gcn1(node_features1, adj1)
        head3 = self.gcn2(node_features2, adj2)

        x = torch.cat((head1, head2, head3), dim=1)

        x2 = self.classifier(x)
        
        return x2