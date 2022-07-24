#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#ver 211107
from doctest import OutputChecker
import torch.nn as nn
import torch.nn.functional as F
#from pygcn.layers import GraphConvolution

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

#GATED SKIP CONNECTION 
class GCNIILayer(nn.Module):
    
    def __init__(self, in_features, out_features, residual = True, variant = False, bias=True, init='xavier'):
        super(GCNIILayer, self).__init__()
        
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else :
            self.in_features = in_features
        
        self.out_features = out_features
        self.residual = residual
        self.linear = nn.Linear(self.in_features, self.out_features)
        self.linear_r = nn.Linear(34, out_features)       
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)
        
        if init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        else:
            raise NotImplementedError        
        
    def forward(self, x, adj, h0, lamda, alpha, l):
        """
        feature Information:
        # hi = torch.size([4096, 34]), input * adj matmul value
        # h0 = torch.size([4096, 34]), initial input       
        # support : torch_size([4096,16])
        # adj : torch.Size([4096, 4096]), adjencent matrix
        # out : torch.size([4096, 16], GCNII Layer's Output
        """
        theta = math.log(lamda/l+1)
        
        x = torch.matmul(x, self.weight)
        hi = torch.sparse.mm(adj, x)
        
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        
        # hi = torch.size([4096, 34])
        # h0 = torch.size([4096, 34])

        out = theta*support+(1-theta)*r
        
        # support : torch_size([4096,16])
        # adj : torch.Size([4096, 4096])
        # out : torch.size([4096, 16])
        
        if self.bias is not None:
            return out + self.bias
        else:
            return out
    
    
    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)    

class GCN(nn.Module):
    """
    alpha : GCN Input 및 output 비중 계수
    lamda : 
    """
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer, alpha, lamda, sc='gsc'):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        
        self.gc0 = GCNIILayer(
            in_features=nfeat,  # consequence of concatenation
            out_features=nhid,
            bias=True)
        
        for _ in range(nlayer): 
            self.convs.append(GCNIILayer(
            in_features=nhid,  # consequence of concatenation
            out_features=nhid,
            bias=True))
        
        self.fc0 = nn.Linear(nfeat, nhid)
        self.fc1 = nn.Linear(nhid, nclass)

        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        
        self.relu = nn.ReLU()
        
        if sc=='gsc':
            self.sc = GatedSkipConnection(nfeat, nhid)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."       
     
    def forward(self, x, adj):
        _layers = []
        
        x = F.dropout(x, training=self.training)
        x = self.fc0(x)
        layer_inner = self.relu(x)
        _layers.append(x)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.relu(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i+1))
        
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        output = self.fc1(layer_inner)

        return output
    
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
        
        self.classifier = nn.Linear(48, 1)
        
    def forward(self, node_features0, node_features1, node_features2, adj0, adj1, adj2):
        head1 = self.gcn0(node_features0, adj0)
        head2 = self.gcn1(node_features1, adj1)
        head3 = self.gcn2(node_features2, adj2)

        x = torch.cat((head1, head2, head3), dim=1)
#
        x2 = self.classifier(x)
        
        return x2