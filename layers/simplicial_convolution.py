import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SimplicialConvolution(nn.Module):

    def __init__(self, in_relations, out_relations) -> None:
        super(SimplicialConvolution, self).__init__()
        self.lin = nn.Linear(in_relations, out_relations)

    def forward(self, L:torch.Tensor, H:torch.Tensor, B_low:torch.Tensor, H_low:torch.Tensor, B_high:torch.Tensor, H_high:torch.Tensor):
        H1 = self.lin(H)
        H2 = self.lin(torch.matmul(L,H))
        H3 = self.lin(torch.matmul(B_low.transpose(0,1),H_low)) if B_low is not None else torch.zeros_like(H1)
        H4 = self.lin(torch.matmul(B_high,H_high)) if B_high is not None else torch.zeros_like(H1)
        H5 = H1 + H2 + H3 + H4
        return F.leaky_relu(H5)

class SimplicialAttentionLayer(nn.Module):

    def __init__(self, in_relations, out_relations) -> None:
        super(SimplicialAttentionLayer, self).__init__()
        self.lin1 = nn.Linear(in_relations, in_relations, bias=False)
        self.lin2 = nn.Linear(in_relations, out_relations)
        self.a_l = nn.Parameter(torch.FloatTensor(size=(1, in_relations)))
        self.a_r = nn.Parameter(torch.FloatTensor(size=(1, in_relations)))
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)
    
    def attn(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, attn_mask:torch.Tensor):
        Q = torch.mm(self.a_l,Q.transpose(0,1)).transpose(0,1)
        K = torch.mm(self.a_r,K.transpose(0,1))
        Q = Q.repeat(1,K.shape[1])
        K = K.repeat(Q.shape[0],1)
        A = F.leaky_relu(Q+K, negative_slope=0.2)
        A = torch.softmax(A * attn_mask, dim=1)
        return torch.matmul(A, V), A

    def forward(self, L:torch.Tensor, H:torch.Tensor, B_low:torch.Tensor, H_low:torch.Tensor, B_high:torch.Tensor, H_high:torch.Tensor):
        mask = (L!=0)
        H_k = H

        if  H_low is not None:
            B_low = (B_low!=0)
            mask = torch.cat((mask, B_low.transpose(0,1)), dim=1)
            H_k = torch.cat((H_k, H_low), dim=0)

        if  H_high is not None:
            B_high = (B_high!=0)
            mask = torch.cat((mask, B_high), dim=1)
            H_k = torch.cat((H_k, H_high), dim=0)
        
        H, A = self.attn(self.lin1(H), self.lin1(H_k), self.lin2(H_k), attn_mask=mask)

        return F.leaky_relu(H), A



