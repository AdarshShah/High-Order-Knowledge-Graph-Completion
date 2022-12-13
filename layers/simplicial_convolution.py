import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

#Tested
class SimplicialConvolution(nn.Module):

    def __init__(self, in_relations, out_relations) -> None:
        super(SimplicialConvolution, self).__init__()
        self.W_L = nn.parameter.Parameter(torch.rand((in_relations, out_relations)))
        self.W_low = nn.parameter.Parameter(torch.rand((in_relations, out_relations)))
        self.W_high = nn.parameter.Parameter(torch.rand((in_relations, out_relations)))
        self.bias = nn.parameter.Parameter(torch.rand((out_relations,)))

    def forward(self, L:torch.Tensor, H:torch.Tensor, B_low:torch.Tensor, H_low:torch.Tensor, B_high:torch.Tensor, H_high:torch.Tensor):
        '''
        Parameters:
        L : Laplacian Matrix
        H : Simplicial Embedding Matrix
        B_low : Boundary Matrix
        H_low : Lower order Simplicial Embedding Matrix
        B_high : Co-boundary Matrix
        H_high : Higher order Simplicial Embedding Matrix
        '''
        result = torch.matmul(torch.matmul(L,H), self.W_L) + self.bias.repeat(L.shape[0], 1)
        result += torch.matmul(torch.matmul(B_low.transpose(0,1),H_low),self.W_low) if B_low is not None else 0
        result += torch.matmul(torch.matmul(B_high,H_high),self.W_high) if B_high is not None else 0
        return result

#Tested
#Describe this in overleaf later
#Can A be Laplacian ? Yes
class SimplicialAttentionLayer(nn.Module):

    def __init__(self, in_relations) -> None:
        super(SimplicialAttentionLayer, self).__init__()
        self.W_Q = nn.parameter.Parameter(torch.rand((in_relations, in_relations)))
        self.W_K = nn.parameter.Parameter(torch.rand((in_relations, in_relations)))
        self.W_V = nn.parameter.Parameter(torch.rand((in_relations, in_relations)))

    def forward(self, H:torch.Tensor, K:torch.Tensor, V:torch.Tensor, attn_mask:torch.Tensor):
        '''
        Parameters:
        H : Embedding Matrix
        A : Adjacency or Laplacian Matrix
        '''
        # H_Q = torch.matmul(H, self.W_Q)
        # H_K = torch.transpose(torch.matmul(H, self.W_K), 0, 1)
        H_2 = torch.matmul(H, K.transpose(0, 1))
        H_3 = H_2/(np.sqrt(H.shape[1]))
        H_4 = torch.softmax(torch.softmax(H_3, dim=1) * 1 * attn_mask, dim=1)
        # return torch.matmul(H_4, torch.matmul(H, self.W_V))
        return torch.matmul(H_4, V)

class SimplicialAttentionLayer2(nn.Module):

    def __init__(self, in_relations) -> None:
        super(SimplicialAttentionLayer2, self).__init__()
        # self.attn = nn.MultiheadAttention(in_relations, 1)
        # self.attn_l = nn.MultiheadAttention(in_relations, 1)
        # self.attn_h = nn.MultiheadAttention(in_relations, 1)
        self.attn = SimplicialAttentionLayer(in_relations)

    def forward(self, L:torch.Tensor, H:torch.Tensor, B_low:torch.Tensor, H_low:torch.Tensor, B_high:torch.Tensor, H_high:torch.Tensor):
        
        L = (L!=0)
        H1= self.attn(H, H, H, attn_mask = L)
        f = 1
        if  H_low is not None:
            B_low = (B_low!=0)
            H2 = self.attn(H, H_low, H_low, attn_mask = B_low.transpose(0,1))
            H1 = H1 + H2
            f +=1
        
        if  H_high is not None:
            B_high = (B_high!=0)
            H3 = self.attn(H, H_high, H_high, attn_mask = B_high)
            H1 = H1 + H3
            f +=1
        return H1/f




