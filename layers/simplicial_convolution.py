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

    def forward(self, H:torch.Tensor, A:torch.Tensor):
        '''
        Parameters:
        H : Embedding Matrix
        A : Adjacency or Laplacian Matrix
        '''
        H_Q = torch.matmul(H, self.W_Q)
        H_K = torch.transpose(torch.matmul(H, self.W_K), 0, 1)
        H_2 = torch.matmul(H_Q, H_K)
        H_3 = H_2 * A/(np.sqrt(H.shape[1]))
        H_4 = torch.softmax(H_3, dim=1)
        return torch.matmul(H_4, torch.matmul(H, self.W_V))


