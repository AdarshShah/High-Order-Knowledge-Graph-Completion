import torch
from torch import nn
from torch.nn import functional as F

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
        result = torch.matmul(torch.matmul(L,H), self.W_L) + self.bias.repeat(1,L.shape[0]) 
        result += torch.matmul(torch.matmul(B_low,H_low),self.W_low) if B_low is not None else 0
        result += torch.matmul(torch.matmul(B_high,H_high),self.W_high) if B_high is not None else 0
        return result