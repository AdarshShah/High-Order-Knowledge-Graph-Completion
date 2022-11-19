from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer
from torch import nn
from torch.nn import functional as F
import torch

class SimplicialModel1(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialModel1, self).__init__()
        self.dim = 4
        self.conv1 = SimplicialConvolution(classes, classes)
        self.attn1 = SimplicialAttentionLayer(classes)
        self.conv2 = SimplicialConvolution(classes, classes)
        self.lin = nn.Linear(classes, classes)
        self.device = device
    
    def forward(self, embeddings, laplacians, boundaries, order, idx):
        embeddings1 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings1.append(None)
            elif i==0:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1])))
            elif i==3:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], None, None)))
            else:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], boundaries[i+1], embeddings[i+1])))
        embeddings2 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings2.append(None)
            elif laplacians[i] is not None:
                embeddings2.append(self.attn1(embeddings1[i], laplacians[i]))
        embeddings3 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings3.append(None)
            elif i==0:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], None, None, boundaries[i+1], embeddings2[i+1])))
            elif i==3:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], boundaries[i], embeddings2[i-1], None, None)))
            else:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], boundaries[i], embeddings2[i-1], boundaries[i+1], embeddings2[i+1])))
        if laplacians[order] is None:
            return torch.zeros((20,)).to(self.device)
        return self.lin(embeddings3[order][idx])


