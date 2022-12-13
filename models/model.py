from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer2
from torch import nn
from torch.nn import functional as F
import torch
from dgl.nn import pytorch as gnn

class SimplicialModel1(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialModel1, self).__init__()
        self.dim = 4
        self.attn1 = SimplicialAttentionLayer2(classes)

        self.conv1 = SimplicialConvolution(classes, 2*classes)
        self.conv2 = SimplicialConvolution(2*classes, 4*classes)
        self.conv3 = SimplicialConvolution(4*classes, 4*classes)
        self.lin = nn.Linear(4*classes, classes)
        self.device = device

    def conv(self, op, embeddings, laplacians, boundaries):
        embeddings1 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings1.append(None)
            elif i==0:
                embeddings1.append(F.tanh(op(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1])))
            elif i==3:
                embeddings1.append(F.tanh(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], None, None)))
            else:
                embeddings1.append(F.tanh(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], boundaries[i+1], embeddings[i+1])))
        return embeddings1

    def attn(self, op, embeddings, laplacians, boundaries):
        embeddings1 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings1.append(None)
            elif i==0:
                embeddings1.append(op(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1]))
            elif i==3:
                embeddings1.append(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], None, None))
            else:
                embeddings1.append(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], boundaries[i+1], embeddings[i+1]))
        return embeddings1

    def forward(self, embeddings, laplacians, boundaries, order, idx):
        embeddings1 = self.conv(self.conv1, embeddings, laplacians, boundaries)
        embeddings2 = self.attn(self.attn1, embeddings1, laplacians, boundaries)
        embeddings3 = self.conv(self.conv2, embeddings2, laplacians, boundaries)
        embeddings4 = self.attn(self.attn1, embeddings3, laplacians, boundaries)
        embeddings5 = self.conv(self.conv3, embeddings4, laplacians, boundaries)
        embeddings6 = self.attn(self.attn1, embeddings5, laplacians, boundaries)

        if laplacians[order] is None:
            return torch.zeros((20,)).to(self.device)
        return self.lin(embeddings6[order][idx])

class BaseGNN(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(BaseGNN, self).__init__()
        self.gcn1 = gnn.GraphConv(classes, classes, activation=nn.LeakyReLU(inplace=True), allow_zero_in_degree=True)
        self.gcn2 = gnn.GraphConv(classes, classes, activation=nn.LeakyReLU(inplace=True), allow_zero_in_degree=True)
        self.linear = nn.Linear(classes, classes)

    def forward(self, graph, feat, order):
        feat = self.gcn1(graph, feat)
        embeddings = self.gcn2(graph, feat)[: order+1]
        pooling = torch.mean(embeddings, dim=0)
        return self.linear(pooling)

