from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer
from torch import nn
from torch.nn import functional as F
import torch
from dgl.nn import pytorch as gnn

class SimplicialAttentionModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialAttentionModel, self).__init__()
        self.dim = dim

        self.attn1 = SimplicialAttentionLayer(classes, 2*classes)
        self.attn2 = SimplicialAttentionLayer(2*classes, 4*classes)
        self.attn3 = SimplicialAttentionLayer(4*classes, 8*classes)
        self.attn4 = SimplicialAttentionLayer(8*classes, 16*classes)
        self.rel_lin = nn.Linear(16*classes, classes)

    def attn(self, op, embeddings, laplacians, boundaries):
        embeddings1 = []
        A_w = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings1.append(None)
                A_w.append(None)
            elif i==0:
                em, A = op(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1])
                embeddings1.append(em)
                A_w.append(A)
            elif i==self.dim-1:
                em, A = op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], None, None)
                embeddings1.append(em)
                A_w.append(A)
            else:
                em, A = op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], boundaries[i+1], embeddings[i+1])
                embeddings1.append(em)
                A_w.append(A)
        return embeddings1, A_w

    def forward(self, embeddings, laplacians, boundaries, order, idx, rel):
        embeddings1, A_w_1 = self.attn(self.attn1, embeddings, laplacians, boundaries)
        embeddings2, A_w_2 = self.attn(self.attn2, embeddings1, laplacians, boundaries)
        embeddings3, A_w_3 = self.attn(self.attn3, embeddings2, laplacians, boundaries)
        embeddings4, A_w_4 = self.attn(self.attn4, embeddings3, laplacians, boundaries)
        # to process A_w later
        return self.rel_lin(embeddings4[order][idx]).squeeze()[rel.nonzero()]

class SimplicialConvolutionModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialConvolutionModel, self).__init__()
        self.dim = dim

        self.conv1 = SimplicialConvolution(classes, 2*classes)
        self.conv2 = SimplicialConvolution(2*classes, 4*classes)
        self.conv3 = SimplicialConvolution(4*classes, 8*classes)
        self.conv4 = SimplicialConvolution(8*classes, 16*classes)
        self.rel_lin = nn.Linear(16*classes, classes)

        self.device = device

    def conv(self, op, embeddings, laplacians, boundaries):
        embeddings1 = []
        for i in range(self.dim):
            if laplacians[i] is None:
                embeddings1.append(None)
            elif i==0:
                embeddings1.append(op(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1]))
            elif i==self.dim-1:
                embeddings1.append(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], None, None))
            else:
                embeddings1.append(op(laplacians[i], embeddings[i], boundaries[i], embeddings[i-1], boundaries[i+1], embeddings[i+1]))
        return embeddings1

    def forward(self, embeddings, laplacians, boundaries, order, idx, rel):
        embeddings1 = self.conv(self.conv1, embeddings, laplacians, boundaries)
        embeddings2 = self.conv(self.conv2, embeddings1, laplacians, boundaries)
        embeddings3 = self.conv(self.conv3, embeddings2, laplacians, boundaries)
        embeddings4 = self.conv(self.conv4, embeddings3, laplacians, boundaries)
        return self.rel_lin(embeddings4[order][idx]).squeeze()[rel.nonzero()]

class GATModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(GATModel, self).__init__()
        nodes = 2*classes
        self.gcn1 = gnn.GATConv(classes, 2*classes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.gcn2 = gnn.GATConv(2*classes, 4*classes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.gcn3 = gnn.GATConv(4*classes, 8*classes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.gcn4 = gnn.GATConv(8*classes, 16*classes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.rel_lin = nn.Linear(16*classes, classes)
        self.device = device

    def forward(self, graph, feat, order, rel):
        feat = self.gcn1(graph, feat)
        feat = self.gcn2(graph, feat)
        feat = self.gcn3(graph, feat)
        embeddings = self.gcn4(graph, feat).squeeze()[: order+1]
        pooling = torch.mean(embeddings, dim=0)
        return self.rel_lin(pooling).squeeze()[rel.nonzero()]
