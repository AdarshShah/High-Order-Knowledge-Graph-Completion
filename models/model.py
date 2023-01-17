from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer
from torch import nn
from torch.nn import functional as F
import torch
from dgl.nn import pytorch as gnn

class SimplicialAttentionModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialAttentionModel, self).__init__()
        self.dim = dim
        self.lin_layer = nn.Linear(classes, 2 * classes)
        self.attn_layers = nn.ModuleList()
        for i in range(dim):
            self.attn_layers.append(SimplicialAttentionLayer(2 * classes, 2 * classes))
        self.rel_lin = nn.Linear(4 * classes, classes)
    
    def lin(self, op, embeddings):
        embeddings1 = []
        for i in range(self.dim):
            if embeddings[i] is not None:
                embeddings1.append(op(embeddings[i]))
            else:
                embeddings1.append(None)
        return embeddings1

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
        embeddings = self.lin(self.lin_layer, embeddings)
        for i in range(self.dim):
            embeddings, A_w_1 = self.attn(self.attn_layers[i], embeddings, laplacians, boundaries)
        # to process A_w later
        pooling = torch.zeros_like(embeddings[order][idx])
        for em in embeddings:
            if em is not None:
                pooling += torch.sum(em, dim=0)
        return self.rel_lin(torch.cat((pooling, embeddings[order][idx]))).squeeze()[rel.nonzero()]

class SimplicialConvolutionModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialConvolutionModel, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(1, dim+1):
            self.layers.append(SimplicialConvolution(i * classes, (i+1) * classes))
        self.rel_lin = nn.Linear((1+dim) * classes, classes)
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
        for i in range(self.dim):
            embeddings = self.conv(self.layers[i], embeddings, laplacians, boundaries)
        return self.rel_lin(embeddings[order][idx]).squeeze()[rel.nonzero()]

class GATModel(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(GATModel, self).__init__()
        self.dim = dim
        self.lin_layer = nn.Linear(classes, 2 * classes)
        self.attn_layers = nn.ModuleList()
        for i in range(dim):
            self.attn_layers.append(gnn.GATConv(2 * classes, 2 * classes, num_heads=1, activation=nn.LeakyReLU(), allow_zero_in_degree=True))
        self.rel_lin = nn.Linear(2 * classes, classes)        
        self.device = device

    def forward(self, graph, feat, order, rel):
        feat = self.lin_layer(feat)
        for i in range(self.dim):
            feat = self.attn_layers[i](graph, feat)
        embeddings = feat.squeeze()[: order+1]
        pooling = torch.mean(embeddings, dim=0)
        return self.rel_lin(pooling).squeeze()[rel.nonzero()]
