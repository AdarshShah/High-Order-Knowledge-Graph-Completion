from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer2, SimplicialAttentionLayer3
from torch import nn
from torch.nn import functional as F
import torch
from dgl.nn import pytorch as gnn

class SimplicialModel1(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialModel1, self).__init__()
        self.dim = dim
        
        nodes = 2*classes
        self.conv1 = SimplicialConvolution(classes, nodes)
        self.attn1 = SimplicialAttentionLayer2(nodes)

        self.conv2 = SimplicialConvolution(nodes, 2*nodes)
        self.attn2 = SimplicialAttentionLayer2(2*nodes)

        self.conv3 = SimplicialConvolution(2*nodes, 4*nodes)
        self.attn3 = SimplicialAttentionLayer2(4*nodes)

        self.lin1 = nn.Linear(4*nodes, nodes)
        self.lin2 = nn.Linear(4*nodes, nodes)
        self.rel_lin = nn.Linear(3*nodes, 1)
        self.rel_embed = nn.Embedding(classes, nodes)
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

    def forward(self, embeddings, laplacians, boundaries, order, idx, rel):
        embeddings1 = self.conv(self.conv1, embeddings, laplacians, boundaries)
        embeddings2 = self.attn(self.attn1, embeddings1, laplacians, boundaries)
        embeddings3 = self.conv(self.conv2, embeddings2, laplacians, boundaries)
        embeddings4 = self.attn(self.attn2, embeddings3, laplacians, boundaries)
        embeddings5 = self.conv(self.conv3, embeddings4, laplacians, boundaries)
        embeddings6 = self.attn(self.attn3, embeddings5, laplacians, boundaries)

        # pooling = embeddings6[0].mean(dim=0)
        pooling = torch.stack([em.mean(dim=0) if em is not None else torch.zeros_like(embeddings6[0][0]) for em in embeddings6]).mean(dim=0)
        final_embedding = torch.concat((F.tanh(self.lin1(pooling)),F.tanh(self.lin2(embeddings6[order][idx]))))
        # final_embedding = F.tanh(self.lin2(embeddings6[order][idx]))
        return self.rel_lin(torch.cat((final_embedding.repeat(self.rel_embed.num_embeddings,1),self.rel_embed.weight), dim=1)).squeeze()[rel.nonzero()]
        # return self.rel_lin(final_embedding).squeeze()[rel.nonzero()]

class BaseGNN(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(BaseGNN, self).__init__()
        nodes = 2*classes
        self.gcn1 = gnn.GATConv(classes, nodes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        #GAT
        #R-GCN
        self.gcn2 = gnn.GATConv(nodes, 2*nodes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.gcn3 = gnn.GATConv(2*nodes, 4*nodes, num_heads=1, activation=nn.Tanh(), allow_zero_in_degree=True)
        self.lin1 = nn.Linear(4*nodes, nodes)
        self.lin2 = nn.Linear(2*nodes, 1)
        self.rel_embed = nn.Embedding(classes, nodes)
        self.device = device

    def forward(self, graph, feat, order, rel):
        feat = self.gcn1(graph, feat)
        feat = self.gcn2(graph, feat)
        embeddings = self.gcn3(graph, feat).squeeze()[: order+1]
        pooling = torch.mean(embeddings, dim=0)
        return self.lin2(torch.concat((F.tanh(self.lin1(pooling)).repeat(self.rel_embed.num_embeddings,1), self.rel_embed.weight), dim=1)).squeeze()[rel.nonzero()]


class SimplicialModel2(nn.Module):

    def __init__(self, classes, dim=4, device='cuda:0') -> None:
        super(SimplicialModel2, self).__init__()

        self.dim = dim
        
        nodes = 2*classes
        self.attn1 = SimplicialAttentionLayer3(classes, nodes)
        self.attn2 = SimplicialAttentionLayer3(nodes, 2*nodes)
        self.attn3 = SimplicialAttentionLayer3(2*nodes, 4*nodes)

        self.lin1 = nn.Linear(4*nodes, nodes)
        self.rel_lin = nn.Linear(2*nodes, 1)
        self.rel_embed = nn.Embedding(classes, nodes)
        self.device = device
    
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
    
    def forward(self, embeddings, laplacians, boundaries, order, idx, rel):
        embeddings1 = self.attn(self.attn1, embeddings, laplacians, boundaries)
        embeddings2 = self.attn(self.attn2, embeddings1, laplacians, boundaries)
        embeddings3 = self.attn(self.attn3, embeddings2, laplacians, boundaries)


        # pooling = embeddings6[0].mean(dim=0)
        # pooling = torch.stack([em.mean(dim=0) if em is not None else torch.zeros_like(embeddings3[0][0]) for em in embeddings3]).mean(dim=0)
        # final_embedding = torch.concat((F.tanh(self.lin1(pooling)),F.tanh(self.lin2(embeddings3[order][idx]))))
        final_embedding = F.tanh(self.lin1(embeddings3[order][idx]))
        return self.rel_lin(torch.cat((final_embedding.repeat(self.rel_embed.num_embeddings,1),self.rel_embed.weight), dim=1)).squeeze()[rel.nonzero()]
        # return self.rel_lin(final_embedding).squeeze()[rel.nonzero()]

