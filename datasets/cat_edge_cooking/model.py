from layers.simplicial_convolution import SimplicialConvolution, SimplicialAttentionLayer
from torch import nn
from torch.nn import functional as F

class CookingModel(nn.Module):

    def __init__(self) -> None:
        super(CookingModel, self).__init__()
        self.conv1 = SimplicialConvolution(20, 20)
        self.attn1 = SimplicialAttentionLayer(20)
        self.conv2 = SimplicialConvolution(20, 20)
        self.lin = nn.Linear(20, 10)
    
    def forward(self, embeddings, laplacians, boundaries, order, idx):
        embeddings1 = []
        for i in range(4):
            if i==0:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], None, None, boundaries[i+1], embeddings[i+1])))
            elif i==3:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], boundaries[i-1], embeddings[i-1], None, None)))
            else:
                embeddings1.append(F.tanh(self.conv1(laplacians[i], embeddings[i], boundaries[i-1], embeddings[i-1], boundaries[i+1], embeddings[i+1])))
        embeddings2 = []
        for i in range(4):
            if laplacians[i] is not None:
                embeddings2.append(self.attn1(embeddings1[i], laplacians[i]))
        embeddings3 = []
        for i in range(4):
            if i==0:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], None, None, boundaries[i+1], embeddings[i+1])))
            elif i==3:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], boundaries[i-1], embeddings[i-1], None, None)))
            else:
                embeddings3.append(F.tanh(self.conv2(laplacians[i], embeddings2[i], boundaries[i-1], embeddings[i-1], boundaries[i+1], embeddings[i+1])))
        
        return nn.Linear(embeddings3[order][idx])


