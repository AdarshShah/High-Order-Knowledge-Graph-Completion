from torch.utils.data import Dataset, DataLoader
from preprocess.process_dataset import get_dgl_graph
from preprocess.subgraph_extraction import extract_subgraph
from preprocess.graph_to_simplicial_complex import get_simplicial_complex, get_embeddings, _get_simplices, random_sample
import torch


class MyDataset(Dataset):

    max_nodes_per_hop = {
        'cat_edge_cooking' : [10, 100, 150, 200],
        'cat_edge_MAG_10' : [10, 80, 100, 150],
        'cat_edge_algebra_questions' : [3, 20, 25, 30]
    }

    def __init__(self, dataset, num_classes, max_dim=4, dim=None, iterations=10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_dim = max_dim
        self.dataset = dataset
        self.num_classes = num_classes
        self.graph, self.nx_graph = get_dgl_graph(dataset)
        self.iterations = iterations

    def __getitem__(self, index):
        b = 1
        while b!=0:
            simplex, order, pos_label, neg_label = random_sample(self.dataset, num_classes=self.num_classes, max_dim=self.max_dim, dim=self.dim)  # randomly sample 
            to_remove = frozenset(simplex)
            try:
                subgraph = extract_subgraph(simplex, self.graph, h=4, enclosing_sub_graph=True, max_nodes_per_hop=self.max_nodes_per_hop[self.dataset][order])
                isolated_nodes = ((subgraph.in_degrees() == 0) & (subgraph.out_degrees() == 0)).nonzero().squeeze(1)
                subgraph.remove_nodes(isolated_nodes)
                pos_simplex_labels = get_simplicial_complex(subgraph, self.graph, self.nx_graph, self.dataset, self.num_classes, positive=True)
                neg_simplex_labels = get_simplicial_complex(subgraph, self.graph, self.nx_graph, self.dataset, self.num_classes, positive=False)
                pos_embeddings, laplacians, boundaries, idx = get_embeddings(pos_simplex_labels, to_remove, self.num_classes, dim=self.max_dim)
                neg_embeddings, _, _, _ = get_embeddings(neg_simplex_labels, to_remove, self.num_classes, dim=self.max_dim)
                b = 0
            except:
                pass
        return pos_embeddings, neg_embeddings, laplacians, boundaries, order, idx, pos_label, subgraph
    
    def __len__(self):
        return self.iterations