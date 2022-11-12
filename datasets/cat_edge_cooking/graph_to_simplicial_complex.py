from subgraph_extraction import extract_subgraph
from process_dataset import get_dgl_graph
import dgl
import networkx as nx
from functools import lru_cache
import torch
import os

dirname = os.path.dirname(__file__)

@lru_cache()
def _get_simplices(hyperedges_path = 'hyperedges.txt', labels_path = 'hyperedge-labels.txt'):
    labels = []
    with open(os.path.join(dirname,labels_path),'r') as label_file:
        for line in label_file:
            labels.append(int(line)-1)
    simplices = []
    with open(os.path.join(hyperedges_path),'r') as hyperedges_file:
        for line in hyperedges_file:
            simplices.append(set([ int(node)-1 for node in line.split() ]))
    return simplices, labels

@torch.no_grad()
def get_simplicial_complex(subgraph:dgl.DGLGraph, graph:dgl.DGLGraph, nx_graph:nx.Graph):
    simplices, label = _get_simplices()
    edges = torch.stack(graph.find_edges(subgraph.edata['_ID'])).permute(1,0).numpy()
    nodes = set(subgraph.ndata['_ID'].numpy())
    simplicial_complex = []
    labels = []
    visited = set()
    for u,v in edges:
        for simplex_index in nx_graph[u][v]['hyperedge_index']:
            if simplex_index not in visited:
                visited.add(simplex_index)
                simplicial_complex.append(set.intersection(nodes, simplices[simplex_index]))
                labels.append(label[simplex_index])
    return simplicial_complex, labels

graph, nx_graph = get_dgl_graph()
subgraph = extract_subgraph([5930,3243,3671], graph, h=4, enclosing_sub_graph=True, max_nodes_per_hop=500)
simplicial_complex, labels = get_simplicial_complex(subgraph, graph, nx_graph)
import pdb; pdb.set_trace()