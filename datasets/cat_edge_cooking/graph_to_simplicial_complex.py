import dgl
import networkx as nx
from functools import lru_cache
import torch
import os
from hodgelaplacians import HodgeLaplacians

dirname = os.path.dirname(__file__)

@lru_cache()
def _get_simplices(hyperedges_path = 'hyperedges.txt', labels_path = 'hyperedge-labels.txt'):
    '''
    Utility function to extract simplices and their labels
    '''
    labels = []
    with open(os.path.join(dirname,labels_path),'r') as label_file:
        for line in label_file:
            labels.append(int(line)-1)
    simplices = []
    with open(os.path.join(dirname, hyperedges_path),'r') as hyperedges_file:
        for line in hyperedges_file:
            simplices.append(set([ int(node)-1 for node in line.split() ]))
    return simplices, labels

@torch.no_grad()
def get_simplicial_complex(subgraph:dgl.DGLGraph, graph:dgl.DGLGraph, nx_graph:nx.Graph):
    '''
    The function generate simplicial complex from subgraph
    Returns:
    Simplicial Complex : List of Set
    Labels : List of Int
    '''
    simplices, labels = _get_simplices()
    edges = torch.stack(graph.find_edges(subgraph.edata['_ID'])).permute(1,0).numpy()
    nodes = set(subgraph.ndata['_ID'].numpy())
    simplex_labels = {}
    visited = set()
    for u,v in edges:
        for simplex_index in nx_graph[u][v]['hyperedge_index']:
            if simplex_index not in visited:
                visited.add(simplex_index)
                simplex = frozenset(set.intersection(nodes, simplices[simplex_index]))
                simplex_labels[simplex] = torch.zeros(20)
                hl = HodgeLaplacians([simplex])
                for face in hl.face_set:
                    face = frozenset(face)
                    if face not in simplex_labels.keys():
                        simplex_labels[face] = torch.zeros(20)
            hl = HodgeLaplacians([simplex])
            for face in hl.face_set:
                face = frozenset(face)
                simplex_labels[face][labels[simplex_index]] = 1
            simplex_labels[simplex][labels[simplex_index]] = 1
    return simplex_labels

@torch.no_grad()
def get_embeddings(simplex_labels, dim=4):
    '''
    This function is used to obtain initial face embeddings.
    Parameters:
    simplex_labels : Dict where key is face and value is embedding tensor
    Returns:
    embeddings : List of embedding tensor corresponding to each dimension
    '''
    simplicial_complex = HodgeLaplacians(simplex_labels.keys())
    embeddings = []
    for i in range(dim):
        H = []
        for face in simplicial_complex.n_faces(i):
            face = frozenset(face)
            H.append(simplex_labels[face])
        embeddings.append(torch.stack(H))
    return embeddings

