import dgl
import networkx as nx
from functools import lru_cache
import torch
import os
from hodgelaplacians import HodgeLaplacians
import numpy as np

dirname = os.path.dirname(__file__)
device = 'cuda:0'

@lru_cache()
def _get_simplices(dataset, num_classes):
    '''
    Utility function to extract simplices and their labels
    '''
    labels = []
    with open(os.path.join(dirname,f'../datasets/{dataset}/hyperedge-labels.txt'),'r') as label_file:
        for line in label_file:
            labels.append(min(int(line)-1, num_classes-1))
    simplices = []
    with open(os.path.join(dirname, f'../datasets/{dataset}/hyperedges.txt'),'r') as hyperedges_file:
        for line in hyperedges_file:
            simplices.append(set([ int(node)-1 for node in line.split() ]))
    return simplices, labels

@lru_cache()
@torch.no_grad()
def random_sample(dataset, num_classes, max_dim=4):
    dim = np.random.randint(max_dim)
    simplicies, labels = _get_simplices(dataset, num_classes)
    simplex = np.random.choice(simplicies)
    while len(simplex) < dim+1:
        simplex = np.random.choice(simplicies)
    simplex = set(np.random.choice(list(simplex), size=dim+1, replace=False).tolist())
    label = torch.zeros((num_classes,)).to(device)
    for sim, lab in zip(simplicies, labels):
        if simplex.issubset(sim):
            label[lab] = 1
    return simplex, dim, label


@torch.no_grad()
def get_simplicial_complex(subgraph:dgl.DGLGraph, graph:dgl.DGLGraph, nx_graph:nx.Graph, dataset, num_classes):
    '''
    The function generate simplicial complex from subgraph
    Returns:
    Simplicial Complex : List of Set
    Labels : List of Int
    '''
    simplices, labels = _get_simplices(dataset, num_classes)
    edges = torch.stack(graph.find_edges(subgraph.edata['_ID'])).permute(1,0).numpy()
    nodes = set(subgraph.ndata['_ID'].numpy())
    simplex_labels = {}
    visited = set()
    for u,v in edges:
        for simplex_index in nx_graph[u][v]['hyperedge_index']:
            simplex = frozenset(set.intersection(nodes, simplices[simplex_index]))
            if simplex_index not in visited:
                visited.add(simplex_index)
                simplex_labels[simplex] = torch.zeros(num_classes)
                hl = HodgeLaplacians([simplex])
                for face in hl.face_set:
                    face = frozenset(face)
                    if face not in simplex_labels.keys():
                        simplex_labels[face] = torch.zeros(num_classes)            
            hl = HodgeLaplacians([simplex])
            for face in hl.face_set:
                face = frozenset(face)
                simplex_labels[face][labels[simplex_index]] = 1
            simplex_labels[simplex][labels[simplex_index]] = 1
    return simplex_labels

@torch.no_grad()
def get_embeddings(simplex_labels, to_remove, num_classes, dim=4):
    '''
    This function is used to obtain initial face embeddings.
    Parameters:
    simplex_labels : Dict where key is face and value is embedding tensor
    Returns:
    embeddings : List of embedding tensor corresponding to each dimension
    '''
    if type(to_remove)!=frozenset:
        to_remove = frozenset(to_remove)
    simplex_labels[to_remove] = torch.zeros(num_classes)
    simplicial_complex = HodgeLaplacians(simplex_labels.keys(), maxdimension=dim)
    embeddings = []
    laplacians = []
    boundaries = []
    idx = 0
    for i in range(dim):
        H = []
        for z, face in enumerate(simplicial_complex.n_faces(i)):
            face = frozenset(face)
            H.append(simplex_labels[face]) if face in simplex_labels.keys() else H.append(torch.zeros((num_classes,)))
            if face == to_remove:
                idx = z
        try:
            embeddings.append(torch.stack(H).float().to(device))
        except:
            embeddings.append(None)
        try:
            laplacians.append(torch.tensor(simplicial_complex.getHodgeLaplacian(i).todense()).float().to(device))
            boundaries.append(torch.tensor(simplicial_complex.getBoundaryOperator(i).todense()).float().to(device))
        except:
            laplacians.append(None)
            boundaries.append(None)
    return embeddings, laplacians, boundaries, idx