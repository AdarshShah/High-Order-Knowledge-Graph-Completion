import dgl
import networkx as nx
from functools import lru_cache
import torch
import os
from hodgelaplacians import HodgeLaplacians
import numpy as np
from global_parameters import timeout

dirname = os.path.dirname(__file__)

@lru_cache()
def _get_simplices(dataset, num_classes, negative=False):
    '''
    Utility function to extract simplices and their labels.

    Returns:
    --------
    simplices : List[Set], list of simplicies represented by a set.
    labels: List[int], labels corresponding to each simplex.
    '''
    simplices = []
    with open(os.path.join(dirname, f'../datasets/{dataset}/hyperedges.txt'),'r') as hyperedges_file:
        for line in hyperedges_file:
            simplices.append(set([ int(node)-1 for node in line.split() ]))
    labels = []
    with open(os.path.join(dirname,f'../datasets/{dataset}/hyperedge-labels.txt'),'r') as label_file:
        for line in label_file:
            labels.append(min(int(line)-1, num_classes-1))
    if negative:
        np.random.shuffle(labels)
    return simplices, labels

@torch.no_grad()
def random_sample(dataset, num_classes, positive=True, max_dim=4, dim=None):
    '''
    Parameters:
    -----------
    dataset : str, name of the dataset e.g. cat_edge_cooking, cat_edge_DAWN etc.
    num_classes : int, number of classes to which a simplex can be classified into
    max_dim : int, the highest order of simplex allowed.

    Returns:
    --------
    simplex: set, a set of vertices
    dim: int, dimension of the sampled simplex
    pos_label: tensor, a binary tensor of size (num_classes,)
    neg_label: tensor, a binary tensor of size (num_classes,)
    '''
    dim = dim or np.random.randint(max_dim)
    simplicies, pos_labels = _get_simplices(dataset, num_classes)
    _, neg_labels = _get_simplices(dataset, num_classes, negative=True)
    simplex = np.random.choice(simplicies)

    while len(simplex) < dim+1:
        simplex = np.random.choice(simplicies)
    simplex = set(np.random.choice(list(simplex), size=dim+1, replace=False).tolist())

    pos_label = torch.zeros((num_classes,))
    neg_label = torch.zeros((num_classes,))
    for sim, pos_lab, neg_lab in zip(simplicies, pos_labels, neg_labels):
        if simplex.issubset(sim):
            pos_label[pos_lab] = 1
            neg_label[neg_lab] = 1

    return simplex, dim, pos_label, neg_label

@timeout(2)
@torch.no_grad()
def get_simplicial_complex(subgraph:dgl.DGLGraph, graph:dgl.DGLGraph, nx_graph:nx.Graph, dataset, num_classes, positive=True):
    '''
    The function generate simplicial complex from subgraph

    Parameters:
    -----------
    subgraph : dgl.Graph, the sampled subgraph
    graph : dgl.Graph, the original training graph
    nx_graph : nx.Graph, the original training graph with edge attributes 'hyperedge_index'.

    Returns:
    --------
    Simplicial Complex : List of Set
    Labels : List of Int
    '''
    if positive:
        simplices, labels = _get_simplices(dataset, num_classes)
    else:
        simplices, labels = _get_simplices(dataset, num_classes, negative=True)
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

@timeout(2) # prevents large sampled subgraphs 
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
            embeddings.append(torch.stack(H).float())
        except:
            embeddings.append(None)
        try:
            laplacians.append(torch.tensor(simplicial_complex.getHodgeLaplacian(i).todense()).float())
            boundaries.append(torch.tensor(simplicial_complex.getBoundaryOperator(i).todense()).float())
        except:
            laplacians.append(None)
            boundaries.append(None)
    return embeddings, laplacians, boundaries, idx