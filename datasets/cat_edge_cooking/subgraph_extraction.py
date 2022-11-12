import random

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from tqdm import tqdm
import os

dirname = os.path.dirname(__file__)

def extract_subgraph(ind, graph, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    '''
    The main function to perform step 1 (Local Subgraph Extraction)
    ind : node ids list
    graph : dgl graph
    h : predefined hop
    '''
    # extract the h-hop enclosing subgraphs around link 'ind'
    Adj = graph.adj(scipy_fmt='coo')
    A_incidence = incidence_matrix(Adj)
    A_incidence += A_incidence.T

    neighbors = [ get_neighbor_nodes(set([u]), A_incidence, h, max_nodes_per_hop) for u in ind ]

    subgraph_nei_nodes_int = set.intersection(*neighbors)
    subgraph_nei_nodes_un = set.union(*neighbors)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    return dgl.node_subgraph(graph, subgraph_nodes)

def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)

def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)

def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def incidence_matrix(adj):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj.shape
    rows += adj.row.tolist()
    cols += adj.col.tolist()
    dats += adj.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)