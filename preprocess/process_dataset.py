# Generate DGL graph from hyperedges.txt

import os
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

dirname = os.path.dirname(__file__)

@torch.no_grad()
def _generate_nx_graph(dataset:str)->dgl.DGLGraph:
    '''
    Parameters:
    -----------
    dataset : str, name of the dataset e.g. cat_edge_cooking, cat_edge_DAWN etc.
    
    Returns:
    --------
    nx.Graph with edge attribute 'hyperedge_index' pointing to index of simplices to which it belongs to in 'hyperedges.txt'.
    '''
    print(f'> generating networkx graph from {dataset}/hyperedges.txt')
    graphs = []
    edge2idx = {}
    path = os.path.join(dirname, f'../datasets/{dataset}/hyperedges.txt')
    max_node = 0
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            nodes = [ int(node)-1 for node in line.split() ]
            max_node = max(max_node, np.max(nodes))
            graph = nx.complete_graph(nodes, nx.Graph())
            for edge in graph.edges:
                if edge in edge2idx.keys():
                    edge2idx[edge] = edge2idx[edge] + [i]
                else:
                    edge2idx[edge] = [i]
            graphs.append(graph)
    graphs.append(nx.empty_graph(range(max_node+1), nx.Graph()))
    graph = nx.compose_all(graphs)
    nx.set_edge_attributes(graph, edge2idx, 'hyperedge_index')
    return graph.to_directed()

@torch.no_grad()
def get_dgl_graph(dataset:str, skip_cache=False):
    '''
    Parameters:
    -----------
    dataset : str, name of the dataset e.g. cat_edge_cooking, cat_edge_DAWN etc.
    skip_cache : bool, to use cached dataset or not. (default=False)

    Returns: 
    --------
    Directed dgl graph, 
    networkx graph wth edge attributes hyperedge_index = {(u,v):[indices]} 
    '''
    nx_graph = None
    graph = None
    if os.path.exists(os.path.join(dirname,f'../datasets/{dataset}/cache/nx_graph.pkl')) and not skip_cache:
        nx_graph = pickle.load(open(os.path.join(dirname,f'../datasets/{dataset}/cache/nx_graph.pkl'),'rb'))
    if os.path.exists(os.path.join(dirname,f'../datasets/{dataset}/cache/dgl_graph.pkl')) and not skip_cache:
        graph = pickle.load(open(os.path.join(dirname,f'../datasets/{dataset}/cache/dgl_graph.pkl'),'rb'))
    if nx_graph is None:
        nx_graph = _generate_nx_graph(dataset)
        pickle.dump(nx_graph,open(os.path.join(dirname,f'../datasets/{dataset}/cache/nx_graph.pkl'),'wb'))
    else:
        print('> loading nx_graph from cache')
    if graph is None:
        print('> generating dgl undirected graph from networkx graph')
        graph = dgl.from_networkx(nx_graph)
        pickle.dump(graph,open(os.path.join(dirname,f'../datasets/{dataset}/cache/dgl_graph.pkl'),'wb'))
    else:
        print('> loading dgl_graph from cache')
    return graph, nx_graph