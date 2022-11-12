# Generate DGL graph from hyperedges.txt

import random

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from tqdm import tqdm
from subgraph_extraction import extract_subgraph
import pickle
import os


def _generate_nx_graph(path = '/home/adarsh/H-KGC/datasets/cat-edge-Cooking/hyperedges.txt'):
    '''
    Returns 
    nx.Graph with edge attribute 'hyperedge_index' pointing to index of simplices to which it belongs to in 'hyperedges.txt'.
    '''
    print('> generating networkx graph from hyperedges.txt')
    graphs = []
    edge2idx = {}
    with open(path, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            nodes = [ int(node) for node in line.split() ]
            graph = nx.complete_graph(nodes, nx.Graph())
            for edge in graph.edges:
                if edge in edge2idx.keys():
                    edge2idx[edge] = edge2idx[edge] + [i]
                else:
                    edge2idx[edge] = [i]
            graphs.append(graph)
    graph = nx.compose_all(graphs)
    nx.set_edge_attributes(graph, edge2idx, 'hyperedge_index')
    return graph.to_directed()

def get_dgl_graph():
    '''
    Returns 
    Directed dgl graph, 
    networkx graph wth edge attributes hyperedge_index = {(u,v):[indices]} 
    '''
    nx_graph = None
    graph = None
    if os.path.exists('datasets/cat-edge-Cooking/cache/nx_graph.pkl'):
        nx_graph = pickle.load(open('datasets/cat-edge-Cooking/cache/nx_graph.pkl','rb'))
    if os.path.exists('datasets/cat-edge-Cooking/cache/dgl_graph.pkl'):
        graph = pickle.load(open('datasets/cat-edge-Cooking/cache/dgl_graph.pkl','rb'))
    if nx_graph is None:
        nx_graph = _generate_nx_graph()
        pickle.dump(nx_graph,open('datasets/cat-edge-Cooking/cache/nx_graph.pkl','wb'))
    else:
        print('> loading nx_graph from cache')
    if graph is None:
        print('> generating dgl undirected graph from networkx graph')
        graph = dgl.from_networkx(nx_graph)
        pickle.dump(graph,open('datasets/cat-edge-Cooking/cache/dgl_graph.pkl','wb'))
    else:
        print('> loading dgl_graph from cache')
    return graph, nx_graph

graph, nx_graph = get_dgl_graph()
subgraph = extract_subgraph([5930,3243,3671], graph, h=4, enclosing_sub_graph=True, max_nodes_per_hop=50)
import pdb; pdb.set_trace()