"""Data reading and writing utilities."""

import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable

def normalize_adjacency(graph, args):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    for node in graph.nodes():
        graph.add_edge(node, node)
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    L = sparse.csr_matrix(nx.laplacian_matrix(graph),
                          dtype=np.float32)

    degs = sparse.csr_matrix(sparse.coo_matrix((degs, (ind, ind)),
                                               shape=L.shape,
                                               dtype=np.float32))

    propagator = sparse.eye(L.shape[0])-args.gamma*degs.dot(L)
    return propagator

def read_graph(args):
    """
    Method to read graph and create a target matrix with adjacency matrix powers.
    :param args: Arguments object.
    :return powered_P: Target matrix.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(args.edge_path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    P = normalize_adjacency(graph, args)
    powered_P = P
    if args.order > 1:
        for _ in tqdm(range(args.order-1), desc="Adjacency matrix powers"):
            powered_P = powered_P.dot(P)
    return powered_P

def read_features(args):
    if args.features == "sparse":
        features = read_sparse_features(args.feature_path)
    else:
        features = read_dense_features(args.feature_path)
    return features

def read_dense_features(feature_path):
    """
    Method to get node feaures.
    :param feature_path: Path to the node features.
    :return X: Node features.
    """
    features = pd.read_csv(feature_path)
    features = np.array(features)[:, 1:]
    return features

def read_sparse_features(feature_path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    features = json.load(open(feature_path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
                                                   shape=(node_count, feature_count),
                                                   dtype=np.float32))
    return features

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())
