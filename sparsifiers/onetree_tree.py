import networkx as nx
import random
import numpy as np
from helpers.timer import time_execution
from functools import partial


def normalize_edge(u, v):
    return (u, v) if u <= v else (v, u)


def compute_forest(graph, tree_func):
    forest_trees = set()
    # Iterate over each connected component
    for component in nx.connected_components(graph):
        # Extract the subgraph for the current component
        subgraph = graph.subgraph(component)
        # Choose an arbitrary root node for the BFS
        root = list(subgraph.nodes)[0]
        # Compute BFS tree and add to the list of trees
        tree = list(tree_func(subgraph, root))

        forest_trees.update(normalize_edge(u, v) for u, v in tree)
    return forest_trees


def get_tree_func(tree_func_name: str):
    match tree_func_name:
        case "dfs":
            return nx.dfs_edges
        case "bfs":
            return nx.bfs_edges
        case _:
            return None


def onetree_tree(nx_graph, k=0, tree_func_name=None, **kwargs):
    print("Enters one tree")
    print("K:", k)
    n = nx_graph.number_of_nodes()
    tree_func = get_tree_func(tree_func_name)

    spanner_edges = set()

    edges = compute_forest(nx_graph, tree_func)
    nx_graph.remove_edges_from(edges)
    spanner_edges.update(edges)

    k_new = k - len(spanner_edges)

    if k_new > 0:
        edges = compute_forest(nx_graph, tree_func)
        spanner_edges.update(list(edges)[:k_new])

    print("Size of spanner edges:", len(spanner_edges))
    return spanner_edges
