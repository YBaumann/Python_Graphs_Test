import networkx as nx
import numpy as np
import random
from helpers.timer import time_execution
from functools import partial


def get_tree_func(tree_func_name: str):
    match tree_func_name:
        case "dfs":
            return nx.dfs_edges
        case "bfs":
            return nx.bfs_edges
        case _:
            return None


# k is the number of edges in the sparsifier
def ktree_random(
    nx_graph,
    tree_func_name=None,
    k=0,
    **kwargs,
):
    print("Final Graph should have:", k)
    tree_function = get_tree_func(tree_func_name)

    def normalize_edge(u, v):
        return (u, v) if u <= v else (v, u)

    spanner_edges = set()

    # Get connected components as subgraphs
    components = [
        nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)
    ]
    max_component = max(components, key=lambda x: x.number_of_nodes())
    index_max = components.index(max_component)

    n_graph = nx_graph.number_of_nodes()

    # For each component, compute 'd' DFS trees
    for i, component in enumerate(components):
        n = component.number_of_nodes()
        if component == max_component:
            nr_trees = int(k / (n - 1))
            print("nr trees:", nr_trees)
            for _ in range(nr_trees):
                source = list(component.nodes())[0]
                tree_edges = list(tree_function(component, source=source))
                spanner_edges.update(normalize_edge(u, v) for u, v in tree_edges)
                # delete edges from the graph
                component.remove_edges_from(tree_edges)

            component_edges = set(normalize_edge(u, v) for u, v in component.edges())
            non_tree_edges = component_edges - set(
                normalize_edge(u, v) for u, v in tree_edges
            )
            k_new = k - nr_trees * (n - 1)
            print("k_new:", k_new)
            if k_new > 0:
                k_new = min(k_new, len(non_tree_edges))
                random_edges = random.sample(sorted(non_tree_edges), k_new)
                spanner_edges.update(random_edges)
        else:
            source = list(component.nodes())[0]
            tree_edges = list(tree_function(component, source=source))
            spanner_edges.update(normalize_edge(u, v) for u, v in tree_edges)
    print("Size of spanner edges:", len(spanner_edges))
    return spanner_edges
