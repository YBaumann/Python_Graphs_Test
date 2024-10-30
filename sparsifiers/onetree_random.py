import networkx as nx
import random
import numpy as np
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


def one_tree_random(nx_graph, k=0, tree_func_name=None, **kwargs):
    print("Enters one tree")
    print("K:", k)
    n = nx_graph.number_of_nodes()
    k -= n - 1
    tree_func = get_tree_func(tree_func_name)

    def normalize_edge(u, v):
        return (u, v) if u <= v else (v, u)

    spanner_edges = set()

    # Get connected components as subgraphs
    components = [
        nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)
    ]

    # Calculate component sizes
    component_sizes = np.array(
        [component.number_of_nodes() for component in components]
    )
    component_sizes = np.zeros(len(component_sizes))
    indices_of_max = int(np.argmax(component_sizes))
    component_sizes[indices_of_max] = k

    for idx, component in enumerate(components):
        k_component = component_sizes[idx]

        # Compute tree and add its edges, use any sourc enode in the component
        tree_edges = list(tree_func(component, source=list(component.nodes())[0]))
        spanner_edges.update(normalize_edge(u, v) for u, v in tree_edges)

        # Identify non-tree edges
        component_edges = set(normalize_edge(u, v) for u, v in component.edges())
        non_tree_edges = component_edges - set(
            normalize_edge(u, v) for u, v in tree_edges
        )

        # Sample random non-tree edges
        sample_size = int(
            min(k_component, len(non_tree_edges))
        )  # This sets the sample size to zero for non max
        if sample_size > 0:
            random_edges = random.sample(sorted(non_tree_edges), sample_size)
            spanner_edges.update(random_edges)

    return spanner_edges
