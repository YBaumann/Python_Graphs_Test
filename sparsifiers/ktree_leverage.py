import networkx as nx
import numpy as np
from collections import defaultdict
from helpers.timer import time_execution
from sparsifiers.onetree_random import get_tree_func
import random


def pick_tuples_from_list(list_of_tuples, k, probabilities):
    """Picks k tuples from a list of tuples."""
    if k == 0:
        return []
    choice = np.random.choice(
        a=len(list_of_tuples),
        size=k,
        p=probabilities,
        replace=False,
    ).tolist()
    return [list_of_tuples[i] for i in choice]


# @time_execution
def tree_to_leverage_scores(graph, tree, root) -> dict[tuple[int, int], float]:
    """
    Given a tree, return the leverage scores of the edges
    """
    leverage_scores = defaultdict(float)
    pairs = graph.edges() - tree.edges()
    ap_lca = nx.tree_all_pairs_lowest_common_ancestor(tree, root=root, pairs=pairs)
    distances = nx.single_source_shortest_path_length(tree, root)
    for (source, target), lca in ap_lca:
        sorted_key = tuple(sorted((source, target)))
        leverage_scores[sorted_key] = (
            distances[source] + distances[target] - 2 * distances[lca]
        )
    return dict(leverage_scores)


def approximate_leverage_scores(
    graph, tree_function, nr_trees: int = 1
) -> dict[tuple[int, int], float]:
    """
    Approximate the leverage scores with nr_trees many trees
    """
    stretches = defaultdict(list)
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    # Get all leverage score estimates
    for _ in range(nr_trees):
        root = np.random.choice(nodes)
        tree = tree_function(graph, source=root)
        tree_leverage_scores = tree_to_leverage_scores(graph, tree, root)
        for key, value in tree_leverage_scores.items():
            stretches[key].append(value)

    # Get final leverage score approximate
    leverage_scores = defaultdict(float)
    for key, values in stretches.items():
        inverses = 1 / np.array(values)
        approx_leverage_score = 1 / np.sum(inverses)
        leverage_scores[key] = float(approx_leverage_score)
    return leverage_scores


def check_leverage_scores(graph):
    # Get max component of the graph
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    max_component = max(components, key=lambda x: x.number_of_nodes())
    tree_function = nx.bfs_tree
    nr_trees = 5
    leverage_scores = approximate_leverage_scores(
        max_component, tree_function, nr_trees
    )
    print(leverage_scores)


def ktree_leverage(nx_graph, k=0, tree_func_name=None, nr_trees=5, **kwargs):
    # print number of nodes edges and k
    print("Number of nodes:", nx_graph.number_of_nodes())
    print("Number of edges:", nx_graph.number_of_edges())
    print("K:", k)
    n = nx_graph.number_of_nodes()
    tree_func = get_tree_func(tree_func_name)
    spanner_edges = set()

    def normalize_edge(u, v):
        return (u, v) if u <= v else (v, u)

    # Get connected components as subgraphs
    components = [
        nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)
    ]

    # Calculate component sizes
    component_sizes = np.array(
        [component.number_of_nodes() for component in components]
    )
    component_sizes = np.zeros(len(component_sizes))
    index_of_max = int(np.argmax(component_sizes))
    component_sizes[index_of_max] = k

    # For each component, compute 'd' DFS trees
    for i, component in enumerate(components):
        n = component.number_of_nodes()
        if i == index_of_max:
            # Compute the number of trees we want to add and the remaining random edges
            nr_trees = int(k / (n - 1))
            print("nr trees:", nr_trees)
            k_new = k - nr_trees * (n - 1)
            if k_new > 0:
                tree_function = nx.bfs_tree
                leverage_scores = approximate_leverage_scores(
                    component, tree_function, nr_trees
                )

            for _ in range(nr_trees):
                source = list(component.nodes())[0]
                tree_edges = list(tree_func(component, source=source))
                spanner_edges.update(normalize_edge(u, v) for u, v in tree_edges)
                # delete edges from the graph
                component.remove_edges_from(tree_edges)

            component_edges = set(normalize_edge(u, v) for u, v in component.edges())
            non_tree_edges = component_edges - set(
                normalize_edge(u, v) for u, v in tree_edges
            )
            if k_new > 0:
                non_tree_edges = list(sorted(non_tree_edges))
                sampling_probabilities = [
                    leverage_scores[tuple(sorted(edge))] for edge in non_tree_edges
                ]

                normalized_sampling_probabilities = np.array(
                    sampling_probabilities
                ) / np.sum(sampling_probabilities)

                random_edges = random.choices(
                    sorted(non_tree_edges),
                    k=k_new,
                    weights=normalized_sampling_probabilities,
                )

                spanner_edges.update(random_edges)
        else:
            source = list(component.nodes())[0]
            tree_edges = list(tree_func(component, source=source))
            spanner_edges.update(normalize_edge(u, v) for u, v in tree_edges)

    print("Size of spanner edges:", len(spanner_edges))
    return spanner_edges
