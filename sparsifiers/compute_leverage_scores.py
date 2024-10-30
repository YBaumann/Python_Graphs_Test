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
