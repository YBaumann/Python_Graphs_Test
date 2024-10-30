### The idea here is to have a class that can implement both
import networkx as nx
import random
import numpy as np
from collections import defaultdict
from helpers.timer import time_execution


class tree_sparsifier:
    tree_func = None
    random_sampler = None
    nr_trees = 1

    def _get_tree_func(self, tree_func_name: str):
        match tree_func_name:
            case "dfs":
                return nx.dfs_edges
            case "bfs":
                return nx.bfs_edges
            case _:
                return None

    def _normalize_edge(self, u, v):
        return (u, v) if u <= v else (v, u)

    def _sample_random_edges(self, graph, k):
        component_edges = set(self._normalize_edge(u, v) for u, v in graph.edges())
        return random.sample(sorted(component_edges), k)

    def _pick_tuples_from_list(self, list_of_tuples, k, probabilities):
        if k == 0:
            return []
        choice = np.random.choice(
            a=len(list_of_tuples),
            size=k,
            p=probabilities,
            replace=False,
        ).tolist()
        return [list_of_tuples[i] for i in choice]

    def _tree_to_leverage_scores(
        self, graph, tree, root
    ) -> dict[tuple[int, int], float]:
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
        self, graph, tree_function, nr_trees: int = 1
    ) -> dict[tuple[int, int], float]:
        stretches = defaultdict(list)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        # Get all leverage score estimates
        for _ in range(nr_trees):
            root = np.random.choice(nodes)
            tree = tree_function(graph, source=root)
            tree_leverage_scores = self._tree_to_leverage_scores(graph, tree, root)
            for key, value in tree_leverage_scores.items():
                stretches[key].append(value)

        # Get final leverage score approximate
        leverage_scores = defaultdict(float)
        for key, values in stretches.items():
            inverses = 1 / np.array(values)
            approx_leverage_score = 1 / np.sum(inverses)
            leverage_scores[key] = float(approx_leverage_score)
        return leverage_scores

    def _sample_with_leverage_scores(self, graph, k):
        leverage_scores = self.approximate_leverage_scores(
            graph, self.tree_func, self.nr_trees
        )
        probabilities = np.array(list(leverage_scores.values()))
        probabilities /= np.sum(probabilities)
        return self._pick_tuples_from_list(
            list(leverage_scores.keys()), k, probabilities
        )

    def _sample_with_stump(self, graph, k):
        tree_edges = list(self.tree_func(graph, source=list(graph.nodes())[0]))
        return sorted(self._normalize_edge(u, v) for u, v in tree_edges[:k])

    def _get_random_sampler(self, random_sampler_name: str):
        match random_sampler_name:
            case "random":
                return self._sample_random_edges
            case "leverage":
                return self._sample_with_leverage_scores
            case "tree":
                return self._sample_with_stump
            case _:
                return None

    def __init__(self, tree_func: str, random_sampler: str, nr_trees: int) -> None:
        self.tree_func = self._get_tree_func(tree_func)
        self.random_sampler = self._get_random_sampler(random_sampler)
        self.nr_trees = nr_trees

    def _compute_one_tree_sparsifier(self, graph, m):
        # m includes all the edges in the trees
        n = graph.number_of_nodes()
        spanner_edges = set()
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        # Calculate component sizes
        component_sizes = np.array(
            [component.number_of_nodes() for component in components]
        )
        max_index = int(np.argmax(component_sizes))

        for idx, component in enumerate(components):
            k_component = component_sizes[idx]
            tree_edges = list(
                self._tree_func(component, source=list(component.nodes())[0])
            )
            spanner_edges.update(self._normalize_edge(u, v) for u, v in tree_edges)
            component_edges = set(
                self._normalize_edge(u, v) for u, v in component.edges()
            )
            non_tree_edges = component_edges - set(
                self._normalize_edge(u, v) for u, v in tree_edges
            )

            # Sample random non-tree edges
            if idx == max_index:
                sample_size = int(min(k_component, len(non_tree_edges)))
                sampled_edges = self.random_sampler(component, sample_size)
                spanner_edges.update(sampled_edges)
        return spanner_edges

    def compute_sparsifier(self, graph, m):
        if self.nr_trees == 1:
            return self._compute_one_tree_sparsifier(graph, m)
        else:
            return self._compute_k_tree_sparsifier(graph, m)
