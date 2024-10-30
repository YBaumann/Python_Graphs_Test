import networkx as nx
import random
from collections import defaultdict
import numpy as np


class TreeSparsifier:
    graph = None
    sparsified_graph = None
    sparsifier_edges = None
    tree_function = None
    random_sampler = None
    nr_trees: str = None
    nr_sparsifier_edges: int = None
    leverage_scores = None

    def __init__(
        self,
        graph,
        tree_function_name,
        nr_trees,
        random_distribution_name,
        nr_sparsifier_edges,
    ) -> None:
        self.graph = graph
        self.nr_sparsifier_edges = nr_sparsifier_edges
        if nr_trees == "k":
            self.nr_trees = nr_sparsifier_edges // (graph.number_of_nodes() - 1)
        else:
            self.nr_trees = 1
        self.sparsifier_edges = set()
        self.leverage_scores = defaultdict(float)
        self.tree_function = self._get_tree_func(tree_function_name)
        self.random_sampler = self._get_random_sampler(random_distribution_name)

    def get_info(self):
        print("Tree function:", self.tree_function.__name__)
        print("Random sampler:", self.random_sampler.__name__)
        print("Nr trees:", self.nr_trees)
        print("Nr sparsifier edges:", self.nr_sparsifier_edges)
        print("Graph nodes:", self.graph.number_of_nodes())
        print("Graph edges:", self.graph.number_of_edges())
        if self.sparsified_graph is not None:
            print("Sparsified graph nodes:", self.sparsified_graph.number_of_nodes())
            print("Sparsified graph edges:", self.sparsified_graph.number_of_edges())

    def _get_tree_func(self, tree_func_name: str):
        match tree_func_name:
            case "dfs":
                return nx.dfs_edges
            case "bfs":
                return nx.bfs_edges
            case _:
                return None

    def _get_random_sampler(self, random_sampler_name: str):
        match random_sampler_name:
            case "random":
                return self._sample_random_edges
            case "leverage":
                # TODO: Here we approximate with 5 trees, we can also pass this as a parameter
                self._approximate_leverage_scores_for_all_edges()
                return self._sample_with_leverage_scores
            case "tree":
                return self._sample_with_stump
            case "low_degree":
                return self._sample_low_degree_vertices
            case _:
                return None

    def _normalize_edge(self, u, v):
        return (u, v) if u <= v else (v, u)

    def _compute_forest(self, graph):
        forest_trees = set()
        # Iterate over each connected component
        for component in nx.connected_components(graph):
            # Extract the subgraph for the current component
            subgraph = graph.subgraph(component)
            # Choose an arbitrary root node for the BFS
            root = list(subgraph.nodes)[0]
            # Compute BFS tree and add to the list of trees
            tree = list(self.tree_function(subgraph, root))

            forest_trees.update(self._normalize_edge(u, v) for u, v in tree)
        return forest_trees

    def _sample_low_degree_vertices(self, graph, k):
        # Here we simply add edges to vertices proportional to their degree in self.sparisifer_edges
        adjacent_counts = defaultdict(int)

        for u, v in self.sparsifier_edges:
            adjacent_counts[u] += 1
            adjacent_counts[v] += 1

        # Add edges to the k vertices with the lowest degree
        edges_sorted_by_degree = [
            (adjacent_counts[u] + adjacent_counts[v], u, v) for u, v in graph.edges()
        ]
        edges_sorted_by_degree.sort()
        return [self._normalize_edge(u, v) for _, u, v in edges_sorted_by_degree[:k]]

    # TODO: I believe the sorting below is not necessary, but not sure how to optimally fix.
    def _sample_random_edges(self, graph, k):
        component_edges = set(self._normalize_edge(u, v) for u, v in graph.edges())
        return random.sample(sorted(component_edges), min(k, len(component_edges)))

    def _sample_with_stump(self, graph, k):
        tree_edges = list(self._compute_forest(graph))
        return [self._normalize_edge(u, v) for u, v in tree_edges[:k]]

    def _sample_with_leverage_scores(self, graph, k):
        leverage_scores_remaining_edges = [
            self.leverage_scores[self._normalize_edge(u, v)] for (u, v) in graph.edges()
        ]

        sampling_probabilities = leverage_scores_remaining_edges / np.sum(
            leverage_scores_remaining_edges
        )

        sampling_probabilities_bounded = np.nan_to_num(sampling_probabilities, nan=0)

        graph_edges = list(graph.edges())

        sampled_indices = np.random.choice(
            len(graph_edges),
            size=min(k, len(graph_edges)),
            replace=False,
            p=sampling_probabilities_bounded,
        )

        sampled_edges = [graph_edges[i] for i in sampled_indices]

        return sampled_edges

    def _tree_to_leverage_scores(
        self, tree, root, pairs
    ) -> dict[tuple[int, int], float]:
        leverage_scores = defaultdict(float)
        ap_lca = nx.tree_all_pairs_lowest_common_ancestor(tree, root=root, pairs=pairs)
        distances = nx.single_source_shortest_path_length(tree, root)
        for (source, target), lca in ap_lca:
            sorted_key = self._normalize_edge(source, target)
            leverage_scores[sorted_key] = (
                distances[source] + distances[target] - 2 * distances[lca]
            )
        return dict(leverage_scores)

    def _approximate_leverage_scores_for_all_edges(
        self, nr_trees: int = 10
    ) -> dict[tuple[int, int], float]:
        stretches = defaultdict(list)

        for component in nx.connected_components(self.graph):
            subgraph = self.graph.subgraph(component)
            pairs = subgraph.edges()
            for _ in range(nr_trees):
                root = np.random.choice(list(subgraph.nodes))
                tree = nx.dfs_tree(subgraph, source=root)
                tree_leverage_scores = self._tree_to_leverage_scores(tree, root, pairs)
                for key, value in tree_leverage_scores.items():
                    stretches[key].append(value)

        # Get final leverage score approximate
        for key, values in stretches.items():
            inverses = 1 / np.array(values)
            approx_leverage_score = 1 / np.sum(inverses)
            self.leverage_scores[key] = float(approx_leverage_score)

    def compute_sparsifier(self):
        working_graph = self.graph.copy()

        # Compute k trees
        # TODO: This is very awkward code, essentially, we either want to compute k trees or 1 tree, but I want to also leave it open to specify the exact number...
        while (
            self.nr_sparsifier_edges - len(self.sparsifier_edges)
            > self.graph.number_of_nodes()
        ):
            edges = self._compute_forest(working_graph)
            working_graph.remove_edges_from(edges)
            self.sparsifier_edges.update(edges)
            if self.nr_trees == 1:
                break

        # Compute the random sampling
        nr_random_edges_to_add = self.nr_sparsifier_edges - len(self.sparsifier_edges)
        assert nr_random_edges_to_add >= 0
        if nr_random_edges_to_add > 0:
            sampled_edges = self.random_sampler(working_graph, nr_random_edges_to_add)
            self.sparsifier_edges.update(sampled_edges)

    def get_sparsifier(self):
        G = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            G.add_node(node, **data)

        G.add_edges_from(self.sparsifier_edges)
        self.sparsified_graph = G
