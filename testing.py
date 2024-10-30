from sparsifiers.sparsifier_class import *
from torch_geometric.datasets import (
    CitationFull,
    GNNBenchmarkDataset,
    Coauthor,
    Reddit2,
    AttributedGraphDataset,
    Amazon,
    Planetoid,
    KarateClub,
)
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


def main():
    dataset = Planetoid(root="dataset/", name="Cora")
    graph = dataset[0]
    nx_graph = to_networkx(graph, to_undirected=True)
    trees = ["dfs", "bfs"]
    random_samplers = ["leverage", "tree", "random"]
    for tree in trees:
        for random_sampler in random_samplers:
            sparsifier = TreeSparsifier(
                nx_graph, tree, 1, random_sampler, nx_graph.number_of_nodes() + 10
            )
            sparsifier.compute_sparsifier()
            sparsifier.get_sparsifier()
            sparsifier.get_info()
            del sparsifier
            print("--------------------------------------------------")
    return


if __name__ == "__main__":
    main()
