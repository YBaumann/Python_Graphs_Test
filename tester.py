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

from torch_geometric.datasets import Amazon, Planetoid, Coauthor
from torch_geometric.datasets import AttributedGraphDataset, CitationFull, KarateClub

import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from ogb.nodeproppred import PygNodePropPredDataset


def main():
    datasets2 = [
        Amazon(root="dataset/", name="Photo"),             # n = 7650, e = 119081
        Planetoid(root="dataset/", name="Cora"),           # n = 2708, e = 10556
        Amazon(root="dataset/", name="Computers"),         # n = 13752, e = 245778
        Planetoid(root="dataset/", name="Pubmed"),         # n = 19717, e = 88651
        Coauthor(root="dataset/", name="CS"),              # n = 18333, e = 81894
        Coauthor(root="dataset/", name="Physics"),         # n = 34493, e = 247962
        Planetoid(root="dataset/", name="CiteSeer"),       # n = 3327, e = 9104
        PygNodePropPredDataset(name="ogbn-arxiv"),         # Open Graph Benchmark arxiv dataset
        PygNodePropPredDataset(name="ogbn-products"),      # Open Graph Benchmark products dataset
        PygNodePropPredDataset(name="ogbn-proteins"),      # Open Graph Benchmark proteins dataset
        AttributedGraphDataset(root="dataset/", name="BlogCatalog"),  # Social network with node attributes
        ]
    return


if __name__ == "__main__":
    main()
