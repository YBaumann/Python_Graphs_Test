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
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
import numpy as np
import torch
from functools import partial
from sparsifiers.sparsifier_class import TreeSparsifier


from trainer.GCN import GCN
from trainer.GeometricMedianGCN import GeometricMedianGCN
from trainer.simple_train import train_GCN_multisparse
import os

from torch_geometric.datasets import KarateClub

import csv


def train_on_multiple_sparsifiers_refactor_to_csv(
    datasets,
    tree_function_names,
    sampler_names,
    one_or_k,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    output_dir="output_csv",
):
    for dataset in datasets:
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        print("Full", graph)
        n = graph.x.shape[0]
        # It is currently directed, we will process it as undirected
        nx_graph = to_networkx(graph, to_undirected=True, node_attrs=["x", "y"])

        m = nx_graph.number_of_edges()
        min_number_of_edges = n - 1
        remaining_edges = m - min_number_of_edges
        extra_edges = [
            min_number_of_edges + remaining_edges // i
            for i in range(nr_dividers + 1, 1, -1)
        ]
        print("edges to add", extra_edges)
        print("total edges", m)

        # Initialize list to store all the individual results
        results = []
        for algo in one_or_k:
            for tree_func_name in tree_function_names:
                for sampler_name in sampler_names:
                    for nr_extra_edges in extra_edges:
                        for repeat_idx in range(repeats):
                            print("New Sparsifier")
                            sparse_graphs = []
                            for sparsifier_idx in range(nr_sparsifiers):
                                sparsifier = TreeSparsifier(
                                    nx_graph,
                                    tree_func_name,
                                    algo,
                                    sampler_name,
                                    nr_extra_edges,
                                )
                                sparsifier.compute_sparsifier()
                                sparsifier.get_sparsifier()
                                sparse_graphs.append(
                                    from_networkx(sparsifier.sparsified_graph)
                                )

                                # Train on GCN with the sparse graphs
                            train_acc, val_acc = train_GCN_multisparse(
                                sparse_graphs,
                                dataset.num_features,
                                dataset.num_classes,
                                train_split=0.25,
                                pref=0,
                                plot_flag=False,
                            )

                            # Append individual results (final training/validation accuracy)
                            results.append(
                                {
                                    "dataset": dataset.name,
                                    "tree_func": sparsifier.tree_function.__name__,
                                    "sampler": sparsifier.random_sampler.__name__,
                                    "one_or_k": algo,
                                    "nr_extra_edges": nr_extra_edges,
                                    "run_idx": repeat_idx + 1,
                                    "train_acc": np.mean(
                                        train_acc[-10:]
                                    ),  # The final training accuracy of the run
                                    "val_acc": np.mean(
                                        val_acc[-10:]
                                    ),  # The final validation accuracy of the run
                                    "nr_nodes": n,
                                    "nr_edges": m,
                                    "nr_edges_sparsifier": sparse_graphs[
                                        0
                                    ].edge_index.shape[1],
                                }
                            )

        # Save all individual results to CSV
        csv_file_path = f"{output_dir}/{dataset.name}_sparsifiers_results.csv"
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {csv_file_path}")


def main():
    datasets2 = [
        # Amazon(root="dataset/", name="Photo"),  # n = 7650, e = 119081
        Planetoid(root="dataset/", name="Cora"),  # n = 2708, e = 10556
        # Amazon(root="dataset/", name="Computers"),  # n = 13752, e = 245778
        # Planetoid(root="dataset/", name="Pubmed"),  # n = 19717 e = 88651
        # Coauthor(root="dataset/", name="CS"),  # n = 18333, e = 81894
        # Coauthor(root="dataset/", name="Physics"),  # n = 34493, e = 247962
        # Planetoid(root="dataset/", name="CiteSeer"),  # n = 3327, e = 9104
        # PygNodePropPredDataset(name="ogbn-arxiv"),
        # AttributedGraphDataset(root="dataset/", name="BlogCatalog"),
        ### CitationFull(root="dataset/", name="Cora"),
        # KarateClub(),
    ]

    train_on_multiple_sparsifiers_refactor_to_csv(
        datasets2,
        [
            "dfs",
            "bfs",
        ],
        [
            "low_degree",
            "leverage",
            "tree",
            "random",
        ],
        ["k"],
        nr_dividers=1,
        repeats=1,
        nr_sparsifiers=1,
    )

    # train_with_different_depths(datasets2[0], compute_four_add_spanner)


if __name__ == "__main__":
    main()
    # just_plot()
