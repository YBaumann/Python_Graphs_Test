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
import time
import sys

# Read input arguments
if len(sys.argv) < 9:
    raise ValueError("Please provide the required arguments: dataset_name, nr_dividers, nr_repeats, nr_sparsifier, tree_function_names, sampler_names, and one_or_k.")

dataset_name = sys.argv[1]
nr_dividers_input = int(sys.argv[2])
nr_repeats_input = int(sys.argv[3])
nr_sparsifier_input = int(sys.argv[4])
nr_epochs_input = int(sys.argv[5])

# Parse tree function names, sampler names, and one_or_k from the command line as comma-separated lists
tree_function_names = sys.argv[6].split(",")
sampler_names = sys.argv[7].split(",")
one_or_k = sys.argv[8].split(",")

def train_on_multiple_sparsifiers_refactor_to_csv(
    datasets,
    tree_function_names,
    sampler_names,
    one_or_k,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    nr_epochs=100,
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
                                sparsifier_start_time = time.time()
                                sparsifier.compute_sparsifier()
                                sparsifier_end_time = time.time()
                                sparsification_time = sparsifier_end_time - sparsifier_start_time
                                sparsifier.get_sparsifier()
                                sparse_graphs.append(
                                    from_networkx(sparsifier.sparsified_graph)
                                )

                                # Train on GCN with the sparse graphs
                            start_time = time.time()  # Start timing
                            train_acc, val_acc = train_GCN_multisparse(
                                sparse_graphs,
                                dataset.num_features,
                                dataset.num_classes,
                                nr_epochs,
                                train_split=0.25,
                                pref=0,
                                plot_flag=False,
                            )
                            end_time = time.time()  # End timing
                            elapsed_time = end_time - start_time  # Calculate elapsed time

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
                                        train_acc[-5:]
                                    ),
                                    "val_acc": np.mean(
                                        val_acc[-5:]
                                    ),
                                    "nr_nodes": n,
                                    "nr_edges": m,
                                    "nr_edges_sparsifier": sparse_graphs[
                                        0
                                    ].edge_index.shape[1],
                                    "time_taken_training": elapsed_time,
                                    "nr_sparsifiers": nr_sparsifiers,
                                    "sparsification_time": sparsification_time,
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
    # Select the dataset based on the provided argument
    if dataset_name == "Cora":
        datasets = [Planetoid(root="dataset/", name="Cora")]
    elif dataset_name == "Photo":
        datasets = [Amazon(root="dataset/", name="Photo")]
    elif dataset_name == "Computers":
        datasets = [Amazon(root="dataset/", name="Computers")]
    elif dataset_name == "Pubmed":
        datasets = [Planetoid(root="dataset/", name="Pubmed")]
    elif dataset_name == "CS":
        datasets = [Coauthor(root="dataset/", name="CS")]
    elif dataset_name == "Physics":
        datasets = [Coauthor(root="dataset/", name="Physics")]
    elif dataset_name == "CiteSeer":
        datasets = [Planetoid(root="dataset/", name="CiteSeer")]
    elif dataset_name == "ogbn-arxiv":
        datasets = [PygNodePropPredDataset(name="ogbn-arxiv")]
    elif dataset_name == "ogbn-products":
        datasets = [PygNodePropPredDataset(name="ogbn-products")]
    elif dataset_name == "ogbn-proteins":
        datasets = [PygNodePropPredDataset(name="ogbn-proteins")]
    elif dataset_name == "BlogCatalog":
        datasets = [AttributedGraphDataset(root="dataset/", name="BlogCatalog")]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    train_on_multiple_sparsifiers_refactor_to_csv(
        datasets,
        tree_function_names=tree_function_names,
        sampler_names=sampler_names,
        one_or_k=one_or_k,
        nr_dividers=nr_dividers_input,
        repeats=nr_repeats_input,
        nr_sparsifiers=nr_sparsifier_input,
        nr_epochs=nr_epochs_input,
    )

if __name__ == "__main__":
    main()