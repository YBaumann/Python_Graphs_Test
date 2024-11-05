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
import os
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
import pandas as pd

output_direc = sys.argv[1] if len(sys.argv) > 1 else "."

def run_timings(
    datasets,
    tree_function_names,
    sampler_names,
    one_or_k,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    output_dir="timing_csv",
):
    # Initialize list to store all the individual results
    output_dir = output_direc
    results = []

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

        for algo in one_or_k:
            for tree_func_name in tree_function_names:
                for sampler_name in sampler_names:
                    for nr_extra_edges in extra_edges:
                        for repeat_idx in range(repeats):
                            start_time = time.time()  # Start timing

                            # Initialize and compute sparsifier
                            sparsifier = TreeSparsifier(
                                nx_graph,
                                tree_func_name,
                                algo,
                                sampler_name,
                                nr_extra_edges,
                            )
                            sparsifier.compute_sparsifier()
                            sparsifier.get_sparsifier()

                            end_time = time.time()  # End timing
                            elapsed_time = end_time - start_time  # Calculate elapsed time

                            # Store the results
                            results.append({
                                "tree_function": tree_func_name,
                                "sampler": sampler_name,
                                "nr_extra_edges": nr_extra_edges,
                                "nr_initial_edges": m,
                                "nr_nodes": n,
                                "time_taken": elapsed_time,
                                "dataset": dataset.name,
                            })

    # Convert results to DataFrame for easy analysis
    print("Current Path:", os.getcwd())
    csv_file_path = f"{output_dir}/timing_csv/{dataset.name}_timings_results.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_file_path}")


def main():
    datasets2 = [
        # Amazon(root="dataset/", name="Photo"),             # n = 7650, e = 119081
        # Planetoid(root="dataset/", name="Cora"),           # n = 2708, e = 10556
        # Amazon(root="dataset/", name="Computers"),         # n = 13752, e = 245778
        # Planetoid(root="dataset/", name="Pubmed"),         # n = 19717, e = 88651
        # Coauthor(root="dataset/", name="CS"),              # n = 18333, e = 81894
        # Coauthor(root="dataset/", name="Physics"),         # n = 34493, e = 247962
        # Planetoid(root="dataset/", name="CiteSeer"),       # n = 3327, e = 9104
        # AttributedGraphDataset(root="dataset/", name="BlogCatalog"),  # Social network with node attributes
        PygNodePropPredDataset(name="ogbn-arxiv"),         # Open Graph Benchmark arxiv dataset
        PygNodePropPredDataset(name="ogbn-products"),      # Open Graph Benchmark products dataset
        PygNodePropPredDataset(name="ogbn-proteins"),      # Open Graph Benchmark proteins dataset
        ]
    
    run_timings(datasets2,
        [
            "dfs",
            "bfs",
        ],
        [
            "low_degree",
            #"leverage",
            "tree",
            "random",
        ],
        ["k"],
        nr_dividers=10,
        repeats=2,
        nr_sparsifiers=1,)







if __name__ == "__main__":
    main()