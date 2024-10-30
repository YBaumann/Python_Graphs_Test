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
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import random
import torch
from functools import partial


from sparsifiers.ktree_tree import ktree_tree
from sparsifiers.onetree_tree import onetree_tree
from sparsifiers.onetree_leverage import one_tree_leverage
from sparsifiers.onetree_random import one_tree_random
from sparsifiers.ktree_random import ktree_random
from sparsifiers.ktree_leverage import ktree_leverage
from sparsifiers.archive.k_tree_google_only_tree import compute_k_tree_google_modified
from sparsifiers.archive.two_add_sparsifier import compute_two_add_spanner
from sparsifiers.archive.four_add_sparsifier import compute_four_add_spanner
from sparsifiers.archive.mst_n_random import compute_random_tree_and_n_edges
from sparsifiers.archive.bfs_n_random import (
    compute_bfs_and_n_random_edges,
    compute_dfs_and_n_random_edges,
    compute_dfs_and_n_random_edges_disconnected,
)
from sparsifiers.archive.sparsifiers_google import compute_k_tree_google
from sparsifiers.archive.tree_k_approx_leverage import (
    tree_k_approx_leverage_based,
    tree_k_approx_leverage_based_disconnected,
)

from sparsifiers.archive.bfs_k_leverage_score import (
    bfs_k_leverage_rank,
    m_bfs_k_leverage_score,
    bfs_k_leverage_vertex_based,
)
from trainer.GCN import GCN, GCN_custom
from trainer.simple_train import train_GCN, train_GCN_multisparse
from sparsifiers.generate_sparsifier import (
    generate_sparsifier,
    generate_sparsifier_with_sampling,
)
from helpers.timer import time_execution
import os

from torch_geometric.datasets import KarateClub

import csv


def train_with_different_depths(dataset, sparsifier_algo, depths=[2, 3, 4, 5]):
    pref = random.randint(0, 1000)
    os.makedirs("training_plots/" + str(pref))
    graph = dataset[0]
    if len(graph.y.shape) > 1:
        graph.y = graph.y.squeeze()
    print("Full", graph)
    print("Name", dataset.name)
    train_accs = []
    val_accs = []
    for depth in depths:
        GCN = GCN_custom(dataset.num_features, dataset.num_classes, depth, 16)
        sparse_graph = generate_sparsifier(
            to_networkx(graph, to_undirected=True, node_attrs=["x", "y"]),
            sparsifier_algo,
            sparsifier_algo.__name__,
        )
        sparse_graph = from_networkx(sparse_graph)
        train_acc, val_acc = train_GCN(
            sparse_graph,
            dataset.num_features,
            dataset.num_classes,
            sparsifier_algo.__name__,
            dataset.name,
            pref=pref,
            plot_flag=True,
            model=GCN,
        )
        train_accs.append(train_acc[-1])
        val_accs.append(val_acc[-1])
    plt.clf()
    plt.plot(depths, train_accs, label="Train")
    plt.plot(depths, val_accs, label="Validation")
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.savefig("acc_decreasing_with_depth.png")


def train_on_multiple_sparsifiers_refactor(
    datasets,
    sparsifier_algos,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    include_general=False,
    nr_edges_multiplier=200,
):
    for dataset in datasets:
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        print("Full", graph)
        n = graph.x.shape[0]
        nr_edges_multiplier = n
        extra_edges = [nr_edges_multiplier * i for i in range(1, nr_dividers + 1)]
        print("edges to add", extra_edges)
        # extra_edges = [n - 200, 2 * n - 200, 3 * n - 200]
        nx_graph = to_networkx(graph, to_undirected=True, node_attrs=["x", "y"])

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Total number of sparsifiers
        n_sparsifiers = len(sparsifier_algos)
        # Calculate an offset step size (adjust as needed)
        offset_step = 0.02 * nr_edges_multiplier
        # Generate offsets for each sparsifier, centered around 0
        sparsifier_offsets = np.linspace(
            -offset_step * (n_sparsifiers - 1) / 2,
            offset_step * (n_sparsifiers - 1) / 2,
            n_sparsifiers,
        )
        print(sparsifier_offsets)

        for idx, sparsifier in enumerate(sparsifier_algos):
            offset = sparsifier_offsets[idx]
            # Lists to store mean accuracies and standard deviations
            train_accs_means = []
            train_accs_stds = []
            val_accs_means = []
            val_accs_stds = []

            for nr_extra_edges in extra_edges:
                print(
                    "Sparsifier:",
                    sparsifier.__name__,
                    "Nr of extra edges:",
                    nr_extra_edges,
                )

                # Generate sparse graphs
                sparse_graphs = []

                # Ensure unique sources by setting replace=False
                sources = np.random.choice(n, nr_sparsifiers, replace=False)
                for source in sources:
                    sparse_graph = from_networkx(
                        generate_sparsifier_with_sampling(
                            nx_graph, sparsifier, nr_extra_edges, source=source
                        )
                    )
                    sparse_graphs.append(sparse_graph)
                    print("Sparse Graphs", sparse_graph)

                # Lists to collect accuracies from each repeat
                train_acc_runs = []
                val_acc_runs = []

                for _ in range(repeats):
                    train_acc, val_acc = train_GCN_multisparse(
                        sparse_graphs,
                        dataset.num_features,
                        dataset.num_classes,
                        sparsifier.__name__,
                        dataset.name,
                        pref=0,
                        plot_flag=False,
                    )
                    # Collect the final accuracy from each run
                    train_acc_runs.append(train_acc[-1])
                    val_acc_runs.append(val_acc[-1])

                # Compute mean and standard deviation across repeats
                mean_train_acc = np.mean(train_acc_runs)
                std_train_acc = np.std(train_acc_runs)
                mean_val_acc = np.mean(val_acc_runs)
                std_val_acc = np.std(val_acc_runs)

                # Store the computed statistics
                train_accs_means.append(mean_train_acc)
                train_accs_stds.append(std_train_acc)
                val_accs_means.append(mean_val_acc)
                val_accs_stds.append(std_val_acc)

            # Adjust x-values by offset
            x_vals = [x + offset for x in extra_edges]
            print(x_vals)
            # Plotting the mean accuracies with error bars
            plt.errorbar(
                x_vals,
                train_accs_means,
                yerr=train_accs_stds,
                label=f"{sparsifier.__name__} Training",
                capsize=3,
                marker="o",
                linestyle="--",
            )
            plt.errorbar(
                x_vals,
                val_accs_means,
                yerr=val_accs_stds,
                label=f"{sparsifier.__name__} Validation",
                capsize=3,
                marker="s",
                linestyle="-",
            )

        # Include the general (full) graph performance if specified
        if include_general:
            train_acc_full, val_acc_full = train_GCN(
                graph,
                dataset.num_features,
                dataset.num_classes,
                "full",
                dataset.name,
                plot_flag=False,
            )
            # Compute the mean validation accuracy over the last 10 epochs
            mean_val_acc_full = np.mean(val_acc_full[-10:])
            # Plot as a horizontal line
            plt.axhline(
                y=mean_val_acc_full,
                color="k",
                linestyle=":",
                label="Full Graph Validation",
            )

        # Final plot adjustments
        plt.xlabel("Number of Extra Edges")
        plt.ylabel("Accuracy")
        # Title should say the dataset name, number of vertices in the graph and the number of edges
        plt.title(
            f"Performance on {dataset.name}, n = {n}, e = {graph.edge_index.shape[1]}"
        )
        plt.xticks(extra_edges)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()

        # Save and clear the plot
        plt.savefig(
            f"plots/multiple_sparsifiers/{dataset.name}.png", bbox_inches="tight"
        )
        plt.clf()


def plot_from_csv(csv_file):
    data = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    # Get unique sparsifiers and extra edges
    sparsifiers = data["sparsifier"].unique()
    extra_edges = data["nr_extra_edges"].unique()

    for sparsifier in sparsifiers:
        subset = data[data["sparsifier"] == sparsifier]

        # Plotting mean training accuracies with error bars
        plt.errorbar(
            subset["nr_extra_edges"],
            subset["mean_train_acc"],
            yerr=subset["std_train_acc"],
            label=f"{sparsifier} Training",
            capsize=3,
            marker="o",
            linestyle="--",
        )

        # Plotting mean validation accuracies with error bars
        plt.errorbar(
            subset["nr_extra_edges"],
            subset["mean_val_acc"],
            yerr=subset["std_val_acc"],
            label=f"{sparsifier} Validation",
            capsize=3,
            marker="s",
            linestyle="-",
        )

    plt.xlabel("Number of Extra Edges")
    plt.ylabel("Accuracy")
    plt.xticks(extra_edges)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def train_on_multiple_sparsifiers_refactor_to_csv(
    datasets,
    sparsifier_algos,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    include_general=False,
    nr_edges_multiplier=200,
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
        print("nx_graph", nx_graph)

        m = nx_graph.number_of_edges()
        min_number_of_edges = n - 1
        remaining_edges = m - min_number_of_edges
        extra_edges = [
            min_number_of_edges + remaining_edges // i
            for i in range(nr_dividers + 1, 1, -1)
        ]
        print("edges to add", extra_edges)

        # Initialize list to store all the individual results
        results = []

        for idx, sparsifier in enumerate(sparsifier_algos):
            for nr_extra_edges in extra_edges:
                print(
                    "Sparsifier:",
                    sparsifier.__name__,
                    "Nr of extra edges:",
                    nr_extra_edges,
                )

                # Generate sparse graphs
                sparse_graphs = []
                sources = np.random.choice(n, nr_sparsifiers, replace=False)
                for source in sources:
                    sparse_graph = from_networkx(
                        generate_sparsifier_with_sampling(
                            nx_graph, sparsifier, nr_extra_edges, source=source
                        )
                    )
                    sparse_graphs.append(sparse_graph)

                for repeat_idx in range(repeats):
                    # Train on GCN with the sparse graphs
                    train_acc, val_acc = train_GCN_multisparse(
                        sparse_graphs,
                        dataset.num_features,
                        dataset.num_classes,
                        sparsifier.__name__,
                        dataset.name,
                        pref=0,
                        plot_flag=False,
                    )

                    # Append individual results (final training/validation accuracy)
                    results.append(
                        {
                            "dataset": dataset.name,
                            "sparsifier": sparsifier.__name__,
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
                            "nr_edges_sparsifier": sparse_graphs[0].edge_index.shape[1],
                        }
                    )

        # Save all individual results to CSV
        csv_file_path = f"{output_dir}/{dataset.name}_sparsifiers_results.csv"
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {csv_file_path}")


def train_on_multiple_sparsifiers(
    datasets,
    sparsifier_algos,
    nr_dividers,
    repeats=3,
    nr_sparsifiers=3,
    include_general=False,
):
    for dataset in datasets:
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        print("Full", graph)
        for sparsifier in sparsifier_algos:
            train_accs, val_accs = [], []
            n = graph.x.shape[0]
            extra_edges = [2 * i for i in range(1, nr_dividers + 1)]
            for nr_extra_edges in extra_edges:
                print("Nr of extra edges:", nr_extra_edges)
                nx_graph = to_networkx(graph, to_undirected=True, node_attrs=["x", "y"])
                sparse_graphs = []
                for source in np.random.choice(n, nr_sparsifiers):
                    sparse_graph = from_networkx(
                        generate_sparsifier_with_sampling(
                            nx_graph, sparsifier, nr_extra_edges, source=source
                        )
                    )
                    sparse_graphs.append(sparse_graph)
                    print(sparsifier.__name__, sparse_graph)
                avg_train_acc = 0
                avg_val_acc = 0
                for _ in range(repeats):
                    train_acc, val_acc = train_GCN_multisparse(
                        sparse_graphs,
                        dataset.num_features,
                        dataset.num_classes,
                        sparsifier.__name__,
                        dataset.name,
                        pref=0,
                        plot_flag=False,
                    )
                    avg_train_acc += train_acc[-1]
                    avg_val_acc += val_acc[-1]
                train_accs.append(avg_train_acc / repeats)
                val_accs.append(avg_val_acc / repeats)
            plt.plot(train_accs, label=sparsifier.__name__ + " Training Accuracy")
            plt.plot(val_accs, label=sparsifier.__name__ + " Validation Accuracy")
            plt.xticks(range(nr_dividers), extra_edges)
            plt.xlabel("Extra_edges")
            plt.ylabel("Accuracy")

        if include_general:
            train_acc, val_acc = train_GCN(
                graph,
                dataset.num_features,
                dataset.num_classes,
                "full",
                dataset.name,
                plot_flag=False,
            )
            plt.plot(
                [np.mean(val_acc[-10:]) for _ in range(nr_dividers)],
                label="True Validation Accuracy",
            )
        # plt.show()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(dataset.name)
        plt.savefig(
            "plots/multiple_sparsifiers/" + dataset.name + ".png", bbox_inches="tight"
        )
        plt.clf()


def train_different_ks(
    datasets, sparsifier_algos, nr_dividers, repeats=3, include_general=False
):
    for dataset in datasets:
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        print("Full", graph)

        for sparsifier in sparsifier_algos:
            train_accs, val_accs = [], []
            n = graph.x.shape[0]
            extra_edges = [4 * i for i in range(1, nr_dividers + 1)]
            assert len(extra_edges) == nr_dividers
            for nr_extra_edges in extra_edges:
                print("Extra edges:", nr_extra_edges)
                nx_graph = to_networkx(graph, to_undirected=True, node_attrs=["x", "y"])
                # n = nx_graph.number_of_nodes()

                source = np.random.choice(n)
                sparse_graph = generate_sparsifier_with_sampling(
                    nx_graph, sparsifier, nr_extra_edges, source=source
                )
                sparse_graph = from_networkx(sparse_graph)
                print(sparsifier.__name__, sparse_graph)
                avg_train_acc = 0
                avg_val_acc = 0
                for _ in range(repeats):
                    train_acc, val_acc = train_GCN(
                        sparse_graph,
                        dataset.num_features,
                        dataset.num_classes,
                        sparsifier.__name__,
                        dataset.name,
                        pref=0,
                        plot_flag=False,
                    )
                    avg_train_acc += np.mean(train_acc[-10:])
                    avg_val_acc += np.mean(val_acc[-10:])
                train_accs.append(avg_train_acc / repeats)
                val_accs.append(avg_val_acc / repeats)
            # plt.plot(train_accs, label=sparsifier.__name__ + " Training Accuracy")
            plt.plot(val_accs, label=sparsifier.__name__ + " Validation Accuracy")
            plt.xticks(range(nr_dividers), extra_edges)
            plt.xlabel("Extra_edges")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(dataset.name)
        if include_general:
            train_acc, val_acc = train_GCN(
                graph,
                dataset.num_features,
                dataset.num_classes,
                "full",
                dataset.name,
                plot_flag=False,
            )
            plt.plot(
                [np.mean(val_acc[-10:]) for _ in range(nr_dividers)],
                label="True Validation Accuracy",
            )
        # plt.show()
        plt.savefig(
            "plots/extra_extra_edges/"
            + dataset.name
            + "_"
            + str(nr_dividers)
            + "_"
            + str(repeats)
            + ".png"
        )
        plt.clf()


def train_on_datasets(datasets, sparsifier_algos, include_general=False):
    pref = random.randint(0, 1000)
    os.makedirs("training_plots/" + str(pref))
    for dataset in datasets:
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        print("Full", graph)
        for sparsifier in sparsifier_algos:
            nx_graph = to_networkx(graph, to_undirected=True, node_attrs=["x", "y"])
            Gcc = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
            nx_graph = nx_graph.subgraph(Gcc[0])
            sparse_graph = generate_sparsifier(
                nx_graph,
                sparsifier,
                sparsifier.__name__,
            )
            sparse_graph = from_networkx(sparse_graph)
            print(sparsifier.__name__, sparse_graph)
            train_acc, val_acc = train_GCN(
                sparse_graph,
                dataset.num_features,
                dataset.num_classes,
                sparsifier.__name__,
                dataset.name,
                pref=pref,
                plot_flag=True,
            )
        if include_general:
            train_acc, val_acc = train_GCN(
                graph,
                dataset.num_features,
                dataset.num_classes,
                "full",
                dataset.name,
                pref=pref,
                plot_flag=True,
            )
        plt.clf()


def train_different_densities(dataset, sparsifier_algo, repeats=2):
    train_accs = []
    val_accs = []
    plt.clf()
    for density in np.linspace(0.2, 0.9, 8):
        print(density)
        graph = dataset[0]
        if len(graph.y.shape) > 1:
            graph.y = graph.y.squeeze()
        sparse_graph = generate_sparsifier(
            to_networkx(graph, to_undirected=True, node_attrs=["x", "y"]),
            sparsifier_algo,
            sparsifier_algo.__name__,
        )
        sparse_graph = from_networkx(sparse_graph)
        print(sparsifier_algo.__name__, sparse_graph)
        avg_train_acc = 0
        avg_val_acc = 0
        for _ in range(repeats):
            train_acc, val_acc = train_GCN(
                sparse_graph,
                dataset.num_features,
                dataset.num_classes,
                sparsifier_algo.__name__,
                dataset.name,
                density,
                random.randint(0, 1000),
                False,
            )
            avg_train_acc += train_acc[-1]
            avg_val_acc += val_acc[-1]
        train_accs.append(avg_train_acc / repeats)
        val_accs.append(avg_val_acc / repeats)

    current_plot = random.randint(0, 1000)
    plt.clf()
    plt.plot(np.linspace(0.2, 0.9, 8), train_accs, label="Train")
    plt.plot(np.linspace(0.2, 0.9, 8), val_accs, label="Validation")
    plt.legend()
    plt.xlabel("Density")
    plt.ylabel("Accuracy")
    # plt.show()
    plt.savefig(
        "plots/density_plots/"
        + str(current_plot)
        + "_"
        + dataset.name
        + "_"
        + sparsifier_algo.__name__
        + ".png"
    )
    print("Plot is in", current_plot)


def main():
    datasets2 = [
        # Amazon(root="dataset/", name="Photo"),  # n = 7650, e = 119081
        # Amazon(root="dataset/", name="Computers"),  # n = 13752, e = 245778
        # # # # # # GNNBenchmarkDataset(root="dataset/", name="CIFAR10"),
        # # # # # # GNNBenchmarkDataset(root="dataset/", name="MNIST"),
        Planetoid(root="dataset/", name="Cora"),  # n = 2708, e = 10556
        # Planetoid(root="dataset/", name="CiteSeer"),  # n = 3327, e = 9104
        # Planetoid(root="dataset/", name="Pubmed"),  # n = 19717 e = 88651
        # Coauthor(root="dataset/", name="CS"),  # n = 18333, e = 81894
        # Coauthor(root="dataset/", name="Physics"),  # n = 34493, e = 247962
        ### PygNodePropPredDataset(name="ogbn-arxiv"),
        # AttributedGraphDataset(root="dataset/", name="BlogCatalog"),
        ### CitationFull(root="dataset/", name="Cora"),
        # KarateClub(),
    ]

    # train_different_ks(
    #     datasets2,
    #     [
    #         # # compute_random_tree_and_n_edges,
    #         # bfs_k_leverage_vertex_based,
    #         # # bfs_k_leverage_rank,
    #         # # compute_bfs_and_n_random_edges,
    #         # compute_dfs_and_n_random_edges,
    #     ],
    #     4,
    #     repeats=1,
    #     include_general=True,
    # )

    kdfs_random_partial = partial(ktree_random, tree_func_name="dfs")
    kdfs_random_partial.__name__ = "kdfs_random"

    kbfs_random_partial = partial(ktree_random, tree_func_name="bfs")
    kbfs_random_partial.__name__ = "kbfs_random"

    kbfs_leverage_partial = partial(ktree_leverage, tree_func_name="bfs")
    kbfs_leverage_partial.__name__ = "kbfs_leverage"

    kdfs_leverage_partial = partial(ktree_leverage, tree_func_name="dfs")
    kdfs_leverage_partial.__name__ = "kdfs_leverage"

    dfs_random_partial = partial(one_tree_random, tree_func_name="dfs")
    dfs_random_partial.__name__ = "onedfs_random"

    bfs_random_partial = partial(one_tree_random, tree_func_name="bfs")
    bfs_random_partial.__name__ = "onebfs_random"

    dfs_leverage_partial = partial(one_tree_leverage, tree_func_name="dfs")
    dfs_leverage_partial.__name__ = "onedfs_leverage"

    bfs_leverage_partial = partial(one_tree_leverage, tree_func_name="bfs")
    bfs_leverage_partial.__name__ = "onebfs_leverage"

    dfs_treetree_partial = partial(onetree_tree, tree_func_name="dfs")
    dfs_treetree_partial.__name__ = "onedfs_tree"

    bfs_treetree_partial = partial(onetree_tree, tree_func_name="bfs")
    bfs_treetree_partial.__name__ = "onebfs_tree"

    kdfs_treetree_partial = partial(ktree_tree, tree_func_name="dfs")
    kdfs_treetree_partial.__name__ = "kdfs_tree"

    kbfs_treetree_partial = partial(ktree_tree, tree_func_name="bfs")
    kbfs_treetree_partial.__name__ = "kbfs_tree"

    train_on_multiple_sparsifiers_refactor_to_csv(
        datasets2,
        [
            dfs_treetree_partial,
            bfs_treetree_partial,
            kdfs_treetree_partial,
            kbfs_treetree_partial,
            kdfs_random_partial,
            kbfs_random_partial,
            kbfs_leverage_partial,
            kdfs_leverage_partial,
            dfs_random_partial,
            bfs_random_partial,
            dfs_leverage_partial,
            bfs_leverage_partial,
        ],
        nr_dividers=1,
        repeats=1,
        nr_sparsifiers=1,
        include_general=True,
        nr_edges_multiplier=1,
    )

    # train_with_different_depths(datasets2[0], compute_four_add_spanner)


def just_plot():
    plot_from_csv("output_csv/Photo_sparsifiers_results.csv")
    plot_from_csv("output_csv/Computers_sparsifiers_results.csv")
    plot_from_csv("output_csv/Cora_sparsifiers_results.csv")
    plot_from_csv("output_csv/CiteSeer_sparsifiers_results.csv")
    plot_from_csv("output_csv/Pubmed_sparsifiers_results.csv")
    plot_from_csv("output_csv/CS_sparsifiers_results.csv")
    plot_from_csv("output_csv/Physics_sparsifiers_results.csv")
    plot_from_csv("output_csv/ogbn-arxiv_sparsifiers_results.csv")


if __name__ == "__main__":
    main()
    # just_plot()
