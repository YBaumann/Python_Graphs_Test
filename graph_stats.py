import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
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
from sparsifiers.sparsifier_class import TreeSparsifier


import networkx as nx
import numpy as np
import time

def compute_graph_statistics(graph):
    # Number of nodes and edges
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    # Degree sequence
    degrees = [degree for node, degree in graph.degree()]
    
    # Average degree and standard deviation
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    
    # 90th percentile degree
    percentile_90_degree = np.percentile(degrees, 90)

    percentile_95_degree = np.percentile(degrees, 95)
    
    percentile_99_degree = np.percentile(degrees, 99)
    percentile_999_degree = np.percentile(degrees, 99.9)
    
    # Node with the highest degree
    highest_degree_node = max(graph.degree(), key=lambda x: x[1])[0]
    
    # Average clustering coefficient
    avg_clustering_coeff = nx.average_clustering(graph)
    
    # Diameter (only if the graph is connected; may be slow)
    # diameter = nx.diameter(graph) if nx.is_connected(graph) else float('inf')
    
    # # Average shortest path length (only if the graph is connected; may be slow)
    # avg_shortest_path_length = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
    
    # # Assortativity
    # assortativity = nx.degree_assortativity_coefficient(graph)
    
    # # Degree centrality statistics
    # degree_centrality = nx.degree_centrality(graph)
    # max_degree_centrality = max(degree_centrality.values())
    # min_degree_centrality = min(degree_centrality.values())
    # median_degree_centrality = np.median(list(degree_centrality.values()))
    
    # Return as a dictionary
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "std_degree": std_degree,
        "90th_percentile_degree": percentile_90_degree,
        "95th_percentile_degree": percentile_95_degree,
        "99th_percentile_degree": percentile_99_degree,
        "99.9th_percentile_degree": percentile_999_degree,
        "highest_degree_node": highest_degree_node,
        "avg_clustering_coeff": avg_clustering_coeff,
        # "diameter": diameter,  # Potentially slow
        # "avg_shortest_path_length": avg_shortest_path_length,  # Potentially slow
        # "assortativity": assortativity,
        # "max_degree_centrality": max_degree_centrality,
        # "min_degree_centrality": min_degree_centrality,
        # "median_degree_centrality": median_degree_centrality
    }


datasets = [
        Planetoid(root="dataset/", name="Cora"),
        Amazon(root="dataset/", name="Photo"),
        Amazon(root="dataset/", name="Computers"),
        Planetoid(root="dataset/", name="Pubmed"),
        Coauthor(root="dataset/", name="CS"),
        Coauthor(root="dataset/", name="Physics"),
        Planetoid(root="dataset/", name="CiteSeer"),
        #PygNodePropPredDataset(name="ogbn-arxiv"),
        AttributedGraphDataset(root="dataset/", name="BlogCatalog"),
    ]





    

from tqdm import tqdm

    
tree_function_names=["dfs", "pseudo_random_spanning_tree","low_degree_spanning_tree", "bfs"]
sampler_names=["low_degree",  "random", "tree", "leverage",]
one_or_k=["k"]
nr_dividers=10

# List to store statistics for each dataset
stats_list = []
    
for dataset in datasets:
    dataset_name = dataset.name
    print(dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    min_number_of_edges = n - 1
    remaining_edges = m - min_number_of_edges
    extra_edges = [
        min_number_of_edges + remaining_edges // i
        for i in range(nr_dividers + 1, 1, -1)
    ]
    for tree_func in tqdm(tree_function_names):
        print(tree_func)
        for sampler in sampler_names:
            for k in one_or_k:
                for i in tqdm(range(nr_dividers)):
                    for nr_extra_edges in extra_edges:
                        time_start = time.time()
                        sparsifier = TreeSparsifier(
                                        graph,
                                        tree_func,
                                        k,
                                        sampler,
                                        nr_extra_edges,)
                        sparsifier.compute_sparsifier()
                        sparsifier.get_sparsifier()

                        graph_statistics = compute_graph_statistics(sparsifier.sparsified_graph)
                        time_end = time.time()
                        total_time = time_end - time_start
        
                        # Add dataset name to the statistics dictionary
                        graph_statistics["dataset_name"] = dataset_name
                        graph_statistics["tree_func"] = tree_func
                        graph_statistics["sampler"] = sampler
                        graph_statistics["nr_edges_sparsifier"] = sparsifier.sparsified_graph.number_of_edges()
                        graph_statistics["sparsification_time"] = total_time
                        stats_list.append(graph_statistics)
    
# Convert list of dictionaries to a DataFrame and save as CSV
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv("graph_statistics_sparsifier.csv", index=False)

