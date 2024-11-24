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
    
    # Node with the highest degree
    highest_degree_node = max(graph.degree(), key=lambda x: x[1])[0]
    
    # Average clustering coefficient
    avg_clustering_coeff = nx.average_clustering(graph)
    
    # Diameter and average shortest path length for the largest connected component
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        avg_shortest_path_length = nx.average_shortest_path_length(graph)
    else:
        # Find the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        
        diameter = nx.diameter(subgraph)
        avg_shortest_path_length = nx.average_shortest_path_length(subgraph)

    # Homophily
    
    
    # Assortativity
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
        "highest_degree_node": highest_degree_node,
        "avg_clustering_coeff": avg_clustering_coeff,
        "diameter": diameter,
        "avg_shortest_path_length": avg_shortest_path_length,
        # "assortativity": assortativity,
        # "max_degree_centrality": max_degree_centrality,
        # "min_degree_centrality": min_degree_centrality,
        # "median_degree_centrality": median_degree_centrality
    }


datasets = [
        Amazon(root="dataset/", name="Photo"),
        Planetoid(root="dataset/", name="Cora"),
        Amazon(root="dataset/", name="Computers"),
        Planetoid(root="dataset/", name="Pubmed"),
        Coauthor(root="dataset/", name="CS"),
        # Coauthor(root="dataset/", name="Physics"),
        Planetoid(root="dataset/", name="CiteSeer"),
        # PygNodePropPredDataset(name="ogbn-arxiv"),
        AttributedGraphDataset(root="dataset/", name="BlogCatalog"),
    ]

def compute_homophily(dataset):
    data = dataset[0]
    edge_index = data.edge_index
    labels = data.y

    # Compute homophily
    same_label_edges = 0
    total_edges = edge_index.size(1)

    for i in range(total_edges):
        node1 = edge_index[0, i].item()
        node2 = edge_index[1, i].item()
        if labels[node1] == labels[node2]:
            same_label_edges += 1

    homophily = same_label_edges / total_edges
    return homophily


from tqdm import tqdm
# List to store statistics for each dataset
stats_list = []
    
for dataset in datasets:
    dataset_name = dataset.name
    print(dataset_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    graph_statistics = compute_graph_statistics(graph)
        
    # Add dataset name to the statistics dictionary
    graph_statistics["dataset_name"] = dataset_name
    graph_statistics["homophily"] = compute_homophily(dataset)
    #print(graph_statistics)
    stats_list.append(graph_statistics)
    
# Convert list of dictionaries to a DataFrame and save as CSV
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv("graph_statistics_sparsifier_new.csv", index=False)