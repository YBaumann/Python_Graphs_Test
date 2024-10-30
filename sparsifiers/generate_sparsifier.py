import networkx as nx
import time


# Given a graph, a sparsification function and a name for the graph, generate the sparsifier accordingly
def generate_sparsifier(graph, func, name):
    nx_graph_copy = graph.copy()

    sparse_edgelist = func(nx_graph_copy)

    # Create a new graph and add all nodes with their attributes
    G = nx.Graph()
    for node, data in graph.nodes(data=True):
        G.add_node(node, **data)

    # Add the spanner edges to the new graph
    G.add_edges_from(sparse_edgelist)
    return G


def generate_sparsifier_with_sampling(graph, func, k, source=0, nr_trees=1):
    print("Gets to generate")
    nx_graph_copy = graph.copy()
    sparse_edgelist = func(nx_graph_copy, k=k, source=source, nr_trees=nr_trees)

    # Create a new graph and add all nodes with their attributes
    G = nx.Graph()
    for node, data in graph.nodes(data=True):
        G.add_node(node, **data)

    # Add the spanner edges to the new graph
    G.add_edges_from(sparse_edgelist)
    # Print the 10 highest degree nodes in the graph
    # print(sorted(G.degree, key=lambda x: x[1], reverse=True)[:10])
    print(G)
    return G


def generate_sparsifier_with_sampling_and_time(graph, func, k, source=0, nr_trees=1):
    nx_graph_copy = graph.copy()
    start_sparsify = time.time()
    sparse_edgelist = func(nx_graph_copy, k=k, source=source, nr_trees=nr_trees)
    end_sparsify = time.time()
    sparsify_time = end_sparsify - start_sparsify

    # Create a new graph and add all nodes with their attributes
    G = nx.Graph()
    for node, data in graph.nodes(data=True):
        G.add_node(node, **data)

    # Add the spanner edges to the new graph
    G.add_edges_from(sparse_edgelist)
    return G, sparsify_time
