#!/bin/bash

# Define the list of dataset names to be processed
datasets=("ogbn-arxiv") # "Cora"  "Pubmed" "Photo" "Computers" "BlogCatalog") # "Physics" "ogbn-arxiv" )

# Define your integer parameters
nr_dividers=10
nr_repeats=10
nr_sparsifier=1 # 4
nr_epochs=200 # 50
# Define tree function names, sampler names, and one_or_k as comma-separated strings
tree_function_names="pseudo_random_spanning_tree,dfs,bfs,low_degree_spanning_tree"
sampler_names="low_degree,random,leverage,tree"
one_or_k="k" #1"

# The input order is: dataset, nr_dividers, nr_repeats, nr_sparsifier, tree_function_names, sampler_names, one_or_k
for dataset in "${datasets[@]}"; do
    sbatch run_one_with_input.sh "$dataset" "$nr_dividers" "$nr_repeats" "$nr_sparsifier" "$nr_epochs" "$tree_function_names" "$sampler_names" "$one_or_k"
done
