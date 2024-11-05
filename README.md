# Graph Neural Network Sparsification

This repository contains the code supporting the research presented in the paper. In this work, we explore methods to sparsify graph-based datasets for efficient training and inference in graph neural networks (GNNs) without compromising performance. We propose an improved algorithm for finding graph lottery tickets that yield competitive results with fewer edges.

## Introduction

Graph datasets are increasingly important across various domains, from social networks to healthcare. However, the sheer size of these datasets often demands methods to reduce computational costs in GNN training. By leveraging sparsification techniques, our work aims to reduce graph size while maintaining performance through the "graph lottery ticket hypothesis." This repository includes implementations, experiments, and algorithms to identify and utilize graph lottery tickets efficiently.

## Getting Started

### Prerequisites

To set up and run the code, ensure you have the following installed:
- Python 3.13
- Libraries: PyTorch, DGL (Deep Graph Library), and other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YBaumann/Python_Graphs_Test.git
   cd Python_Graphs_Test
   ```
2. Install requirements:
   ```bash
    pip install -r requirements.txt
   ```
   