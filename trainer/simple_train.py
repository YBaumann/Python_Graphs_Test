from trainer.GCN import GCN
from trainer.GeometricMedianGCN import GeometricMedianGCN
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from helpers.timer import time_execution


# Train with multiple sparsifiers
@time_execution
def train_GCN_multisparse(
    sparse_graphs,
    num_features,
    num_classes,
    nr_epochs=100,
    train_split=0.25,
    **kwargs,
):
    # Check for MPS device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M1/M2 GPU with MPS backend.")
    else:
        device = torch.device("cpu")
        print("MPS backend not available. Using CPU.")

    # Define model and move it to the device
    model = GCN(num_features, num_classes).to(device)
    # model = GeometricMedianGCN(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_acc_history = []
    val_acc_history = []

    for graph in sparse_graphs:
        # Move graph data to the device
        graph = graph.to(device)

        # Define train/test masks
        num_nodes = graph.x.size(0)
        indices = torch.randperm(num_nodes)
        train_size = int(train_split * num_nodes)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_indices] = True
        test_mask[test_indices] = True

        graph.train_mask = train_mask
        graph.test_mask = test_mask

        for epoch in range(nr_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)

            loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

            # Calculate training accuracy
            model.eval()
            with torch.no_grad():
                logits = model(graph.x, graph.edge_index)
                pred = logits.argmax(dim=1)

                correct_train = pred[graph.train_mask] == graph.y[graph.train_mask]
                train_acc = correct_train.sum().item() / graph.train_mask.sum().item()
                train_acc_history.append(train_acc)

                # Calculate validation accuracy
                correct_val = pred[graph.test_mask] == graph.y[graph.test_mask]
                val_acc = correct_val.sum().item() / graph.test_mask.sum().item()
                val_acc_history.append(val_acc)

    # Optionally, test the model on one of the graphs
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        pred = logits.argmax(dim=1)
        correct = pred[graph.test_mask] == graph.y[graph.test_mask]
        acc = correct.sum().item() / graph.test_mask.sum().item()
        print(f"Test Accuracy: {acc}")

    return train_acc_history, val_acc_history
