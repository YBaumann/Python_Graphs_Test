from trainer.GCN import GCN, GCN_logits
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

    # Define train/test masks
    num_nodes = sparse_graphs[0].x.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(train_split * num_nodes)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    for graph in sparse_graphs:
        # Move graph data to the device
        graph = graph.to(device)

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


@time_execution
def train_GCN_multisparse_ensemble(
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

    train_acc_history = []
    val_acc_history = []
    ensemble_outputs = []

    # Define train/test masks
    num_nodes = sparse_graphs[0].x.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(train_split * num_nodes)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    model = GCN(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for graph in sparse_graphs:
        # Move graph data to the device
        graph = graph.to(device)

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
                print(
                    f"Sparsifier: {sparse_graphs.index(graph)}, Epoch: {epoch}, Loss: {loss.item()}"
                )

        # Collect predictions from this sparsifier's model
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            pred = logits.argmax(dim=1)
            ensemble_outputs.append(pred)

            # Calculate training accuracy
            correct_train = pred[graph.train_mask] == graph.y[graph.train_mask]
            train_acc = correct_train.sum().item() / graph.train_mask.sum().item()
            train_acc_history.append(train_acc)

            # Calculate validation accuracy
            correct_val = pred[graph.test_mask] == graph.y[graph.test_mask]
            val_acc = correct_val.sum().item() / graph.test_mask.sum().item()
            val_acc_history.append(val_acc)

    # Combine outputs using the specified ensemble method
    final_output = None
    print(ensemble_outputs)

    ensemble_outputs_tensor = torch.stack(ensemble_outputs, dim=0)
    final_output = torch.mode(ensemble_outputs_tensor, dim=0).values

    # Evaluate ensemble performance on the test set
    with torch.no_grad():
        correct_ensemble = final_output[graph.test_mask] == graph.y[graph.test_mask]
        ensemble_acc = correct_ensemble.sum().item() / graph.test_mask.sum().item()
        print(f"Ensemble Test Accuracy: {ensemble_acc}")
        print(f"validation Accuracy: {val_acc_history[-1]}")

    return train_acc_history, val_acc_history, ensemble_acc


@time_execution
def train_GCN_multisparse_ensemble_fine_grained(
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

    train_acc_history = []
    val_acc_history = []
    ensemble_outputs = []

    # Define train/test masks
    num_nodes = sparse_graphs[0].x.size(0)
    indices = torch.randperm(num_nodes)
    train_size = int(train_split * num_nodes)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    model = GCN_logits(num_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    graphs_on_device = [graph.to(device) for graph in sparse_graphs]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    for epoch in range(nr_epochs):
        for graph in graphs_on_device:
            graph.train_mask = train_mask
            graph.test_mask = test_mask
            model.train()
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)
            out = F.log_softmax(out, dim=1)

            loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(
                    f"Sparsifier: {sparse_graphs.index(graph)}, Epoch: {epoch}, Loss: {loss.item()}"
                )

        # Collect predictions from this sparsifier's model
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            pred = logits.argmax(dim=1)
            # print(pred)
            ensemble_outputs.append(logits)
            # Calculate training accuracy
            correct_train = pred[graph.train_mask] == graph.y[graph.train_mask]
            train_acc = correct_train.sum().item() / graph.train_mask.sum().item()
            train_acc_history.append(train_acc)

            # Calculate validation accuracy
            correct_val = pred[graph.test_mask] == graph.y[graph.test_mask]
            val_acc = correct_val.sum().item() / graph.test_mask.sum().item()
            val_acc_history.append(val_acc)

    # Combine outputs using the specified ensemble method
    final_output = None

    ensemble_logits_sum = torch.sum(torch.stack(ensemble_outputs, dim=0), dim=0)
    final_output = ensemble_logits_sum.argmax(dim=1)

    # Evaluate ensemble performance on the test set
    with torch.no_grad():
        correct_ensemble = final_output[graph.test_mask] == graph.y[graph.test_mask]
        ensemble_acc = correct_ensemble.sum().item() / graph.test_mask.sum().item()
        print(f"Ensemble Test Accuracy: {ensemble_acc}")
        print(f"validation Accuracy: {val_acc_history[-1]}")

    return train_acc_history, val_acc_history, ensemble_acc
