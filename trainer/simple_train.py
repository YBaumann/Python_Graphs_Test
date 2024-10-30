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

        for epoch in range(100):
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


# Train GCN model an plot result
@time_execution
def train_GCN(
    graph,
    num_features,
    num_classes,
    name,
    dataset_name,
    train_split=0.25,
    pref=random.randint(0, 1000),
    plot_flag=True,
    model=None,
):
    # Define model
    if model == None:
        model = GCN(num_features, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Define train/test masks 0.75 and 0.25 splits
    num_nodes = graph.x.size(0)
    graph.train_mask = np.zeros(num_nodes, dtype=bool)
    # graph.test_mask = np.zeros(num_nodes, dtype=bool)

    graph.train_mask[: int(train_split * num_nodes)] = True
    # graph.test_mask[int(train_split * num_nodes) :] = True

    # Shuffle train mask
    np.random.shuffle(graph.train_mask)
    graph.test_mask = ~graph.train_mask

    # Train model
    model.train()

    train_acc_history = []
    val_acc_history = []

    for epoch in range(400):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        # if epoch % 50 == 0:
        #     print(f"Epoch: {epoch}, Loss: {loss}")

        # Calculate training accuracy
        model.eval()
        _, pred = model(graph.x, graph.edge_index).max(dim=1)
        correct_train = int(
            pred[graph.train_mask].eq(graph.y[graph.train_mask]).sum().item()
        )
        train_acc = correct_train / int(graph.train_mask.sum())
        train_acc_history.append(train_acc)

        # Calculate validation accuracy
        _, pred = model(graph.x, graph.edge_index).max(dim=1)
        correct_val = int(
            pred[graph.test_mask].eq(graph.y[graph.test_mask]).sum().item()
        )
        val_acc = correct_val / int(graph.test_mask.sum())
        val_acc_history.append(val_acc)

    # Test model
    model.eval()
    _, pred = model(graph.x, graph.edge_index).max(dim=1)
    correct = int(pred[graph.test_mask].eq(graph.y[graph.test_mask]).sum().item())
    acc = correct / int(graph.test_mask.sum())
    print(f"Test Accuracy: {acc}")

    if not plot_flag:
        return train_acc_history, val_acc_history
    # Plot training and validation accuracy history
    plt.plot(train_acc_history, label=name + " Training Accuracy")
    plt.plot(val_acc_history, label=name + " Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(dataset_name)
    plt.savefig(
        "training_plots/" + str(pref) + "/" + dataset_name + "_current_accuracy.png"
    )

    return train_acc_history, val_acc_history
