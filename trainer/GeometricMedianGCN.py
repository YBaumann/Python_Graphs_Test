import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# TODO: This doesn't work yet
# The problem is that the first dimension of out is too large, it should be the number of nodes
class GeometricMedianGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GeometricMedianGCN, self).__init__()
        self.conv1 = GeometricMedianConv(num_features, 16)
        self.conv2 = GeometricMedianConv(16, num_classes)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GeometricMedianConv(MessagePassing):
    def __init__(self, in_channels, out_channels, max_iters=10, eps=1e-5):
        super(GeometricMedianConv, self).__init__(
            aggr=None
        )  # Disable default aggregation
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.max_iters = max_iters  # Number of iterations for Weiszfeld's algorithm
        self.eps = eps  # Small constant to avoid division by zero

        # Initialize the layer weights
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix to include self-features in aggregation
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply linear transformation
        x = self.linear(x)

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # Pass neighbor features

    def aggregate(self, inputs, index):
        # Custom aggregation using the geometric median via Weiszfeld's algorithm

        # Create zero tensor to hold the aggregated features
        aggr_out = torch.zeros_like(inputs)

        # Get unique nodes (target nodes for aggregation)
        unique_nodes = torch.unique(index)

        # Iterate over each unique node to calculate its geometric median
        for node in unique_nodes:
            neighbors = inputs[index == node]  # Get features of neighbors for this node

            # Initialize median estimate as mean of neighbors
            median = neighbors.mean(dim=0)

            # Weiszfeld's algorithm to find geometric median
            for _ in range(self.max_iters):
                # Calculate distances to current median estimate
                distances = torch.norm(neighbors - median, dim=1).clamp(min=self.eps)

                # Compute new estimate of median
                weighted_sum = (neighbors / distances.view(-1, 1)).sum(dim=0)
                weight_total = (1.0 / distances).sum()
                new_median = weighted_sum / weight_total

                # Check for convergence
                if torch.norm(new_median - median) < self.eps:
                    break

                median = new_median  # Update median

            # Store the computed median for the node
            aggr_out[node] = median

        return aggr_out

    def update(self, aggr_out):
        # Apply an activation function (optional)
        return F.relu(aggr_out)
