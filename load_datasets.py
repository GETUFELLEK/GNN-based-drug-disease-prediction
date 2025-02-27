import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx
import numpy as np

# Define nodes (drugs and diseases)
drug_nodes = ["Aspirin", "Metformin", "Ibuprofen"]
disease_nodes = ["Diabetes", "Hypertension", "Inflammation"]

# Create numerical mappings
node_list = drug_nodes + disease_nodes
node_to_idx = {node: i for i, node in enumerate(node_list)}

# Define edges (drug-disease interactions)
edges = [
    ("Aspirin", "Inflammation"),
    ("Metformin", "Diabetes"),
    ("Ibuprofen", "Inflammation"),
    ("Metformin", "Hypertension")
]

# Convert edges to indices
edge_index = torch.tensor([[node_to_idx[src], node_to_idx[dst]] for src, dst in edges], dtype=torch.long).t()

# Define random node features (e.g., molecular features, embeddings)
num_nodes = len(node_list)
node_features = torch.rand((num_nodes, 16))  # 16-dimensional feature vectors

# Create PyG Data object
graph_data = Data(x=node_features, edge_index=edge_index)

print("Graph Data:", graph_data)
