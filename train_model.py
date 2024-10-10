
import torch
import torch.optim as optim
from gnn_model import GCN
from build_graph import convert_to_torch_geometric
from torch.nn import BCEWithLogitsLoss

# Training function
def train_model(data, num_epochs=200):
    model = GCN(in_channels=64, hidden_channels=128, out_channels=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = BCEWithLogitsLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits.squeeze(), data.y.float())  # Example loss function
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    return model

# Example usage
if __name__ == "__main__":
    from build_graph import build_graph
    from load_dataset import load_drug_disease_data

    # Load dataset and build graph
    data_df = load_drug_disease_data()
    G = build_graph(data_df)
    data = convert_to_torch_geometric(G)

    # Train the model
    trained_model = train_model(data)
t