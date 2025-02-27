from torch.optim import Adam

# Generate labels (1 for connected, 0 for no interaction)
labels = torch.tensor([1, 1, 1, 1], dtype=torch.long)  # All defined edges exist

# Train-Test Split (simple 75%-25% split)
train_ratio = 0.75
train_size = int(len(edges) * train_ratio)
train_idx, test_idx = torch.arange(train_size), torch.arange(train_size, len(edges))

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(graph_data.x, graph_data.edge_index)

    # Select only edge embeddings
    edge_embeddings = output[edge_index[0]]

    loss = loss_fn(edge_embeddings, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training complete!")
