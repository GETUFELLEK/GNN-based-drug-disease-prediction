from sklearn.metrics import accuracy_score

# Predict on test edges
model.eval()
with torch.no_grad():
    test_output = model(graph_data.x, graph_data.edge_index)
    test_edge_embeddings = test_output[edge_index[0]]
    predictions = torch.argmax(test_edge_embeddings, dim=1)

# Compute accuracy
accuracy = accuracy_score(labels[test_idx].cpu(), predictions[test_idx].cpu())
print(f"Test Accuracy: {accuracy:.2f}")
