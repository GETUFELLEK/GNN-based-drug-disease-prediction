
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Evaluation function
def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data).squeeze()
        predictions = torch.sigmoid(logits) > 0.5

        accuracy = accuracy_score(data.y, predictions)
        precision = precision_score(data.y, predictions)
        recall = recall_score(data.y, predictions)
        auc = roc_auc_score(data.y, torch.sigmoid(logits))

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'ROC AUC: {auc:.4f}')

# Example usage
if __name__ == "__main__":
    from train_model import train_model
    from build_graph import build_graph
    from load_dataset import load_drug_disease_data

    # Load dataset and build graph
    data_df = load_drug_disease_data()
    G = build_graph(data_df)
    data = convert_to_torch_geometric(G)

    # Train model
    model = train_model(data)

    # Evaluate model
    evaluate_model(model, data)
