to
import networkx as nx
from torch_geometric.utils import from_networkx


# Function to create the graph from drug-disease data
def build_graph(drug_disease_df):
    G = nx.Graph()
    for _, row in drug_disease_df.iterrows():
        G.add_edge(row['drug_id'], row['disease_id'])
    return G


# Convert the NetworkX graph to a PyTorch Geometric graph
def convert_to_torch_geometric(G):
    data = from_networkx(G)
    return data


# Example usage
if __name__ == "__main__":
    from load_dataset import load_drug_disease_data

    # Load dataset
    data_df = load_drug_disease_data()

    # Build and convert the graph
    G = build_graph(data_df)
    torch_geometric_data = convert_to_torch_geometric(G)

    print(torch_geometric_data)
