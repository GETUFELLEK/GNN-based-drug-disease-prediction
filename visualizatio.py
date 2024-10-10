import matplotlib.pyplot as plt
import networkx as nx


# Visualize the knowledge graph
def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    nx.draw(G, pos, node_size=50, node_color="lightblue", with_labels=True, font_size=8)
    plt.title('Drug-Disease Knowledge Graph')
    plt.show()


# Example usage
if __name__ == "__main__":
    from build_graph import build_graph
    from load_dataset import load_drug_disease_data

    # Load dataset and build graph
    data_df = load_drug_disease_data()
    G = build_graph(data_df)

    # Visualize the graph
    visualize_graph(G)
