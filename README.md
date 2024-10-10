 This document outlines the project’s purpose, installation steps, usage, and further details that will help anyone who wants to understand or run the project.

-# Drug-Disease Interaction Prediction using Graph Neural Networks (GNN)

## Overview

This project demonstrates how to use Graph Neural Networks (GNNs) to predict drug-disease interactions. The goal is to leverage publicly available datasets such as DrugBank and build a graph-based AI model to predict relationships between drugs and diseases, such as which drug treats or causes a disease. We use **PyTorch Geometric** to implement the GNN and handle the graph structure, while **NetworkX** helps to visualize the knowledge graph.

## Project Structure

```
|-- medical_ai_project
    |-- load_dataset.py         # Code to load and preprocess the dataset
    |-- build_graph.py          # Code to build the drug-disease graph
    |-- gnn_model.py            # Definition of the GCN model
    |-- train_model.py          # Training logic for the GNN
    |-- evaluate_model.py       # Code for evaluating the GNN model
    |-- README.md               # This file (project documentation)
    |-- requirements.txt        # List of dependencies for easy installation
```

## Dependencies

This project uses the following Python packages:

- `torch` (PyTorch for deep learning)
- `torch_geometric` (for graph-based neural networks)
- `networkx` (for graph manipulation)
- `pandas` (for data manipulation)
- `scikit-learn` (for model evaluation)
- `matplotlib` (for graph visualization)

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

To get started, download a dataset that contains drug-disease interactions. One such dataset can be obtained from **DrugBank**. You will need a CSV file with at least two columns: one for drug IDs and one for disease IDs. The file path should be updated in `load_dataset.py`.

### Example CSV Structure:
```csv
drug_id,disease_id,relationship
DB001,DS001,treats
DB002,DS002,causes
...
```

## Steps to Run the Project

### 1. Load the Dataset

The first step is to load the dataset and preprocess it. The dataset is loaded using the `load_dataset.py` script.

```bash
python load_dataset.py
```

### 2. Build the Knowledge Graph

Using the preprocessed data, a graph is built in **NetworkX** and then converted to **PyTorch Geometric** format. Run the `build_graph.py` script to construct the graph.

```bash
python build_graph.py
```

### 3. Define and Train the GNN Model

The GNN model is defined in `gnn_model.py`. We use a **Graph Convolutional Network (GCN)** to predict the interaction between drugs and diseases. You can train the model using `train_model.py`:

```bash
python train_model.py
```

### 4. Evaluate the Model

After training, evaluate the model's performance using metrics such as accuracy, precision, recall, and ROC-AUC. You can run the evaluation script:

```bash
python evaluate_model.py
```

### 5. Visualize the Knowledge Graph

The drug-disease interaction graph can be visualized using **NetworkX** and **Matplotlib**. The function in `build_graph.py` can be used to generate a graphical representation of the knowledge graph.

```bash
python build_graph.py
```

### Example Graph Visualization:

![Graph Example](graph_visualization.png)

## Model Architecture

We use a simple **Graph Convolutional Network (GCN)** with two layers:

- **Input Layer**: Accepts node features (64-dimensional random features).
- **Hidden Layer**: A graph convolution layer with ReLU activation.
- **Output Layer**: A fully connected layer for binary classification (predicting interactions).

The model is trained using **Binary Cross-Entropy Loss (BCE Loss)** and **Adam Optimizer**.

## Results

The performance of the model is evaluated using:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Measures the ability of the model to predict positive samples correctly.
- **Recall**: Measures the ability of the model to find all relevant positive samples.
- **ROC-AUC**: Measures the trade-off between true positives and false positives.

## Future Work

- **Advanced GNNs**: We can experiment with **Graph Attention Networks (GAT)** or **GraphSAGE** for better performance.
- **Improved Features**: Include more node and edge features such as drug properties, disease severity, and more.
- **Expanded Dataset**: Use larger datasets to improve model generalization.

## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [DrugBank Database](https://www.drugbank.ca/)

## License

This project is open-source and available under the MIT License.

---

# GNN-based-drug-disease-prediction
