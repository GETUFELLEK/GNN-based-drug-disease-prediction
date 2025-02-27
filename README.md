# Drug-Disease Interaction Prediction using Graph Neural Networks (GNN)

## Overview

This project demonstrates how to use **Graph Neural Networks (GNNs)** to predict drug-disease interactions on **Apple Silicon (M1/M2/M3) Macs** using **PyTorch Geometric (PyG)**. The goal is to leverage publicly available datasets such as **DrugBank** and build a **graph-based AI model** to predict relationships between drugs and diseases, such as which drug treats or causes a disease. 

We utilize **PyTorch Geometric** for graph-based deep learning, **NetworkX** for graph visualization, and **Apple's Metal (MPS) backend** for hardware acceleration.

---

## **Installation Instructions (Apple Silicon M1/M2/M3)**

Since CUDA is **not available** on Apple Silicon, we install PyTorch with **MPS support** and PyTorch Geometric (PyG) using CPU-compatible wheels.

### **1️⃣ Install PyTorch (CPU/MPS Version)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **2️⃣ Install Required Build Tools** *(Optional for Compilation)*
```bash
xcode-select --install  # Ensures Apple Clang compiler is installed
```

If using **Conda (Miniforge)**:
```bash
conda install -c conda-forge clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
```
Set macOS deployment target:
```bash
export MACOSX_DEPLOYMENT_TARGET=12.3
export CC=clang CXX=clang++
```

### **3️⃣ Install PyTorch Geometric (Core Library)**
```bash
pip install torch-geometric
```

### **4️⃣ Install PyG Dependencies (Graph Operations)**
```bash
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

### **5️⃣ Verify Installation**
```python
import torch
import torch_geometric
print("Torch Version:", torch.__version__)
print("Torch Geometric Version:", torch_geometric.__version__)
print("Is CUDA available?", torch.cuda.is_available())  # Expected: False
```

---

## **Dataset**
Download a dataset that contains **drug-disease interactions**. Example source: **DrugBank**. The dataset should be a CSV file containing **drug IDs and disease IDs**.

### **Example CSV Structure:**
```csv
drug_id,disease_id,relationship
DB001,DS001,treats
DB002,DS002,causes
...
```

---

## **Steps to Run the Project**

### **1️⃣ Load the Dataset**
```bash
python load_dataset.py
```

### **2️⃣ Build the Knowledge Graph**
```bash
python build_graph.py
```

### **3️⃣ Train the GNN Model**
```bash
python train_model.py
```

### **4️⃣ Evaluate the Model**
```bash
python evaluate_model.py
```

### **5️⃣ Visualize the Knowledge Graph (optional)**
```bash
python build_graph.py
```


## **Model Architecture**
We implement a **Graph Convolutional Network (GCN)** with the following layers:
- **Input Layer:** Accepts node features (e.g., drug properties, disease embeddings)
- **Hidden Layer:** A graph convolution layer with ReLU activation
- **Output Layer:** A fully connected layer for binary classification

The model is trained using **Binary Cross-Entropy Loss (BCE Loss)** and **Adam Optimizer**.

---

## **Results**
- **Accuracy**: Measures correct predictions.
- **Precision & Recall**: Measures classification performance.
- **ROC-AUC**: Evaluates model effectiveness in distinguishing interactions.

---

## **Future Work**
✅ **Use Advanced GNN Architectures** (GAT, GraphSAGE, HGT)
✅ **Incorporate Molecular Features**
✅ **Expand to Larger Datasets (Hetionet, DisGeNET)**

---

## **References**
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [DrugBank Database](https://www.drugbank.ca/)

---

## **License**
This project is open-source under the **MIT License**.
