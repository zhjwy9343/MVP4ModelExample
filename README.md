# MVP (Minimum Value Product) of DGL GNN Model Example Development

This project is designed for an MVP for GNN model development, and as a tutorial for new interns who will implement
GNN-related research papers with DGL.

GNN model used in this MVP comes from the paper of [CompositionGCN](https://arxiv.org/abs/1911.03082), and was simplified
for the education purpose. In this project, this is called simplified CompGCN.

The Graph Data to Tackle
--------------------------
- Graph data (g): the DGL's built-in CoraGraphDataset.

The Model and Algorithm
-------------------------

For the simplified CompGCN, the detailed math formulas are same as the original CompGCN paper, except that we did not
use basis vector to represent the relation embedding because in the Cora dataset, there is only one relation. Therefore
we used a learnable parameter as this relation's embedding with the same dimension as the input feature's.

The Structure of Example Files
--------------------------------
- **compgcn_node_classification.py:** The python file for training and inference(if required) using a full graph.
  
- **utils.py:** helper file, providing the circular correlation computation function.
