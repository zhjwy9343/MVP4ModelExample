# MVP (Minimum Value Product) of DGL GNN Model Example Development

This project is designed for an MVP for GNN model development, and as a tutorial for new interns who will implement
GNN-related research papers with DGL.

GNN model used in this MVP comes from the paper of [CompositionGCN](https://arxiv.org/abs/1911.03082), and with modification
for our education purpose. In this project, this is called CompGCN.

The Graph Data to Tackle
--------------------------
- Graph data (g): Homogenous graph
- Node features (n_feats): Could be any dimensions
- Edge features (e_feats): Only one dimension with 3 options, 0,1,or 2

The Model and Algorithm
-------------------------

For the modified CompGCN, the detailed math formulas are in the CompGCN.ipynb notebook file. So bad github README.md 
does not support Letax coding.

The Structure of Example Files
--------------------------------

- **Model folder:** Files about GNN models and related files are put in this folder.

- **compgcn_node_classification.py:** The python file for training and inference(if required) using a full graph.

- **compgcn_node_classification_mb.py:** The python file for training and inference(if required) using sampling 
  and minibatch method.
 
 
## Community

.


