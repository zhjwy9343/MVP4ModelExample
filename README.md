# DGL Implementation of the CompGCN Paper

This DGL example implements the GNN model proposed in the paper [CompositionGCN](https://arxiv.org/abs/1911.03082). 
The author's codes of implementation is in [here](https://github.com/malllabiisc/CompGCN)

**Note**: Please replace the paper and author's codes with the paper you implemented.

The graph dataset used in this example 
---------------------------------------
The DGL's built-in CoraGraphDataset. Dataset summary:
- NumNodes: 2708
- NumEdges: 10556
- NumFeats: 1433
- NumClasses: 7
- NumTrainingSamples: 140
- NumValidationSamples: 500
- NumTestSamples: 1000

How to run example files
--------------------------------
In the MVP4ModelExample folder, run

```python
python main.py
```

If want to use a GPU, run

```python
python main.py --gpu 0
```

Performance
-------------------------
**Note**: If your implementation needs to reproduce the SOTA performance, please put these performance here.
