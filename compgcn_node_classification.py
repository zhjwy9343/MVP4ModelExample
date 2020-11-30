#-*- coding:utf-8 -*-

"""
    The main file to train an Simplified CompGCN model using a full graph.
"""

import argparse
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import dgl
from dgl import DGLError
from dgl.data import CoraGraphDataset
import dgl.function as fn

from utils import ccorr


class CompGraphConv(nn.Module):
    """
        One layer of simplified CompGCN.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 comp_fn='sub',
                 activation=None,
                 batchnorm=False,
                 dropout=0,
                 ):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # define weights of 3 node matrices
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        # define weights of the 1 relation matrix
        self.W_R = nn.Linear(self.in_dim, self.out_dim)


    def forward(self, g, n_in_feats, r_feats):
        """
            Compute one layer of composition transfer for one relation only in a homogeneous graph with additional
            reversed edges.
        """
        with g.local_scope():
            # assign values to source nodes. In a homo graph, this is equal to assigning them to all nodes.
            g.srcdata['h'] = n_in_feats

            # assign feature to all edges with the same value, the r_feats.
            g.edata['h'] = th.stack([r_feats] * g.number_of_edges())

            # compute composition function in 4 steps
            # Step 1, compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_v('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_v('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.dst['h'])})
            else:
                raise DGLError('Only supports sub, mul, and ccorr')

            # Step 2, take advantage of dgl's edges and reversed-edges order
            comp_h = g.edata['comp_h']
            half = int(comp_h.shape[0] / 2)
            comp_h_O = self.W_O(comp_h[:half])
            comp_h_I = self.W_I(comp_h[half:])
            new_comp_h = th.cat([comp_h_O, comp_h_I], dim=0)
            g.edata['new_comp_h'] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e('new_comp_h', 'm'), fn.sum('m', 'comp_edge'))

            # Step 4: add results of self-loop
            if self.comp_fn == 'sub':
                comp_h_s = n_in_feats - r_feats
            elif self.comp_fn == 'mul':
                comp_h_s = n_in_feats * r_feats
            elif self.comp_fn == 'ccorr':
                comp_h_s = ccorr(n_in_feats, r_feats)
            else:
                raise DGLError('Only supports sub, mul, and ccorr')

            # sum all of the comp results as output
            n_out_feats = self.W_S(comp_h_s) + g.ndata['comp_edge']

            # compute relation output
            r_out_feats = self.W_R(r_feats)

            # Use batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Use drop out
            n_out_feats = self.dropout(n_out_feats)
            r_out_feats = self.dropout(r_out_feats)

            # Use activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)
                r_out_feats = self.actvation(r_out_feats)

        return n_out_feats, r_out_feats


class CompGCN(nn.Module):
    """
        The model of the simplified CompGCN, without using basis vector, for a homograph
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=2,
                 comp_fn='sub',
                 dropout=0.0,
                 activation=None,
                 batchnorm=False
                 ):
        super(CompGCN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layer = num_layers
        self.comp_fn = comp_fn
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()

        # Input layer and initial relation embedding
        self.r_embedding = nn.Parameter(th.Tensor(self.in_dim))

        self.layers.append(CompGraphConv(self.in_dim,
                                         self.hid_dim,
                                         comp_fn = self.comp_fn,
                                         activation=self.activation,
                                         batchnorm=self.batchnorm,
                                         dropout=self.dropout))

        # Hidden layers with n - 1 CompGraphConv layers
        for i in range(self.num_layer - 2):
            self.layers.append(CompGraphConv(self.hid_dim,
                                             self.hid_dim,
                                             comp_fn=self.comp_fn,
                                             activation=self.activation,
                                             batchnorm=self.batchnorm,
                                             dropout=self.dropout))

        # Output layer with the output class
        self.layers.append(CompGraphConv(self.hid_dim,
                                         self.out_dim,
                                         comp_fn = self.comp_fn))

        # initialize relation embeddings
        th.nn.init.uniform_(self.r_embedding)

    def forward(self, graph, n_feats):

        # For full graph training, directly use the graph

        # Forward of n layers of CompGraphConv
        r_feats = self.r_embedding

        for layer in self.layers:
            n_feats, r_feats = layer(graph, n_feats, r_feats)

            # print(r_feats.shape)

        return n_feats


def main(args):

    # Step 1： Prepare graph data and split into train/validation ============================= #
    # Load from DGL dataset
    if args.dataset == 'cora':
        dataset = CoraGraphDataset(force_reload=True)
        graph = dataset[0]
    else:
        raise NotImplementedError

    # check cuda
    use_cuda = (args.gpu >= 0 and th.cuda.is_available())
    print("If use GPU: {}".format(use_cuda))

    if use_cuda:
        device = 'cuda:{}'.format(args.gpu)
        graph = graph.to(device)

    # retrieve the number of classes
    num_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata['label'].long()

    # Extract node features
    n_feats = graph.ndata.pop('feat')
    in_dim = n_feats.shape[-1]

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask).squeeze()
    val_idx = th.nonzero(val_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()

    # In this Cora dataset, we only have one relationship in a homogenous graph, which has direction of edges.
    # So our model will have one relation embedding, and take advantage of DGL's build-in APIs to get the reverse direction
    # for computing the 3 directional weights, and 1 relation weight.
    graph = dgl.add_reverse_edges(graph)

    # Step 2: Create model =================================================================== #
    compgcn_model = CompGCN(in_dim=in_dim,
                            hid_dim=args.hid_dim,
                            out_dim=num_classes,
                            num_layers=args.num_layers,
                            comp_fn=args.comp_fn,
                            dropout=args.drop_out,
                            activation=F.relu,
                            batchnorm=True
                            )

    if use_cuda:
        compgcn_model = compgcn_model.to('cuda:{}'.format(args.gpu))

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': compgcn_model.parameters(), 'lr':args.lr, 'weight_decay':5e-4}])

    # Step 4: training epoches =============================================================== #
    for epoch in range(args.max_epoch):

        # Training and validation using a full graph
        compgcn_model.train()

        logits = compgcn_model.forward(graph, n_feats)

        # compute loss
        tr_loss = loss_fn(logits[train_idx], labels[train_idx])
        tr_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

        valid_loss = loss_fn(logits[val_idx], labels[val_idx])
        valid_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Print out performance
        print("In epoch {}, Train Acc: {:.4f} | Train Loss: {:.4f}; Valid Acc: {:.4f} | Valid loss: {:.4f}".
              format(epoch, tr_acc, tr_loss.item(), valid_acc, valid_loss.item()))

    # Test with mini batch after all epoch
    compgcn_model.eval()

    # forward
    logits = compgcn_model.forward(graph, n_feats)

    # compute loss
    test_loss = loss_fn(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

    print()

    # Step 5: Optional, save model to file ============================================================== #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVP Mod-CompGCN Full Graph')
    parser.add_argument("--dataset", type=str, default="cora", help="DGL dataset for this MVP")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    parser.add_argument("--hid_dim", type=int, default=100, help="Hidden layer dimensionalities")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--comp_fn", type=str, default='ccorr', help="Composition function")
    parser.add_argument("--max_epoch", type=int, default=500, help="The max number of epoches")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out rate")

    args = parser.parse_args()
    print(args)

    main(args)
