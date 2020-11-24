#-*- coding:utf-8 -*-

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
from dgl import DGLError

from .utils import ccorr

class CompGraphConv(nn.Module):
    """
        One layer of modified CompGCN
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

        # define weights of 3 node matrics
        self.W_0 = nn.Linear(self.in_dim, self.out_dim)
        self.W_1 = nn.Linear(self.in_dim, self.out_dim)
        self.W_2 = nn.Linear(self.in_dim, self.out_dim)
        self.W_h = nn.Linear(self.in_dim, self.out_dim)


    def forward(self, g, n_in_feats, e_feats_name):
        """
            Compute one layer of composition transfer
        """
        # assign values to destination nodes
        n_h_dst = n_in_feats[:g.number_of_dst_nodes()]

        with g.local_scope():
            # assign values to source nodes
            g.srcdata['h'] = n_in_feats

            # get e_features
            e_feats = g.edata[e_feats_name]

            # compute composition function in two steps
            # Step 1, compute composition by edge and store results in edges.
            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_v('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_v('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.dst['h'])})
            else:
                raise DGLError('Only supports sub, mul, and ccorr')

            # Step 2, linear transfer by UDF that uses different weight matrixes based on edge values
            comp_h = g.edata['comp_h']
            comp_h0 = self.W_0(comp_h)
            comp_h1 = self.W_1(comp_h)
            comp_h2 = self.W_2(comp_h)

            # use edge value as mask to filter different edge values
            mask_0 = th.unsqueeze((e_feats == 0) * 1, 1)
            comp_h0 = comp_h0 * mask_0
            mask_1 = th.unsqueeze((e_feats == 1) * 1, 1)
            comp_h1 = comp_h1 * mask_1
            mask_2 = th.unsqueeze((e_feats == 2) * 1, 1)
            comp_h2 = comp_h2 * mask_2

            # Add the three together to edge features and assign them to graph edges
            comp_m = comp_h0 + comp_h1 + comp_h2
            g.edata['comp_m'] = comp_m

            # Send comp_m to destination nodes and aggregate
            g.update_all(fn.copy_e('comp_m', 'm'), fn.sum('m', 'out'))

            # Post process
            n_out_feats = g.dstdata['out'] + self.W_h(n_h_dst)

            # Use drop out
            n_out_feats = self.dropout(n_out_feats)

            # Use batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Use activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats


class CompGCN(nn.Module):
    """

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

        # Input layer and hidden layers
        self.layers = nn.ModuleList()
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


    def forward(self, bipartites, n_feats, e_feats_name):

        # Forward of n layers of CompGraphConv
        for layer, bipartite in zip(self.layers, bipartites):
            n_feats = layer(bipartite, n_feats, e_feats_name)

        return n_feats


if __name__ == '__main__':
    # Test with the original paper's diagram
    pass
