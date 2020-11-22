#-*- coding:utf-8 -*-

"""
    The main file to train an GNN model.
"""

import os
import sys
import numpy as np
import argparse
import dgl
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading import NodeDataLoader

from models.model import CompGCN


def main(args):

    # Step 1： Prepare graph data and split into train/validation ============================= #
    # Load from example dataset
    graph = args.dataset[0]

    # retrieve masks for train/test
    train_mask = graph.ndata.pop('train_mask')
    test_mask = graph.ndata.pop('test_mask')
    train_idx = th.nonzero(train_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()

    # retrieve labels of target
    labels = graph.ndata.pop('labels')

    # split dataset into train and validate
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # Extract node
    n_feats = graph.ndata.pop('f')
    in_dim = n_feats.shape(-1)

    # Check if given e_feats_name exist in the graph's edata
    e_feats_name = args.e_feats_name
    try:
        e_feats = graph.edata[e_feats_name]
    except Exception:
        print("The given feature name {} does NOT exist in the graph' edge data...".format(e_feats_name))
        sys.exit(-1)

    e_feats = graph.edata.pop('f')

    # check cuda
    use_cuda = (args.gpu >= 0 and th.cuda.is_available())
    print("If use GPU: {}".format(use_cuda))

    if use_cuda:
        # graph = graph.to('cuda:{}'.format(args.gpu))     ? should we have graph in GPU when we can only sample it in CPU?
        n_feats = n_feats.to('cuda:{}'.format(args.gpu))
        e_feats = e_feats.to('cuda:{}'.format(args.gpu))
        labels = labels.to('cuda:{}'.format(args.gpu))

    # Step 2: Create sampler and dataloader for mini batch
    sampler = MultiLayerFullNeighborSampler()

    train_dataloader = NodeDataLoader(g=graph,
                                      nids=train_idx,
                                      block_sampler=sampler,
                                      shuffle=True,
                                      batch_size=args.batchsize,
                                      num_workers=0)

    valid_dataloader = NodeDataLoader(g=graph,
                                      nids=val_idx,
                                      block_sampler=sampler,
                                      batch_size=args.batchsize,
                                      shuffle=False,
                                      num_workers=0)

    test_dataloader = NodeDataLoader(g=graph,
                                     nids=test_idx,
                                     block_sampler=sampler,
                                     batch_size=args.batchsize,
                                     shuffle=False,
                                     num_workers=0)

    # Step 3: Create model =================================================================== #
    compgcn_model = CompGCN(in_dim=in_dim,
                            hid_dim=args.hid_dim,
                            out_dim=args.num_classes,
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

        # Training with mini batch
        compgcn_model.train()
        for i, (input_nodes, output_nodes, bipartites) in enumerate(train_dataloader):
            # retrive node features of the
            n_feats_mb = n_feats[input_nodes]
            labels_mb = labels[output_nodes]

            if use_cuda:
                bipartites = [b.to('cuda:{}'.format(args.gpu)) for b in bipartites]
                n_feats_mb = n_feats_mb.to('cuda:{}'.format(args.gpu))
                labels_mb = labels_mb.to('cuda:{}'.format(args.gpu))

            # forward
            tr_logits = compgcn_model.forward(bipartites, n_feats_mb, e_feats_name)

            # compute loss
            tr_loss = loss_fn(tr_logits, labels_mb)
            tr_acc = th.sum(tr_logits.argmax(dim=1) == labels_mb).item() / len(output_nodes)

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            # Print out performance
            print("In epoch|batch {}|{}, Train Acc: {:.4f} | Train Loss: {:.4f}".
                  format(epoch, i, tr_acc, tr_loss.item()))

        # Validation with mini batch
        compgcn_model.eval()
        for v, (input_nodes, output_nodes, bipartites) in enumerate(valid_dataloader):
            # retrive node features of the
            n_feats_mb = n_feats[input_nodes]
            labels_mb = labels[output_nodes]

            if use_cuda:
                bipartites = [b.to('cuda:{}'.format(args.gpu)) for b in bipartites]
                n_feats_mb = n_feats_mb.to('cuda:{}'.format(args.gpu))
                labels_mb = labels_mb.to('cuda:{}'.format(args.gpu))

            # forward
            valid_logits = compgcn_model.forward(bipartites, n_feats_mb, e_feats_name)

            # compute loss
            valid_loss = loss_fn(valid_logits, labels_mb)
            valid_acc = th.sum(valid_logits.argmax(dim=1) == labels_mb).item() / len(train_idx)

            print("In epoch|batch {}|{}, Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                  format(epoch, v, valid_acc, valid_loss.item()))

    # Test with mini batch after all epoch
    compgcn_model.eval()
    for i, (input_nodes, output_nodes, bipartites) in enumerate(test_dataloader):
        # retrive node features of the
        n_feats_mb = n_feats[input_nodes]
        labels_mb = labels[output_nodes]

        if use_cuda:
            bipartites = [b.to('cuda:{}'.format(args.gpu)) for b in bipartites]
            n_feats_mb = n_feats_mb.to('cuda:{}'.format(args.gpu))
            labels_mb = labels_mb.to('cuda:{}'.format(args.gpu))

        # forward
        test_logits = compgcn_model.forward(bipartites, n_feats_mb, e_feats_name)

        # compute loss
        test_loss = loss_fn(test_logits, labels_mb)
        test_acc = th.sum(test_logits.argmax(dim=1) == labels_mb).item() / len(output_nodes)

        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

    print()

    # Step 5: If need, save model to file ============================================================== #
    model_stat_dict = compgcn_model.state_dict()
    model_path = args.save_path
    th.save(model_stat_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoSH CompGCN Full Graph')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimensionalities")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of prediction classes")
    parser.add_argument("--e_feats_name", type=str, default='f', help="The key name of edge features")
    parser.add_argument("--comp_fn", type=str, default='sub', help="Composition function")
    parser.add_argument("--max_epoch", type=int, default=200, help="The max number of epoches")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--drop_out", type=float, default=0.1, help="Drop out rate")
    parser.add_argument("--save_path", type=str, default='./model.pth', help="File path of the model to be saved.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)

    np.random.seed(123456)
    th.manual_seed(123456)

    main(args)