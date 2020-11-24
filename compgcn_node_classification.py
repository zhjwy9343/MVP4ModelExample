#-*- coding:utf-8 -*-

"""
    The main file to train an GNN model using a full graph.
"""

import sys
import numpy as np
import argparse
import torch as th
import torch.optim as optim
import torch.nn.functional as F

from dgl import load_graphs
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
    labels = graph.ndata.pop('labels').long()

    # split dataset into train and validate
    if args.validation:
        valid_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        valid_idx = train_idx

    # Extract node
    n_feats = graph.ndata.pop('f')
    in_dim = n_feats.shape[-1]

    # Check if given e_feats_name exist in the graph's edata
    e_feats_name = args.e_feats_name
    try:
        e_feats = graph.edata[e_feats_name]
    except Exception:
        print("The given feature name {} does NOT exist in the graph' edge data...".format(e_feats_name))
        sys.exit(-1)

    # check cuda
    use_cuda = (args.gpu >= 0 and th.cuda.is_available())
    print("If use GPU: {}".format(use_cuda))

    if use_cuda:
        graph = graph.to('cuda:{}'.format(args.gpu))
        n_feats = n_feats.to('cuda:{}'.format(args.gpu))
        labels = labels.to('cuda:{}'.format(args.gpu))

    # Step 2: Create model =================================================================== #
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

        # Training and validation using a full graph
        compgcn_model.train()

        # forward
        graphs = [graph] * args.num_layers
        logits = compgcn_model.forward(graphs, n_feats, e_feats_name)

        # compute loss
        tr_loss = loss_fn(logits[train_idx], labels[train_idx])
        tr_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

        valid_loss = loss_fn(logits[valid_idx], labels[valid_idx])
        valid_acc = th.sum(logits[valid_idx].argmax(dim=1) == labels[valid_idx]).item() / len(valid_idx)

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
    logits = compgcn_model.forward(graphs, n_feats, e_feats_name)

    # compute loss
    test_loss = loss_fn(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

    print()

    # Step 5: If need, save model to file ============================================================== #
    model_stat_dict = compgcn_model.state_dict()
    model_path = args.save_path
    th.save(model_stat_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoSH CompGCN Full Graph')
    # parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index")
    parser.add_argument("--hid_dim", type=int, default=10, help="Hidden layer dimensionalities")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of prediction classes")
    parser.add_argument("--e_feats_name", type=str, default='f', help="The key name of edge features")
    parser.add_argument("--comp_fn", type=str, default='sub', help="Composition function")
    parser.add_argument("--max_epoch", type=int, default=20, help="The max number of epoches")
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

    args.dataset, _ = load_graphs('./syn1_ba_500_500.bin')

    main(args)