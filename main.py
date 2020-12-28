import warnings
import argparse
import torch as th
import torch.optim as optim
import dgl
from dgl.data import CoraGraphDataset
import torch.nn.functional as F
from model import CompGCN
from utils import extract_cora_edge_direction

warnings.filterwarnings('ignore')   


def train(args):

    if args.gpu != -1 and th.cuda.is_available:
        device = 'cuda:%s'%args.gpu
    else:
        device = 'cpu'

    dataset = CoraGraphDataset(raw_dir='')
    graph = dataset[0]

    num_classes = dataset.num_classes
    labels = graph.ndata.pop('label').to(device).long()

    node_feats = graph.ndata.pop('feat').to(device)
    feat_dim = node_feats.shape[-1]

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    in_edges_mask, out_edges_mask = extract_cora_edge_direction(graph)
    in_edges_mask = th.Tensor(in_edges_mask)
    out_edges_mask = th.Tensor(out_edges_mask)
    graph.edata['out'] = out_edges_mask
    graph.edata['in'] = in_edges_mask

    edge_label_idx = {'cite': th.Tensor([i for i in range(graph.number_of_edges())]).long()}
    etypes = list(edge_label_idx.keys())

    ''' data setting & mdoel hyper-parameters '''

    in_units = feat_dim
    hid_units = args.hid_units
    out_units = num_classes
    num_basis = args.num_basis
    num_layer = args.num_layers
    act = F.relu
    batchnorm = False
    composition = args.composition
    dropout_rate = args.dropout

    model = CompGCN(in_units, hid_units, out_units, etypes, num_basis, num_layer, act, batchnorm, composition, device, dropout_rate)

    model.reset_parameters()
    model = model.to(device)
    graph = graph.to(device)

    loss_fn = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.max_epoch):

        # Training and validation using a full graph
        model.train()

        logits = model(graph, node_feats, edge_label_idx)

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


    model.eval()

    # forward
    logits = model.forward(graph, node_feats, edge_label_idx)

    # compute loss
    test_loss = loss_fn(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CompGCN Full Graph')
    parser.add_argument("--dataset", type=str, default="cora", help="DGL dataset for this MVP")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--hid_units", type=int, default=100, help="Hidden layer dimensionalities")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--num_basis", type=int, default=1, help="Number of basis")
    parser.add_argument("--composition", type=str, default='sub', help="Composition function. "
                                                                   "Valid options: sub, mul and ccorr")
    parser.add_argument("--max_epoch", type=int, default=100, help="The max number of epoches. Default: 100")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default: 3e-1")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate. Default: 0.0")

    args = parser.parse_args()
    print(args)

    train(args)
