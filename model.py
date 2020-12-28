import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
from torch.nn import Linear, Parameter
from utils import ccorr

class CompGCN(nn.Module):
    """ CompGCN model.
    Parameters：
    ----------
    in_units : int
        input dimension
    hid_units : int
        hidden dimension
    out_units : int
        output dimension
    num_basis : int
        number of basis for initialize the representation of relationship
    num_layer : init
        number of Comp Graph Conv layers in the model
    act : function
        activation function
    batchnorm : bool
        If True, apply a batch normalization
    composition : str
        the composition function used in calculating the message: 'add', 'sub', 'mul'
    device: str, optional
        which device to put data in
    dropout_rate: float
        dropout rate
    ----------

    Outputs:
    --------
    h_u ： Tensor: [n_nodes, n_classes]
        the final repr of the nodes in the graph
    --------
    """
    def __init__(self,
                 in_units,
                 hid_units,
                 out_units,
                 etypes,
                 num_basis,
                 num_layer = 4,
                 act = 'relu',
                 batchnorm = False,
                 composition  = 'sub',
                 device = 'cpu',
                 dropout_rate = 0.0):
        super(CompGCN, self).__init__()

        self.in_units = in_units
        self.etypes = etypes

        self.composition = composition
        self.device = device

        self.dropout = nn.Dropout(dropout_rate)

        self.num_layer = num_layer
        self.act = act

        # Initialize relation embeddings
        num_relation = len(self.etypes)
        self.Basis = Parameter(th.Tensor(num_basis, in_units))
        self.alpha = Parameter(th.Tensor(num_relation + 1, num_basis))


        self.Layers = nn.ModuleList()

        # Input Layer
        self.Layers.append(CompGCNLayer(in_units, hid_units, etypes, act, batchnorm,
                                        composition, device, dropout_rate))

        # Hidden Layers
        for i in range(num_layer-1):
            self.Layers.append(CompGCNLayer(hid_units, hid_units, etypes, act, batchnorm,
                                            composition, device, dropout_rate))
        # Output Layer
        self.Layers.append(CompGCNLayer(hid_units, out_units, etypes, None, batchnorm,
                                        composition, device, dropout_rate))

    def reset_parameters(self):
        init.xavier_uniform_(self.Basis)
        init.xavier_uniform_(self.alpha)

        for layer in self.Layers:
            layer.reset_parameters()

    # generate relation embeddings using basis vectors
    def get_rel_repr(self):

        z = dict()
        for idx, etype in enumerate(self.etypes):
            z[etype] = th.mm(self.alpha[idx].unsqueeze(0), self.Basis)
        z['self_loop'] = th.mm(self.alpha[-1].unsqueeze(0), self.Basis)

        return z

    def forward(self, graph, h_u, edge_label_idx):

        # get relation embedding
        h_r = self.get_rel_repr()

        # Model forward
        for layer in self.Layers:
            h_u, h_r = layer(graph, h_u, h_r, edge_label_idx)

        return h_u


class CompGCNLayer(nn.Module):
    """ Graph Convolution module used in CompGCN model.
    Parameters：
    ----------
    in_units : int
        input dimension
    out_units : int
        output dimension
    etypes: list: [str, str]
        edge types
    act : function
        activation function
    batchnorm : bool
        If True, apply a batch normalization
    composition : str
        the composition function used in calculating the message: 'add', 'sub', 'mul'
    device: str, optional
        which device to put data in
    dropout_rate: float
        dropout rate
    ----------

    Outputs:
    --------
    h_v ： Tensor: [n_nodes, embed_dim]
    h_r : Tensor: [n_relation, embed_dim]
    --------
    """
    def __init__(self,
                 in_units,
                 out_units,
                 etypes,
                 act = None,
                 batchnorm=False,
                 composition = 'sub',
                 device = 'cpu',
                 dropout_rate = 0.0):
        super(CompGCNLayer, self).__init__()

        self.in_units = in_units
        self.out_units = out_units
        self.etypes = etypes

        self.composition = composition
        self.device = device

        self.act = act
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn = nn.BatchNorm1d(out_units)

        # Define weight matrixs of 3 kinds of relations
        self.W = nn.ModuleDict()

        self.W['O'] = Linear(in_units, out_units, bias = False)   # W_O
        self.W['I'] = Linear(in_units, out_units, bias = False)   # W_I
        self.W['S'] = Linear(in_units, out_units, bias = False)   # W_S

        # Define weight for relation repr transformation
        self.W['rel'] = Linear(in_units, out_units, bias = False)    # W_rel


    def reset_parameters(self):
        for key in self.W.keys():
            self.W[key].reset_parameters()
        # self.W_rel.reset_parameters()

    def forward(self, graph, h_u, h_r, edge_label_idx):

        with graph.local_scope():

            n_edge = graph.number_of_edges()
            graph.edata['h'] = th.zeros(n_edge, self.in_units).to(self.device)
            graph.ndata['h'] = h_u

            ''' assign relation embedding to edges '''
            for etype in self.etypes:
                idx = edge_label_idx[etype]
                graph.edata['h'][idx] = h_r[etype]

            if self.composition == 'sub':
                graph.apply_edges(fn.u_sub_e('h', 'h', 'comp_m'))
                comp_s = h_u - h_r['self_loop']

            elif self.composition == 'mul':
                graph.apply_edges(fn.u_mul_e('h', 'h', 'comp_m'))
                comp_s = th.mul(h_u, h_r['self_loop'])

            elif self.composition == 'ccorr':
                graph.apply_edges(lambda edges: {'comp_m': ccorr(edges.src['h'], edges.data['h'])})
                comp_s = ccorr(h_u, h_r['self_loop'])
            else:
                raise Exception('Composition function not supported.')

            ''' message transformation '''
            comp_m = graph.edata.pop('comp_m')

            out_idx = th.nonzero(graph.edata['out']).squeeze()
            in_idx = th.nonzero(graph.edata['in']).squeeze()

            new_comp_m = th.zeros(graph.number_of_edges(), self.out_units).to(self.device)
            new_comp_m[out_idx] = self.W['O'](comp_m[out_idx])
            new_comp_m[in_idx] = self.W['I'](comp_m[in_idx])

            graph.edata['new_comp_m'] = new_comp_m

            ''' message passing '''
            graph.update_all(fn.copy_e('new_comp_m', 'm'), fn.sum('m', 'new_h'))

            # node embedding update
            h_v = self.W['S'](comp_s) + graph.ndata['new_h']

            # relation embedding update
            for etype in self.etypes:
                h_r[etype] = self.W['rel'](h_r[etype])
            h_r['self_loop'] = self.W['rel'](h_r['self_loop'])

            # batch normalization
            if self.batchnorm:
                h_v = self.bn(h_v)

            # drop out
            h_v = self.dropout(h_v)
            for etype in self.etypes:
                h_r[etype] = self.dropout(h_r[etype])
            h_r['self_loop'] = self.dropout(h_r['self_loop'])


            # activation
            if self.act:
                h_v = self.act(h_v)
                for etype in self.etypes:
                    h_r[etype] = self.act(h_r[etype])
                h_r['self_loop'] = self.act(h_r['self_loop'])

        return h_v, h_r




