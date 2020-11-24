#-*- coding:utf-8 -*-

"""
    A utility file to build an example graph data for the MVP.
"""

import os
import math
import dgl
import argparse

import torch as th
import numpy as np
import networkx as nx

from dgl import save_graphs

def house(start, role_start=0):
    """Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles

def ba(start, width, role_start=0, m=5):
    """Builds a BA preferential attachment graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles


def build_graph(
            width_basis,
            basis_type,
            list_shapes,
            start=0,
            rdm_basis_plugins=False,
            add_random_edges=0,
            m=5,
        ):
    """This function creates a basis (scale-free, path, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    m                :      number of edges to attach to existing node (for BA graph)
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :      a nx graph with the particular shape
    role_ids         :      labels for each role
    plugins          :      node ids with the attached shapes
    """
    if basis_type == "ba":
        basis, role_id = eval(basis_type)(start, width_basis, m=m)
    else:
        basis, role_id = eval(basis_type)(start, width_basis)

    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis  # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motifs
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(n_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    seen_shapes = {"basis": [0, n_basis]}

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape) > 1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        if shape_type == "cycle":
            if np.random.random() > 0.5:
                a = np.random.randint(1, 4)
                b = np.random.randint(1, 4)
                basis.add_edges_from([(a + start, b + plugins[shape_id])])
        temp_labels = [r + col_start for r in roles_graph_s]
        # temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis), 2, replace=False)
            print(src, dest)
            basis.add_edges_from([(src, dest)])

    return basis, role_id, plugins


def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    G, role_id, _ = build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    #     if feature_generator is None:
    #         feature_generator = ConstFeatureGen(1)
    #     feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def create_sample_data(args):
    """
    Create a sample graph with given number of nodes.
    Reuse the synthetic data generation from GNN Explainer codes

    :return:
        save the graph data in dgl graph format with features and addtional information.
    """
    g, labels, name = gen_syn1(nb_shapes=args.num_nodes, width_basis=500)
    file_name = 'syn1' + '_' + name + '.bin'

    # covert multiple classes to binary, 0->0; else->1
    bin_labels = th.Tensor([1 if e>0 else 0 for e in labels])

    # create dgl graph and convert to bidirected
    graph = dgl.from_networkx(g)
    graph = dgl.to_bidirected(graph)

    # create mask of labels
    num_nodes = len(bin_labels)
    num_tr = int(num_nodes * 0.8)
    tr_nodes_idx = np.random.choice(np.arange(num_nodes), num_tr)
    tr_mask = np.array([False] * num_nodes)
    tr_mask[tr_nodes_idx] = True
    test_mask = np.invert(tr_mask)

    graph.ndata['train_mask'] = th.from_numpy(tr_mask)
    graph.ndata['test_mask'] = th.from_numpy(test_mask)

    graph.ndata['labels'] = bin_labels

    # create node with small random number
    graph.ndata['f'] = th.randn([num_nodes, 10]) * 0.01

    # create edge features according to nodes' labels, 0: 0->0, 1: 0->1, 2: 1->1
    u, v = graph.edges()
    u_label = bin_labels[u]
    v_label = bin_labels[v]
    e_feats = u_label + v_label
    graph.edata['f'] = e_feats

    # save graph data to local disk
    file_path = os.path.join(args.save_path, file_name)
    save_graphs(file_path, [graph])

    print("Sample graph data is saved to {}...".format(file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MVP data generation")
    parser.add_argument("--save_path", type=str, default='./', help="The folder where to save the graph data")
    parser.add_argument("--num_nodes", type=int, default=500, help="The number of nodes to generate")

    args = parser.parse_args()

    create_sample_data(args)
