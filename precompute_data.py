import argparse
import networkx as nx
import itertools
import os
from gudhi.simplex_tree import SimplexTree

from torch_geometric.utils.convert import to_networkx
from prod_filtrations import dataset_entry_to_input, thm4_ph

import torch
from datasets import get_tu_datasets, get_zinc

parser = argparse.ArgumentParser(description='Pre-computing the datasets!')
parser.add_argument('--folder', type=str, default='pre_processed', help='Log directory')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['DHFR', 'COX2', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY', 'ZINC', 'IMDB-MULTI'])
parser.add_argument('--filtration_type', type=str, default='vertex_prod', choices=['vertex_prod', 'full_prod'])
parser.add_argument('--filtration_fn', type=str, default='betweenness', choices=['degree', 'betweenness'])

family = [
    nx.cycle_graph(4),
    nx.complete_graph(5),
    nx.path_graph(4),
    nx.star_graph(5),
    nx.lollipop_graph(3, 3),
    nx.watts_strogatz_graph(10, 2, 0.2, seed=42),
    nx.house_graph()
]


def get_data(name, filtration_fn, filtration_type):

    if filtration_type == 'vertex_prod':
        func = vertex_prod_filtration
    elif filtration_type == 'full_prod':
        func = full_prod

    if name == "ZINC":
        train_data, data_val, data_test = get_zinc()
        train_data = func(train_data, filtration_fn)
        val_data = func(data_val, filtration_fn)
        test_data = func(data_test, filtration_fn)
        return train_data, val_data, test_data
    else:
        dataset = get_tu_datasets(name)
        return func(dataset, filtration_fn)


def full_prod(dataset, filtration_fn):
    print("Computing full product filtrations...")
    data_list = []
    j = 0
    for data in dataset:
        g = to_networkx(data).to_undirected()
        list_dgms = {}
        j = j + 1
        for i, anchor in enumerate(family):
            h = anchor
            g_box_h = nx.cartesian_product(g, h)
            g_box_h = nx.convert_node_labels_to_integers(g_box_h, first_label=0)

            if filtration_fn == 'betweenness':
                fv = list(nx.betweenness_centrality(g_box_h).values())
            elif filtration_fn == 'degree':
                fv = list(nx.degree_centrality(g_box_h).values())

            st = SimplexTree()
            for n in g_box_h.nodes:
                st.insert([n], filtration=fv[n])

            for e in g_box_h.edges:
                st.insert([e[0], e[1]], filtration=max(fv[e[0]], fv[e[1]]))

            st.make_filtration_non_decreasing()
            diagrams = st.persistence(min_persistence=-1, persistence_dim_max=False)
            dgms_tensor = torch.tensor([[x, y] for _, (x, y) in diagrams]).view(-1, 2)
            dgms_tensor[dgms_tensor == float('inf')] = 1  # replaces infinity tuples with the maximum
            list_dgms[i] = dgms_tensor

        data.diagms = list_dgms
        data_list.append(data)
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


def vertex_prod_filtration(dataset, filtration_fn):
    print("Computing vertex product filtrations...")
    product = itertools.product(dataset, family)
    data_list = []
    count, total = 0, 0
    for item in product:
        g_networkx = to_networkx(item[0]).to_undirected()
        h_networkx = item[1]

        if filtration_fn == 'betweenness':
            g_fv = list(nx.betweenness_centrality(g_networkx).values())
            h_fv = list(nx.betweenness_centrality(h_networkx).values())
        elif filtration_fn == 'degree':
            g_fv = list(nx.degree_centrality(g_networkx).values())
            h_fv = list(nx.degree_centrality(h_networkx).values())

        if count % len(family) == 0:
            count = 0
            internal_list = {}
            total = total+1

        # code that computes the PDs
        entry_impl, entry_gd = dataset_entry_to_input((g_networkx, h_networkx), g_fv, h_fv)
        output = thm4_ph(entry_gd)

        internal_list[count] = torch.tensor(output)
        if count % len(family) == len(family)-1:
            item[0].diagms = internal_list
            data_list.append(item[0])

        count = count + 1
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


args = parser.parse_args()
if not os.path.exists(f"{args.folder}"):
    os.makedirs(f"{args.folder}")

name = f'{args.folder}/{args.dataset}_{args.filtration_type}_{args.filtration_fn}.data'
if args.dataset != 'ZINC':
    torch.save(get_data(args.dataset, args.filtration_fn, args.filtration_type), name)
else:
    train, val, test = get_data(args.dataset, args.filtration_fn, args.filtration_type)
    torch.save({
        'train': train,
        'val': val,
        'test': test
    }, name)