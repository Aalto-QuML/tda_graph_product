import torch
from sklearn.model_selection import StratifiedShuffleSplit
import os.path as osp
from torch_geometric.datasets import ZINC, TUDataset


class FilterConstant(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, self.dim)
        return data


def get_tu_datasets(name, feat_replacement='constant'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), './datasets', name)
    dataset = TUDataset(name=name, root=path)
    if not hasattr(dataset, 'x'):
        if feat_replacement == 'constant':
            dataset.transform = FilterConstant(10)
    return dataset


def get_zinc():
  path = osp.join(osp.dirname(osp.realpath(__file__)), './datasets', 'ZINC')
  train_data = ZINC(path, subset=True, split='train')
  data_val = ZINC(path, subset=True, split='val')
  data_test = ZINC(path, subset=True, split='test')

  return train_data, data_val, data_test


def get_data(folder, name, filtration_fn, filtration_type='vertex_prod', seed=42):
    path = f'{folder}/{name}_{filtration_type}_{filtration_fn}.data'
    if name == 'ZINC':
        if filtration_type != 'none':
            data_dict = torch.load(path, weights_only=False)
            train_data, val_data, test_data = data_dict['train'], data_dict['val'], data_dict['test']
        else:
            train_data, val_data, test_data = get_zinc()
        num_classes = 1
    else:
        if filtration_type != 'none':
            data = torch.load(path, weights_only=False)
        else:
            data = get_tu_datasets(name)
        num_classes = data.num_classes
        train_data, val_data, test_data = data_split(data, seed=seed)

    stats = dict()
    stats['num_features'] = train_data.num_node_features
    stats['num_classes'] = num_classes

    return train_data, val_data, test_data, stats


def data_split(dataset, seed=42):
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
    train_data = dataset[train_idx]
    val_data = dataset[val_test_idx[val_idx]]
    test_data = dataset[val_test_idx[test_idx]]
    return train_data, val_data, test_data

