import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from layers.gcn_factory import GcnCreator
from layers.gin_factory import GinCreator
from layers.sage_factory import SageCreator


class GNN(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        global_pooling,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)
        elif gnn == "sage":
            gnn_instance = SageCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x.float())

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)

        x = self.pooling_fun(x, batch)
        return x



class GNNClassifier(nn.Module):

    def __init__(self, width_final_mlp, n_layers_final_mlp, num_classes,
                 gnn, gnn_hidden, gnn_depth, bn, num_features):
        super().__init__()

        self.gnn = GNN(gnn=gnn, hidden_dim=gnn_hidden, depth=gnn_depth,
                       num_node_features=num_features, global_pooling='sum', batch_norm=bn)
        layers = []

        for i in range(n_layers_final_mlp):
            layers.append(nn.Linear(
                in_features=width_final_mlp if i != 0 else (gnn_hidden),
                out_features=width_final_mlp))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=width_final_mlp if n_layers_final_mlp != 0 else (
            gnn_hidden),
                                out_features=num_classes))
        self.rho = nn.Sequential(*layers)

    def forward(self, inputs):
        gnn_embedding = self.gnn(inputs.x, inputs.edge_index, inputs.batch)
        return self.rho(gnn_embedding)
