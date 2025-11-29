import argparse
from torch import tensor
import numpy as np
import os
import json

from gnn import GNNClassifier
from train import *

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader

from datasets import get_data
from vectorization import PersLay
import torch.optim as optim

parser = argparse.ArgumentParser(description='Topology of Graph Products!')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--folder', type=str, default='pre_processed', help='Folder for pre-processed data')
parser.add_argument('--logdir', type=str, default='results', help='Log directory')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['DHFR', 'MUTAG', 'COX2', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY', 'ZINC', 'IMDB-MULTI'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--width_weight_fn', type=int, default=64)
parser.add_argument('--width_final_mlp', type=int, default=64)
parser.add_argument('--n_layers_weight_fn', type=int, default=1)
parser.add_argument('--n_layers_final_mlp', type=int, default=2)
parser.add_argument('--q', type=int, default=64)
parser.add_argument('--gnn_hidden', type=int, default=64)
parser.add_argument('--gnn_depth', type=int, default=4)
parser.add_argument("--no-bn", dest="bn", action="store_false")

parser.add_argument('--agg_type', type=str, default='mean', choices=['mean', 'sum', 'max'])
parser.add_argument('--gnn', type=str, default='gin', choices=['gin', 'gcn', 'sage'])
parser.add_argument("--no-gnn", dest="use_gnn", action="store_false")
parser.add_argument('--use_weight_fn',  action="store_true")

parser.add_argument('--filtration_type', type=str, default='full_prod', choices=['vertex_prod', 'full_prod', 'none'])
parser.add_argument('--filtration_fn', type=str, default='betweenness', choices=['degree', 'betweenness'])

parser.add_argument("--early_stop_patience", type=int, default=40)
parser.add_argument("--lr_decay_patience", type=int, default=10)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.logdir = f'{args.logdir}/{args.dataset}/{args.agg_type}/' \
                f'{args.gnn}/{args.gnn_depth}/' \
                f'{args.n_layers_weight_fn}/{args.width_weight_fn}/{args.n_layers_final_mlp}/{args.width_final_mlp}/' \
              f'{args.q}/filtration_type{args.filtration_type}/filtration_fn{args.filtration_fn}'

if not os.path.exists(f"{args.logdir}"):
    os.makedirs(f"{args.logdir}")

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

train_data, val_data, test_data, stats = get_data(args.folder, args.dataset, args.filtration_fn, args.filtration_type, args.seed)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# Model
if args.filtration_type == 'none':
    model = GNNClassifier(args.width_final_mlp, args.n_layers_final_mlp, stats["num_classes"],
    args.gnn, args.gnn_hidden, args.gnn_depth, args.bn, stats["num_features"])
else:
    model = PersLay(width_weight_fn=args.width_weight_fn, width_final_mlp=args.width_final_mlp,
                n_layers_weight_fn=args.n_layers_weight_fn, n_layers_final_mlp=args.n_layers_final_mlp,
                num_classes=stats["num_classes"],
                q=args.q, agg_type=args.agg_type,
                use_gnn=args.use_gnn, gnn=args.gnn, gnn_hidden=args.gnn_hidden, gnn_depth=args.gnn_depth, bn=args.bn, num_features=stats["num_features"], use_weight=args.use_weight_fn)


model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    min_lr=1e-6,
    patience=args.lr_decay_patience,
)

loss_fn = torch.nn.CrossEntropyLoss()
if args.dataset == "ZINC":
    loss_fn = torch.nn.L1Loss(reduction='mean')

train_losses = []
train_signed_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
val_losses = []
val_accuracies = []

models = []

for epoch in range(1, args.max_epochs + 1):
    # train
    train_loss = train(train_loader, model, loss_fn, optimizer, device)

    # evaluation
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    test_accuracies.append(test_acc)
    test_losses.append(test_loss)  # test losses

    val_accuracies.append(val_acc)
    val_losses.append(val_loss)  # test losses

    train_losses.append(torch.tensor(train_loss).mean())  # train losses

    if (epoch - 1) % args.interval == 0:
        print(
            f"{epoch:3d}: Train Loss: {torch.tensor(train_loss).mean():.3f},"
            f" Val Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, "
            f"Test Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}"
        )

    scheduler.step(val_acc)

    if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2 - epochs_no_improve]:
        epochs_no_improve = epochs_no_improve + 1

    else:
        epochs_no_improve = 0

    if epochs_no_improve >= args.early_stop_patience:
        print("Early stopping!")
        break


results = {
    "train_losses": tensor(train_losses),
    "test_accuracies": tensor(test_accuracies),
    "test_losses": tensor(test_losses),
    "val_accuracies": tensor(val_accuracies),
    "val_losses": tensor(val_losses),
}

torch.save(results, f'{args.logdir}/gprod_{args.seed}.results')
