from impl import models, SubGDataset, train, metrics, config
import datasets
import torch
from torch.optim import Adam, lr_scheduler
import optuna
from torch.nn import BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')

# Node feature setting
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# Train settings
parser.add_argument('--repeat', type=int, default=1)
# Optuna Settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--abl', action='store_true')

parser.add_argument('--optruns', type=int, default=100)
parser.add_argument('--path', type=str, default="Emb/")
parser.add_argument('--name', type=str, default="opt")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
config.set_device(args.device)

baseG = datasets.load_dataset(args.dataset)

max_deg, max_z, output_channels = 0, 1, 1


def split():
    global trn_dataset, val_dataset
    global max_deg, max_z, output_channels, loader_fn, tloader_fn
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError
    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    x, ei, ea, pos, y = baseG.get_LPdataset()
    idx = torch.randperm(pos.shape[0], device=pos.device)
    trn_len = int(0.95 * idx.shape[0])
    trn_idx = idx[:trn_len]
    val_idx = idx[trn_len:]
    trn_dataset = SubGDataset.GDataset(x, ei, ea, pos[trn_idx], y[trn_idx])
    val_dataset = SubGDataset.GDataset(x, ei, ea, pos[val_idx], y[val_idx])

    def loader_fn(ds, bs):
        return SubGDataset.GDataloader(ds, bs)

    def tloader_fn(ds, bs):
        return SubGDataset.GDataloader(ds, bs, shuffle=False)


split()


def buildModel(hidden_dim, conv_layer, dropout, jk):
    tmp2 = hidden_dim * (conv_layer) if jk else hidden_dim
    conv = models.EmbGConv(hidden_dim,
                           hidden_dim,
                           hidden_dim,
                           conv_layer,
                           max_deg=max_deg,
                           activation=nn.ReLU(inplace=True),
                           jk=jk,
                           dropout=dropout,
                           conv=functools.partial(models.MyGCNConv,
                                                  aggr=args.aggr),
                           gn=True)

    edge_ssl = models.MLP(tmp2,
                          hidden_dim,
                          1,
                          2,
                          dropout=dropout,
                          activation=nn.ReLU(inplace=True))

    gnn = models.EdgeGNN(conv, nn.ModuleList([edge_ssl]),
                         nn.ModuleList([models.MeanPool()])).to(config.device)
    return gnn


def work(hidden_dim, conv_layer, dropout, jk, lr, batch_size):
    trn_loader = loader_fn(trn_dataset, batch_size)
    val_loader = tloader_fn(val_dataset, val_dataset.y.shape[0])
    outs = []
    loss_fn = lambda x, y: BCEWithLogitsLoss()(x.flatten(), y.flatten())
    for _ in range(args.repeat):
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk)
        with torch.no_grad():
            emb = gnn.NodeEmb(trn_dataset.x, trn_dataset.edge_index,
                              trn_dataset.edge_attr).detach().cpu()
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=0.7,
                                             min_lr=5e-5,
                                             patience=50)
        best_score = 0.0
        early_stop = 0
        for i in range(100):  # 400
            gnn.train()
            losss = []
            for ib, batch in enumerate(trn_loader):
                optimizer.zero_grad()
                emb = gnn.NodeEmb(trn_dataset.x, trn_dataset.edge_index,
                                  trn_dataset.edge_attr)
                edge_emb = gnn.Pool(emb, batch[-2], None)
                edge_pred = gnn.preds[0](edge_emb)
                loss = loss_fn(edge_pred, batch[-1])
                loss.backward()
                scd.step(loss)
                losss.append(loss.item())
                optimizer.step()
                if ib >= 9:
                    break
            if i % 5 == 0:
                score, _ = train.test(gnn, val_loader, metrics.binaryf1, loss_fn)
                print(f"iter {i} loss {np.average(losss)} score {score}",
                      flush=True)
                early_stop += 1
                if score > best_score:
                    with torch.no_grad():
                        emb = gnn.NodeEmb(
                            trn_dataset.x, trn_dataset.edge_index,
                            trn_dataset.edge_attr).detach().cpu()
                    best_score = score
                    early_stop = 0
                if early_stop >= 3:
                    break
            else:
                print(f"iter {i} loss {np.average(losss)}", flush=True)

        outs.append(best_score)

    return np.average(outs) - np.std(outs), emb


best_score = 0


def obj(trial):
    global trn_dataset, val_dataset, tst_dataset, args
    global input_channels, output_channels, loader_fn, tloader_fn
    global loss_fn, best_score
    hidden_dim = 64
    conv_layer = trial.suggest_int('conv_layer', 2, 5, step=1)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    args.aggr = trial.suggest_categorical("aggr", ["sum", "mean", "gcn"])
    jk = 0
    lr = 1e-3
    batch_size = 131072
    jk = (jk == 1)
    score, emb = work(hidden_dim, conv_layer, dropout, jk, lr, batch_size)
    # save best embeddings
    if score > best_score:
        torch.save(emb, f"{args.path}{args.name}_{hidden_dim}.pt")
        best_score = score
    return score


print(args)
# tuning hyperparameters of pretrained GNNs.
study = optuna.create_study(direction="maximize",
                            storage="sqlite:///" + args.path + args.name +
                            ".db",
                            study_name=args.name,
                            load_if_exists=True)
study.optimize(obj, n_trials=args.optruns)
print("best params ", study.best_params)
print("best valf1 ", study.best_value)
