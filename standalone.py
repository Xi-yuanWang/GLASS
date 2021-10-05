from impl import modelx, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, SGD, lr_scheduler
import optuna
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Sequential
import argparse
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv
import functools
import numpy as np
import time
import random

parser = argparse.ArgumentParser(description='')
# Data settings
parser.add_argument('--dataset', type=str, default='ppi_bp')

# Label settings
parser.add_argument('--use_degree', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# Train settings
parser.add_argument('--repeat', type=int, default=1)
# Optuna Settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--optruns', type=int, default=100)
parser.add_argument('--path', type=str, default="")
parser.add_argument('--name', type=str, default="opt")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


def set_repeat_seed(repeat: int):
    a = [7, 15, 31, 63, 127, 255, 511, 0, 1, 3]
    set_seed(a[repeat])


if args.use_seed:
    set_seed(1024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, max_z, output_channels, input_channels = 0, 1, 1, 1
score_fn = None


def bloss(loss_fn, b):
    return lambda *kargs, **kwargs: torch.abs(loss_fn(*kargs, **kwargs) - b
                                              ) + b


if baseG.y.unique().shape[0] == 2:

    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    global trn_dataset, val_dataset, tst_dataset
    global max_deg, max_z, output_channels, loader_fn, tloader_fn, input_channels
    if args.use_degree:
        baseG.addDegreeFeature()
    elif args.use_one:
        baseG.addOneFeature()
    elif args.use_nodeid:
        baseG.addIDFeature()
    input_channels = baseG.x.shape[-1]
    #max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    t_trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    t_val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    t_tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))

    trn_dataset = SubGDataset.GsDataset(
        SubGDataset.todatalist(t_trn_dataset, 0))
    val_dataset = SubGDataset.GsDataset(
        SubGDataset.todatalist(t_val_dataset, 0))
    tst_dataset = SubGDataset.GsDataset(
        SubGDataset.todatalist(t_tst_dataset, 0))

    def tfunc(ds, bs, shuffle=True):
        return SubGDataset.GsDataloader(ds,
                                        bs,
                                        shuffle=shuffle)

    def loader_fn(ds, bs):
        return tfunc(ds, bs)

    def tloader_fn(ds, bs):
        return tfunc(ds, bs, False)


split()


def buildModel(hidden_dim,
               conv_layer,
               dropout,
               jk,
               lr,
               conv="GCN",
               pool=sum):
    conv2conv_fn = {"GCN": functools.partial(GCNConv, add_self_loops=False), "GIN": functools.partial(modelx.MyGINConv,activation=nn.ELU(inplace=True))}
    conv_fn = conv2conv_fn[conv]
    tmp2 = hidden_dim * (conv_layer) if jk else hidden_dim
    conv = modelx.GConv(input_channels,
                        hidden_dim,
                        hidden_dim,
                        conv_layer,
                        conv=conv_fn,
                        activation=nn.ELU(inplace=True),
                        jk=jk,
                        dropout=dropout,
                        gn=True)

    mlp = modelx.MLP(tmp2,
                     hidden_dim,
                     output_channels,
                     2,
                     dropout=dropout,
                     activation=nn.SELU(inplace=True),
                     bn=False)

    gnn = modelx.GNN(conv, mlp, aggr=pool).to(config.device)
    return gnn


def work(hidden_dim, conv_layer, dropout, jk, lr, batch_size, conv="GCN", pool="sum"):
    trn_loader = loader_fn(trn_dataset, batch_size)
    val_loader = tloader_fn(val_dataset, batch_size)
    outs = []

    for _ in range(args.repeat):
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, lr, conv, pool)
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=0.7,
                                             min_lr=5e-5)
        val_score = 0
        early_stop = 0
        for i in range(200):  # 400
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            scd.step(loss)
            if (i + 1) % 5 == 0:
                early_stop += 1
                score = train.test(gnn, val_loader, score_fn)
                if score > val_score:
                    early_stop = 0
                    val_score = score
                if early_stop > 50:
                    break
        outs.append(val_score)

    return sum(outs) / args.repeat


def obj(trial):
    global trn_dataset, val_dataset, tst_dataset
    global input_channels, output_channels, loader_fn, tloader_fn
    global loss_fn
    if args.dataset in ["cut_ratio","coreness","component","density"]:
        hidden_dim = trial.suggest_int('hidden_dim', 4, 16, step=4)
        conv_layer = trial.suggest_int('conv_layer', 1, 2)
    else:
        hidden_dim = trial.suggest_int('hidden_dim', 8, 64, step=8)
        conv_layer = trial.suggest_int('conv_layer', 1, 2)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    jk = 1  
    lr = trial.suggest_loguniform("lr", 5e-5, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    jk = (jk == 1)
    delta = trial.suggest_float('delta', 0.0, 0.0, step=0.1)
    conv = trial.suggest_categorical("conv", ["GIN", "GCN"])
    pool = trial.suggest_categorical("pool", ["sum", "mean"])
    return work(hidden_dim, conv_layer, dropout, jk, lr, batch_size)


def test(hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         batch_size=160,
         gamma=0.0,
         delta=0.0,
         conv="GCN",
         pool="sum"):
    trn_loader = loader_fn(trn_dataset, batch_size)
    val_loader = tloader_fn(val_dataset, batch_size)
    tst_loader = tloader_fn(tst_dataset, batch_size)

    for pos_w in [1]:
        outs = []
        for _ in range(args.repeat):
            print(f"repeat {_}")
            t1 = time.time()
            gnn = buildModel(hidden_dim, conv_layer, dropout, jk, lr,
                             conv, pool)
            optimizer = Adam(gnn.parameters(), lr=lr)
            scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.7,
                                                 min_lr=5e-5)
            val_score = 0
            print("build model ", time.time() - t1)
            early_stop = 0
            for i in range(500):
                t1 = time.time()
                loss = train.train(optimizer, gnn, trn_loader, loss_fn)
                print("trn time", time.time() - t1)
                scd.step(loss)

                t1 = time.time()
                if i % 5 == 0:
                    score = train.test(gnn,
                                       val_loader,
                                       score_fn,
                                       verbose=True,
                                       loss_fn=loss_fn)
                    early_stop += 1
                    if score > val_score:
                        val_score = score
                        score = train.test(gnn,
                                           tst_loader,
                                           score_fn,
                                           verbose=True,
                                           loss_fn=loss_fn)
                        tst_score = score
                        print(
                            f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                            flush=True)
                        early_stop /= 2
                    elif score >= val_score - 1e-5:
                        score = train.test(gnn,
                                           tst_loader,
                                           score_fn,
                                           verbose=True,
                                           loss_fn=loss_fn)
                        tst_score = max(score, tst_score)
                        print(
                            f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
                            flush=True)
                        early_stop /= 2
                    else:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, verbose=True, loss_fn=loss_fn):.4f}",
                            flush=True)
                    if early_stop > 10:
                        break
                print("val and tst time", time.time() - t1)
            print(f"end: val {val_score:.4f} tst {tst_score:.4f}", flush=True)
            outs.append(tst_score)
        print("tst scores", outs)
        print(np.average(outs), np.std(outs) / np.sqrt(len(outs)))
    return np.average(outs)


print(args)

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///" + args.path + args.name +
                            ".db",
                            study_name=args.name,
                            load_if_exists=True)
if args.test:
    print("best params", study.best_params, flush=True)
    print(test(**(study.best_params)))
else:
    study.optimize(obj, n_trials=args.optruns)
    print("best params ", study.best_params)
    print("best valf1 ", study.best_value)
