from impl import modelx, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, SGD, lr_scheduler, AdamW
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

# X setting
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
parser.add_argument('--noSSL', action='store_true')
# Label settings
parser.add_argument('--use_maxzeroone', action='store_true')

# Train settings
parser.add_argument('--repeat', type=int, default=1)
# Optuna Settings
parser.add_argument('--test', action='store_true')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--seed', type=int, default=0)

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
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, max_z, output_channels = 0, 1, 1
score_fn = None

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
node_ssl_dataset = None
edge_ssl_dataset = None
subg_ssl_dataset = None


def split():
    global trn_dataset, val_dataset, tst_dataset
    global max_deg, max_z, output_channels, loader_fn, tloader_fn
    global node_ssl_dataset, edge_ssl_dataset, subg_ssl_dataset
    if args.use_deg:
        baseG.setPosDegreeFeature()
    elif args.use_one:
        baseG.setPosOneFeature()
    elif args.use_nodeid:
        baseG.setPosNodeIdFeature()
    else:
        raise NotImplementedError
    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    #subg_ssl_dataset = SubGDataset.GDataset(*(baseG.get_SubGdataset()))

    if args.use_maxzeroone:
        max_z = 1

        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:

        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


split()


def buildModel(hidden_dim, conv_layer, dropout, jk, lr, batch_size, pool,
               z_ratio, aggr):
    tmp2 = hidden_dim * (conv_layer) if jk else hidden_dim
    conv_fn = modelx.GLASSConv
    conv = modelx.NZGConv(hidden_dim,
                          hidden_dim,
                          conv_layer,
                          max_deg=max_deg,
                          max_z=max_z,
                          activation=nn.ELU(inplace=True),
                          jk=jk,
                          dropout=dropout,
                          conv=functools.partial(conv_fn,
                                                 aggr=aggr,
                                                 z_ratio=z_ratio,
                                                 dropout=dropout),
                          gn=True)

    if args.use_nodeid and not args.noSSL:
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb,
                                                      freeze=False,
                                                      scale_grad_by_freq=True)

    mlp = modelx.MLP(tmp2,
                     hidden_dim,
                     output_channels,
                     2,
                     dropout=dropout,
                     activation=nn.SELU(inplace=True),
                     bn=False)

    subg_ssl = modelx.MLP(tmp2,
                          hidden_dim,
                          4,
                          2,
                          dropout=dropout,
                          activation=nn.ReLU(inplace=True),
                          bn=False)
    pool_fn_fn = {
        "mean": modelx.MeanPool,
        "max": modelx.MaxPool,
        "sum": modelx.AddPool,
        "size": modelx.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
        pool_fn2 = pool_fn_fn[pool]()
    elif pool == "attention":
        pool_fn1 = modelx.AttentionPool(tmp2, dropout)
        pool_fn2 = modelx.AttentionPool(tmp2, dropout)
    else:
        raise NotImplementedError
    gnn = modelx.GLASS(conv, torch.nn.ModuleList([mlp, subg_ssl]),
                       torch.nn.ModuleList([pool_fn1,
                                            pool_fn2])).to(config.device)
    return gnn


def test(pool="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         alpha=0.0,
         beta=0.0,
         gamma=0.0,
         leak=0.2,
         resi=0.7):
    num_node = torch.sum(trn_dataset.pos > 0) / trn_dataset.pos.shape[0]
    outs = []
    loss_fns = [loss_fn, lambda x, y: nn.MSELoss()(x.flatten(), y.flatten())]
    t1 = time.time()
    num_div = tst_dataset.y.shape[0] / batch_size
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5
    t2 = time.time()
    dist_ssl_dataset = SubGDataset.GDataset(*(baseG.get_Distdataset()))
    dist_ssl_dataloader = SubGDataset.ZGDataloader(dist_ssl_dataset,
                                                   int(batch_size * num_node /
                                                       2),
                                                   z_fn=utils.MaxZOZ,
                                                   drop_last=True,
                                                   shuffle=True)
    print("dist ssl ", time.time() - t2)

    outs = []
    for _ in range(args.repeat):
        set_repeat_seed(_)
        print(f"repeat {_}")
        t1 = time.time()
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, lr, batch_size,
                         pool, z_ratio, aggr)
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)
        optimizer = Adam(gnn.parameters(), lr=lr)

        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)

        val_score = 0
        tst_score = 0
        print("build model ", time.time() - t1)
        early_stop = 0
        for i in range(300):
            t1 = time.time()
            loss = train.train_modelx(optimizer,
                                      gnn, [trn_loader, dist_ssl_dataloader],
                                      loss_fns, [1.0],
                                      ssl_ids=[],
                                      verbose=True)
            print("trn time", time.time() - t1)
            scd.step(loss)
            t1 = time.time()
            if i >= 100 / num_div:
                score = train.test(gnn,
                                   val_loader,
                                   score_fn,
                                   verbose=True,
                                   loss_fn=loss_fn)
                if score > val_score:
                    early_stop = 0
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
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, verbose=True, loss_fn=loss_fn):.4f}",
                            flush=True)
            print("val and tst time", time.time() - t1)
            if early_stop > 100 / num_div:
                break
            if val_score >= 1.00:
                break
        print(f"end: val {val_score:.4f} tst {tst_score:.4f}", flush=True)
        outs.append(tst_score)
    print("tst scores", outs)
    print(np.average(outs), np.std(outs) / np.sqrt(len(outs)))
    return np.average(outs) - np.std(outs)


print(args)

params = {}
if args.dataset == "em_user":
    params = {
        'aggr': 'gcn',
        'batch_size': 6,
        'conv_layer': 1,
        'dropout': 0.15000000000000002,
        'gamma': 0.0,
        'hidden_dim': 56,
        'lr': 0.0011311645595824163,
        'pool': 'size',
        'resi': 0.7,
        'z_ratio': 0.75
    }
elif args.dataset == "ppi_bp":
    params = {
        'aggr': 'mean',
        'batch_size': 80,
        'conv_layer': 2,
        'dropout': 0.35000000000000003,
        'gamma': 0.0,
        'hidden_dim': 32,
        'lr': 0.002535512066861446,
        'pool': 'size',
        'resi': 0.9,
        'z_ratio': 0.6
    }
elif args.dataset == "hpo_metab":
    params = {
        'aggr': 'gcn',
        'batch_size': 29,
        'conv_layer': 1,
        'dropout': 0.30000000000000004,
        'gamma': 0.0,
        'hidden_dim': 48,
        'lr': 0.00326,
        'pool': 'sum',
        'resi': 0.9,
        'z_ratio': 0.8
    }
elif args.dataset == "hpo_neuro":
    params = {
        'aggr': 'gcn',
        'batch_size': 49,
        'conv_layer': 1,
        'dropout': 0.1,
        'gamma': 0.0,
        'hidden_dim': 48,
        'lr': 0.0020168118786811425,
        'pool': 'size',
        'resi': 0.9,
        'z_ratio': 0.55
    }
elif args.dataset == "density":
    #'hidden_dim': 16, 'conv_layer': 1, 'dropout': 0.2, 'aggr': 'sum', 'pool': 'mean',  'z_ratio': 1.0, 'batch_size': 4, 'gamma': 0.0,
    params = {
        'aggr': 'sum',
        'batch_size': 4,
        'conv_layer': 1,
        'dropout': 0.12687905889103224,
        'lr': 0.002559226759914785,
        'gamma': 0.0,
        'hidden_dim': 16,
        'pool': 'mean',
        'z_ratio': 1.00
    }
elif args.dataset == "cut_ratio":
    params = {
        'hidden_dim': 8,
        'conv_layer': 1,
        'dropout': 0.30000000000000004,
        'aggr': 'sum',
        'pool': 'mean',
        'lr': 0.01,
        'z_ratio': 0.75,
        'batch_size': 4,
        'gamma': 0.0,
        'resi': 0.8834483446349967
    }
elif args.dataset == "coreness":
    params = {
        'aggr': 'sum',
        'batch_size': 7,
        'conv_layer': 2,
        'dropout': 0.15000000000000002,
        'gamma': 0.0,
        'hidden_dim': 16,
        'lr': 0.002445039624607346,
        'pool': 'size',
        'resi': 0.9,
        'z_ratio': 0.8
    }

elif args.dataset == "component":
    params = {
        'hidden_dim': 16,
        'conv_layer': 1,
        'dropout': 0.0,
        'aggr': 'gcn',
        'pool': 'sum',
        'z_ratio': 0.75,
        'batch_size': 8,
        'gamma': 0.0
    }

print("best params", params, flush=True)
print(test(**(params)))
