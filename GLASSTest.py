from impl import models, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np
import time
import random
import yaml

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
# Node feature settings. 
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn
    baseG = datasets.load_dataset(args.dataset)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)
    # initialize node features
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
    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    # choice of dataloader
    if args.use_maxzeroone:

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


def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn. 
    '''
    conv = models.EmbZGConv(hidden_dim,
                            hidden_dim,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(models.GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout),
                            gn=True)

    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim,
                    output_channels)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError

    gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                       torch.nn.ModuleList([pool_fn1])).to(config.device)
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
         resi=0.7):
    '''
    Test a set of hyperparameters in a task.
    Args:
        jk: whether to use Jumping Knowledge Network.
        z_ratio: see GLASSConv in impl/model.py. A hyperparameter of GLASS.
        resi: the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        split()
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio,
                         aggr)
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []
        for i in range(300):
            t1 = time.time()
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 100 / num_div:
                score, _ = train.test(gnn,
                                      val_loader,
                                      score_fn,
                                      loss_fn=loss_fn)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                elif score >= val_score - 1e-5:
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)[0]:.4f}",
                            flush=True)
            if val_score >= 1 - 1e-5:
                early_stop += 1
            if early_stop > 100 / num_div:
                break
        print(
            f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score)
    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )


print(args)
# read configuration
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
split()
test(**(params))
