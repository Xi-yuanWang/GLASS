from torch_geometric.data import DataLoader as pygDataloader
from torch.utils.data import DataLoader
import torch
from . import config
from .utils import batch2pad
import time
import numpy as np


def train(optimizer, model, dl, loss_fn, accumulate=1):
    model.train()
    total_loss = 0
    num_sample = 0
    num_batch = 0
    for batch in dl:
        if num_batch == 0:
            optimizer.zero_grad()
        pred = model(*batch[:-1])
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        num_batch+=1
        if num_batch == accumulate:
            num_batch = 0
            optimizer.step()
        num_sample += batch[-1].shape[0]
        total_loss += loss.item() * batch[-1].shape[0]
    return total_loss / num_sample


def train_modelx(optimizer, modelx, dls, loss_fns, coefs, ssl_ids=[1, 2], verbose=False, max_iter=100000):
    modelx.train()
    total_loss = []
    ssl_loss = {i: [] for i in ssl_ids}
    ssl_ids = [i for i in ssl_ids if coefs[i] > 0.01]

    def op():
        nonlocal total_loss, ssl_loss, modelx, optimizer, dls, loss_fns, coefs, ssl_ids
        for _, batch in enumerate(dls[0]):
            if _ >=max_iter:
                break
            optimizer.zero_grad()
            pred = modelx(*batch[:-1], id=0)
            loss = loss_fns[0](pred, batch[-1])
            loss.backward()
            total_loss.append(loss.detach().item())
            for i in ssl_ids:
                try:
                    batch = next(dls[i])
                except:
                    batch = next(iter(dls[i]))
                pred = modelx(*batch[:-1], id=i)
                tloss = loss_fns[i](pred, batch[-1])
                loss = coefs[i]*tloss
                loss.backward()
                ssl_loss[i].append(tloss.detach().item())
            optimizer.step()

    if verbose:
        op()
    else:
        op()
    return sum(total_loss)/len(total_loss)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate




@torch.no_grad()
def test(model, dl, metrics, verbose=False, loss_fn=None):
    model.eval()
    preds = []
    ys = []
    for batch in dl:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    if verbose:
        print(f"val and tst loss {loss_fn(pred, y):.4f}")
    return metrics(pred.cpu().numpy(), y.cpu().numpy())
