from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Tuple
from torch import Tensor

'''
labeling tricks
'''

def batch2pad(batch):
    # batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    # print(batch)
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni>=0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    # pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    batch = torch.arange(pad.shape[0])
    batch = batch.reshape(-1, 1)
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]
    batch = batch.to(pad.device).flatten()
    pos = pad.flatten()
    idx = pos >= 0
    return batch[idx], pos[idx]


@torch.jit.script
def ZeroOneZ(x, Sbool):
    z = torch.zeros((x.shape[0], x.shape[1]),
                    device=x.device,
                    dtype=torch.int64)
    z[Sbool] = 1
    return z


@torch.jit.script
def ZeroOneZs(x, S_len: int, Ss):
    '''
    x (|V|,F)
    S (|V'|)
    '''
    z = torch.zeros((x.shape[0], x.shape[1], S_len),
                    device=x.device, dtype=torch.int64)
    for sidx, s in enumerate(Ss):
        z[s[s >= 0], :, sidx] = 1
    return z


@torch.jit.script
def SumZOZ(x, S_len: int, Ss):
    z = ZeroOneZs(x, S_len, Ss)
    z = torch.sum(z, dim=-1)
    return z


@torch.jit.script
def MaxZOZ(x, Ss):
    #print("Ss= ",Ss)
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
    s = Ss.flatten()
    ss = s[s>=0].to(z.device)
    #print("ss=", ss)
    z[ss] = 1
    return z


def pos2sp(pos, n_node: int):
    coord_2 = torch.arange(
        pos.shape[0]).reshape(-1, 1)[:, torch.zeros(pos.shape[1], dtype=torch.int64)].to(pos.device)
    coord = torch.stack([coord_2.flatten(), pos.flatten()])
    coord = coord[:, coord[1] >= 0]
    vec_pos = torch.sparse_coo_tensor(
        coord, torch.ones(coord.shape[1], device=pos.device), size=(pos.shape[0], n_node), device=pos.device)
    return vec_pos
