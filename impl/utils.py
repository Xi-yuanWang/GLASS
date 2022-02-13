from torch.nn.utils.rnn import pad_sequence
import torch


def batch2pad(batch):
    '''
    The j-th element in batch vector is i if node j is in the i-th subgraph.
    The i-th row of pad matrix contains the nodes in the i-th subgraph.
    batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    '''
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni >= 0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    '''
    pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    '''
    batch = torch.arange(pad.shape[0])
    batch = batch.reshape(-1, 1)
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]
    batch = batch.to(pad.device).flatten()
    pos = pad.flatten()
    idx = pos >= 0
    return batch[idx], pos[idx]


@torch.jit.script
def MaxZOZ(x, pos):
    '''
    produce max-zero-one label
    x is node feature
    pos is a pad matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph.
    -1 is padding value.
    '''
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
    pos = pos.flatten()
    # pos[pos >= 0] removes -1 from pos
    tpos = pos[pos >= 0].to(z.device)
    z[tpos] = 1
    return z
