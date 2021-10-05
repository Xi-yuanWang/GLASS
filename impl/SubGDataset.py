from torch_geometric.data import Dataset, Data, InMemoryDataset
import torch
from torch_geometric.utils import k_hop_subgraph, is_undirected, to_undirected, negative_sampling
from torch.nn.functional import one_hot
from .utils import batch2pad
from torch_geometric.data import DataLoader as pygDataloader
from torch.utils.data import DataLoader
from . import config
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from scipy.sparse.csgraph import dijkstra
import networkx as nx
import numpy as np


class GDataset(Data):
    '''
    x : node feature
    pos : the node set of target subgraphs
    subG_label : the target
    '''
    def __init__(self, x, edge_index, edge_attr, pos, subG_label):
        super(GDataset, self).__init__(x=x,
                                       edge_index=edge_index,
                                       edge_attr=edge_attr,
                                       y=subG_label,
                                       pos=pos)
        self.num_nodes = x.shape[0]
        # self.to_undirected()

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        return self


class GsDataset(InMemoryDataset):
    '''
    designed for GNN-seg.
    '''
    def __init__(self, datalist):
        self.datalist = datalist
        super(GsDataset, self).__init__()
        self.data, self.slices = self.collate(self.datalist)

    def __len__(self):
        return len(self.datalist)

    def to(self, device):
        self.data = self.data.to(device)
        for i in self.slices:
            self.slices[i] = self.slices[i].to(device)
        return self


class GDataloader(DataLoader):
    '''
    Dataloader for GDataset
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False):
        super(GDataloader,
              self).__init__(torch.arange(len(Gdataset)).to(Gdataset.x.device),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
        self.Gdataset = Gdataset

    def get_x(self):
        return self.Gdataset.x

    def get_ei(self):
        return self.Gdataset.edge_index

    def get_ea(self):
        return self.Gdataset.edge_attr

    def get_pos(self):
        return self.Gdataset.pos

    def get_y(self):
        return self.Gdataset.y

    def __iter__(self):
        self.iter = super(GDataloader, self).__iter__()
        return self

    def __next__(self):
        perm = next(self.iter)
        return self.get_x(), self.get_ei(), self.get_ea(
        ), self.get_pos()[perm], self.get_y()[perm]


class ZGDataloader(GDataloader):
    '''
    Dataloader for GDataset.
    z_fn assigns node label.
    '''
    def __init__(self,
                 Gdataset,
                 batch_size=64,
                 shuffle=True,
                 drop_last=False,
                 z_fn=lambda x, y: torch.zeros(
                     (x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(ZGDataloader, self).__init__(Gdataset, batch_size, shuffle,
                                           drop_last)
        self.z_fn = z_fn

    def __next__(self):
        perm = next(self.iter)
        tpos = self.get_pos()[perm]
        return self.get_x(), self.get_ei(), self.get_ea(), tpos, self.z_fn(
            self.get_x(), tpos), self.get_y()[perm]


class GsDataloader(pygDataloader):
    '''
    dataloader for GsDataset
    '''
    def __init__(self,
                 Gsdataset,
                 batch_size=64,
                 shuffle=True,
                 full_subG_pool=False):
        super(GsDataloader, self).__init__(Gsdataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)

    def __iter__(self):
        self.iter = super(GsDataloader, self).__iter__()
        return self

    def __next__(self):
        batch = next(self.iter)
        x = batch.x
        ei = batch.edge_index
        ea = batch.edge_attr
        pos = batch2pad(batch.batch)
        y = batch.y
        return x, ei, ea, pos, y


class ZGsDataloader(GsDataloader):
    def __init__(self,
                 Gsdataset,
                 batch_size=64,
                 shuffle=True,
                 z_fn=lambda x, y: torch.zeros(
                     (x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(ZGsDataloader, self).__init__(Gsdataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle)
        self.z_fn = z_fn

    def __iter__(self):
        self.iter = super(GsDataloader, self).__iter__()
        return self

    def __next__(self):
        batch = next(self.iter)
        x = batch.x
        ei = batch.edge_index
        ea = batch.edge_attr
        pos = batch2pad(batch.batch)
        y = batch.y
        return x, ei, ea, pos, self.z_fn(x, batch.pos), y


'''
function to help segregating subgraphs from the whole graph
'''


def todata(x, edge_index, edge_attr, centre, hop, y):
    node, edge, inv, edge_mask = k_hop_subgraph(centre,
                                                hop,
                                                edge_index,
                                                relabel_nodes=True)
    if node.shape[0] == 0:
        print("empty", centre)
    npos = torch.zeros_like(node, device=node.device)
    npos[inv] = 1
    if not torch.any(npos):
        print("empty", centre)
    return Data(x[node], edge, edge_attr[edge_mask], y=y, pos=npos)


def todatalist(gd: GDataset, hop):
    return [
        todata(gd.x, gd.edge_index, gd.edge_attr, gd.pos[i][gd.pos[i] >= 0],
               hop, gd.y[i]) for i in range(len(gd))
    ]
