import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.glob.set2set import Set2Set
from torch_geometric.nn.glob.attention import GlobalAttention
from torch_geometric.utils import add_self_loops
from .utils import pos2sp, pad2batch


class MLP(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 bn=False,
                 gn=False,
                 bias=True):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(
                nn.Linear(input_channels, output_channels, bias=bias))
        else:
            self.lins.append(
                nn.Linear(input_channels, hidden_channels, bias=bias))
            for _ in range(num_layers - 2):
                self.lins.append(
                    nn.Linear(hidden_channels, hidden_channels, bias=bias))
            self.lins.append(
                nn.Linear(hidden_channels, output_channels, bias=bias))

        self.bns = None
        self.gns = None
        if bn:
            self.bns = nn.ModuleList()
            for layer in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        if gn:
            self.gns = nn.ModuleList()
            for layer in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))

        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if not (self.bns is None):
            for bn in self.bns:
                bn.reset_parameters()
        if not (self.gns is None):
            for gn in self.bns:
                gn.reset_parameters()

    def forward(self, x):
        if len(self.lins) > 1:
            for layer, lin in enumerate(self.lins[:-1]):
                x = lin(x)
                if not (self.bns is None):
                    x = self.bns[layer](x)
                if not (self.gns is None):
                    x = self.gns[layer](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.lins[-1](x)
        else:
            return F.dropout(self.activation(self.lins[0](x)),
                             p=self.dropout,
                             training=self.training)


class GConv(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super(GConv, self).__init__()
        self.convs = nn.ModuleList()
        self.jk = jk
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for layer in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]


class GLASSConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def buildAdj(self, edge_index, edge_weight, n_node: int):
        adj = torch.sparse_coo_tensor(edge_index,
                                      edge_weight,
                                      size=(n_node, n_node))
        deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
        deg[deg < 0.5] += 1.0
        if self.aggr == "mean":
            deg = 1.0 / deg
            return torch.sparse_coo_tensor(edge_index,
                                           deg[edge_index[0]] * edge_weight,
                                           size=(n_node, n_node))
        elif self.aggr == "sum":
            return torch.sparse_coo_tensor(edge_index,
                                           edge_weight,
                                           size=(n_node, n_node))
        elif self.aggr == "gcn":
            deg = torch.pow(deg, -0.5)
            return torch.sparse_coo_tensor(edge_index,
                                           deg[edge_index[0]] * edge_weight *
                                           deg[edge_index[1]],
                                           size=(n_node, n_node))
        else:
            raise NotImplementedError

    def forward(self, x_, edge_index, edge_weight, mask):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = self.buildAdj(edge_index, edge_weight, n_node)
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        x = self.adj @ x
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class MyGINConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=nn.ReLU(inplace=True),
                 mlp_fn=None,
                 train_eps=False):
        super(MyGINConv, self).__init__()
        if mlp_fn is None:
            def mlp_fn(x, y):
                return MLP(x, y, y, 2, activation=activation)

        def gin_fn(in_channels, out_channels, **kwargs):
            return GINConv(mlp_fn(in_channels, out_channels),
                           train_eps=train_eps)

        self.conv = gin_fn(in_channels, out_channels)

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight):
        return self.conv(x, edge_index)


class NZGConv(nn.Module):
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 max_z=1,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=GLASSConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=True)
        self.emb_gn = GraphNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     **kwargs))
        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for layer in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z):
        # x = self.input_norm(self.input_emb(x))
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        else:
            mask = (z > 0.5).reshape(-1, 1)
        #print("mask",torch.sum(mask))
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, mask)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)
        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x


class GLASS(nn.Module):
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)
    
    def get_params(self, z:int):
        if z==-1:
            return [_ for _ in self.preds.parameters()]+ [_ for _ in self.pools.parameters()]
        return self.conv.get_params(z)



class PoolModule(nn.Module):
    def __init__(self, pool_fn, trans_fn=None):
        super(PoolModule, self).__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class AttentionPool(GlobalAttention):
    def __init__(self, size: int, dropout: float):
        super(AttentionPool, self).__init__(
            MLP(size,
                size,
                1,
                2,
                dropout=dropout,
                activation=torch.tanh,
                bn=False),
            MLP(size,
                size,
                size,
                1,
                dropout=dropout,
                activation=torch.tanh,
                bn=False))


class GNN(torch.nn.Module):
    def __init__(self, conv: GConv, pred: MLP, aggr="sum"):
        super(GNN, self).__init__()
        self.mods = nn.ModuleList()
        self.mods.append(conv)
        self.mods.append(pred)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_weight, subG_node):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.mods[0](x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                               edge_index, edge_weight)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        sp = pos2sp(subG_node, emb.shape[0])
        emb = sp@emb
        if self.aggr == "mean":
            norm = (1/(torch.sparse.sum(sp, dim=1).to_dense())).reshape(-1, 1)
            emb = torch.multiply(norm, emb)
        emb = self.mods[1](emb)
        return emb
