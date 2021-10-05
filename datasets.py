import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import torch
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, to_networkx
from torch_geometric.data import Data
import networkx as nx
import os


class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_label,
                 mask):
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.mask = mask
        self.to_undirected()

    def addDegreeFeature(self):
        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_attr,
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        self.x = torch.cat((self.x, one_hot(degree).to(torch.float).reshape(
            self.x.shape[0], 1, -1)),
                           dim=-1)

    def addOneFeature(self):
        self.x = torch.cat(
            (self.x, torch.ones(self.x.shape[0], self.x.shape[1], 1)), dim=-1)

    def addIDFeature(self):
        self.x = torch.cat((self.x, torch.arange(self.x.shape[0]).reshape(
            -1, 1)[:, torch.ones(self.x.shape[1], 1)].reshape(
                self.x.shape[0], self.x.shape[1], 1)),
                           dim=-1)

    def setPosDegreeFeature(self, mod=1):
        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_attr,
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        degree = torch.div(degree, mod, rounding_mode='floor')
        degree = torch.unique(degree, return_inverse=True)[1]
        self.x = degree.reshape(self.x.shape[0], 1, -1)

    def setPosOneFeature(self):
        self.x = torch.ones((self.x.shape[0], 1, 1), dtype=torch.int64)

    def setPosNodeIdFeature(self):
        self.x = torch.arange(self.x.shape[0], dtype=torch.int64).reshape(
            self.x.shape[0], 1, -1)

    def get_split(self, split: str):
        # train,valid,test
        tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        return self.x, self.edge_index, self.edge_attr, self.pos[
            self.mask == tar_mask], self.y[self.mask == tar_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr)

    def get_LPdataset(self, use_loop=False):
        neg_edge = negative_sampling(self.edge_index)
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        pos = torch.cat((self.edge_index, neg_edge), dim=1).t()
        y = torch.cat((torch.ones(ei.shape[1]),
                       torch.zeros(neg_edge.shape[1]))).to(ei.device)
        if use_loop:
            mask = (ei[0] == ei[1])
            pos_loops = ei[0][mask]
            all_loops = torch.arange(x.shape[0],
                                     device=x.device).reshape(-1, 1)[:, [0, 0]]
            y_loop = torch.zeros(x.shape[0], device=y.device)
            y_loop[pos_loops] = 1
            pos = torch.cat((pos, all_loops), dim=0)
            y = torch.cat((y, y_loop), dim=0)
        return x, ei, ea, pos, y

    def get_Lindataset(self, poly: int):
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        pos = torch.arange(self.x.shape[0], device=x.device).reshape(-1, 1)

        from torch_geometric.nn.models import LabelPropagation
        with torch.no_grad():
            mod = LabelPropagation(1, 0.0)
            ys = [x.to(torch.float).reshape(x.shape[0], -1)]
            for i in range(poly - 1):
                ys.append(mod(ys[-1], ei))
            y = torch.cat(ys, dim=-1)
            y = y - torch.mean(y, dim=0, keepdim=True)
            y /= (torch.std(y, dim=0, keepdim=True) + 1e-5)
        return x, ei, ea, pos, y

    def get_Distdataset(self):
        setpos = self.pos.unique()
        dev = setpos.device
        setpos = setpos[setpos >= 0]
        from scipy.sparse.csgraph import dijkstra
        from torch_geometric.utils import to_scipy_sparse_matrix
        spG = to_scipy_sparse_matrix(self.edge_index)
        idx = torch.randint(setpos.shape[0], (500, ))
        selectedSoc = setpos[idx].cpu().numpy()
        out = dijkstra(spG,
                       directed=False,
                       unweighted=True,
                       indices=selectedSoc,
                       limit=5)
        pos1 = torch.arange(out.shape[1], device=dev).reshape(
            1, -1)[torch.zeros(out.shape[0], dtype=torch.int64, device=dev)]
        pos0 = torch.from_numpy(selectedSoc).to(dev).reshape(
            -1, 1)[:,
                   torch.zeros(out.shape[1], dtype=torch.int64, device=dev)]
        pos = torch.stack((pos1.flatten(), pos0.flatten())).t()
        y = torch.from_numpy(out.flatten()).to(torch.int64).to(dev)

        pos = pos[y > 0]
        y = y[y > 0]
        y -= 1
        sety = torch.unique(y)
        idxs = [torch.arange(y.shape[0])[y == i] for i in sety]
        minlen = min([len(i) for i in idxs])
        idxs = [i[torch.randint(i.shape[0], (minlen, ))] for i in idxs]
        idx = torch.cat(idxs)

        return self.x, self.edge_index, self.edge_attr, pos[idx], y[idx]

    def get_SubGdataset(self):
        pos = self.pos.cpu().numpy()
        y = []
        G = to_networkx(self, to_undirected=True)
        for i in pos:
            out_edge = nx.cut_size(G, i[i >= 0])
            subg = nx.subgraph(G, i[i >= 0])
            in_edge = subg.number_of_edges()
            num_node = len(subg)
            cut_ratio = in_edge / (in_edge + out_edge + 1)
            density = in_edge / (num_node * (num_node - 1) / 2 + 1)
            component = nx.components.number_connected_components(subg)
            self_loop = nx.number_of_selfloops(subg)
            self_loop_ratio = self_loop / num_node
            y.append([cut_ratio, density, component, self_loop_ratio])
        y = torch.FloatTensor(y).to(self.x.device)
        y = y - torch.mean(y, dim=0, keepdim=True)
        y /= (torch.std(y, dim=0, keepdim=True) + 1e-5)
        pos = self.pos
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        return x, ei, ea, pos, y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        return self


def load_dataset(name: str):
    if name in [
            "coreness", "cut_ratio", "density", "component"
    ]:  #"./dataset_/{name}/tmp.npy""./biggerSynSubG/SynG_{name}_0.npy"
        obj = np.load(f"./dataset_/{name}/tmp.npy", allow_pickle=True).item()
        edge = np.array([[i[0] for i in obj['G'].edges],
                         [i[1] for i in obj['G'].edges]])
        degree = obj['G'].degree
        node = [n for n in obj['G'].nodes]
        subG = obj["subG"]
        subG_pad = pad_sequence([torch.tensor(i) for i in subG],
                                batch_first=True,
                                padding_value=-1)
        subGLabel = torch.tensor([ord(i) - ord('A') for i in obj["subGLabel"]])
        #mask = torch.tensor(obj['mask'])
        cnt = subG_pad.shape[0]
        mask = torch.cat(
            (torch.zeros(cnt - cnt // 2, dtype=torch.int64),
             torch.ones(cnt // 4, dtype=torch.int64),
             2 * torch.ones(cnt // 2 - cnt // 4, dtype=torch.int64)))
        mask = mask[torch.randperm(mask.shape[0])]
        return BaseGraph(torch.empty(
            (len(node), 1, 0)), torch.from_numpy(edge),
                         torch.ones(edge.shape[1]), subG_pad, subGLabel, mask)
    elif name in ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user"]:
        multilabel = False

        # copy from SubGNN/subgraph_utils.py

        def read_subgraphs(sub_f, split=True):
            label_idx = 0
            labels = {}
            train_sub_G, val_sub_G, test_sub_G = [], [], []
            train_sub_G_label, val_sub_G_label, test_sub_G_label = [], [], []
            train_mask, val_mask, test_mask = [], [], []
            nonlocal multilabel
            # Parse data
            with open(sub_f) as fin:
                subgraph_idx = 0
                for line in fin:
                    nodes = [
                        int(n) for n in line.split("\t")[0].split("-")
                        if n != ""
                    ]
                    if len(nodes) != 0:
                        if len(nodes) == 1:
                            print(nodes)
                        l = line.split("\t")[1].split("-")
                        if len(l) > 1:
                            multilabel = True
                        for lab in l:
                            if lab not in labels.keys():
                                labels[lab] = label_idx
                                label_idx += 1
                        if line.split("\t")[2].strip() == "train":
                            train_sub_G.append(nodes)
                            train_sub_G_label.append(
                                [labels[lab] for lab in l])
                            train_mask.append(subgraph_idx)
                        elif line.split("\t")[2].strip() == "val":
                            val_sub_G.append(nodes)
                            val_sub_G_label.append([labels[lab] for lab in l])
                            val_mask.append(subgraph_idx)
                        elif line.split("\t")[2].strip() == "test":
                            test_sub_G.append(nodes)
                            test_sub_G_label.append([labels[lab] for lab in l])
                            test_mask.append(subgraph_idx)
                        subgraph_idx += 1
            if not multilabel:
                train_sub_G_label = torch.tensor(train_sub_G_label).squeeze()
                val_sub_G_label = torch.tensor(val_sub_G_label).squeeze()
                test_sub_G_label = torch.tensor(test_sub_G_label).squeeze()

            if len(val_mask) < len(test_mask):
                return train_sub_G, train_sub_G_label, test_sub_G, test_sub_G_label, val_sub_G, val_sub_G_label

            return train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label

        if os.path.exists(
                f"./dataset/{name}/train_sub_G.pt") and name != "hpo_neuro":
            train_sub_G = torch.load(f"./dataset/{name}/train_sub_G.pt")
            train_sub_G_label = torch.load(
                f"./dataset/{name}/train_sub_G_label.pt")
            val_sub_G = torch.load(f"./dataset/{name}/val_sub_G.pt")
            val_sub_G_label = torch.load(
                f"./dataset/{name}/val_sub_G_label.pt")
            test_sub_G = torch.load(f"./dataset/{name}/test_sub_G.pt")
            test_sub_G_label = torch.load(
                f"./dataset/{name}/test_sub_G_label.pt")
        else:
            train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label = read_subgraphs(
                f"./dataset/{name}/subgraphs.pth")
            torch.save(train_sub_G, f"./dataset/{name}/train_sub_G.pt")
            torch.save(train_sub_G_label,
                       f"./dataset/{name}/train_sub_G_label.pt")
            torch.save(val_sub_G, f"./dataset/{name}/val_sub_G.pt")
            torch.save(val_sub_G_label, f"./dataset/{name}/val_sub_G_label.pt")
            torch.save(test_sub_G, f"./dataset/{name}/test_sub_G.pt")
            torch.save(test_sub_G_label,
                       f"./dataset/{name}/test_sub_G_label.pt")
        mask = torch.cat(
            (torch.zeros(len(train_sub_G_label), dtype=torch.int64),
             torch.ones(len(val_sub_G_label), dtype=torch.int64),
             2 * torch.ones(len(test_sub_G_label))),
            dim=0)
        if multilabel:
            tlist = train_sub_G_label + val_sub_G_label + test_sub_G_label
            max_label = max([max(i) for i in tlist])
            label = torch.zeros(len(tlist), max_label + 1)
            for idx, ll in enumerate(tlist):
                label[idx][torch.LongTensor(ll)] = 1
        else:
            label = torch.cat(
                (train_sub_G_label, val_sub_G_label, test_sub_G_label))
        pos = pad_sequence(
            [torch.tensor(i) for i in train_sub_G + val_sub_G + test_sub_G],
            batch_first=True,
            padding_value=-1)
        rawedge = nx.read_edgelist(f"./dataset/{name}/edge_list.txt").edges
        edge_index = torch.tensor([[int(i[0]), int(i[1])]
                                   for i in rawedge]).t()
        '''
        x = torch.load(f"./dataset/{name}/gin_embeddings.pth",
                       map_location=torch.device('cpu')).detach()
        x = x.reshape(x.shape[0], 1, x.shape[1])
        '''

        num_node = max([torch.max(pos), torch.max(edge_index)]) + 1
        x = torch.empty((num_node, 1, 0))

        return BaseGraph(x, edge_index, torch.ones(edge_index.shape[1]), pos,
                         label.to(torch.float), mask)
    else:
        raise NotImplementedError()
