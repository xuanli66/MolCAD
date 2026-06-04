
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3

EDGE_EMB_BINS = 32
SELF_LOOP_BOND_TYPE = 4


def _pool_fn(name: str):
    name = str(name).lower()
    if name == "mean":
        return "mean"
    if name == "add":
        return "add"
    if name == "max":
        return "max"
    if name in ("meanmax", "mean_max", "mean+max"):
        return "meanmax"
    raise ValueError(f"Unknown pool='{name}', choose from ['mean','add','max','meanmax'].")


class EdgeGINEConv(MessagePassing):


    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.emb_dim = int(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, 2 * self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.emb_dim, self.emb_dim),
        )

        self.edge_embedding1 = nn.Embedding(EDGE_EMB_BINS, self.emb_dim)
        self.edge_embedding2 = nn.Embedding(EDGE_EMB_BINS, self.emb_dim)
        self.edge_embedding3 = nn.Embedding(EDGE_EMB_BINS, self.emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    @staticmethod
    def _sanitize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:

        if edge_attr is None:

            return None

        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()

        ea = edge_attr.clone()

        if ea.numel() > 0:
            if int(ea[:, 0].min().item()) >= 1:
                ea[:, 0] = ea[:, 0] - 1

        ea[:, 0] = ea[:, 0].clamp(0, EDGE_EMB_BINS - 1)
        ea[:, 1] = ea[:, 1].clamp(0, EDGE_EMB_BINS - 1)
        ea[:, 2] = ea[:, 2].clamp(0, EDGE_EMB_BINS - 1)
        return ea

    def forward(self, x, edge_index, edge_attr):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        ea = self._sanitize_edge_attr(edge_attr)

        self_loop_attr = torch.zeros((x.size(0), 3), device=x.device, dtype=torch.long)
        self_loop_attr[:, 0] = SELF_LOOP_BOND_TYPE
        self_loop_attr[:, 1] = 0
        self_loop_attr[:, 2] = 0

        if ea is None:
            ea2 = self_loop_attr
        else:
            ea2 = torch.cat([ea.to(x.device), self_loop_attr], dim=0)

        e = (
            self.edge_embedding1(ea2[:, 0])
            + self.edge_embedding2(ea2[:, 1])
            + self.edge_embedding3(ea2[:, 2])
        )

        out = self.propagate(edge_index=edge_index, x=x, edge_attr=e)
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINet(nn.Module):

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 512,
        drop_ratio: float = 0.0,
        pool: str = "mean",
        edge_drop: float = 0.05,
        feat_mask: float = 0.05,
        residual: bool = True,
        jk: str = "last",
    ):
        super().__init__()
        self.num_layer = int(num_layer)
        self.emb_dim = int(emb_dim)
        self.feat_dim = int(feat_dim)
        self.drop_ratio = float(drop_ratio)

        self.pool = _pool_fn(pool)
        self.edge_drop = float(edge_drop)
        self.feat_mask = float(feat_mask)
        self.residual = bool(residual)
        self.jk = str(jk).lower()

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList([EdgeGINEConv(self.emb_dim) for _ in range(self.num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.emb_dim) for _ in range(self.num_layer)])

        in_dim = 2 * self.emb_dim if self.pool == "meanmax" else self.emb_dim
        self.feat_lin = nn.Linear(in_dim, self.feat_dim)

    @staticmethod
    def _drop_edges(edge_index, edge_attr, p: float, training: bool):
        if (not training) or p <= 0:
            return edge_index, edge_attr
        E = edge_index.size(1)
        if E == 0:
            return edge_index, edge_attr
        keep = torch.rand(E, device=edge_index.device) > p

        if keep.sum().item() == 0:
            keep[torch.randint(0, E, (1,), device=edge_index.device)] = True
        edge_index = edge_index[:, keep]
        if edge_attr is not None:
            edge_attr = edge_attr[keep]
        return edge_index, edge_attr

    @staticmethod
    def _feature_mask(h: torch.Tensor, p: float, training: bool):
        if (not training) or p <= 0:
            return h

        mask = (torch.rand_like(h) > p).float()
        return h * mask

    def _pool_graph(self, node_h, batch):
        if self.pool == "mean":
            return global_mean_pool(node_h, batch)
        if self.pool == "add":
            return global_add_pool(node_h, batch)
        if self.pool == "max":
            return global_max_pool(node_h, batch)
        if self.pool == "meanmax":
            return torch.cat([global_mean_pool(node_h, batch), global_max_pool(node_h, batch)], dim=1)
        raise RuntimeError("unreachable")

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = data.batch

        if x.dim() == 1:
            x = x.view(-1, 1)
        if x.size(1) == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)

        h = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long())
        h = self._feature_mask(h, self.feat_mask, self.training)
        layer_graph_embs = []

        for layer in range(self.num_layer):
            ei, ea = self._drop_edges(edge_index, edge_attr, self.edge_drop, self.training)

            h_in = h
            h = self.gnns[layer](h, ei, ea)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = F.relu(h, inplace=True)

            h = F.dropout(h, p=self.drop_ratio, training=self.training)

            if self.residual and h_in.shape == h.shape:
                h = h + h_in

            pooled = self._pool_graph(h, batch)
            layer_graph_embs.append(pooled)

        if self.jk == "last":
            g = layer_graph_embs[-1]
        elif self.jk == "sum":
            g = torch.stack(layer_graph_embs, dim=0).sum(dim=0)
        elif self.jk == "cat":
            g = torch.cat(layer_graph_embs, dim=1)
        else:
            raise ValueError("jk must be one of ['last','sum','cat']")

        if self.jk == "cat":
            self_proj = nn.Linear(g.size(1), self.feat_dim).to(g.device)
            g = self_proj(g)
        else:
            g = self.feat_lin(g)

        return g
