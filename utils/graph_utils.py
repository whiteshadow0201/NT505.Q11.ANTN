import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from utils import *

# --- Dùng FMoE (high-level) ---
from fmoe import FMoE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- Lớp EGraphSAGELayer -----------------
class EGraphSAGELayer(nn.Module):
    def __init__(self, ndim_in, edim_in, ndim_out, edim_out, activation,
                 moe=False,
                 num_experts=4,
                 top_k=1,
                 hhsize_time=None,
                 moe_use_linear=False):
        super(EGraphSAGELayer, self).__init__()
        self.activation = activation
        self.hhsize_time = hhsize_time
        self.moe_use_linear = moe_use_linear
        self.node_out_dim = ndim_out
        self.edge_out_dim = edim_out

        if not moe:
            self.W_apply = nn.Linear(ndim_in + ndim_in + edim_in, ndim_out)
            self.W_edge = nn.Linear(ndim_out * 2, edim_out)
        else:
            print("Running FMOE")
            in_apply = ndim_in + ndim_in + edim_in
            in_edge = ndim_out * 2

            def make_apply_expert(d_model):
                return nn.Linear(d_model, ndim_out)

            def make_edge_expert(d_model):
                return nn.Linear(d_model, edim_out)

            self.W_apply = FMoE(
                num_expert=num_experts,
                d_model=in_apply,
                expert=make_apply_expert,
                top_k=top_k,
                world_size=1,
                moe_group=None,
            )

            self.W_edge = FMoE(
                num_expert=num_experts,
                d_model=in_edge,
                expert=make_edge_expert,
                top_k=top_k,
                world_size=1,
                moe_group=None,
            )

    def message_func(self, edges):
        return {'m': torch.cat([edges.src['h'], edges.data['h']], dim=1)}

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            x_apply = torch.cat([nfeats, g.ndata['h_neigh']], dim=1)
            h_nodes_new = self.W_apply(x_apply)
            if self.activation:
                h_nodes_new = self.activation(h_nodes_new)

            g.ndata['h_new'] = h_nodes_new
            u, v = g.edges()
            edge_input = torch.cat([g.ndata['h_new'][u], g.ndata['h_new'][v]], dim=1)

            h_edges_new = self.W_edge(edge_input)
            if self.activation:
                h_edges_new = self.activation(h_edges_new)

            return h_nodes_new, h_edges_new

# ----------------- Encoder -----------------
class EGraphSAGE_GraphAlign(nn.Module):
    def __init__(self, ndim_in, edim, n_hidden, n_out, n_layers, activation,
                 num_experts=4, top_k=1, hhsize_time=1, moe_use_linear=False, moe_layer=None):
        super(EGraphSAGE_GraphAlign, self).__init__()
        self.layers = nn.ModuleList()
        if moe_layer is None:
            moe_layer = list(range(n_layers))

        moe_params_0 = {
            "moe": (0 in moe_layer),
            "num_experts": num_experts,
            "top_k": top_k,
            "hhsize_time": hhsize_time,
            "moe_use_linear": moe_use_linear
        }

        if n_layers == 1:
            self.layers.append(EGraphSAGELayer(ndim_in, edim, n_out, n_out, activation, **moe_params_0))
        else:
            self.layers.append(EGraphSAGELayer(ndim_in, edim, n_hidden, n_hidden, activation, **moe_params_0))
            for l in range(1, n_layers - 1):
                moe_params_l = {
                    "moe": (l in moe_layer),
                    "num_experts": num_experts,
                    "top_k": top_k,
                    "hhsize_time": hhsize_time,
                    "moe_use_linear": moe_use_linear
                }
                self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_hidden, n_hidden, activation, **moe_params_l))
            moe_params_last = {
                "moe": (n_layers - 1 in moe_layer),
                "num_experts": num_experts,
                "top_k": top_k,
                "hhsize_time": hhsize_time,
                "moe_use_linear": moe_use_linear
            }
            self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_out, n_out, activation, **moe_params_last))

    def forward(self, g, nfeats, efeats, corrupt=False):
        if corrupt:
            perm = torch.randperm(efeats.shape[0], device=efeats.device)
            efeats_to_use = efeats[perm]
        else:
            efeats_to_use = efeats

        h_nodes = nfeats
        h_edges = efeats_to_use

        for layer in self.layers:
            h_nodes, h_edges = layer(g, h_nodes, h_edges)

        return h_nodes, h_edges

# ----------------- Discriminator -----------------
class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(n_hidden, n_hidden, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features, summary):
        summary_expanded = summary.expand_as(features)
        scores = self.bilinear(features, summary_expanded)
        return scores

# ----------------- DGI Model -----------------
class DGI_GraphAlign(nn.Module):
    def __init__(self, encoder):
        super(DGI_GraphAlign, self).__init__()
        self.encoder = encoder
        last_layer_out_dim = getattr(encoder.layers[-1], "edge_out_dim", None)
        if last_layer_out_dim is None:
            last_layer = encoder.layers[-1].W_edge
            last_layer_out_dim = getattr(last_layer, "out_features", None)
        if last_layer_out_dim is None:
            raise RuntimeError("Không xác định được kích thước đầu ra của W_edge ở tầng cuối cùng.")
        self.discriminator = Discriminator(last_layer_out_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, nfeats, efeats):
        _, positive_edges = self.encoder(g, nfeats, efeats, corrupt=False)
        _, negative_edges = self.encoder(g, nfeats, efeats, corrupt=True)
        summary = torch.sigmoid(positive_edges.mean(dim=0))
        positive_scores = self.discriminator(positive_edges, summary)
        negative_scores = self.discriminator(negative_edges, summary)
        l1 = self.loss(positive_scores, torch.ones_like(positive_scores))
        l2 = self.loss(negative_scores, torch.zeros_like(negative_scores))
        return l1 + l2
