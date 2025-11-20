import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from utils import *
import os





class EGraphSAGELayer(nn.Module):
    def __init__(self, ndim_in, edim_in, ndim_out, edim_out, activation):
        super(EGraphSAGELayer, self).__init__()
        self.activation = activation
        self.node_out_dim = ndim_out
        self.edge_out_dim = edim_out

        # Chỉ sử dụng các lớp Linear tiêu chuẩn
        self.W_apply = nn.Linear(ndim_in + ndim_in + edim_in, ndim_out)
        self.W_edge = nn.Linear(ndim_out * 2, edim_out)

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
class EGraphSAGE(nn.Module):
    def __init__(self, ndim_in, edim, n_hidden, n_out, n_layers, activation, device):
        super(EGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # Đã loại bỏ tất cả logic và tham số liên quan đến MoE

        if n_layers == 1:
            self.layers.append(EGraphSAGELayer(ndim_in, edim, n_out, n_out, activation))
        else:
            # Lớp đầu tiên
            self.layers.append(EGraphSAGELayer(ndim_in, edim, n_hidden, n_hidden, activation))
            # Các lớp ẩn
            for l in range(1, n_layers - 1):
                self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_hidden, n_hidden, activation))
            # Lớp cuối cùng
            self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_out, n_out, activation))

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
class DGI(nn.Module):
    def __init__(self, encoder):
        super(DGI, self).__init__()
        self.encoder = encoder

        # Logic này vẫn hoạt động vì EGraphSAGELayer vẫn lưu self.edge_out_dim
        last_layer_out_dim = getattr(encoder.layers[-1], "edge_out_dim", None)

        if last_layer_out_dim is None:
            # Fallback (phòng trường hợp) - cũng sẽ hoạt động vì W_edge là nn.Linear
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
        # print("--",summary.shape, "--")

        positive_scores = self.discriminator(positive_edges, summary)
        negative_scores = self.discriminator(negative_edges, summary)
        l1 = self.loss(positive_scores, torch.ones_like(positive_scores))
        l2 = self.loss(negative_scores, torch.zeros_like(negative_scores))
        # print(l1.shape)
        # print(l2.shape)

        return l1 + l2


# ----------------- Export -----------------
import os
import torch
import yaml
import networkx as nx
from networkx.readwrite import json_graph

def save_graph_env(experiment_id, G_nx, nfeats, efeats, node_order, base_dir='graphs'):
    """
    Lưu môi trường đồ thị và các thông tin liên quan vào file .pth để Agent sử dụng.

    Args:
        experiment_id (int/str): ID của thí nghiệm (tên thư mục con).
        G_nx (networkx.Graph): Đồ thị NetworkX gốc.
        nfeats (Tensor): Đặc trưng node (gốc).
        efeats (Tensor): Đặc trưng cạnh (gốc).
        node_order (list): Danh sách tên các node theo thứ tự index.
        base_dir (str): Thư mục cha chứa các graph (mặc định là 'graphs').
    """
    # 1. Tạo đường dẫn thư mục
    save_path = os.path.join(base_dir, str(experiment_id))
    os.makedirs(save_path, exist_ok=True)
    print(f"Đang xử lý lưu dữ liệu vào thư mục: {save_path}")

    # 2. Tạo ánh xạ Tên -> Index
    node_map = {name: i for i, name in enumerate(node_order)}

    # 3. Đóng gói dữ liệu
    # Lưu ý: Key 'g1' được giữ nguyên để tương thích với code load của Agent cũ
    env_data = {
        "G": G_nx,
        "nfeats": nfeats,
        "efeats": efeats,
        "node_order": node_order,
        "node_map": node_map,
    }

    # 4. Lưu file môi trường
    env_file_path = os.path.join(save_path, "graph_environment.pth")
    torch.save(env_data, env_file_path)

    print(f" >> Đã lưu Môi trường Tĩnh (env_data) vào: {env_file_path}")

    return save_path