import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np


# --- Phần 1: Triển khai Mixture-of-feature-expert (MoF) ---
# (Không thay đổi)
class MoFLayer(nn.Module):
    """
    Lớp Mixture-of-Feature-Expert (MoF) đơn giản.
    """

    def __init__(self, in_dim, out_dim, num_experts, top_k=1):
        super(MoFLayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_experts)
        ])
        self.gating = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gating_scores = self.gating(x)
        top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=-1)
        mask = torch.zeros_like(gating_scores).scatter_(-1, top_k_indices, 1)
        gating_weights = F.softmax(
            gating_scores.masked_fill(mask == 0, -float('inf')),
            dim=-1
        )
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )
        final_output = torch.sum(
            gating_weights.unsqueeze(-1) * expert_outputs,
            dim=1
        )
        return final_output


# --- Phần 2: EGraphSAGE (Đã sửa đổi) ---

class EGraphSAGELayer(nn.Module):
    """
    [SỬA ĐỔI] Tầng E-GraphSAGE hiện chấp nhận edim_in và edim_out
    để cho phép truyền (propagation) đặc trưng cạnh.
    """

    def __init__(self, ndim_in, edim_in, ndim_out, edim_out, activation):
        super(EGraphSAGELayer, self).__init__()
        self.activation = activation

        # [SỬA ĐỔI] W_apply hiện chấp nhận edim_in
        # Kích thước đầu vào = (đặc trưng nút) + (đặc trưng nút nguồn + đặc trưng cạnh VÀO)
        self.W_apply = nn.Linear(ndim_in + ndim_in + edim_in, ndim_out)

        # [SỬA ĐỔI] W_edge hiện tạo ra edim_out
        self.W_edge = nn.Linear(ndim_out * 2, edim_out)

    def message_func(self, edges):
        # (Không thay đổi)
        return {'m': torch.cat([edges.src['h'], edges.data['h']], dim=1)}

    def forward(self, g, nfeats, efeats):
        # (Không thay đổi)
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats  # efeats giờ là đặc trưng từ tầng trước

            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            h_nodes_new = self.W_apply(torch.cat([nfeats, g.ndata['h_neigh']], dim=1))
            if self.activation:
                h_nodes_new = self.activation(h_nodes_new)

            g.ndata['h_new'] = h_nodes_new
            u, v = g.edges()
            edge_input = torch.cat([g.ndata['h_new'][u], g.ndata['h_new'][v]], dim=1)
            h_edges_new = self.W_edge(edge_input)
            if self.activation:
                h_edges_new = self.activation(h_edges_new)

            return h_nodes_new, h_edges_new


# --- Phần 3: Tích hợp GraphAlign vào EGraphSAGE (Đã sửa đổi) ---

class EGraphSAGE_GraphAlign(nn.Module):
    """
    [SỬA ĐỔI] Mô hình E-GraphSAGE hiện khởi tạo các tầng EGraphSAGELayer
    với đúng kích thước edim_in/edim_out và truyền đặc trưng cạnh trong forward.
    """

    def __init__(self, ndim_in, edim, n_hidden, n_out, n_layers, activation,
                 num_experts=4, top_k=1):
        super(EGraphSAGE_GraphAlign, self).__init__()

        # --- TÍCH HỢP GRAPHALIGN (Không thay đổi) ---
        self.mof_nodes = MoFLayer(ndim_in, ndim_in, num_experts, top_k)
        self.mof_edges = MoFLayer(edim, edim, num_experts, top_k)
        # --- KẾT THÚC TÍCH HỢP ---

        self.layers = nn.ModuleList()

        # --- [SỬA ĐỔI] Khởi tạo tầng ---
        # Tầng đầu tiên: (N_in, E_in) -> (N_hidden, E_hidden)
        # E_hidden phải = N_hidden để W_edge hoạt động
        self.layers.append(EGraphSAGELayer(ndim_in, edim, n_hidden, n_hidden, activation))

        # Các tầng ẩn: (N_hidden, E_hidden) -> (N_hidden, E_hidden)
        for _ in range(n_layers - 2):
            self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_hidden, n_hidden, activation))

        # Tầng cuối cùng: (N_hidden, E_hidden) -> (N_out, E_out)
        # E_out phải = N_out
        if n_layers > 1:
            self.layers.append(EGraphSAGELayer(n_hidden, n_hidden, n_out, n_out, activation))
        else:
            # Ghi đè tầng đầu tiên nếu n_layers = 1
            self.layers[0] = EGraphSAGELayer(ndim_in, edim, n_out, n_out, activation)

    def forward(self, g, nfeats, efeats, corrupt=False):
        """
        [SỬA ĐỔI] Hàm forward hiện truyền cả h_nodes và h_edges qua các tầng.
        """

        nfeats_aligned = self.mof_nodes(nfeats)
        efeats_aligned = self.mof_edges(efeats)

        if corrupt:
            perm = torch.randperm(efeats_aligned.shape[0])
            efeats_to_use = efeats_aligned[perm]
        else:
            efeats_to_use = efeats_aligned

        h_nodes = nfeats_aligned
        h_edges = efeats_to_use  # [SỬA ĐỔI] Đây là biến sẽ được cập nhật

        # [SỬA ĐỔI] Lặp qua từng tầng của mô hình
        for i, layer in enumerate(self.layers):
            # Cả h_nodes và h_edges đều được cập nhật và truyền đi
            h_nodes, h_edges = layer(g, h_nodes, h_edges)

        # Trả về embedding node và embedding cạnh từ tầng cuối cùng
        return h_nodes, h_edges


# --- Phần 4: DGI (Sử dụng Encoder mới) ---

class Discriminator(nn.Module):
    # (Không thay đổi)
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


class DGI_GraphAlign(nn.Module):
    """
    [SỬA ĐỔI] DGI hiện lấy kích thước đầu ra từ EGraphSAGELayer cuối cùng
    một cách chính xác (vì edim_out = ndim_out = n_out).
    """

    def __init__(self, encoder):
        super(DGI_GraphAlign, self).__init__()
        self.encoder = encoder

        # [SỬA ĐỔI] Lấy kích thước đầu ra từ tầng cuối cùng
        # (Giờ đây ndim_out và edim_out là giống nhau)
        last_layer_out_dim = encoder.layers[-1].W_apply.out_features
        self.discriminator = Discriminator(last_layer_out_dim)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, nfeats, efeats):
        # (Không thay đổi)
        _, positive_edges = self.encoder(g, nfeats, efeats, corrupt=False)
        _, negative_edges = self.encoder(g, nfeats, efeats, corrupt=True)
        summary = torch.sigmoid(positive_edges.mean(dim=0))
        positive_scores = self.discriminator(positive_edges, summary)
        negative_scores = self.discriminator(negative_edges, summary)
        l1 = self.loss(positive_scores, torch.ones_like(positive_scores))
        l2 = self.loss(negative_scores, torch.zeros_like(negative_scores))

        return l1 + l2


