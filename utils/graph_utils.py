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


# --- Phần 5: Ví dụ Huấn luyện Đa Đồ thị (Đã Cập nhật) ---

if __name__ == "__main__":
    # (Định nghĩa hàm normalize_and_pad_batch)
    MAX_N_FEATURES = 50
    MAX_E_FEATURES = 50
    MARKER_VALUE = -999.0


    def normalize_and_pad_batch(feats_list, max_dim):
        padded_tensors = []
        for feats in feats_list:
            num_nodes_or_edges, num_feats = feats.shape
            if num_feats > max_dim:
                raise ValueError(f"Đồ thị có {num_feats} đặc trưng, nhiều hơn MAX={max_dim}")
            padded_tensor = torch.full((num_nodes_or_edges, max_dim), MARKER_VALUE, dtype=torch.float32)
            padded_tensor[:, :num_feats] = feats
            padded_tensors.append(padded_tensor)

        full_batch_tensor = torch.cat(padded_tensors, dim=0)
        normalized_batch_tensor = full_batch_tensor.clone()

        for i in range(max_dim):
            col = full_batch_tensor[:, i]
            valid_mask = (col != MARKER_VALUE)
            if valid_mask.any():
                valid_values = col[valid_mask]
                mean = valid_values.mean()
                std = valid_values.std() + 1e-6
                normalized_batch_tensor[valid_mask, i] = (valid_values - mean) / std
                normalized_batch_tensor[~valid_mask, i] = 0.0
            else:
                normalized_batch_tensor[:, i] = 0.0
        return normalized_batch_tensor


    # --- 1. Tạo dữ liệu cho hai đồ thị mạng khác nhau ---
    g1 = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 0]))
    nfeats1 = torch.tensor([[0, 2], [0, 1], [0, 0]], dtype=torch.float32)
    efeats1 = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.8, 0.1], [0.7, 0.2]], dtype=torch.float32)
    g1.ndata['h'] = nfeats1  # Gán để DGL biết
    g1.edata['h'] = efeats1

    g2 = dgl.graph(([0, 0, 1, 2, 3], [1, 2, 3, 1, 0]))
    nfeats2 = torch.tensor([[0, 80], [1, 100], [0, 50], [0, 0]], dtype=torch.float32)
    efeats2 = torch.tensor([[5, 4], [8, 1], [2, 2], [9, 1], [7, 3]], dtype=torch.float32)
    g2.ndata['h'] = nfeats2
    g2.edata['h'] = efeats2

    # --- 2. BƯỚC QUAN TRỌNG: Chuẩn hóa (Normalization) + Padding ---
    print("--- Bước chuẩn hóa (GraphAlign) + Padding ---")
    nfeats_batch = normalize_and_pad_batch([nfeats1, nfeats2], MAX_N_FEATURES)
    efeats_batch = normalize_and_pad_batch([efeats1, efeats2], MAX_E_FEATURES)

    # --- 3. Tạo Batch huấn luyện ---
    g_batch = dgl.batch([g1, g2])
    print(f"\nĐã tạo batch: {g_batch.num_nodes()} nodes, {g_batch.num_edges()} edges")
    print(f"Kích thước NFeats Batch: {nfeats_batch.shape}")
    print(f"Kích thước EFeats Batch: {efeats_batch.shape}")

    # --- 4. Khởi tạo và Huấn luyện Mô hình ---
    NDIM_IN = MAX_N_FEATURES  # 50
    EDIM = MAX_E_FEATURES  # 50
    N_HIDDEN = 16
    N_OUT = 24  # Kích thước embedding cuối cùng
    N_LAYERS = 2
    NUM_EXPERTS = 4
    TOP_K = 1

    # Tạo bộ mã hóa EGraphSAGE với MoF (Đã sửa đổi)
    encoder = EGraphSAGE_GraphAlign(
        NDIM_IN, EDIM, N_HIDDEN, N_OUT, N_LAYERS,
        F.relu, NUM_EXPERTS, TOP_K
    )

    dgi_model = DGI_GraphAlign(encoder)
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.01)

    print("\n--- Bắt đầu huấn luyện DGI + GraphAlign (Có truyền đặc trưng cạnh) ---")
    dgi_model.train()
    optimizer.zero_grad()
    loss = dgi_model(g_batch, nfeats_batch, efeats_batch)
    loss.backward()
    optimizer.step()
    print(f"GraphAlign + DGI Loss (1 batch): {loss.item():.4f}")

    # --- 5. Kiểm tra đầu ra ---
    dgi_model.eval()
    with torch.no_grad():
        final_nodes, final_edges = dgi_model.encoder(g_batch, nfeats_batch, efeats_batch)
        print(f"\nKích thước Node Embedding đầu ra: {final_nodes.shape}")
        print(f"Kích thước Edge Embedding đầu ra: {final_edges.shape}")

    # Kích thước đầu ra của node và cạnh từ tầng cuối cùng sẽ là N_OUT (24)
    assert final_nodes.shape == (g_batch.num_nodes(), N_OUT)
    assert final_edges.shape == (g_batch.num_edges(), N_OUT)
    print("\n[THÀNH CÔNG] Kích thước đầu ra của Node và Cạnh đã khớp.")

