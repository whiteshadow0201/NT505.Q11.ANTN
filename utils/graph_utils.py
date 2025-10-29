import math
import random
import numpy as np
from queue import PriorityQueue
from copy import deepcopy
import torch
import torch.nn as nn
from collections import deque, defaultdict
import dgl
import dgl.function as fn


class EGraphSAGELayer(nn.Module):
    """
    Một tầng của mô hình E-GraphSAGE (Edge-Aware GraphSAGE).
    Tầng này cập nhật embedding của các nút dựa trên thông tin từ các nút lân cận và các cạnh nối tới chúng,
    sau đó tính toán embedding cho các cạnh.
    """
    def __init__(self, ndim_in, edim, ndim_out, activation):
        """
        Khởi tạo tầng.

        Args:
            ndim_in (int): Kích thước vector đặc trưng đầu vào của nút.
            edim (int): Kích thước vector đặc trưng của cạnh.
            ndim_out (int): Kích thước embedding đầu ra cho nút.
            activation (function): Hàm kích hoạt (ví dụ: F.relu).
        """
        super(EGraphSAGELayer, self).__init__()
        self.activation = activation

        # Lớp tuyến tính để cập nhật embedding của nút.
        # Đầu vào là sự kết hợp của đặc trưng gốc của nút và thông điệp tổng hợp từ các nút lân cận.
        # Kích thước đầu vào = (đặc trưng nút) + (đặc trưng nút nguồn + đặc trưng cạnh)
        self.W_apply = nn.Linear(ndim_in + ndim_in + edim, ndim_out)

        # Lớp tuyến tính để tạo embedding cho cạnh.
        # Đầu vào là sự kết hợp của embedding (đã được cập nhật) của nút nguồn và nút đích.
        self.W_edge = nn.Linear(ndim_out * 2, ndim_out)

    def message_func(self, edges):
        """
        Hàm tạo thông điệp (message).
        Thông điệp được gửi từ nút nguồn tới nút đích, chứa thông tin của cả nút nguồn và cạnh.
        """
        # Kết hợp đặc trưng của nút nguồn (edges.src['h']) và đặc trưng của cạnh (edges.data['h'])
        return {'m': torch.cat([edges.src['h'], edges.data['h']], dim=1)}

    def forward(self, g, nfeats, efeats):
        """
        Hàm lan truyền tiến (forward pass).

        Args:
            g (DGLGraph): Đồ thị.
            nfeats (Tensor): Tensor chứa đặc trưng của các nút.
            efeats (Tensor): Tensor chứa đặc trưng của các cạnh.

        Returns:
            tuple[Tensor, Tensor]: Một cặp tensor chứa graphs mới cho các nút và các cạnh.
        """
        # Gán đặc trưng vào đồ thị để tính toán
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            # Bước 1: Truyền và tổng hợp thông điệp
            # - message_func: tạo thông điệp từ mỗi nút nguồn.
            # - fn.mean('m', 'h_neigh'): lấy trung bình tất cả thông điệp 'm' nhận được và lưu vào 'h_neigh'.
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            # Bước 2: Cập nhật embedding cho các nút
            # Kết hợp đặc trưng gốc của nút (nfeats) và thông tin từ hàng xóm (g.ndata['h_neigh'])
            h_nodes_new = self.W_apply(torch.cat([nfeats, g.ndata['h_neigh']], dim=1))
            if self.activation:
                h_nodes_new = self.activation(h_nodes_new)

            # Bước 3: Tính toán embedding cho các cạnh
            # Gán embedding mới của nút vào đồ thị để lấy thông tin nút nguồn/đích
            g.ndata['h_new'] = h_nodes_new
            u, v = g.edges()

            # Kết hợp embedding mới của nút nguồn và nút đích
            edge_input = torch.cat([g.ndata['h_new'][u], g.ndata['h_new'][v]], dim=1)
            h_edges_new = self.W_edge(edge_input)
            if self.activation:
                h_edges_new = self.activation(h_edges_new)

            return h_nodes_new, h_edges_new

class EGraphSAGE(nn.Module):
    """
    Mô hình E-GraphSAGE hoàn chỉnh, bao gồm nhiều tầng EGraphSAGELayer.
    """
    def __init__(self, ndim_in, edim, n_hidden, n_out, n_layers, activation):
        """
        Khởi tạo mô hình.

        Args:
            ndim_in (int): Kích thước đặc trưng đầu vào của nút.
            edim (int): Kích thước đặc trưng của cạnh.
            n_hidden (int): Kích thước của các tầng ẩn.
            n_out (int): Kích thước embedding đầu ra cuối cùng.
            n_layers (int): Số lượng tầng EGraphSAGELayer.
            activation (function): Hàm kích hoạt.
        """
        super(EGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # Tầng đầu tiên: chuyển từ kích thước đầu vào sang kích thước ẩn
        self.layers.append(EGraphSAGELayer(ndim_in, edim, n_hidden, activation))

        # Các tầng ẩn tiếp theo
        for _ in range(n_layers - 2):
            self.layers.append(EGraphSAGELayer(n_hidden, edim, n_hidden, activation))

        # Tầng cuối cùng: chuyển từ kích thước ẩn sang kích thước đầu ra
        # Nếu chỉ có 1 tầng, tầng này cũng là tầng đầu tiên
        if n_layers > 1:
            self.layers.append(EGraphSAGELayer(n_hidden, edim, n_out, activation))
        else:
             # Ghi đè tầng đầu tiên nếu n_layers = 1
            self.layers[0] = EGraphSAGELayer(ndim_in, edim, n_out, activation)

    def forward(self, g, nfeats, efeats, corrupt=False):
        """
        Hàm lan truyền tiến của toàn bộ mô hình.

        Args:
            g (DGLGraph): Đồ thị.
            nfeats (Tensor): Đặc trưng nút ban đầu.
            efeats (Tensor): Đặc trưng cạnh ban đầu.
            corrupt (bool): Nếu True, xáo trộn đặc trưng cạnh để tạo negative samples.
        """
        # THÊM logic xử lý `corrupt`
        if corrupt:
            # Tạo một hoán vị ngẫu nhiên của các chỉ số cạnh
            perm = torch.randperm(efeats.shape[0])
            # Xáo trộn đặc trưng cạnh theo hoán vị đó
            efeats = efeats[perm]

        h_nodes = nfeats
        h_edges = efeats
        # Lặp qua từng tầng của mô hình
        for i, layer in enumerate(self.layers):
            # Đặc trưng cạnh (efeats) ban đầu (hoặc đã bị xáo trộn) được sử dụng cho tất cả các tầng
            # Đặc trưng nút (h_nodes) được cập nhật qua mỗi tầng
            h_nodes, h_edges = layer(g, h_nodes, efeats)

        return h_nodes, h_edges

# DGI classes
class Discriminator(nn.Module):
    """
    Bộ phân biệt (Discriminator) cho DGI.
    Nó tính toán một điểm số cho cặp (embedding, tóm tắt đồ thị).
    """
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
        # Mở rộng summary để có cùng số chiều với features
        summary_expanded = summary.expand_as(features)
        scores = self.bilinear(features, summary_expanded)
        return scores

class DGI(nn.Module):
    """
    Mô hình Deep Graph Infomax (DGI).
    """
    def __init__(self, encoder):
        super(DGI, self).__init__()
        self.encoder = encoder
        # Kích thước đầu ra của encoder chính là kích thước ẩn cho discriminator
        self.discriminator = Discriminator(encoder.layers[-1].W_edge.out_features)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, nfeats, efeats):
        # 1. Tạo embedding "dương" từ đồ thị gốc
        _, positive_edges = self.encoder(g, nfeats, efeats, corrupt=False)

        # 2. Tạo embedding "âm" từ đồ thị bị làm hỏng
        _, negative_edges = self.encoder(g, nfeats, efeats, corrupt=True)

        # 3. Tạo một vector tóm tắt cho toàn bộ đồ thị
        # Ở đây, chúng ta dùng trung bình của các embedding cạnh "dương"
        summary = torch.sigmoid(positive_edges.mean(dim=0))

        # 4. Tính điểm cho cả mẫu dương và âm
        positive_scores = self.discriminator(positive_edges, summary)
        negative_scores = self.discriminator(negative_edges, summary)

        # 5. Tính loss
        # Mô hình cần dự đoán điểm cao (gần 1) cho mẫu dương
        # và điểm thấp (gần 0) cho mẫu âm.
        l1 = self.loss(positive_scores, torch.ones_like(positive_scores))
        l2 = self.loss(negative_scores, torch.zeros_like(negative_scores))

        return l1 + l2
