import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
# from dgl.nn import SAGEConv  # Cần import SAGEConv nếu bạn dùng nó (mặc dù EGraphSAGE của bạn không dùng)


# ===================================================================
# PHẦN 1: CÁC CLASS BẠN ĐÃ CUNG CẤP (Giữ nguyên)
# ===================================================================

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
        # Kích thước đầu vào = (đặc trưng nút gốc) + (đặc trưng nút nguồn + đặc trưng cạnh)
        # Lưu ý: ndim_in ở đây là kích thước đầu vào của TẦNG NÀY (có thể đã bao gồm memory)
        self.W_apply = nn.Linear(ndim_in + ndim_in + edim, ndim_out)

        # Lớp tuyến tính để tạo embedding cho cạnh.
        self.W_edge = nn.Linear(ndim_out * 2, ndim_out)

    def message_func(self, edges):
        """
        Hàm tạo thông điệp (message).
        """
        # Kết hợp đặc trưng của nút nguồn (edges.src['h']) và đặc trưng của cạnh (edges.data['h'])
        return {'m': torch.cat([edges.src['h'], edges.data['h']], dim=1)}

    def forward(self, g, nfeats, efeats):
        """
        Hàm lan truyền tiến (forward pass).
        """
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats  # Sử dụng efeats được truyền vào

            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            # Xử lý các nút không có hàng xóm (và do đó không có 'h_neigh')
            if 'h_neigh' not in g.ndata:
                g.ndata['h_neigh'] = torch.zeros(g.num_nodes(), nfeats.shape[1] + efeats.shape[1],
                                                 device=nfeats.device, dtype=nfeats.dtype)

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


class EGraphSAGE(nn.Module):
    """
    Mô hình E-GraphSAGE hoàn chỉnh, bao gồm nhiều tầng EGraphSAGELayer.
    """

    def __init__(self, ndim_in, edim, n_hidden, n_out, n_layers, activation):
        """
        Khởi tạo mô hình.
        Lưu ý: ndim_in ở đây là kích thước đã MỞ RỘNG (bao gồm cả memory)
        """
        super(EGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        # Khởi tạo kích thước hiện tại
        current_ndim_in = ndim_in
        current_edim = edim

        # --- SỬA LOGIC KHỞI TẠO ---
        # Tầng đầu tiên: (in/in -> hidden/hidden)
        self.layers.append(EGraphSAGELayer(current_ndim_in, current_edim, n_hidden, activation))

        # Cập nhật dim cho tầng tiếp theo
        current_ndim_in = n_hidden
        current_edim = n_hidden  # Vì W_edge output dim = ndim_out (là n_hidden)

        # Các tầng ẩn: (hidden/hidden -> hidden/hidden)
        for _ in range(n_layers - 2):
            self.layers.append(EGraphSAGELayer(current_ndim_in, current_edim, n_hidden, activation))
            # Dim vẫn là n_hidden cho các tầng ẩn tiếp theo
            current_ndim_in = n_hidden
            current_edim = n_hidden

        # Tầng cuối: (hidden/hidden -> out/out)
        self.layers.append(EGraphSAGELayer(current_ndim_in, current_edim, n_out, activation))

        final_node_out = n_out
        final_edge_out = n_out  # W_edge output dim = ndim_out (là n_out)

        self.out_feats_node = final_node_out
        self.out_feats_edge = final_edge_out

    def forward(self, g, nfeats, efeats, corrupt=False):
        """
        Hàm lan truyền tiến của toàn bộ mô hình.
        """
        local_efeats = efeats
        if corrupt:
            perm = torch.randperm(local_efeats.shape[0], device=local_efeats.device)
            local_efeats = local_efeats[perm]

        h_nodes = nfeats
        h_edges = local_efeats  # Khởi tạo

        # Xử lý node và cạnh cập nhật qua các layer trong EGraphSage
        for i, layer in enumerate(self.layers):
            h_nodes, h_edges = layer(g, h_nodes, h_edges)

        # Trả về kết quả từ tầng cuối cùng
        return h_nodes, h_edges

# DGI classes
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


class DGI(nn.Module):
    def __init__(self, encoder):
        super(DGI, self).__init__()
        self.encoder = encoder

        # Kích thước đầu ra CẠNH của encoder
        # Cần truy cập vào EGraphSAGE bên trong TGNWrapper
        if isinstance(encoder, TGNWrapperEncoder):
            sage_encoder = encoder.sage_encoder
        else:
            sage_encoder = encoder  # Nếu dùng EGraphSAGE trực tiếp

        self.discriminator = Discriminator(sage_encoder.out_feats_edge)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, nfeats, efeats):
        # 1. Tạo embedding "dương"
        # TGNWrapper.forward sẽ được gọi ở đây
        _, positive_edges = self.encoder(g, nfeats, efeats, corrupt=False)

        # 2. Tạo embedding "âm"
        _, negative_edges = self.encoder(g, nfeats, efeats, corrupt=True)

        # 3. Tạo vector tóm tắt
        summary = torch.sigmoid(positive_edges.mean(dim=0))

        # 4. Tính điểm
        positive_scores = self.discriminator(positive_edges, summary)
        negative_scores = self.discriminator(negative_edges, summary)

        # 5. Tính loss
        l1 = self.loss(positive_scores, torch.ones_like(positive_scores))
        l2 = self.loss(negative_scores, torch.zeros_like(negative_scores))

        return l1 + l2


# ===================================================================
# PHẦN 2: TGN WRAPPER ENCODER (ĐÃ ĐIỀU CHỈNH)
# ===================================================================

class TGNWrapperEncoder(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim,
                 memory_dim, msg_dim,
                 sage_n_hidden, sage_n_out, sage_n_layers, sage_activation):
        """
        Khởi tạo TGN Wrapper.
        Wrapper này sẽ tự khởi tạo EGraphSAGE bên trong nó.
        """
        super().__init__()

        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.num_nodes = num_nodes

        # --- TGN Components ---
        self._memory_data = torch.zeros((num_nodes, memory_dim))
        self.memory = nn.Parameter(self._memory_data, requires_grad=False)

        # --- CÁC HÀM MESSAGE (MESSAGE FUNCTIONS) ---

        # 1. Message cho sự kiện tương tác (thêm cạnh)
        self.msg_gen_fn = nn.Linear(memory_dim * 2 + edge_feat_dim, msg_dim)

        # 2. Bộ cập nhật bộ nhớ (GRU)
        self.memory_updater = nn.GRUCell(input_size=msg_dim, hidden_size=memory_dim)

        # 3. [MỚI] Message cho sự kiện XÓA CẠNH (theo Phụ lục A.1)
        # Giả sử chúng ta dùng chung msg_dim
        # Paper dùng msgs và msgd, ở đây ta dùng chung một hàm cho đơn giản
        self.edge_del_msg_fn = nn.Linear(memory_dim * 2 + edge_feat_dim, msg_dim)

        # 4. [MỚI] Message cho sự kiện XÓA NÚT (theo Phụ lục A.1)
        # Message "tạm biệt" chỉ dựa trên trạng thái của nút bị xóa
        self.node_del_msg_fn = nn.Linear(memory_dim, msg_dim)

        # --- HẾT PHẦN MỚI ---

        # --- EGraphSAGE Encoder (Internal) ---
        sage_ndim_in = node_feat_dim + memory_dim
        self.sage_encoder = EGraphSAGE(
            ndim_in=sage_ndim_in,
            edim=edge_feat_dim,
            n_hidden=sage_n_hidden,
            n_out=sage_n_out,
            n_layers=sage_n_layers,
            activation=sage_activation
        )

        print(f"TGNWrapperEncoder khởi tạo với {num_nodes} nút.")
        print(f"  Memory dim: {memory_dim}")
        print(f"  EGraphSAGE input dim: {sage_ndim_in} (node={node_feat_dim} + mem={memory_dim})")

    # --- Hàm message cho sự kiện THÊM CẠNH (Tương tác) ---
    def _compute_messages(self, edges):
        src_mem = edges.src['mem']
        dst_mem = edges.dst['mem']
        edge_feat = edges.data['h_edge']
        edge_feat = edge_feat.to(src_mem.dtype)

        input_features = torch.cat([src_mem, dst_mem, edge_feat], dim=1)
        msg = F.relu(self.msg_gen_fn(input_features))
        return {'msg': msg}

    # --- [MỚI] Hàm message cho sự kiện XÓA CẠNH ---
    def _compute_delete_messages(self, edges):
        src_mem = edges.src['mem']
        dst_mem = edges.dst['mem']
        edge_feat = edges.data['h_edge']
        edge_feat = edge_feat.to(src_mem.dtype)

        input_features = torch.cat([src_mem, dst_mem, edge_feat], dim=1)
        # Chúng ta giả định 2 hàm message: 1 cho src (msgs), 1 cho dst (msgd)
        # Để đơn giản, ta dùng chung 1 hàm `edge_del_msg_fn`
        msg = F.relu(self.edge_del_msg_fn(input_features))

        # Paper nói mi(t) và mj(t). Ta sẽ trả về message cho cả hai.
        # Giả sử `msg` là chung, hoặc ta có thể tách:
        # msg_src = F.relu(self.edge_del_msg_src_fn(input_features))
        # msg_dst = F.relu(self.edge_del_msg_dst_fn(input_features))
        # Ở đây ta dùng chung `msg` cho cả src và dst
        return {'msg_src_del': msg, 'msg_dst_del': msg}

    def forward(self, g, nfeats, efeats, corrupt=False):
        if g.num_nodes() != self.num_nodes:
            raise ValueError(
                f"Lỗi kích thước! Đồ thị DGL có {g.num_nodes()} nút, "
                f"nhưng bộ nhớ TGN (self.memory) có {self.num_nodes} nút."
            )

        # --- 1. CẬP NHẬT BỘ NHỚ (TGN Memory Update) ---
        with torch.no_grad():
            with g.local_scope():
                g.ndata['mem'] = self.memory
                g.edata['h_edge'] = efeats

                # Tính toán message TƯƠNG TÁC (thêm cạnh)
                g.apply_edges(self._compute_messages)
                g.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'agg_msg'))

                agg_msg = torch.zeros((g.num_nodes(), self.msg_gen_fn.out_features),
                                      dtype=self.memory.dtype,
                                      device=self.memory.device)

                if 'agg_msg' in g.ndata:
                    dst_nodes_with_msg_mask = (g.in_degrees() > 0)
                    dst_nodes_with_msg = torch.where(dst_nodes_with_msg_mask)[0]

                    if dst_nodes_with_msg.shape[0] > 0:
                        agg_msg[dst_nodes_with_msg] = g.ndata['agg_msg'][dst_nodes_with_msg]

                current_mem = self.memory
                new_memory = self.memory_updater(agg_msg, current_mem)
                self.memory.data = new_memory
                self._memory_data = new_memory.data

        # --- 2. TÍNH TOÁN EMBEDDING (EGraphSAGE) ---
        augmented_nfeats = torch.cat([nfeats, self.memory], dim=1)
        h_nodes, h_edges = self.sage_encoder(g, augmented_nfeats, efeats, corrupt=corrupt)

        return h_nodes, h_edges

    # --- CÁC HÀM TIỆN ÍCH ĐỂ THÍCH ỨNG VỚI THAY ĐỔI ---

    @torch.no_grad()
    def add_node(self, node_id, initial_memory=None):
        # ... (Giữ nguyên hàm add_node) ...
        if node_id != self.memory.shape[0]:
            print(f"Cảnh báo: ID nút mới {node_id} không liên tục. Kỳ vọng {self.memory.shape[0]}")

        if self.memory.device != torch.device('cpu'):
            print(f"Cảnh báo: Bộ nhớ đang ở trên thiết bị {self.memory.device}. "
                  "Đảm bảo initial_memory cũng ở trên thiết bị này nếu được cung cấp.")

        if initial_memory is None:
            new_mem_vector = torch.zeros((1, self.memory_dim),
                                         dtype=self.memory.dtype,
                                         device=self.memory.device)
        else:
            new_mem_vector = initial_memory.view(1, self.memory_dim).to(
                self.memory.device, dtype=self.memory.dtype
            )

        self._memory_data = torch.cat([self._memory_data, new_mem_vector], dim=0)
        self.memory = nn.Parameter(self._memory_data, requires_grad=False)
        self.num_nodes = self.memory.shape[0]
        print(f"Đã thêm nút {node_id}. Kích thước bộ nhớ mới: {self.memory.shape}")

    # --- [CẬP NHẬT] HÀM XỬ LÝ XÓA CẠNH ---
    @torch.no_grad()
    def process_deleted_edges(self, deleted_edges_g):
        """
        Xử lý sự kiện XÓA CẠNH một cách chủ động (theo Phụ lục A.1).
        Hàm này tính toán message xóa và cập nhật bộ nhớ
        của các nút liên quan.

        Args:
            deleted_edges_g (dgl.Graph): Một đồ thị DGL *chỉ* chứa
                các cạnh đã bị xóa. Quan trọng:
                1. Số nút (num_nodes) phải bằng self.num_nodes.
                2. edata['h_edge'] phải chứa đặc trưng của các cạnh bị xóa.
        """
        if deleted_edges_g.num_edges() == 0:
            print("Không có cạnh nào bị xóa.")
            return

        print(f"Đang xử lý {deleted_edges_g.num_edges()} sự kiện XÓA CẠNH (chủ động)...")

        with deleted_edges_g.local_scope():
            # 1. Gán bộ nhớ hiện tại vào đồ thị tạm
            deleted_edges_g.ndata['mem'] = self.memory
            # (Giả định deleted_edges_g.edata['h_edge'] đã được cung cấp)

            # 2. Tính toán message XÓA
            deleted_edges_g.apply_edges(self._compute_delete_messages)

            # 3. Tổng hợp message cho các nút NGUỒN (src)
            deleted_edges_g.update_all(fn.copy_e('msg_src_del', 'm_del'),
                                       fn.mean('m_del', 'agg_msg_src_del'))

            # 4. Tổng hợp message cho các nút ĐÍCH (dst)
            deleted_edges_g.update_all(fn.copy_e('msg_dst_del', 'm_del'),
                                       fn.mean('m_del', 'agg_msg_dst_del'))

            # 5. Cập nhật bộ nhớ cho các nút bị ảnh hưởng
            # Lấy ID các nút nguồn bị ảnh hưởng
            src_nodes_affected_mask = (deleted_edges_g.out_degrees() > 0)
            src_nodes_affected = torch.where(src_nodes_affected_mask)[0]

            # Lấy ID các nút đích bị ảnh hưởng
            dst_nodes_affected_mask = (deleted_edges_g.in_degrees() > 0)
            dst_nodes_affected = torch.where(dst_nodes_affected_mask)[0]

            # Cập nhật nút nguồn
            if src_nodes_affected.shape[0] > 0:
                current_mem_src = self.memory[src_nodes_affected]
                agg_msg_src = deleted_edges_g.ndata['agg_msg_src_del'][src_nodes_affected]

                new_mem_src = self.memory_updater(agg_msg_src, current_mem_src)
                self.memory.data[src_nodes_affected] = new_mem_src
                print(f"  -> Đã cập nhật bộ nhớ (xóa cạnh) cho {len(src_nodes_affected)} nút NGUỒN.")

            # Cập nhật nút đích
            if dst_nodes_affected.shape[0] > 0:
                current_mem_dst = self.memory[dst_nodes_affected]
                agg_msg_dst = deleted_edges_g.ndata['agg_msg_dst_del'][dst_nodes_affected]

                new_mem_dst = self.memory_updater(agg_msg_dst, current_mem_dst)
                self.memory.data[dst_nodes_affected] = new_mem_dst
                print(f"  -> Đã cập nhật bộ nhớ (xóa cạnh) cho {len(dst_nodes_affected)} nút ĐÍCH.")

            # Cập nhật bản sao lưu
            self._memory_data = self.memory.data.clone()

    # --- [CẬP NHẬT] HÀM XỬ LÝ XÓA NÚT ---
    @torch.no_grad()
    def delete_node(self, node_id_to_delete, G_original_before_delete, node_order_map):
        """
        Xóa một nút khỏi bộ nhớ.
        Theo Phụ lục A.1, hàm này cũng sẽ (tùy chọn) cập nhật
        bộ nhớ của các hàng xóm của nút bị xóa.

        Args:
            node_id_to_delete (int): ID (chỉ số) của nút cần xóa.
            G_original_before_delete (nx.Graph): Đồ thị NetworkX
                TRƯỚC KHI nút bị xóa, dùng để tìm hàng xóm.
            node_order_map (dict): Ánh xạ {node_name: node_id}.
        """
        if node_id_to_delete >= self.num_nodes or node_id_to_delete < 0:
            print(f"Lỗi: Không thể xóa nút {node_id_to_delete}. ID không hợp lệ.")
            return

        print(f"Đang xóa nút {node_id_to_delete} (chủ động)...")

        # === PHẦN 1: Cập nhật hàng xóm (Tùy chọn theo A.1) ===

        # 1. Tìm tên nút từ ID
        node_name_to_delete = None
        for name, idx in node_order_map.items():
            if idx == node_id_to_delete:
                node_name_to_delete = name
                break

        if node_name_to_delete is None:
            print(f"Lỗi: Không tìm thấy tên nút cho ID {node_id_to_delete}")
            # Vẫn tiếp tục xóa, nhưng không thể cập nhật hàng xóm
        else:
            # 2. Tìm hàng xóm (cả vào và ra) trong đồ thị CŨ
            try:
                neighbors = list(G_original_before_delete.predecessors(node_name_to_delete)) + \
                            list(G_original_before_delete.successors(node_name_to_delete))
                neighbor_names = list(set(neighbors))  # Loại bỏ trùng lặp

                if neighbor_names:
                    # 3. Lấy ID của các hàng xóm
                    neighbor_ids = [node_order_map[name] for name in neighbor_names]
                    neighbor_ids_tensor = torch.tensor(neighbor_ids,
                                                       dtype=torch.long,
                                                       device=self.memory.device)

                    # 4. Lấy bộ nhớ của nút SẮP BỊ XÓA
                    deleted_node_mem = self.memory[node_id_to_delete].unsqueeze(0)

                    # 5. Tính message "tạm biệt"
                    goodbye_msg = F.relu(self.node_del_msg_fn(deleted_node_mem))

                    # 6. Lấy bộ nhớ hiện tại của các hàng xóm
                    current_neighbor_mems = self.memory[neighbor_ids_tensor]

                    # 7. Cập nhật bộ nhớ hàng xóm
                    # Lặp lại message "tạm biệt" cho mọi hàng xóm
                    goodbye_msg_expanded = goodbye_msg.expand_as(current_neighbor_mems)

                    new_neighbor_mems = self.memory_updater(goodbye_msg_expanded, current_neighbor_mems)

                    # 8. Ghi lại bộ nhớ đã cập nhật
                    self.memory.data[neighbor_ids_tensor] = new_neighbor_mems
                    print(f"  -> Đã cập nhật bộ nhớ cho {len(neighbor_ids)} hàng xóm của nút bị xóa.")

            except Exception as e:
                print(f"  -> Lỗi khi cập nhật hàng xóm: {e}. "
                      "Có thể nút không có trong G_original_before_delete.")

        # === PHẦN 2: Xóa nút khỏi bộ nhớ (Bắt buộc) ===

        indices_to_keep = [i for i in range(self.num_nodes) if i != node_id_to_delete]

        if not indices_to_keep:
            self._memory_data = torch.empty((0, self.memory_dim),
                                            dtype=self.memory.dtype,
                                            device=self.memory.device)
        else:
            indices_to_keep_tensor = torch.tensor(indices_to_keep,
                                                  device=self.memory.device,
                                                  dtype=torch.long)
            self._memory_data = self._memory_data.index_select(0, indices_to_keep_tensor)

        # Tạo lại nn.Parameter với kích thước mới
        self.memory = nn.Parameter(self._memory_data, requires_grad=False)
        old_num_nodes = self.num_nodes
        self.num_nodes = self.memory.shape[0]

        print(f"Đã xóa nút khỏi bộ nhớ. Kích thước bộ nhớ mới: {self.memory.shape}")
        print(f"LƯU Ý QUAN TRỌNG: Các nút có ID > {node_id_to_delete} (trong {old_num_nodes} nút cũ) "
              f"hiện đã bị dịch chuyển chỉ số.")

    @torch.no_grad()
    def process_new_edges(self, new_edges_g):
        # ... (Giữ nguyên hàm process_new_edges) ...
        print(f"Đang xử lý {new_edges_g.num_edges()} cạnh mới...")

        if new_edges_g.num_nodes() != self.num_nodes:
            print(f"Lỗi: Đồ thị cạnh mới có {new_edges_g.num_nodes()} nút, "
                  f"nhưng bộ nhớ có {self.num_nodes} nút.")
            return

        with new_edges_g.local_scope():
            new_edges_g.ndata['mem'] = self.memory
            if 'h_edge' not in new_edges_g.edata:
                print("Cảnh báo: 'new_edges_g' không có 'h_edge'. Sẽ sử dụng efeats rỗng.")
                efeats_zeros = torch.zeros((new_edges_g.num_edges(), self.edge_feat_dim),
                                           dtype=self.memory.dtype,
                                           device=self.memory.device)
                new_edges_g.edata['h_edge'] = efeats_zeros

            new_edges_g.apply_edges(self._compute_messages)
            new_edges_g.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'agg_msg'))

            agg_msg = torch.zeros((self.num_nodes, self.msg_gen_fn.out_features),
                                  dtype=self.memory.dtype,
                                  device=self.memory.device)

            if 'agg_msg' in new_edges_g.ndata:
                affected_nodes_mask = (new_edges_g.in_degrees() > 0)
                affected_nodes = torch.where(affected_nodes_mask)[0]

                if affected_nodes.shape[0] > 0:
                    agg_msg[affected_nodes] = new_edges_g.ndata['agg_msg'][affected_nodes]
            else:
                affected_nodes = torch.tensor([], dtype=torch.long, device=self.memory.device)

            if affected_nodes.shape[0] > 0:
                current_mem_affected = self.memory[affected_nodes]
                agg_msg_affected = agg_msg[affected_nodes]

                new_mem_affected = self.memory_updater(agg_msg_affected, current_mem_affected)
                self.memory.data[affected_nodes] = new_mem_affected
                self._memory_data.data[affected_nodes] = new_mem_affected.data
                print(f"Đã cập nhật bộ nhớ cho {len(affected_nodes)} nút bị ảnh hưởng.")
            else:
                print("Không có nút nào bị ảnh hưởng (ví dụ: đồ thị rỗng).")