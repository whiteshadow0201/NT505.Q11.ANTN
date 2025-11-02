import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import os
from copy import deepcopy
from utils.graph_utils import *


# ===================================================================
# PHẦN 2: KỊCH BẢN KIỂM THỬ (MAIN SCRIPT)
# ===================================================================

if __name__ == "__main__":

    # --- 1. Thiết lập đường dẫn ---

    # TODO: Thay đổi '1' thành thư mục chứa mô hình đã huấn luyện của bạn
    LOAD_DIR = "./graphs/1"
    EXPORT_DIR = "./graph_test"

    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"Sẽ tải mô hình từ: {LOAD_DIR}")
    print(f"Sẽ xuất kết quả vào: {EXPORT_DIR}")

    # --- 2. Tải các thành phần đã lưu ---
    print("\n--- Đang tải các thành phần đã huấn luyện ---")

    try:
        model_config = torch.load(f"{LOAD_DIR}/model_config.pth", weights_only=False)
        graph_env = torch.load(f"{LOAD_DIR}/graph_environment.pth", weights_only=False)
        model_state_dict = torch.load(f"{LOAD_DIR}/dgi_model_state_dict.pth", weights_only=False)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file trong thư mục '{LOAD_DIR}'.")
        print("Vui lòng đảm bảo bạn đã chạy mã huấn luyện và lưu mô hình.")
        exit()

    # --- 3. Khởi tạo lại mô hình ---
    print("--- Đang khởi tạo lại mô hình DGI/TGN ---")

    # Giả định các tham số TGN này khớp với lúc huấn luyện
    # (Vì chúng không được lưu trong model_config.pth của bạn)
    MEMORY_DIM = 32
    MSG_DIM = 32

    # Lấy thông tin từ các file đã tải
    original_g = graph_env['G_original']
    NUM_NODES_ORIGINAL = original_g.number_of_nodes()

    # Khởi tạo encoder và DGI
    encoder = TGNWrapperEncoder(
        num_nodes=NUM_NODES_ORIGINAL,
        node_feat_dim=model_config['NDIM_IN'],
        edge_feat_dim=model_config['EDIM'],
        memory_dim=MEMORY_DIM,
        msg_dim=MSG_DIM,
        sage_n_hidden=model_config['N_HIDDEN'],
        sage_n_out=model_config['N_OUT'],
        sage_n_layers=model_config['N_LAYERS'],
        sage_activation=F.leaky_relu
    )

    dgi_model = DGI(encoder)

    # Tải trọng số đã huấn luyện
    dgi_model.load_state_dict(model_state_dict)
    dgi_model.eval()  # Chuyển sang chế độ suy luận (QUAN TRỌNG)

    # Lấy encoder đã huấn luyện
    encoder = dgi_model.encoder
    print("Tải trọng số và khởi tạo mô hình thành công.")

    # --- 4. Tạo đồ thị mở rộng ---
    print("\n--- Đang tạo đồ thị mở rộng (G_extended) ---")

    G_extended = graph_env['G_original'].copy()
    node_order_ext = deepcopy(graph_env['node_order'])

    # Thêm nút mới
    new_node_name = "Host 4"
    new_node_id = G_extended.number_of_nodes()  # ID sẽ là 8 (vì 8 nút 0-7)
    G_extended.add_node(new_node_name, state=0, priority=0)
    node_order_ext.append(new_node_name)
    node_map_ext = {name: i for i, name in enumerate(node_order_ext)}

    print(f"Đã thêm nút mới: '{new_node_name}' (ID: {new_node_id})")

    # Thêm các cạnh mới
    new_edges_list = [
        ("Host 4", "Host 1", {"user": 0.5, "root": 0.5}),
        ("Web Server", "Host 4", {"user": 0.7, "root": 0.1})
    ]
    G_extended.add_edges_from(new_edges_list)
    print(f"Đã thêm {len(new_edges_list)} cạnh mới.")

    # --- 5. Thích ứng TGN Encoder (KHÔNG HUẤN LUYỆN) ---
    print("\n--- Đang thích ứng bộ nhớ TGN (Memory Adaptation) ---")

    # 5a. Thích ứng với nút mới
    encoder.add_node(new_node_id)

    # 5b. Thích ứng với các cạnh mới
    # Tạo đồ thị DGL *chỉ chứa các cạnh mới*
    num_total_nodes = G_extended.number_of_nodes()  # = 9

    src_nodes = [node_map_ext[u] for u, v, d in new_edges_list]
    dst_nodes = [node_map_ext[v] for u, v, d in new_edges_list]
    new_edge_features = torch.tensor([[d['user'], d['root']] for u, v, d in new_edges_list],
                                     dtype=torch.float32)

    new_edges_g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_total_nodes)
    new_edges_g.edata['h_edge'] = new_edge_features

    # Chạy cập nhật bộ nhớ
    encoder.process_new_edges(new_edges_g)

    # --- 6. Xây dựng đồ thị DGL đầy đủ và Suy luận ---
    print("\n--- Đang suy luận (Inference) trên đồ thị mở rộng ---")

    # Xây dựng DGL graph đầy đủ một cách cẩn thận
    src_ext = [node_map_ext[u] for u, v in G_extended.edges()]
    dst_ext = [node_map_ext[v] for u, v in G_extended.edges()]
    g_dgl_ext = dgl.graph((src_ext, dst_ext), num_nodes=num_total_nodes)

    # Lấy đặc trưng theo đúng thứ tự
    node_features_ext = torch.tensor(
        [[G_extended.nodes[n]['state'], G_extended.nodes[n]['priority']] for n in node_order_ext],
        dtype=torch.float32
    )

    # Lấy đặc trưng cạnh theo đúng thứ tự của g_dgl_ext.edges()
    # (NetworkX G.edges() và DGL G.edges() có thể không cùng thứ tự)
    # Chúng ta xây dựng DGL graph từ (src_ext, dst_ext),
    # nên đặc trưng cạnh phải tương ứng với thứ tự đó
    edge_features_ext = torch.tensor(
        [[G_extended.get_edge_data(u, v)['user'], G_extended.get_edge_data(u, v)['root']]
         for u, v in G_extended.edges()],
        dtype=torch.float32
    )

    g_dgl_ext.ndata['h'] = node_features_ext
    g_dgl_ext.edata['h'] = edge_features_ext

    # Chạy suy luận
    with torch.no_grad():
        final_node_emb, final_edge_emb = encoder(
            g_dgl_ext,
            g_dgl_ext.ndata['h'],
            g_dgl_ext.edata['h'],
            corrupt=False
        )

    print("Suy luận hoàn tất.")
    print(f"Kích thước Node Embeddings mới: {final_node_emb.shape}")
    print(f"Kích thước Edge Embeddings mới: {final_edge_emb.shape}")

    # --- 7. Xuất kết quả ---
    print("\n--- Đang xuất kết quả ra 'graph_test' ---")

    # 7a. Xuất Embeddings
    np.save(f"{EXPORT_DIR}/node_embeddings.npy", final_node_emb.numpy())
    np.save(f"{EXPORT_DIR}/edge_embeddings.npy", final_edge_emb.numpy())
    print("Đã lưu node_embeddings_ext.npy và edge_embeddings_ext.npy")

    # 7b. Xuất Môi trường Đồ thị Mới
    graph_environment_ext = {
        'G_original': G_extended,
        'node_order': node_order_ext,
        'node_map': node_map_ext,
        'node_features_original': node_features_ext,
        'edge_features_original': edge_features_ext
    }
    torch.save(graph_environment_ext, f"{EXPORT_DIR}/graph_environment.pth")
    print("Đã lưu graph_environment.pth")

    print("\n[THÀNH CÔNG] Đã hoàn tất kiểm thử thích ứng.")
