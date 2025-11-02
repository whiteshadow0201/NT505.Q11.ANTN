import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import os
from copy import deepcopy
# Đảm bảo 'utils.graph_utils' chứa TGNWrapperEncoder
# với các hàm delete_node và process_deleted_edges đã được định nghĩa
from utils.graph_utils import *
import subprocess
# ===================================================================
# PHẦN 2: KỊCH BẢN KIỂM THỬ (MAIN SCRIPT) - PHIÊN BẢN XÓA NÚT
# ===================================================================

if __name__ == "__main__":

    # --- 1. Thiết lập đường dẫn ---

    # TODO: Thay đổi '1' thành thư mục chứa mô hình đã huấn luyện của bạn
    LOAD_DIR = "./graphs/1"
    # Thư mục xuất kết quả cho kịch bản xóa

    EXPORT_DIR = "./graph_test_delete"
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
    MEMORY_DIM = 32
    MSG_DIM = 32

    # Lấy thông tin từ các file đã tải
    G_original_loaded = graph_env['G_original'].copy()
    node_order_original = deepcopy(graph_env['node_order'])
    node_map_original = {name: i for i, name in enumerate(node_order_original)}
    NUM_NODES_ORIGINAL = G_original_loaded.number_of_nodes()

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
    dgi_model.load_state_dict(model_state_dict)
    dgi_model.eval()
    encoder = dgi_model.encoder
    print("Tải trọng số và khởi tạo mô hình thành công.")
    print(f"Bộ nhớ Encoder ban đầu: {encoder.memory.shape}")

    # --- 4. [THAY ĐỔI] Tạo đồ thị thu hẹp (G_shrunk) ---
    print("\n--- Đang tạo đồ thị thu hẹp (G_shrunk) ---")

    # Chọn một nút để xóa (ví dụ: 'Host 3')
    node_to_delete_name = "Host 3"

    # Lấy ID của nút TRƯỚC KHI xóa
    if node_to_delete_name not in node_map_original:
        print(f"LỖI: Nút '{node_to_delete_name}' không tồn tại trong đồ thị gốc.")
        exit()

    node_id_to_delete = node_map_original[node_to_delete_name]
    print(f"Đã chọn nút để xóa: '{node_to_delete_name}' (ID gốc: {node_id_to_delete})")

    # Tạo đồ thị "sau khi xóa"
    G_shrunk = G_original_loaded.copy()
    G_shrunk.remove_node(node_to_delete_name)

    # Tạo trật tự nút và map nút MỚI
    node_order_shrunk = [n for n in node_order_original if n != node_to_delete_name]
    node_map_shrunk = {name: i for i, name in enumerate(node_order_shrunk)}
    num_nodes_shrunk = G_shrunk.number_of_nodes()

    print(f"Số nút mới: {num_nodes_shrunk} (Ban đầu: {NUM_NODES_ORIGINAL})")
    print(f"Số cạnh mới: {G_shrunk.number_of_edges()} (Ban đầu: {G_original_loaded.number_of_edges()})")

    # --- 5. [THAY ĐỔI] Thích ứng TGN Encoder (Xóa Nút) ---
    print("\n--- Đang thích ứng bộ nhớ TGN (Node Deletion) ---")

    # Gọi hàm delete_node (phiên bản chủ động từ lần trước)
    # Hàm này cần ID của nút xóa, đồ thị TRƯỚC KHI XÓA, và map nút CŨ
    encoder.delete_node(
        node_id_to_delete,
        G_original_loaded,  # Đồ thị NetworkX gốc
        node_map_original  # Map ánh xạ tên->ID gốc
    )

    print(f"Bộ nhớ Encoder sau khi xóa: {encoder.memory.shape}")
    if encoder.memory.shape[0] != num_nodes_shrunk:
        print(f"LỖI: Kích thước bộ nhớ ({encoder.memory.shape[0]}) "
              f"không khớp số nút mới ({num_nodes_shrunk})!")
        exit()

    # Lưu ý: Kịch bản này không kiểm tra process_deleted_edges riêng lẻ,
    # vì delete_node (phiên bản A.1) đã ngụ ý xử lý các cạnh
    # thông qua việc cập nhật hàng xóm.

    # --- 6. [THAY ĐỔI] Xây dựng đồ thị DGL đầy đủ và Suy luận ---
    print("\n--- Đang suy luận (Inference) trên đồ thị thu hẹp ---")

    # Xây dựng DGL graph đầy đủ từ G_shrunk
    src_shrunk = [node_map_shrunk[u] for u, v in G_shrunk.edges()]
    dst_shrunk = [node_map_shrunk[v] for u, v in G_shrunk.edges()]
    g_dgl_shrunk = dgl.graph((src_shrunk, dst_shrunk), num_nodes=num_nodes_shrunk)

    # Lấy đặc trưng nút theo đúng thứ tự MỚI
    node_features_shrunk = torch.tensor(
        [[G_shrunk.nodes[n]['state'], G_shrunk.nodes[n]['priority']] for n in node_order_shrunk],
        dtype=torch.float32
    )

    # Lấy đặc trưng cạnh theo đúng thứ tự MỚI
    edge_features_shrunk = torch.tensor(
        [[G_shrunk.get_edge_data(u, v)['user'], G_shrunk.get_edge_data(u, v)['root']]
         for u, v in G_shrunk.edges()],
        dtype=torch.float32
    )

    g_dgl_shrunk.ndata['h'] = node_features_shrunk
    g_dgl_shrunk.edata['h'] = edge_features_shrunk

    # Chạy suy luận
    with torch.no_grad():
        final_node_emb, final_edge_emb = encoder(
            g_dgl_shrunk,
            g_dgl_shrunk.ndata['h'],
            g_dgl_shrunk.edata['h'],
            corrupt=False
        )

    print("Suy luận hoàn tất.")
    print(f"Kích thước Node Embeddings mới: {final_node_emb.shape}")
    print(f"Kích thước Edge Embeddings mới: {final_edge_emb.shape}")

    # --- 7. [THAY ĐỔI] Xuất kết quả ---
    print(f"\n--- Đang xuất kết quả ra '{EXPORT_DIR}' ---")

    # 7a. Xuất Embeddings
    np.save(f"{EXPORT_DIR}/node_embeddings.npy", final_node_emb.numpy())
    np.save(f"{EXPORT_DIR}/edge_embeddings.npy", final_edge_emb.numpy())
    print("Đã lưu node_embeddings.npy và edge_embeddings.npy")

    # 7b. Xuất Môi trường Đồ thị Mới
    graph_environment_shrunk = {
        'G_original': G_shrunk,
        'node_order': node_order_shrunk,
        'node_map': node_map_shrunk,
        'node_features_original': node_features_shrunk,
        'edge_features_original': edge_features_shrunk
    }
    torch.save(graph_environment_shrunk, f"{EXPORT_DIR}/graph_environment.pth")
    print("Đã lưu graph_environment.pth")

    print("\n[THÀNH CÔNG] Đã hoàn tất kiểm thử thích ứng (XÓA NÚT).")