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
import numpy as np
import torch
from copy import deepcopy
import random
import itertools  # ### SỬA ĐỔI ###: Thêm thư viện để lặp
import torch.nn.functional as F
import time

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

# --- HÀM HỖ TRỢ: GÁN TIMESTAMP (ĐÃ SỬA) ---
def add_timestamp_to_edges(raw_edges_list):
    """
    Nhận vào danh sách cạnh thô, gán timestamp cách nhau 0.1s
    dựa trên thứ tự trong danh sách và trả về danh sách đã sort.
    """
    processed_edges = []

    # Lấy mốc thời gian hiện tại làm chuẩn
    start_time = time.time()

    # Dùng enumerate để lấy chỉ số i (0, 1, 2...)
    for i, (u, v, attrs) in enumerate(raw_edges_list):
        # Gán timestamp: Mỗi cạnh cách nhau đúng 0.1 giây
        # Cạnh đầu tiên: start_time
        # Cạnh thứ hai: start_time + 0.1
        # Cạnh thứ ba: start_time + 0.2 ...
        attrs['timestamp'] = start_time + (i * 0.1)

        processed_edges.append((u, v, attrs))

    # QUAN TRỌNG: Sort danh sách dựa trên timestamp
    processed_edges.sort(key=lambda x: x[2]['timestamp'])
    return processed_edges