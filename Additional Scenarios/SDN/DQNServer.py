import http.server
import socketserver
import threading
import torch
import torch.nn.functional as F
import networkx as nx
import dgl
import numpy as np
import pickle
import requests
import io
import yaml
import json
import os
from copy import deepcopy

# --- IMPORT CÁC UTILS (Đảm bảo folder utils nằm cùng cấp) ---
from utils.basic_utils import *
from utils.graph_utils import *
from utils.model_utils import *
# Giả sử NetworkEnv nằm trong utils hoặc file riêng, cần import để tái tạo môi trường
from utils.attack_algo_utils import global_weighted_random_attack

# ======================================================
# 1. CẤU HÌNH HỆ THỐNG
# ======================================================
# Cấu hình Mininet (Mục tiêu để tải graph và gửi lệnh deploy)
MININET_IP = "192.168.2.9"
MININET_PORT = 8000
GRAPH_URL = f"http://{MININET_IP}:{MININET_PORT}/network_state.pth"
DEPLOY_URL = f"http://{MININET_IP}:{MININET_PORT}/deploy_honeypots"

# Cấu hình Server AI (Script này)
MY_SERVER_PORT = 9999
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

print("Device:",DEVICE)
# Đường dẫn Model đã train (Từ Original.py)
GNN_CONFIG_PATH = "graphs/model_config.yaml"
GNN_STATE_PATH = "graphs/dgi_model_state_dict.pth"
DQN_MODEL_PATH = "Saved_Model/dqn_model.pth"  # Hoặc đường dẫn file bạn đã lưu

# Biến toàn cục lưu trữ Model đã load để không phải load lại nhiều lần
GLOBAL_MODELS = {
    "encoder": None,
    "policy_net": None,
    "config": None
}


# ======================================================
# 2. CÁC HÀM HỖ TRỢ (UTILS)
# ======================================================

def build_batch_tensor(feats, max_dim):
    """Hàm tiền xử lý feature giống khi train"""
    return feats[:, :max_dim]


def load_static_models():
    """Load model 1 lần duy nhất khi khởi động script"""
    print("--- [INIT] Đang tải các Model đã huấn luyện... ---")

    # 1. Load Config & GNN Encoder
    try:
        with open(GNN_CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
            GLOBAL_MODELS["config"] = config

        encoder = EGraphSAGE(
            config['NDIM_IN'], config['EDIM'], config['N_HIDDEN'],
            config['N_OUT'], config['N_LAYERS'], F.leaky_relu, DEVICE
        )
        dgi = DGI(encoder)
        dgi.load_state_dict(torch.load(GNN_STATE_PATH, map_location=DEVICE, weights_only=False))
        GLOBAL_MODELS["encoder"] = dgi.encoder
        GLOBAL_MODELS["encoder"].eval()
        print(f"[+] GNN Encoder loaded.")
    except Exception as e:
        print(f"[!] Lỗi load GNN: {e}")
        exit()

    # 2. Load DQN Policy Net
    try:
        checkpoint = torch.load(DQN_MODEL_PATH, map_location=DEVICE)

        # Lấy state_dict (xử lý trường hợp lưu full checkpoint hoặc chỉ save state_dict)
        state_dict = checkpoint['policy_net_state_dict'] if 'policy_net_state_dict' in checkpoint else checkpoint

        # --- SỬA LỖI KEY ACCESS ---
        # MultiHeadDQN dùng 'fc1' là lớp đầu tiên, không phải 'layers.0'
        if 'fc1.weight' in state_dict:
            input_w = state_dict['fc1.weight']
        else:
            # Fallback nếu bạn thực sự dùng Sequential ở version khác
            input_w = state_dict['layers.0.weight']

        state_size = input_w.shape[1]  # [out_features, in_features] -> lấy in_features

        # Tính toán num_honeypots dựa trên số lượng đầu ra output_heads
        # Key sẽ có dạng: output_heads.0.weight, output_heads.1.weight, ...
        # Đếm số lượng key bắt đầu bằng 'output_heads' và kết thúc bằng '.weight'
        keys = state_dict.keys()
        head_weights = [k for k in keys if k.startswith('output_heads.') and k.endswith('.weight')]
        num_honeypots = len(head_weights)

        # Lấy output shape từ head đầu tiên để biết max_nodes
        # Key: 'output_heads.0.weight' -> shape [num_nodes, hidden_size]
        first_head_key = f'output_heads.0.weight'
        if first_head_key in state_dict:
            output_w = state_dict[first_head_key]
            max_nodes_out = output_w.shape[0]
        else:
            raise KeyError("Không tìm thấy layer 'output_heads' trong checkpoint.")

        print(
            f"[*] Detected Config from Checkpoint: State_Size={state_size}, HP={num_honeypots}, Max_Out={max_nodes_out}")

        policy_net = MultiHeadDQN(state_size, num_honeypots, max_nodes_out).to(DEVICE)
        policy_net.load_state_dict(state_dict)
        policy_net.eval()
        GLOBAL_MODELS["policy_net"] = policy_net
        print(f"[+] DQN Policy Net loaded.")

    except KeyError as e:
        print(f"[!] Lỗi Key trong checkpoint: {e}")
        print("Hãy kiểm tra lại xem model được train bằng class MultiHeadDQN trong model_utils.py hay không.")
        exit()
    except Exception as e:
        print(f"[!] Lỗi load DQN: {e}")
        exit()


# ======================================================
# 3. CORE LOGIC: FETCH -> RECONSTRUCT -> PREDICT -> DEPLOY
# ======================================================

def fetch_and_reconstruct_graph():
    print(f"[*] Đang tải đồ thị từ: {GRAPH_URL} ...")
    try:
        response = requests.get(GRAPH_URL, timeout=5)
        response.raise_for_status()
        graph_data = pickle.load(io.BytesIO(response.content))
    except Exception as e:
        print(f"[!] Lỗi tải graph: {e}")
        return None, None, None, None

    if len(graph_data.nodes) == 0:
        return None, None, None, None

    # Trích xuất edges và attributes
    raw_edges = []
    for u, v, data in graph_data.edges(data=True):
        edge_data = data.copy()
        if 'user' not in edge_data: edge_data['user'] = 0.0
        if 'root' not in edge_data: edge_data['root'] = 0.0
        raw_edges.append((u, v, edge_data))

    # Xử lý Timestamp & DGL
    sorted_edges_list = add_timestamp_to_edges(raw_edges)

    try:
        g_dgl, node_order, node_map = build_dgl(
            nx_graph=graph_data,
            sorted_edges=sorted_edges_list,
            node_feat_keys=['state', 'priority'],
            edge_feat_keys=['user', 'root']
        )
    except KeyError as e:
        print(f"[!] Thiếu thuộc tính trong graph: {e}")
        return None, None, None, None

    return graph_data, g_dgl, node_map, node_order


def run_inference_pipeline():
    print("\n" + "=" * 40)
    print(">>> BẮT ĐẦU QUY TRÌNH XỬ LÝ UPDATE <<<")

    # 1. Reconstruct Graph
    G_original, g_dgl, node_map, node_order = fetch_and_reconstruct_graph()
    if G_original is None:
        print("[!] Không thể tái tạo đồ thị. Hủy bỏ.")
        return

    # 2. Chuẩn bị Environment
    # Lấy features
    nfeats = g_dgl.ndata['h']
    efeats = g_dgl.edata['h']

    # Xử lý features (Cần khớp logic train: build_batch_tensor)
    MAX_N_FEATURES = 2
    MAX_E_FEATURES = 2
    nfeats_tensor = build_batch_tensor(nfeats, MAX_N_FEATURES).to(DEVICE)
    efeats_tensor = build_batch_tensor(efeats, MAX_E_FEATURES).to(DEVICE)

    # Khởi tạo Env mới với graph mới
    # Lưu ý: max_nodes, max_edges cần khớp với config lúc train để đảm bảo tính nhất quán
    # Nếu đồ thị thực tế lớn hơn max_nodes lúc train, Env sẽ crash hoặc hoạt động sai.
    # Ta giả định max_nodes đủ lớn hoặc Env tự xử lý cắt/pad.
    env = NetworkEnv(
        G_new=deepcopy(G_original),
        attack_fn=global_weighted_random_attack,  # Dummy fn, không quan trọng lúc inference
        g_dgl=g_dgl.to(DEVICE),
        encoder=GLOBAL_MODELS["encoder"],
        original_node_features=nfeats_tensor,
        original_edge_features=efeats_tensor,
        node_map=node_map,
        num_honeypots=GLOBAL_MODELS["policy_net"].num_honeypots_N,
        max_nodes=24,  # Cần khớp với lúc train (Original.py: max_nodes=24)
        max_edges=128
    )

    # 3. Predict (Chạy DQN)
    state = env.reset().to(DEVICE)
    real_node_count = len(env.nodes)
    N_honeypots = env.num_honeypots
    policy_net = GLOBAL_MODELS["policy_net"]

    target_node_names = []

    with torch.no_grad():
        state_tensor = state.flatten().unsqueeze(0).to(DEVICE)  # [1, State_Size]
        q_values_all = policy_net(state_tensor)  # [1, N, M]
        q_values_N_M = q_values_all.squeeze(0)  # [N, M]

        # Masking & Selection
        try:
            action = select_action_multi_head(q_values_N_M, real_node_count)
        except NameError:
            # Fallback nếu thiếu import
            q_values_N_M[:, real_node_count:] = -float('inf')
            action = np.zeros((N_honeypots, q_values_N_M.shape[1]))
            for i in range(N_honeypots):
                idx = torch.argmax(q_values_N_M[i]).item()
                action[i, idx] = 1.0

    # 4. Decode Action
    print(f"[*] Kết quả dự đoán cho {N_honeypots} honeypots:")
    for i in range(N_honeypots):
        idx = np.argmax(action[i])
        if idx < len(env.nodes):
            node_name = env.nodes[idx]
            target_node_names.append(node_name)
            print(f"   -> HP {i + 1}: Node '{node_name}'")
        else:
            print(f"   -> HP {i + 1}: Invalid/Pad")

    # 5. Send Deploy Command
    if target_node_names:
        push_honeypot_deployment(target_node_names)
    else:
        print("[!] Không tìm thấy node hợp lệ để deploy.")

    print(">>> HOÀN TẤT QUY TRÌNH <<<")
    print("=" * 40 + "\n")


def push_honeypot_deployment(honeypot_nodes):
    payload = {"targets": honeypot_nodes}
    print(f"[*] Đang gửi lệnh deploy tới Mininet ({DEPLOY_URL})...")
    try:
        response = requests.post(DEPLOY_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[+] SERVER MININET PHẢN HỒI: {response.json()}")
        else:
            print(f"[!] SERVER LỖI: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[!] KẾT NỐI THẤT BẠI: {e}")


# ======================================================
# 4. HTTP SERVER LISTENER (ĐÃ SỬA)
# ======================================================

class AIRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        """Xử lý request POST từ Mininet"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
        except Exception:
            post_data = ""

        print(f"\n[SERVER] Nhận POST từ {self.client_address}")
        print(f"[SERVER] Payload: '{post_data}'")

        # --- [MỚI] XỬ LÝ TEST CONNECTION ---
        if "TEST CONNECTION" in post_data:
            print(f"[SERVER] >>> Nhận lệnh ping kiểm tra. Trả lời OK.")
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"AI SERVER CONNECTED SUCCESSFULLY!")
            return # Quan trọng: Return ngay để không chạy inference
        # -----------------------------------

        if "UPDATE PLEASE!" in post_data or self.path == "/trigger_update":
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK. Starting Inference Pipeline...")

            # Chạy AI ở luồng riêng
            t = threading.Thread(target=run_inference_pipeline)
            t.start()
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Unknown command")

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"AI Agent Server is RUNNING.")


def start_server():
    # Cho phép reuse port ngay lập tức để không bị lỗi "Address already in use" khi restart nhanh
    socketserver.TCPServer.allow_reuse_address = True

    # Khởi tạo server
    httpd = socketserver.TCPServer(("", MY_SERVER_PORT), AIRequestHandler)

    print(f"\n[SERVER] AI Agent đang lắng nghe tại port {MY_SERVER_PORT}...")
    print("[INFO] Nhấn Ctrl+C để dừng server.")

    try:
        # Chạy server (vòng lặp vô tận)
        httpd.serve_forever()
    except KeyboardInterrupt:
        # Bắt sự kiện nhấn Ctrl+C
        pass
    finally:
        # Dọn dẹp và đóng port dù có lỗi gì xảy ra
        print("\n[STOP] Đang đóng kết nối và giải phóng Port...")
        httpd.server_close()
        print("[STOP] Server đã tắt hoàn toàn.")


# ======================================================
# 5. MAIN
# ======================================================
if __name__ == "__main__":
    # 1. Load Model (DQN + GNN)
    load_static_models()

    # 2. Start Server
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n[!] Dừng server.")