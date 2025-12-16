# Generated from: Original.ipynb
# Converted at: 2025-12-15T15:58:39.780Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# --- CUDA DEVICE SETUP ---
import torch

from utils.basic_utils import add_timestamp_to_edges

# Tự động chọn GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Khi cần chuyển dữ liệu: 
# tensor = tensor.to(device)
# model = model.to(device)
# g = g.to(device)

import networkx as nx
import dgl
import torch
import numpy as np
import pickle
import requests
import io
from utils.basic_utils import * # Giả sử bạn đã có file này

# --- CẤU HÌNH KẾT NỐI ---
MININET_IP = "192.168.2.9"
PORT = 8000
FILENAME = "network_state.pth"
URL = f"http://{MININET_IP}:{PORT}/{FILENAME}"

# --- 1. TẢI VÀ RECONSTRUCT DỮ LIỆU ĐỒ THỊ ---

def fetch_and_load_graph(url):
    print(f"[*] Dang tai do thi tu: {url} ...")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Check lỗi HTTP

        # Load graph từ bytes nhận được (không cần lưu file ra đĩa nếu không muốn)
        # Sử dụng io.BytesIO để đọc dữ liệu binary trực tiếp
        graph_data = pickle.load(io.BytesIO(response.content))
        print("[+] Download va Reconstruct thanh cong!")
        return graph_data
    except Exception as e:
        print(f"[!] Loi khi tai graph: {e}")
        # Trả về đồ thị rỗng hoặc thoát chương trình tùy logic
        return nx.DiGraph()

# >>>> THAY THẾ PHẦN KHỞI TẠO CỨNG <<<<
# Load đồ thị từ máy Mininet
G_original = fetch_and_load_graph(URL)

if len(G_original.nodes) == 0:
    print("Do thi rong hoac loi ket noi. Dung chuong trinh.")
    exit()

# Trích xuất raw_edges từ đồ thị đã load
# Lý do: Code phía sau của bạn (add_timestamp_to_edges) cần đầu vào là list các tuple cạnh
# Cấu trúc trích xuất: [(u, v, {attr}), ...]
raw_edges = []
for u, v, data in G_original.edges(data=True):
    # Đảm bảo các thuộc tính user/root tồn tại, nếu không gán mặc định
    edge_data = data.copy()
    if 'user' not in edge_data: edge_data['user'] = 0.0
    if 'root' not in edge_data: edge_data['root'] = 0.0
    raw_edges.append((u, v, edge_data))

print(f"[*] Da trich xuat {len(raw_edges)} canh tu do thi nhap ve.")

# --- 2. XỬ LÝ & CONVERT SANG DGL ---
# (Phần này giữ nguyên logic của bạn, chỉ đảm bảo input raw_edges chuẩn)

# Bước A: Xử lý timestamp và sort
# Lưu ý: raw_edges trích xuất từ NX thường không có timestamp trừ khi bạn đã lưu nó vào attr trước đó.
# Hàm add_timestamp_to_edges sẽ tự thêm timestamp giả lập nếu logic bên trong nó làm vậy.
sorted_edges_list = add_timestamp_to_edges(raw_edges)

# Bước B: Tạo DGL Graph chuẩn
# Lưu ý: 'state' và 'priority' phải có trong node attributes của G_original (đã được lưu trong file .pth)
# 'user' và 'root' phải có trong edge attributes
try:
    g_dgl, node_order, node_map = build_dgl(
        nx_graph=G_original,
        sorted_edges=sorted_edges_list,
        node_feat_keys=['state', 'priority'], # Lấy từ node attributes
        edge_feat_keys=['user', 'root']       # Lấy từ edge attributes
    )
except KeyError as e:
    print(f"[!] Loi thieu thuoc tinh trong do thi: {e}")
    print("Kiem tra lai file export xem da co du thuoc tinh state, priority, user, root chua.")
    exit()

# Lấy lại feature từ DGL
nfeats = g_dgl.ndata['h']
efeats = g_dgl.edata['h']

# Tạo static priority (Cột số 1 của nfeats: 'priority')
static_priority_features = nfeats[:, 1].unsqueeze(1)

# --- 3. IN KẾT QUẢ ---
print("\n" + "="*30)
print("--- KẾT QUẢ TÁI TẠO ĐỒ THỊ ---")
print(f"Đã tạo g_dgl: {g_dgl.num_nodes()} nodes, {g_dgl.num_edges()} edges.")
print(f"Shape Đặc trưng Node (nfeats): {nfeats.shape}")
print(f"Shape Đặc trưng Edge (efeats): {efeats.shape}")

print(f"\nNode Order g_dgl (Example top 5):\n{node_order[:5]} ...")
print(f"Node map:",node_map)

print("\n" + "="*30 + "\n")
print(f"Tổng số node: {len(node_order)}")

# ... (Phần code trước đó của bạn) ...

# --- 4. KIỂM TRA CHI TIẾT FEATURE (MỚI THÊM) ---
print("\n" + "="*30)
print(">>> CHECK CHI TIẾT DỮ LIỆU BÊN TRONG TENSOR")

# -------------------------------------------------
# A. KIỂM TRA NODE FEATURES
# Giả định nfeats cấu trúc: [State, Priority] (dựa trên node_feat_keys)
# -------------------------------------------------
print(f"\n[1] Node Features (Top 5 nodes):")
print(f"{'DGL_ID':<8} | {'ORIG_ID':<15} | {'State':<10} | {'Priority':<10}")
print("-" * 55)

num_nodes_to_check = min(10, g_dgl.num_nodes()) # Kiểm tra tối đa 10 node

for i in range(num_nodes_to_check):
    # Lấy ID gốc từ danh sách node_order
    orig_id = node_order[i]

    # Lấy feature, chuyển về numpy cho dễ nhìn, làm tròn 2 số
    feats = nfeats[i].detach().numpy()
    state_val = feats[0]
    prio_val = feats[1]

    print(f"{i:<8} | {str(orig_id):<15} | {state_val:<10.2f} | {prio_val:<10.2f}")

# -------------------------------------------------
# B. KIỂM TRA EDGE FEATURES
# Giả định efeats cấu trúc: [User, Root] (dựa trên edge_feat_keys)
# -------------------------------------------------
print(f"\n[2] Edge Features (Top 5 edges):")
print(f"{'Edge_IDX':<8} | {'Link (Orig)':<20} | {'User':<10} | {'Root':<10}")
print("-" * 60)

num_edges_to_check = min(10, g_dgl.num_edges())
src_ids, dst_ids = g_dgl.edges() # Lấy danh sách ID nguồn/đích trong DGL

for i in range(num_edges_to_check):
    # Lấy ID node trong DGL
    u_dgl = src_ids[i].item()
    v_dgl = dst_ids[i].item()

    # Map về tên gốc
    u_orig = node_order[u_dgl]
    v_orig = node_order[v_dgl]
    link_str = f"{u_orig} -> {v_orig}"

    # Lấy feature
    edge_ft = efeats[i].detach().numpy()
    user_val = edge_ft[0]
    root_val = edge_ft[1]

    print(f"{i:<8} | {link_str:<20} | {user_val:<10.2f} | {root_val:<10.2f}")

print("\n" + "="*30)

# ## Sổ tay Tác nhân RL (Đã sửa lỗi)
# 
# Sổ tay này tải môi trường, cấu hình GNN, và trọng số GNN đã huấn luyện (từ `Graph_Canonical_50.ipynb`) để chạy Tác nhân DQN.


import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import dgl
import os
from copy import deepcopy
from collections import defaultdict
import random
import yaml

# Giả định các tệp utils này tồn tại trong thư mục của bạn
from utils.attack_algo_utils import *
from utils.graph_utils import *
from utils.basic_utils import *
from utils.model_utils import *

print("--- CHUẨN BỊ MÔI TRƯỜNG ---")

# --- CHỌN PHIÊN BẢN THÍ NGHIỆM ---
experiment_id = 1
BASE_PATH = f'graphs/{experiment_id}'
print(f"Đang chạy thí nghiệm ID: {experiment_id} tại đường dẫn: {BASE_PATH}")

# ======================================================================
# 2. ĐỊNH NGHĨA HÀM TIỀN XỬ LÝ (PHẢI GIỐNG HỆT FILE HUẤN LUYỆN)
# ======================================================================
# (Chúng ta cần các hằng số và hàm này để xử lý

MAX_N_FEATURES = 2
MAX_E_FEATURES = 2

def build_batch_tensor(feats, max_dim):
    return feats[:, :max_dim]


# ======================================================================
# 3. TẢI MODEL GNN ĐÃ HUẤN LUYỆN (PHẦN ĐÃ SỬA)
# ======================================================================
print("\n--- Đang tải cấu hình và trọng số GNN ---")

MODEL_STATE_PATH = f"graphs/dgi_model_state_dict.pth"
CONFIG_FILE_PATH = f"graphs/model_config.yaml"

try:
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    # --- 3.2: Khởi tạo mô hình rỗng TỪ CẤU HÌNH ĐÃ TẢI ---
    encoder = EGraphSAGE(
        config['NDIM_IN'],       # 50
        config['EDIM'],          # 50
        config['N_HIDDEN'],
        config['N_OUT'],
        config['N_LAYERS'],
        F.leaky_relu,
        device,
    )

    dgi_model_to_load = DGI(encoder)

    # --- 3.3: Tải trọng số đã lưu ---
    dgi_model_to_load.load_state_dict(torch.load(MODEL_STATE_PATH, weights_only=False))

    # --- 3.4: Trích xuất encoder bạn cần ---
    trained_encoder = dgi_model_to_load.encoder
    trained_encoder.to(device)
    trained_encoder.eval() # Chuyển sang chế độ dự đoán

    print(f"[THÀNH CÔNG] Đã tải và trích xuất GNN encoder.")

except Exception as e:
    print(f"\n[LỖI] Có lỗi xảy ra khi tải model: {e}")
    trained_encoder = None

def train_dqn(env, num_episodes, device=None, batch_size=10, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):

    global best_checkpoint, best_episode

    # --- Reset env lấy state ban đầu ---
    state = env.reset().to(device)

    # [FIX] Tính state_size dựa trên kích thước thực tế của dữ liệu
    # State lúc này bao gồm cả Node và Edge embeddings
    # Dù state là 1D hay 2D, numel() sẽ trả về tổng số phần tử (ví dụ 1440)
    state_size = state.numel()

    # Lấy embedding_dim (chỉ để tham khảo hoặc log)
    if len(state.shape) > 1:
        embedding_dim = state.shape[-1]
    else:
        # Nếu state đã flatten rồi thì cần lấy từ env hoặc config
        embedding_dim = env.embedding_dim

    num_honeypots_N = env.num_honeypots
    # Lưu ý: max_nodes_M chỉ dùng cho đầu ra action, không dùng tính state_size nữa
    max_nodes_M = env.max_nodes

    print(f"Embedding Dim: {embedding_dim}")
    print(f"Actual State Shape from Env: {state.shape}")
    print(f"Calculated State Size (Input to DQN): {state_size}")
    print(f"Action Output Shape: ({num_honeypots_N}, {max_nodes_M})")

    # --- Khởi tạo mô hình với state_size đã sửa ---
    policy_net = MultiHeadDQN(state_size, num_honeypots_N, max_nodes_M).to(device)
    target_net = MultiHeadDQN(state_size, num_honeypots_N, max_nodes_M).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = epsilon_start
    total_reward = 0
    dsp = 0
    best_dsp = 0

    interval_check = max(1, num_episodes // 10)
    interval_save = max(1, num_episodes // 5)
    best_checkpoint = None
    best_episode = 0

    for episode in range(1, num_episodes + 1):
        if episode > 1:
            state = env.reset().to(device)

        done = False
        current_num_nodes = len(env.nodes)

        while not done:
            # --- Chọn action ---
            if random.random() < epsilon:
                # Random trong vùng node thật
                action = sample_valid_action_matrix(num_honeypots_N, current_num_nodes)

                # Pad action về đúng kích thước (N, M)
                full_action = np.zeros((num_honeypots_N, max_nodes_M), dtype=np.float32)
                full_action[:, :current_num_nodes] = action
                action = full_action
            else:
                # Policy
                with torch.no_grad():
                    # Flatten state để đưa vào mạng
                    # state có thể là (60, 24) hoặc (1440,), flatten sẽ ra (1440,)
                    # unsqueeze(0) -> (1, 1440)
                    state_tensor = state.flatten().unsqueeze(0).to(device)

                    q_values = policy_net(state_tensor)

                    # Masking
                    if current_num_nodes < q_values.shape[-1]:
                        q_values[:, :, current_num_nodes:] = -float('inf')

                    action = select_action_multi_head(q_values.squeeze(0), current_num_nodes)

            # --- Step ---
            next_state, reward, done, path, captured = env.step(action)
            next_state = next_state.to(device)

            # --- Push Buffer ---
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            if reward == 1:
                dsp += 1

            # --- Training ---
            if len(replay_buffer) >= batch_size:
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)

                states_batch = states_batch.to(device)
                next_states_batch = next_states_batch.to(device)
                actions_batch = actions_batch.to(device)
                rewards_batch = rewards_batch.to(device)
                dones_batch = dones_batch.to(device)

                # Flatten batch: [B, ...] -> [B, state_size]
                states_flat = states_batch.flatten(start_dim=1)
                next_states_flat = next_states_batch.flatten(start_dim=1)

                # 1. Current Q
                q_values_all = policy_net(states_flat)
                current_q_per_head = q_values_all * actions_batch
                current_q_total = current_q_per_head.sum(dim=2).sum(dim=1)

                # 2. Target Q
                with torch.no_grad():
                    next_q_values_all = target_net(next_states_flat)

                    # Masking Target
                    if current_num_nodes < next_q_values_all.shape[-1]:
                        next_q_values_all[..., current_num_nodes:] = -float('inf')

                    next_q_values_per_head = next_q_values_all.max(dim=2)[0]
                    next_q_total = next_q_values_per_head.sum(dim=1)
                    targets = rewards_batch.squeeze(1) + (1 - dones_batch.squeeze(1)) * gamma * next_q_total

                # 3. Loss & Step
                loss = nn.MSELoss()(current_q_total, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # --- Update Target Net ---
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # --- Logging ---
        if episode % interval_check == 0:
            placement = []
            for i in range(num_honeypots_N):
                try:
                    node_idx = np.argmax(action[i])
                    if node_idx < len(env.nodes):
                        node_name = env.nodes[node_idx]
                    else:
                        node_name = "INVALID_PADDING"
                except:
                    node_name = "ERROR"
                placement.append(f"Honeypot {i} -> {node_name}\n")

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}, DSP: {dsp/interval_check*100:.3f}%")
            print("".join(placement))
            print(f"Path: {path}")
            total_reward = 0
            if dsp > best_dsp:
                best_dsp = dsp
                best_episode = episode
                best_checkpoint = {
                    'policy_net_state_dict': deepcopy(policy_net.state_dict()),
                    'target_net_state_dict': deepcopy(target_net.state_dict()),
                    'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                }
            dsp = 0

        # --- Save Model ---
        if (episode + 1) % interval_save == 0 and best_checkpoint is not None:
            os.makedirs('./Saved_Model', exist_ok=True)
            path_save = f'./Saved_Model/dqn_model.pth'
            torch.save({
                'policy_net_state_dict': best_checkpoint['policy_net_state_dict'],
                'target_net_state_dict': best_checkpoint['target_net_state_dict'],
                'optimizer_state_dict': best_checkpoint['optimizer_state_dict'],
                'episode': best_episode},
                path_save)
            print(f'Saved model with best DSP {best_dsp} at episode {best_episode} to {path_save}')
            best_dsp = 0
            best_episode = 0
            best_checkpoint = None

    return policy_net

# ======================================================================
# 5. KHỞI TẠO MÔI TRƯỜNG & HUẤN LUYỆN (PHẦN ĐÃ SỬA)
# ======================================================================
print("\n--- Khởi tạo Môi trường RL ---")

# Initialize environment and train
algo = global_weighted_random_attack
G_new_env = deepcopy(G_original)

# --- SỬA LỖI LOGIC QUAN TRỌNG ---
# Chúng ta phải xử lý (pad + norm) các đặc trưng GỐC
# để chúng khớp với đầu vào 50-dim mà encoder mong đợi.
nfeats_tensor = build_batch_tensor(nfeats, MAX_N_FEATURES)
efeats_tensor = build_batch_tensor(efeats, MAX_E_FEATURES)

nfeats_tensor = nfeats_tensor.to(device)
efeats_tensor = efeats_tensor.to(device)
print(f"Đã xử lý đặc trưng node: {nfeats_tensor.shape}")
print(f"Đã xử lý đặc trưng cạnh: {efeats_tensor.shape}")
target_priority = 2 

# 2. Dùng list comprehension để lọc các node từ G_new_env
#    Chúng ta lặp qua (node, data) trong G.nodes(data=True)
#    và chỉ giữ lại 'node' nếu 'priority' có trong 'data' VÀ data['priority'] == 2
env = NetworkEnv(
    G_new=G_new_env,
    attack_fn=algo,
    g_dgl=g_dgl.to(device), # Sử dụng g_dgl (DGL graph gốc)
    encoder=trained_encoder, # Encoder đã huấn luyện (mong đợi 50-dim)
    
    # --- SỬA LỖI: Truyền vào các đặc trưng ĐÃ XỬ LÝ (50-dim) ---
    original_node_features=nfeats_tensor,
    original_edge_features=efeats_tensor,
    # ---------------------------------------------------------
    
    node_map=node_map,
    num_honeypots = 3,

    max_nodes = 24,
    max_edges = 128)

# --- 3. HUẤN LUYỆN ---
num_episode = 5678
if not os.path.exists('./Saved_Model'):
    os.makedirs('./Saved_Model')

model = train_dqn(env, num_episode,device)

# ========================================================
# THÊM ĐOẠN CODE DƯỚI ĐÂY ĐỂ LƯU TRỌNG SỐ CHO BƯỚC SAU
# ========================================================
OLD_MODEL_PATH = './graphs/1/dqn_model.pth'

print(f"Đang lưu trọng số mô hình cũ vào: {OLD_MODEL_PATH} ...")

# Chỉ cần lưu state_dict của policy_net là đủ cho việc chuyển giao tri thức
torch.save({
    'policy_net_state_dict': model.state_dict(),
    'num_nodes': env.num_nodes,  # Lưu thêm số lượng node (M) để tiện kiểm tra
    'num_honeypots': env.num_honeypots
}, OLD_MODEL_PATH)

print("[HOÀN TẤT] Đã lưu file. Hãy dùng đường dẫn này cho file học tiệm tiến.")

evaluate_model(model, env, 2000, device)

import json
import requests
import numpy as np
import torch
# Đảm bảo đã import select_action_multi_head từ utils hoặc cell trước đó
# from model_utils import select_action_multi_head

# --- 1. CẤU HÌNH KẾT NỐI ---
DEPLOY_URL = f"http://{MININET_IP}:{PORT}/deploy_honeypots"

# --- 2. HÀM GỬI PAYLOAD ---
def push_honeypot_deployment(honeypot_nodes):
    """
    Gửi danh sách các node (VD: ['web', 'h1']) sang Mininet để deploy.
    """
    payload = {
        "targets": honeypot_nodes
    }

    print(f"[*] Đang gửi lệnh deploy tới: {DEPLOY_URL}")
    print(f"[*] Payload: {json.dumps(payload)}")

    try:
        response = requests.post(DEPLOY_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[+] THÀNH CÔNG: {response.json()}")
        else:
            print(f"[!] LỖI SERVER ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"[!] LỖI KẾT NỐI: {e}")

# --- 3. HÀM DỰ ĐOÁN & DEPLOY (LOGIC TỪ EVALUATE_MODEL) ---
def predict_and_deploy_live(env, model, device=None):
    print("\n" + "="*40)
    print("--- LIVE PREDICTION & DEPLOYMENT ---")
    print("="*40)

    # 1. Setup Device & Model
    device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval() # Chế độ đánh giá (không dropout/batchnorm update)

    # 2. Lấy trạng thái mạng hiện tại
    state = env.reset().to(device)

    # Lấy số lượng node thực tế để Masking
    real_node_count = len(env.nodes)
    N_honeypots = env.num_honeypots

    # 3. Chạy Model để lấy Action
    with torch.no_grad():
        # Flatten state [M, D] -> [1, M*D] (Thêm batch dimension = 1)
        state_tensor = state.flatten().unsqueeze(0).to(device)

        # Forward pass: Lấy Q-values [1, N, M]
        q_values_all = model(state_tensor)

        # Squeeze(0) -> [N, M] (Bỏ batch dimension)
        q_values_N_M = q_values_all.squeeze(0)

        # Masking & Selection (Giống hệt evaluate_model)
        # Hàm này trả về ma trận action (thường là one-hot hoặc distribution)
        # Nếu không có hàm này, ta có thể dùng argmax trực tiếp trên q_values đã mask
        try:
            action = select_action_multi_head(q_values_N_M, real_node_count)
        except NameError:
            print("[!] Cảnh báo: Không tìm thấy 'select_action_multi_head'. Tự thực hiện Masking & Argmax...")
            # Fallback logic nếu thiếu hàm utils:
            q_values_N_M[:, real_node_count:] = -float('inf')
            # Tạo one-hot giả lập để code phía dưới chạy được
            action = np.zeros((N_honeypots, q_values_N_M.shape[1]))
            for i in range(N_honeypots):
                idx = torch.argmax(q_values_N_M[i]).item()
                action[i, idx] = 1.0

    # 4. Giải mã Action thành Tên Node
    target_node_names = []
    print(f"-> Số lượng Honeypot cần đặt: {N_honeypots}")

    for i in range(N_honeypots):
        # Lấy index có giá trị lớn nhất từ action của head thứ i
        idx = np.argmax(action[i])

        # Mapping index -> Tên node trong Env
        if idx < len(env.nodes):
            node_name = env.nodes[idx]
            target_node_names.append(node_name)
            print(f"   + Honeypot {i+1}: Index {idx} -> Node '{node_name}'")
        else:
            print(f"   + Honeypot {i+1}: Index {idx} -> PAD/INVALID (Bỏ qua)")

    # 5. Gửi sang Mininet
    if target_node_names:
        push_honeypot_deployment(target_node_names)
    else:
        print("[!] Model không chọn được node hợp lệ nào.")

# --- 4. THỰC THI ---

# Đảm bảo các biến 'env', 'model', 'device' đã tồn tại từ các cell trước
if 'env' in locals() and 'model' in locals():
    # Nếu chưa có device, tự define
    if 'device' not in locals():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    predict_and_deploy_live(env, model, device)
else:
    print("[!] Lỗi: Thiếu biến 'env' hoặc 'model'. Hãy chạy Train/Load model trước.")