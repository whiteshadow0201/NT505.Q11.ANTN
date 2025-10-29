# utils2.py
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

def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

def inverse_cantor(z):
    w = int((math.sqrt(8*z + 1) - 1) // 2)
    t = (w * (w + 1)) // 2
    k2 = z - t
    k1 = w - k2
    return k1, k2

def action_to_index(action, num_nodes):
    first = np.argmax(action[0])
    second = np.argmax(action[1])
    if first == second:
        raise ValueError("Action nodes must be distinct")
    return cantor_pairing(first, second)

def index_to_action(index, num_nodes):
    first, second = inverse_cantor(index)
    if first == second or first >= num_nodes or second >= num_nodes:
        raise ValueError("Invalid index or node count")
    action = np.zeros((2, num_nodes), dtype=np.float32)
    action[0, first] = 1
    action[1, second] = 1
    return action


def is_valid_index(index, num_nodes):
    first, second = inverse_cantor(index)
    return first != second and first < num_nodes and second < num_nodes


def sample_valid_index(action_space_size, num_nodes, exploration_counter, min_explorations=10):
    """
    Samples a random valid index, ensuring each valid index is explored at least
    min_explorations times before exploration is considered complete.

    Args:
        action_space_size (int): The size of the action space
        num_nodes (int): The total number of nodes
        exploration_counter (defaultdict): Counter tracking explorations per index
        min_explorations (int): Minimum number of times each valid index must be explored

    Returns:
        int: A randomly selected valid index with less than min_explorations
    """
    valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, num_nodes)]

    if not valid_indices:
        raise ValueError("No valid indices found")

    # Check if all valid indices have been explored at least min_explorations times
    under_explored = [idx for idx in valid_indices if exploration_counter[idx] < min_explorations]

    if under_explored:
        # Prioritize under-explored indices
        selected_idx = random.choice(under_explored)
        exploration_counter[selected_idx] += 1
        return selected_idx
    else:
        # Switch to pure random sampling after all indices meet the threshold
        selected_idx = random.choice(valid_indices)
        exploration_counter[selected_idx] += 1
        return selected_idx


def sample_exploration_index(new_action_space_size, new_num_honeypot_nodes, old_num_honeypot_nodes, exploration_counter,min_explorations=10):
    """
    Samples a random index from possible new node pairs in an expanded action space.
    Uses Cantor pairing to map node pairs (i, j) to unique indices, considering only
    pairs where at least one node is new (i.e., i or j >= old_num_honeypot_nodes). Tracks exploration
    counts and continues until each valid pair is explored at least min_explorations times.

    Args:
        new_action_space_size (int): The size of the new action space, limiting valid indices
        new_num_honeypot_nodes (int): The number of honeypot-eligible nodes after expansion
        old_num_honeypot_nodes (int): The number of honeypot-eligible nodes before expansion
        exploration_counter (defaultdict): Counter tracking explorations per index
        min_explorations (int): Minimum number of times each valid pair must be explored

    Returns:
        int: A randomly selected index from valid new node pairs with less than min_explorations

    Raises:
        ValueError: If no valid new-node-related actions are found
    """
    new_node_indices = []
    exploration_done = False
    # Generate all valid indices for new node pairs
    for i in range(new_num_honeypot_nodes):
        for j in range(new_num_honeypot_nodes):
            if i != j:
                if i >= old_num_honeypot_nodes or j >= old_num_honeypot_nodes:
                    idx = cantor_pairing(i, j)
                    if idx >= new_action_space_size:
                        continue
                    new_node_indices.append(idx)

    if not new_node_indices:
        raise ValueError("No valid new-node-related actions found")

    # Find indices that have been explored less than min_explorations times
    under_explored = [idx for idx in new_node_indices if exploration_counter[idx] < min_explorations]

    if under_explored:
        selected_idx = random.choice(under_explored)
        exploration_counter[selected_idx] += 1
        return selected_idx, exploration_done
    else:
        selected_idx = random.choice(new_node_indices)
        exploration_done = True
        return selected_idx, exploration_done

class DQN(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return (
#             np.array(state),
#             np.array(action),
#             np.array(reward),
#             np.array(next_state),
#             np.array(done)
#         )
#
#     def __len__(self):
#         return len(self.buffer)

import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_embedding, action, reward, next_state_embedding, done):
        """
        Lưu trữ kinh nghiệm.
        state_embedding và next_state_embedding giờ là các PyTorch Tensor.
        """
        # Detach tensor khỏi graph và chuyển về CPU
        # để giải phóng bộ nhớ GPU và tránh lỗi graph
        state_embedding = state_embedding.detach().cpu()
        next_state_embedding = next_state_embedding.detach().cpu()

        self.buffer.append((state_embedding, action, reward, next_state_embedding, done))

    def sample(self, batch_size):
        """
        Lấy một batch mẫu từ buffer.
        """
        # 1. Lấy một batch các tuple kinh nghiệm
        batch = random.sample(self.buffer, batch_size)

        # 2. Giải nén batch
        # state, action, reward, next_state, done giờ là các TUPLE
        # ví dụ: state là (tensor_emb_1, tensor_emb_2, ...)
        state, action, reward, next_state, done = zip(*batch)

        # 3. Gom các tensor embedding lại thành một batch
        # torch.stack sẽ tạo một chiều mới (batch_size) ở đầu
        state_batch = torch.stack(state)
        next_state_batch = torch.stack(next_state)

        # 4. Chuyển đổi các thành phần khác sang Tensor
        # (Giả sử action là numpy, reward/done là số)
        action_batch = torch.tensor(np.array(action), dtype=torch.float32)
        reward_batch = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done_batch = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch
        )

    def __len__(self):
        return len(self.buffer)

# # 1. Environment Setup
# # Environment class
# class NetworkEnv:
#     def __init__(self, G_new, attack_fn, goal="Data Server"):
#         self.G_new = G_new
#         self.attack_fn = attack_fn
#         self.goal = goal
#         self.nodes = [n for n in G_new.nodes if n not in ["Attacker"]]
#         self.num_nodes = len(self.nodes)
#         self.state = np.zeros(self.num_nodes, dtype=np.float32)
#         self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
#
#         print("node_to_idx:", self.node_to_idx)
#         # Include all nodes as honeypot-eligible (no exclusion of goal)
#         self.honeypot_nodes = self.nodes  # Changed to include "Data Server"
#         self.num_honeypot_nodes = len(self.honeypot_nodes)
#
#     def reset(self):
#         self.state = np.zeros(self.num_nodes, dtype=np.float32)
#         return self.state
#
#     def step(self, action):
#         honeypot_nodes = []
#         G = deepcopy(self.G_new)
#
#         for i in range(2):
#             node_idx = np.argmax(action[i])
#             if action[i, node_idx] == 0:
#                 node_idx = random.randint(0, self.num_honeypot_nodes - 1)
#             node = self.honeypot_nodes[node_idx]
#             honeypot = f"Honeypot {{{node}}}"
#             honeypot_nodes.append(honeypot)
#             G.add_node(honeypot)
#             G.add_edge(node, honeypot, user=0.8, root=0.8)
#             self.graph = deepcopy(G)
#
#         path, captured = self.attack_fn(G, honeypot_nodes, self.goal)
#
#         new_state = np.zeros(self.num_nodes, dtype=np.float32)
#         reward = 0
#         done = False
#
#         if any(h in captured for h in honeypot_nodes):
#             reward = 1
#             done = True
#         elif self.goal in captured:
#             reward = -1
#             done = True
#             new_state[self.node_to_idx[self.goal]] = 1
#
#         for node in captured:
#             if node in self.node_to_idx:
#                 new_state[self.node_to_idx[node]] = 1
#
#         self.state = new_state
#         return new_state, reward, done, path, captured
#
#     def get_action_space_size(self):
#         return 2 * self.num_honeypot_nodes ** 2 + self.num_honeypot_nodes
#

# (Giả sử các class GNN, DGI... đã được định nghĩa ở trên)

class NetworkEnv:
    def __init__(self, G_new, attack_fn, g_dgl, encoder,
                 original_node_features, original_edge_features,
                 node_map,  # <-- Thay đổi: Nhận trực tiếp node_map
                 goal="Data Server"):

        # --- Phần GNN ---
        self.g_dgl = g_dgl
        self.encoder = encoder
        self.original_node_features = original_node_features.clone()
        self.original_edge_features = original_edge_features.clone()
        self.encoder.eval()  # Chuyển sang chế độ dự đoán

        # --- Phần Môi trường và Phát triển nông thôn ---
        self.G_new = G_new
        self.attack_fn = attack_fn
        self.goal = goal

        # --- THAY ĐỔI: Sử dụng trực tiếp node_map ---
        # 1. Sử dụng trực tiếp bản đồ node-to-index đã được tải
        self.node_to_idx = node_map

        # 2. Lấy danh sách nodes từ các keys của bản đồ
        self.nodes = list(node_map.keys())

        # 3. Lấy tổng số node
        self.num_nodes = len(self.nodes)
        # --- Hết thay đổi ---

        print("node_to_idx đã được tải:", self.node_to_idx)

        # State NumPy (nội bộ), được khởi tạo với kích thước chính xác
        self.state_np = np.zeros(self.num_nodes, dtype=np.float32)

        # (Logic honeypot giữ nguyên)
        self.honeypot_nodes = self.nodes
        self.num_honeypot_nodes = len(self.honeypot_nodes)

    def _get_embeddings_from_state(self, numpy_state_array):
        """
        Hàm helper: Chạy GNN encoder để lấy graphs từ mảng state NumPy.
        (Giữ nguyên, không thay đổi)
        """
        new_node_features = self.original_node_features.clone()
        new_state_tensor = torch.tensor(numpy_state_array, dtype=torch.float32)
        new_node_features[:, 0] = new_state_tensor

        with torch.no_grad():
            node_embeddings, _ = self.encoder(
                self.g_dgl,
                new_node_features,
                self.original_edge_features
            )
        return node_embeddings

    def reset(self):
        """
        (Giữ nguyên, không thay đổi)
        """
        self.state_np = np.zeros(self.num_nodes, dtype=np.float32)
        initial_embeddings = self._get_embeddings_from_state(self.state_np)
        return initial_embeddings

    def step(self, action):
        """
        (Logic bên trong giữ nguyên, không thay đổi)
        """
        honeypots = []
        G = deepcopy(self.G_new)

        for i in range(2):
            node_idx = np.argmax(action[i])
            if action[i, node_idx] == 0:
                node_idx = random.randint(0, self.num_honeypot_nodes - 1)
            node = self.honeypot_nodes[node_idx]
            honeypot = f"Honeypot {{{node}}}"
            honeypots.append(honeypot)
            G.add_node(honeypot)
            G.add_edge(node, honeypot, user=0.8, root=0.8)
            self.graph = deepcopy(G)

        path, captured = self.attack_fn(G, honeypots, self.goal)

        new_state_np = np.zeros(self.num_nodes, dtype=np.float32)
        reward = 0
        done = False

        if any(h in captured for h in honeypots):
            reward = 1
            done = True
        elif self.goal in captured:
            reward = -1
            done = True
            # self.node_to_idx hoạt động chính xác vì nó là node_map
            new_state_np[self.node_to_idx[self.goal]] = 1

        for node in captured:
            if node in self.node_to_idx:
                new_state_np[self.node_to_idx[node]] = 1

        self.state_np = new_state_np

        new_state_embeddings = self._get_embeddings_from_state(self.state_np)

        return new_state_embeddings, reward, done, path, captured

    def get_action_space_size(self):
        """
        (Giữ nguyên, không thay đổi)
        """
        return 2 * self.num_honeypot_nodes ** 2 + self.num_honeypot_nodes

# def evaluate_model(model, env, num_episodes=1000):
#     successes = 0
#     action_space_size = env.get_action_space_size()
#     for episode in range(1, num_episodes + 1):
#         state = env.reset()
#         done = False
#         episode_honeypots = []  # Lưu vị trí honeypot trong episode
#
#         while not done:
#             with torch.no_grad():
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 q_values = model(state_tensor).squeeze(0)  # shape: [action_space_size]
#
#                 # Lọc q_values chỉ lấy index hợp lệ
#                 valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, env.num_honeypot_nodes)]
#                 valid_q_values = q_values[valid_indices]
#
#                 # Lấy chỉ số trong valid_indices có q_value max
#                 max_idx_in_valid = torch.argmax(valid_q_values).item()
#
#                 # Map về action_idx thực
#                 action_idx = valid_indices[max_idx_in_valid]
#
#             action = index_to_action(action_idx, env.num_honeypot_nodes)
#             next_state, reward, done, path, captured = env.step(action)
#
#             state = next_state
#             honeypot_nodes = []
#             for i in range(2):
#                 node_idx = np.argmax(action[i])
#                 honeypot_nodes.append(env.honeypot_nodes[node_idx])
#             print("Episode:", episode)
#             if reward == 1:  # Honeypot bẫy được kẻ tấn công
#                 successes += 1
#                 print(path)
#                 print(f"Success\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
#                 break
#             elif reward == -1:  # Kẻ tấn công đạt mục tiêu
#                 print(path)
#                 print(f"Failed\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
#                 break
#
#     dsp = (successes / num_episodes) * 100
#     print(f"\nDefense success probability: {dsp:.2f}%")
#
#

def evaluate_model(model, env, num_episodes=1000):
    successes = 0
    action_space_size = env.get_action_space_size()

    # --- THÊM: Chuyển model sang chế độ đánh giá ---
    # Tắt các cơ chế như Dropout (nếu có)
    model.eval()

    for episode in range(1, num_episodes + 1):
        state = env.reset()  # state là Tensor [N, D]
        done = False

        while not done:
            with torch.no_grad():
                # --- SỬA LỖI CHÍNH ---
                # 1. 'state' đã là Tensor
                # 2. Flatten state từ [N, D] -> [1, N*D]
                state_tensor = state.flatten().unsqueeze(0)

                q_values = model(state_tensor).squeeze(0)  # shape: [action_space_size]

                # (Logic lọc q_values và chọn action giữ nguyên)
                valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, env.num_honeypot_nodes)]
                valid_q_values = q_values[valid_indices]
                max_idx_in_valid = torch.argmax(valid_q_values).item()
                action_idx = valid_indices[max_idx_in_valid]

            action = index_to_action(action_idx, env.num_honeypot_nodes)
            next_state, reward, done, path, captured = env.step(action)

            state = next_state  # state mới là Tensor [N, D]

            # --- SỬA LỖI LOGGING: Chỉ in khi episode kết thúc ---

            if reward == 1:  # Honeypot bẫy được kẻ tấn công
                successes += 1

                # Tính toán honeypots ở đây
                honeypot_nodes = []
                for i in range(2):
                    node_idx = np.argmax(action[i])
                    honeypot_nodes.append(env.honeypot_nodes[node_idx])

                print(f"--- Episode {episode}: Success ---")
                print(path)
                print(f"Honeypots connected to: {honeypot_nodes}\n")
                break  # Thoát vòng while

            elif reward == -1:  # Kẻ tấn công đạt mục tiêu
                # Tính toán honeypots ở đây
                honeypot_nodes = []
                for i in range(2):
                    node_idx = np.argmax(action[i])
                    honeypot_nodes.append(env.honeypot_nodes[node_idx])

                print(f"--- Episode {episode}: Failed ---")
                print(path)
                print(f"Honeypots connected to: {honeypot_nodes}\n")
                break  # Thoát vòng while

            # Nếu reward == 0 (chưa xong), vòng lặp while tiếp tục

    dsp = (successes / num_episodes) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Defense success probability: {dsp:.2f}% ({successes}/{num_episodes})")