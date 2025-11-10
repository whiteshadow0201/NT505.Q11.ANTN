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
import itertools  # ### SỬA ĐỔI ###: Thêm thư viện để lặp


# ### SỬA ĐỔI ###: Hàm cantor_pairing (chỉ cho 2) bị XÓA.
# ### THAY THẾ BẰNG: tuple_to_index (cho N) ###
def tuple_to_index(nodes_tuple, num_nodes_base):
    """
    Ánh xạ một tuple gồm N chỉ số nút (k1, k2, ..., kN) thành một chỉ số
    nguyên duy nhất z, sử dụng hệ cơ số M (num_nodes_base).

    Args:
        nodes_tuple (tuple): Một tuple các chỉ số nút, ví dụ: (k1, k2, ... kN).
        num_nodes_base (int): Số lượng nút có thể (cơ số M).

    Returns:
        int: Một chỉ số nguyên duy nhất z.
    """
    index = 0
    multiplier = 1
    for node_idx in nodes_tuple:
        if not (0 <= node_idx < num_nodes_base):
            raise ValueError(f"Chỉ số nút {node_idx} nằm ngoài giới hạn [0, {num_nodes_base - 1}]")
        index += node_idx * multiplier
        multiplier *= num_nodes_base
    return index


# ### SỬA ĐỔI ###: Hàm inverse_cantor (chỉ cho 2) bị XÓA.
# ### THAY THẾ BẰNG: index_to_tuple (cho N) ###
def index_to_tuple(index, num_honeypots, num_nodes_base):
    """
    Ánh xạ một chỉ số nguyên duy nhất z trở lại tuple N chỉ số nút.
    Đây là phép toán ngược của hệ cơ số M.

    Args:
        index (int): Chỉ số nguyên duy nhất z.
        num_honeypots (int): Số lượng honeypot (N, độ dài của tuple).
        num_nodes_base (int): Số lượng nút có thể (cơ số M).

    Returns:
        tuple: Một tuple các chỉ số nút (k1, k2, ..., kN).
    """
    nodes_tuple = []
    temp_index = index

    # Kiểm tra xem index có nằm trong phạm vi không
    total_space = num_nodes_base ** num_honeypots
    if not (0 <= index < total_space):
        raise ValueError(f"Index {index} nằm ngoài giới hạn [0, {total_space - 1}]")

    for _ in range(num_honeypots):
        node_idx = temp_index % num_nodes_base
        nodes_tuple.append(node_idx)
        temp_index //= num_nodes_base

    return tuple(nodes_tuple)


# ### SỬA ĐỔI ###: Cần thêm `num_honeypots`
def action_to_index(action, num_honeypot_nodes, num_honeypots):
    """
    Chuyển đổi một hành động (N vector one-hot) thành một tuple
    các chỉ số nút và sau đó ánh xạ tuple đó tới một index duy nhất.
    """
    nodes_tuple = []
    for i in range(num_honeypots):
        # np.argmax tìm chỉ số của '1' trong vector one-hot
        nodes_tuple.append(np.argmax(action[i]))

    # Kiểm tra xem các nút có trùng lặp không
    if len(set(nodes_tuple)) != len(nodes_tuple):
        raise ValueError("Các nút trong hành động phải khác nhau")

    return tuple_to_index(tuple(nodes_tuple), num_honeypot_nodes)


# ### SỬA ĐỔI ###: Cần thêm `num_honeypots`
def sample_valid_action_matrix(N, M):
    """
    Tạo một ma trận hành động [N, M] one-hot ngẫu nhiên.
    Chọn N nút khác nhau từ M nút.
    """
    action = np.zeros((N, M), dtype=np.float32)

    # Lấy N chỉ số nút khác nhau ngẫu nhiên từ 0 đến M-1
    try:
        node_indices = random.sample(range(M), N)
    except ValueError:
        print(f"Lỗi: Không thể chọn {N} nút khác nhau từ {M} nút.")
        # Chiến lược dự phòng: chọn N nút đầu tiên (hiếm khi xảy ra)
        node_indices = list(range(N))

    for i in range(N):
        action[i, node_indices[i]] = 1
    return action


def select_action_multi_head(q_values_N_M):
    """
    Chọn hành động [N, M] one-hot tốt nhất từ output Q-value [N, M].
    Đảm bảo các nút được chọn (trục M) là khác nhau.

    Args:
        q_values_N_M (torch.Tensor): Tensor Q-values shape (N, M)
    """
    N = q_values_N_M.shape[0]
    M = q_values_N_M.shape[1]
    action = np.zeros((N, M), dtype=np.float32)

    # Chuyển Q-values (N, M) thành danh sách (Q, head_idx, node_idx)
    q_values_flat = []
    for i in range(N):
        for j in range(M):
            # .item() để chuyển từ tensor sang số Python
            q_values_flat.append((q_values_N_M[i, j].item(), i, j))

            # Sắp xếp giảm dần theo Q-value
    q_values_flat.sort(key=lambda x: x[0], reverse=True)

    honeypot_assigned = [False] * N
    node_chosen = [False] * M
    count = 0

    # Lặp qua danh sách đã sắp xếp
    for q, i, j in q_values_flat:
        if count == N:  # Đã chọn đủ N honeypot
            break

        # Nếu "đầu" (honeypot) 'i' chưa được gán VÀ nút 'j' chưa bị chọn
        if not honeypot_assigned[i] and not node_chosen[j]:
            action[i, j] = 1
            honeypot_assigned[i] = True
            node_chosen[j] = True
            count += 1

    # Nếu vẫn còn honeypot chưa gán (do xung đột), gán ngẫu nhiên
    # vào các nút còn trống
    for i in range(N):
        if not honeypot_assigned[i]:
            for j in range(M):
                if not node_chosen[j]:
                    action[i, j] = 1
                    node_chosen[j] = True
                    break

    return action


# ### SỬA ĐỔI ###: Cần thêm `num_honeypots`
def is_valid_index(index, num_honeypot_nodes, num_honeypots):
    """
    Kiểm tra xem một index có tương ứng với một hành động hợp lệ không
    (tất cả các nút đều khác nhau).
    """
    try:
        nodes_tuple = index_to_tuple(index, num_honeypots, num_honeypot_nodes)
        # Kiểm tra tính duy nhất (không trùng lặp)
        return len(set(nodes_tuple)) == len(nodes_tuple)
    except ValueError:
        # index_to_tuple sẽ báo lỗi nếu index nằm ngoài phạm vi
        return False


# ### SỬA ĐỔI ###: Cần thêm `num_honeypots`
def sample_valid_index(num_honeypot_nodes, num_honeypots, exploration_counter, min_explorations=10):
    """
    Lấy mẫu một index hợp lệ ngẫu nhiên (các nút khác nhau).

    CẢNH BÁO: Nếu M^N (num_nodes^num_honeypots) quá lớn (ví dụ: > 20 triệu),
    việc lặp qua toàn bộ không gian hành động để tìm 'valid_indices'
    sẽ rất chậm.
    """
    action_space_size = num_honeypot_nodes ** num_honeypots

    # Tạo danh sách tất cả các index hợp lệ
    # Cảnh báo: Dòng này có thể rất chậm nếu không gian hành động lớn!
    valid_indices = [idx for idx in range(action_space_size)
                     if is_valid_index(idx, num_honeypot_nodes, num_honeypots)]

    if not valid_indices:
        # Điều này xảy ra nếu N > M
        raise ValueError("Không tìm thấy index hợp lệ (num_honeypot_nodes phải >= num_honeypots)")

    # Logic thăm dò giữ nguyên
    under_explored = [idx for idx in valid_indices if exploration_counter[idx] < min_explorations]

    if under_explored:
        # Ưu tiên các index chưa được khám phá
        selected_idx = random.choice(under_explored)
    else:
        # Chuyển sang lấy mẫu ngẫu nhiên thuần túy
        selected_idx = random.choice(valid_indices)

    exploration_counter[selected_idx] += 1
    return selected_idx


# ### SỬA ĐỔI ###: Hàm này cần thay đổi đáng kể để xử lý N-tuple
def sample_exploration_index(new_num_honeypot_nodes, old_num_honeypot_nodes, num_honeypots, exploration_counter,
                             min_explorations=10):
    """
    Lấy mẫu một index hợp lệ ngẫu nhiên (các nút khác nhau) trong đó
    ít nhất một nút là 'mới' (>= old_num_honeypot_nodes).
    """
    new_node_indices = []
    exploration_done = False

    # Tạo một iterator cho tất cả các N-tuple có thể
    # ví dụ: (0,0,0), (0,0,1), ... (M-1, M-1, M-1)
    all_tuples = itertools.product(range(new_num_honeypot_nodes), repeat=num_honeypots)

    for nodes_tuple in all_tuples:
        # 1. Kiểm tra tính hợp lệ (các nút khác nhau)
        if len(set(nodes_tuple)) != len(nodes_tuple):
            continue

        # 2. Kiểm tra xem có ít nhất một nút là 'mới' không
        is_new = False
        for node_idx in nodes_tuple:
            if node_idx >= old_num_honeypot_nodes:
                is_new = True
                break

        if is_new:
            idx = tuple_to_index(nodes_tuple, new_num_honeypot_nodes)
            new_node_indices.append(idx)

    if not new_node_indices:
        raise ValueError("Không tìm thấy hành động hợp lệ nào liên quan đến nút mới")

    # Logic thăm dò giữ nguyên
    under_explored = [idx for idx in new_node_indices if exploration_counter[idx] < min_explorations]

    if under_explored:
        selected_idx = random.choice(under_explored)
    else:
        selected_idx = random.choice(new_node_indices)
        exploration_done = True

    exploration_counter[selected_idx] += 1
    return selected_idx, exploration_done


# ... (Class DQN không thay đổi) ...
import torch
import torch.nn as nn


class MultiHeadDQN(nn.Module):
    def __init__(self, state_size, num_honeypots_N, num_nodes_M):
        """
        Khởi tạo mạng DQN với N đầu output.

        Args:
            state_size (int): Kích thước của state (ví dụ: 512)
            num_honeypots_N (int): Số lượng honeypot (N)
            num_nodes_M (int): Số lượng nút có thể đặt (M)
        """
        super(MultiHeadDQN, self).__init__()
        self.num_honeypots_N = num_honeypots_N
        self.num_nodes_M = num_nodes_M

        # Lớp thân chung
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

        # Tạo N "đầu" output, mỗi đầu có M nơ-ron
        # Chúng ta dùng ModuleList để lưu trữ các lớp này
        self.output_heads = nn.ModuleList()
        for _ in range(num_honeypots_N):
            # Mỗi đầu là một lớp linear riêng biệt
            self.output_heads.append(nn.Linear(128, num_nodes_M))

    def forward(self, x):
        """
        x: (batch_size, state_size)
        """
        # Đưa qua thân chung
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Đưa qua N đầu output
        # Kết quả sẽ là một list các tensor
        # [ (batch_size, M), (batch_size, M), ... ] (N lần)
        q_values_list = [head(x) for head in self.output_heads]

        # Chúng ta có thể stack chúng lại để dễ xử lý
        # Output shape: (batch_size, N, M)
        return torch.stack(q_values_list, dim=1)

# ... (Class ReplayBuffer không thay đổi) ...
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_embedding, action, reward, next_state_embedding, done):
        state_embedding = state_embedding.detach().cpu()
        next_state_embedding = next_state_embedding.detach().cpu()
        self.buffer.append((state_embedding, action, reward, next_state_embedding, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state_batch = torch.stack(state)
        next_state_batch = torch.stack(next_state)
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


# ... (Class NetworkEnv gần như không đổi, chỉ cần sửa get_action_space_size) ...
class NetworkEnv:
    def __init__(self, G_new, attack_fn, g_dgl, encoder,
                 original_node_features, original_edge_features,
                 node_map, num_honeypots,  # <-- N đây rồi
                 goal=None):

        self.g_dgl = g_dgl
        self.encoder = encoder
        self.original_node_features = original_node_features.clone()
        self.original_edge_features = original_edge_features.clone()
        self.encoder.eval()
        self.num_honeypots = num_honeypots  # <-- Lưu N

        self.G_new = G_new
        self.attack_fn = attack_fn

        if goal is None:
            self.goal = []
        elif not isinstance(goal, list):
            self.goal = [goal]
        else:
            self.goal = goal

        self.node_to_idx = node_map
        self.nodes = list(node_map.keys())
        self.num_nodes = len(self.nodes)

        print("node_to_idx đã được tải:", self.node_to_idx)
        print(f"Mục tiêu (Goals) được thiết lập: {self.goal}")
        print(f"Số lượng honeypot (N) được thiết lập: {self.num_honeypots}")

        self.state_np = np.zeros(self.num_nodes, dtype=np.float32)

        self.honeypot_nodes = self.nodes
        self.num_honeypot_nodes = len(self.honeypot_nodes)  # <-- Đây là M
        print(f"Số lượng nút đặt (M) được thiết lập: {self.num_honeypot_nodes}")

    def _get_embeddings_from_state(self, numpy_state_array):
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
        self.state_np = np.zeros(self.num_nodes, dtype=np.float32)
        initial_embeddings = self._get_embeddings_from_state(self.state_np)
        return initial_embeddings

    def step(self, action):
        """
        Hàm step đã TƯƠNG THÍCH sẵn với N honeypot
        vì nó dùng self.num_honeypots.
        """
        honeypots = []
        G = deepcopy(self.G_new)

        # Vòng lặp này đã đúng (dùng N)
        for i in range(self.num_honeypots):
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
        elif any(g in captured for g in self.goal):
            reward = -1
            done = True
            for g in self.goal:
                if g in self.node_to_idx:
                    new_state_np[self.node_to_idx[g]] = 1

        for node in captured:
            if node in self.node_to_idx:
                new_state_np[self.node_to_idx[node]] = 1

        self.state_np = new_state_np
        new_state_embeddings = self._get_embeddings_from_state(self.state_np)
        return new_state_embeddings, reward, done, path, captured

    def get_action_space_size(self):
        """
        ### SỬA ĐỔI ###: Đây là thay đổi quan trọng nhất trong Env.
        Không gian hành động bây giờ là M^N.
        """
        # M = self.num_honeypot_nodes
        # N = self.num_honeypots
        return self.num_honeypot_nodes ** self.num_honeypots


# ### SỬA ĐỔI ###: Cần cập nhật hàm evaluate để truyền đúng tham số
def evaluate_model(model, env, num_episodes=1000, device=None):
    """
    Đánh giá mô hình DQN (KIẾN TRÚC MULTI-HEAD) đã huấn luyện.
    """

    # --- Chọn device ---
    device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Đang đánh giá trên device: {device}")

    model = model.to(device)
    model.eval()  # Chuy"n model sang chế độ đánh giá

    successes = 0
    # Lấy M (số nút) và N (số honeypot)
    M_nodes = env.num_honeypot_nodes
    N_honeypots = env.num_honeypots

    # ### SỬA ĐỔI ###
    # Không còn action_space_size (M^N) nữa
    print(f"Đang đánh giá... Kiến trúc Multi-Head (N={N_honeypots} honeypots, M={M_nodes} nút)")

    for episode in range(1, num_episodes + 1):
        state = env.reset().to(device)  # Shape [M, D_embed]
        done = False

        while not done:
            with torch.no_grad():
                # Flatten state [M, D] -> [1, M*D]
                state_tensor = state.flatten().unsqueeze(0).to(device)

                # ### SỬA ĐỔI ###
                # Lấy output [1, N, M] từ mạng
                q_values_all = model(state_tensor)

                # Squeeze(0) -> [N, M]
                q_values_N_M = q_values_all.squeeze(0)

                # Chọn hành động tốt nhất, không trùng lặp
                # Output là ma trận numpy [N, M] one-hot
                action = select_action_multi_head(q_values_N_M)

            # ### SỬA ĐỔI ###
            # Không cần 'index_to_action' nữa,
            # vì 'action' đã ở đúng định dạng [N, M]

            next_state, reward, done, path, captured = env.step(action)
            state = next_state.to(device)

            # --- Logging khi kết thúc episode ---
            if reward != 0:
                # Lấy tên các nút đã chọn từ ma trận action
                honeypot_nodes = [env.honeypot_nodes[np.argmax(action[i])] for i in range(N_honeypots)]
                status = "Success" if reward == 1 else "Failed"

                # In log ít hơn để đỡ rối
                if episode % 50 == 0 or num_episodes <= 100:
                    print(f"--- Episode {episode}: {status} ---")
                    print(path)
                    print(f"Honeypots connected to: {honeypot_nodes}\n")

                if reward == 1:
                    successes += 1
                break  # Kết thúc episode

        if episode % 100 == 0:
            print(f"Đã hoàn thành {episode}/{num_episodes} episodes...")

    dsp = (successes / num_episodes) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Defense success probability: {dsp:.2f}% ({successes}/{num_episodes})")