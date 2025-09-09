import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from collections import deque
import heapq


def cantor_pairing(k1, k2):
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2


def inverse_cantor(z):
    w = int(np.floor((np.sqrt(8 * z + 1) - 1) / 2))
    t = (w * w + w) // 2
    k2 = z - t
    k1 = w - k2
    return k1, k2


def action_to_index(action, num_nodes):
    idx = np.where(action == 1)[0]
    if len(idx) != 2:
        raise ValueError("Action must select exactly two nodes")
    return cantor_pairing(idx[0], idx[1])


def index_to_action(index, num_nodes):
    k1, k2 = inverse_cantor(index)
    if k1 >= num_nodes or k2 >= num_nodes:
        raise ValueError("Invalid index for action")
    action = np.zeros((2, num_nodes), dtype=np.float32)
    action[0, k1] = 1
    action[1, k2] = 1
    return action


def is_valid_index(index, num_nodes):
    try:
        k1, k2 = inverse_cantor(index)
        return k1 < num_nodes and k2 < num_nodes and k1 != k2
    except:
        return False


def sample_valid_index(action_space_size, num_nodes, exploration_counter, min_explorations=10):
    valid_indices = [i for i in range(action_space_size) if is_valid_index(i, num_nodes)]
    under_explored = [i for i in valid_indices if exploration_counter.get(i, 0) < min_explorations]
    if under_explored:
        return random.choice(under_explored)
    return random.choice(valid_indices)


def sample_exploration_index(new_action_space_size, new_num_honeypot_nodes, old_num_honeypot_nodes, exploration_counter,
                             min_explorations=10):
    valid_indices = [i for i in range(new_action_space_size) if is_valid_index(i, new_num_honeypot_nodes)]
    new_action_indices = [i for i in valid_indices if not all(x < old_num_honeypot_nodes for x in inverse_cantor(i))]
    under_explored = [i for i in new_action_indices if exploration_counter.get(i, 0) < min_explorations]
    if under_explored:
        return random.choice(under_explored), False
    return random.choice(valid_indices), True


class DQN(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = Fwomen(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class NetworkSecurityEnv:
    def __init__(self, G, attack_fn):
        self.G = G
        self.attack_fn = attack_fn
        self.nodes = [node for node in G.nodes if node != "Attacker"]
        self.goal_nodes = ["Data Server", "File Server"]  # Define two goal nodes
        self.honeypot_nodes = [node for node in self.nodes if node not in self.goal_nodes]
        self.num_nodes = len(self.nodes)
        self.num_honeypot_nodes = len(self.honeypot_nodes)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

    def reset(self):
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        return self.state

    def step(self, action):
        honeypot_indices = np.where(action == 1)[0]
        if len(honeypot_indices) != 2:
            raise ValueError("Action must select exactly two nodes")
        honeypot_nodes = [self.idx_to_node[idx] for idx in honeypot_indices]
        path, captured = self.attack_fn(self.G, honeypot_nodes, self.goal_nodes)

        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        for node in path[1:]:  # Skip Attacker node
            if node in self.node_to_idx:
                self.state[self.node_to_idx[node]] = 1

        if captured in honeypot_nodes:
            reward = 1
            done = True
        elif captured in self.goal_nodes:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, path, honeypot_nodes

    def get_action_space_size(self):
        return 2 * (self.num_honeypot_nodes ** 2)


def global_weighted_random_attack(graph, honeypot_nodes, goal_nodes):
    current_node = "Attacker"
    path = [current_node]
    captured = None

    while current_node not in honeypot_nodes and current_node not in goal_nodes:
        neighbors = list(graph.successors(current_node))
        if not neighbors:
            break
        probabilities = []
        for neighbor in neighbors:
            user_prob = graph[current_node][neighbor].get('user', 0)
            root_prob = graph[current_node][neighbor].get('root', 0)
            probabilities.append(user_prob + root_prob)
        total = sum(probabilities)
        if total == 0:
            break
        probabilities = [p / total for p in probabilities]
        chosen_idx = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
        current_node = neighbors[chosen_idx]
        path.append(current_node)
        captured = current_node

    return path, captured


def greedy_attack_priority_queue(graph, honeypot_nodes, goal_nodes):
    current_node = "Attacker"
    path = [current_node]
    captured = None
    pq = [(0, random.random(), current_node)]
    visited = set()

    while pq and current_node not in honeypot_nodes and current_node not in goal_nodes:
        _, _, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        path.append(current_node)
        captured = current_node
        neighbors = list(graph.successors(current_node))
        for neighbor in neighbors:
            if neighbor not in visited:
                user_prob = graph[current_node][neighbor].get('user', 0)
                root_prob = graph[current_node][neighbor].get('root', 0)
                weight = max(user_prob, root_prob)
                heapq.heappush(pq, (-weight, random.random(), neighbor))

    return path, captured


def evaluate_model(model, env, num_episodes=1000):
    model.eval()
    success_count = 0
    for i in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
        action_idx = q_values.max(1)[1].item()
        if not is_valid_index(action_idx, env.num_honeypot_nodes):
            action_idx = sample_valid_index(env.get_action_space_size(), env.num_honeypot_nodes, {})
        action = index_to_action(action_idx, env.num_honeypot_nodes)
        next_state, reward, done, path, honeypot_nodes = env.step(action)
        print(f"Episode {i + 1}:")
        print(f"Attack path: {path}")
        print(f"Honeypot nodes: {honeypot_nodes}")
        print(f"Success: {reward == 1}\n")
        if reward == 1:
            success_count += 1
    dsp = success_count / num_episodes
    print(f"Defense Success Probability: {dsp:.2%}")
    return dsp