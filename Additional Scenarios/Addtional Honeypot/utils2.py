import math
import random
import numpy as np
from queue import PriorityQueue
from copy import deepcopy
import torch
import torch.nn as nn
from collections import deque, defaultdict

def triple_to_index(k1, k2, k3, num_nodes):
    """
    Map three distinct node indices to a unique index.
    """
    if k1 == k2 or k2 == k3 or k1 == k3 or k1 >= num_nodes or k2 >= num_nodes or k3 >= num_nodes:
        raise ValueError("Node indices must be distinct and within bounds")
    return k1 * num_nodes * num_nodes + k2 * num_nodes + k3

def index_to_triple(index, num_nodes):
    """
    Convert an index to a triplet of node indices, with validity check.
    Returns:
        Tuple (k1, k2, k3, is_valid) where is_valid is False if nodes are not distinct.
    """
    if index >= num_nodes * num_nodes * num_nodes:
        return 0, 0, 0, False
    k1 = index // (num_nodes * num_nodes)
    remainder = index % (num_nodes * num_nodes)
    k2 = remainder // num_nodes
    k3 = remainder % num_nodes
    is_valid = k1 != k2 and k2 != k3 and k1 != k3 and k1 < num_nodes and k2 < num_nodes and k3 < num_nodes
    return k1, k2, k3, is_valid

def action_to_index(action, num_nodes):
    """
    Convert a 3xnum_nodes action matrix to a unique index.
    """
    indices = [np.argmax(action[i]) for i in range(3)]
    return triple_to_index(indices[0], indices[1], indices[2], num_nodes)

def index_to_action(index, num_nodes):
    """
    Convert an index to a 3xnum_nodes action matrix.
    """
    k1, k2, k3, is_valid = index_to_triple(index, num_nodes)
    if not is_valid:
        raise ValueError("Invalid index for action")
    action = np.zeros((3, num_nodes), dtype=np.float32)
    action[0, k1] = 1
    action[1, k2] = 1
    action[2, k3] = 1
    return action

def is_valid_index(index, num_nodes):
    """
    Check if an index corresponds to a valid triplet of distinct nodes.
    """
    k1, k2, k3, is_valid = index_to_triple(index, num_nodes)
    return is_valid

def sample_valid_index(action_space_size, num_nodes, exploration_counter, min_explorations=10):
    """
    Sample a random valid index, ensuring exploration.
    """
    valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, num_nodes)]
    if not valid_indices:
        raise ValueError("No valid indices found")
    under_explored = [idx for idx in valid_indices if exploration_counter[idx] < min_explorations]
    if under_explored:
        selected_idx = random.choice(under_explored)
    else:
        selected_idx = random.choice(valid_indices)
    exploration_counter[selected_idx] += 1
    return selected_idx

def sample_exploration_index(new_action_space_size, new_num_honeypot_nodes, old_num_honeypot_nodes, exploration_counter, min_explorations=10):
    """
    Sample an index involving new nodes for exploration.
    """
    new_node_indices = []
    exploration_done = False
    for i in range(new_num_honeypot_nodes):
        for j in range(new_num_honeypot_nodes):
            for k in range(new_num_honeypot_nodes):
                if i != j and j != k and i != k:
                    if i >= old_num_honeypot_nodes or j >= old_num_honeypot_nodes or k >= old_num_honeypot_nodes:
                        idx = triple_to_index(i, j, k, new_num_honeypot_nodes)
                        if idx < new_action_space_size:
                            new_node_indices.append(idx)
    if not new_node_indices:
        raise ValueError("No valid new-node-related actions found")
    under_explored = [idx for idx in new_node_indices if exploration_counter[idx] < min_explorations]
    if under_explored:
        selected_idx = random.choice(under_explored)
    else:
        selected_idx = random.choice(new_node_indices)
        exploration_done = True
    exploration_counter[selected_idx] += 1
    return selected_idx, exploration_done

def cantor_pairing(k1, k2):
    """
    Original Cantor pairing for two nodes (used in old model).
    """
    return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

def inverse_cantor(z):
    """
    Inverse Cantor pairing for two nodes.
    """
    w = int((math.sqrt(8*z + 1) - 1) // 2)
    t = (w * (w + 1)) // 2
    k2 = z - t
    k1 = w - k2
    return k1, k2

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

class NetworkSecurityEnv:
    def __init__(self, G_new, attack_fn, goal="Data Server"):
        self.G_new = G_new
        self.attack_fn = attack_fn
        self.goal = goal
        self.nodes = [n for n in G_new.nodes if n not in ["Attacker"]]
        self.num_nodes = len(self.nodes)
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        print("node_to_idx:", self.node_to_idx)
        self.honeypot_nodes = [n for n in self.nodes if n != goal]
        self.num_honeypot_nodes = len(self.honeypot_nodes)

    def reset(self):
        self.state = np.zeros(self.num_nodes, dtype=np.float32)
        return self.state

    def step(self, action):
        honeypot_nodes = []
        G = deepcopy(self.G_new)
        for i in range(3):
            node_idx = np.argmax(action[i])
            if action[i, node_idx] == 0:
                node_idx = random.randint(0, self.num_honeypot_nodes - 1)
            node = self.honeypot_nodes[node_idx]
            honeypot = f"Honeypot {{{node}}}"
            honeypot_nodes.append(honeypot)
            G.add_node(honeypot)
            G.add_edge(node, honeypot, user=0.8, root=0.8)
            self.graph = deepcopy(G)
        path, captured = self.attack_fn(G, honeypot_nodes, self.goal)
        new_state = np.zeros(self.num_nodes, dtype=np.float32)
        reward = 0
        done = False
        if any(h in captured for h in honeypot_nodes):
            reward = 1
            done = True
        elif self.goal in captured:
            reward = -1
            done = True
            new_state[self.node_to_idx[self.goal]] = 1
        for node in captured:
            if node in self.node_to_idx:
                new_state[self.node_to_idx[node]] = 1
        self.state = new_state
        return new_state, reward, done, path, captured

    def get_action_space_size(self):
        return self.num_honeypot_nodes * (self.num_honeypot_nodes - 1) * (self.num_honeypot_nodes - 2)

def global_weighted_random_attack(graph, honeypot_nodes, goal):
    captured = {"Attacker"}
    path = ["Attacker"]
    while True:
        neighbors = []
        edge_weights = []
        source_nodes = []
        for compromised_node in captured:
            for neighbor in graph.successors(compromised_node):
                if neighbor not in captured:
                    edge_data = graph[compromised_node][neighbor]
                    weight = edge_data['user'] + edge_data['root']
                    neighbors.append(neighbor)
                    edge_weights.append(weight)
                    source_nodes.append(compromised_node)
        if not neighbors:
            break
        total_weight = sum(edge_weights)
        if total_weight == 0:
            break
        probabilities = [w / total_weight for w in edge_weights]
        chosen_idx = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
        chosen_node = neighbors[chosen_idx]
        path.append(chosen_node)
        captured.add(chosen_node)
        if chosen_node in honeypot_nodes or chosen_node == goal:
            break
    return path, captured

def greedy_attack_priority_queue(graph, honeypot_nodes, goal):
    captured = {"Attacker"}
    path = ["Attacker"]
    pq = PriorityQueue()
    for neighbor in graph.successors("Attacker"):
        weight = max(graph["Attacker"][neighbor]['user'], graph["Attacker"][neighbor]['root'])
        randomizer = random.uniform(0, 1)
        pq.put((-weight, -randomizer, neighbor))
    while not pq.empty():
        neg_weight, neg_randomizer, to_node = pq.get()
        weight = -neg_weight
        randomizer = -neg_randomizer
        if to_node in honeypot_nodes:
            path.append(to_node)
            captured.add(to_node)
            break
        if to_node not in captured:
            captured.add(to_node)
            path.append(to_node)
            if to_node == goal:
                break
            for next_node in graph.successors(to_node):
                if next_node not in captured:
                    next_weight = max(graph[to_node][next_node]['user'], graph[to_node][next_node]['root'])
                    next_randomizer = random.uniform(0, 1)
                    pq.put((-next_weight, -next_randomizer, next_node))
    return path, captured

def evaluate_model(model, env, num_episodes=1000):
    successes = 0
    action_space_size = env.get_action_space_size()
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor).squeeze(0)
                valid_indices = [idx for idx in range(action_space_size) if is_valid_index(idx, env.num_honeypot_nodes)]
                valid_q_values = q_values[valid_indices]
                max_idx_in_valid = torch.argmax(valid_q_values).item()
                action_idx = valid_indices[max_idx_in_valid]
            action = index_to_action(action_idx, env.num_honeypot_nodes)
            next_state, reward, done, path, captured = env.step(action)
            state = next_state
            honeypot_nodes = []
            for i in range(3):
                node_idx = np.argmax(action[i])
                honeypot_nodes.append(env.honeypot_nodes[node_idx])
            print("Episode:", episode)
            if reward == 1:
                successes += 1
                print(path)
                print(f"Success\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
                break
            elif reward == -1:
                print(path)
                print(f"Failed\nHoneypots: {action}\nHoneypots connected to: {honeypot_nodes}\n")
                break
    dsp = (successes / num_episodes) * 100
    print(f"\nDefense success probability: {dsp:.2f}%")