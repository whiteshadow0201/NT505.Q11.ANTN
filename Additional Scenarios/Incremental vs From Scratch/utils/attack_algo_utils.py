import math
import random
import numpy as np
from queue import PriorityQueue
import networkx as nx


# Attacker's greedy attack with randomizer
def global_weighted_random_attack(graph, honeypot_nodes, goal):
    """
    Đã cập nhật: Xử lý trọng số âm (-1) thành 0.
    """
    if goal is None:
        goal = []

    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    if not priority_1_nodes:
        all_nodes = [n for n in graph.nodes if n not in ["Attacker"] + goal + honeypot_nodes]

        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        start_node = random.choice(priority_1_nodes)

    captured = {start_node}
    path = [start_node]

    while True:
        neighbors = []
        edge_weights = []
        source_nodes = []

        for compromised_node in captured:
            for neighbor in graph.successors(compromised_node):
                if neighbor not in captured:
                    edge_data = graph[compromised_node][neighbor]

                    # <--- THAY ĐỔI: Tính tổng, nếu âm thì gán bằng 0
                    raw_weight = edge_data['user'] + edge_data['root']
                    weight = max(0, raw_weight)

                    neighbors.append(neighbor)
                    edge_weights.append(weight)
                    source_nodes.append(compromised_node)

        if not neighbors:
            break

        total_weight = sum(edge_weights)

        # Nếu tất cả các cạnh đều là 0 (hoặc -1 đã bị quy về 0) thì dừng lại
        if total_weight == 0:
            break

        probabilities = [w / total_weight for w in edge_weights]

        chosen_idx = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
        chosen_node = neighbors[chosen_idx]

        path.append(chosen_node)
        captured.add(chosen_node)

        if chosen_node in honeypot_nodes or chosen_node in goal:
            break

    return path, captured


# Attacker's greedy attack with randomizer
def greedy_attack_priority_queue(graph, honeypot_nodes, goal):
    """
    Đã cập nhật: Xử lý trọng số âm (-1) thành 0.
    """
    if goal is None:
        goal = []

    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    if not priority_1_nodes:
        all_nodes = [n for n in graph.nodes if n not in ["Attacker"] + goal + honeypot_nodes]

        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        start_node = random.choice(priority_1_nodes)

    captured = {start_node}
    path = [start_node]

    pq = PriorityQueue()

    for neighbor in graph.successors(start_node):
        if graph.has_edge(start_node, neighbor):
            # <--- THAY ĐỔI: Lấy max user/root, sau đó ép về 0 nếu bị âm
            raw_weight = max(graph[start_node][neighbor]['user'], graph[start_node][neighbor]['root'])
            weight = max(0, raw_weight)

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

            if to_node in goal:
                break

            for next_node in graph.successors(to_node):
                if next_node not in captured and graph.has_edge(to_node, next_node):
                    # <--- THAY ĐỔI: Lấy max user/root, sau đó ép về 0 nếu bị âm
                    raw_next_weight = max(graph[to_node][next_node]['user'], graph[to_node][next_node]['root'])
                    next_weight = max(0, raw_next_weight)

                    next_randomizer = random.uniform(0, 1)
                    pq.put((-next_weight, -next_randomizer, next_node))

    return path, captured