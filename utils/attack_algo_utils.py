
import math
import random
import numpy as np
from queue import PriorityQueue
import networkx as nx

# Attacker's greedy attack with randomizer
def global_weighted_random_attack(graph, honeypot_nodes, goal):
    # 1. Tự động tìm tất cả các node có thuộc tính 'priority' == 1
    #    (Chúng ta dùng attrs.get('priority') để tránh lỗi nếu node không có thuộc tính)
    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    # 2. Xử lý trường hợp không tìm thấy node nào (để tránh lỗi)
    if not priority_1_nodes:
        # Bạn có thể chọn một hành vi dự phòng, ví dụ:
        # raise ValueError("Không tìm thấy node nào có priority = 1.")
        # Hoặc chọn từ tất cả các node (ngoại trừ 'Attacker' nếu có)
        all_nodes = [n for n in graph.nodes if n not in ["Attacker", goal] + honeypot_nodes]
        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        # 3. Chọn ngẫu nhiên một node từ danh sách priority 1 đã tìm thấy
        start_node = random.choice(priority_1_nodes)

    # --- THAY ĐỔI KẾT THÚC ---

    captured = {start_node}
    path = [start_node]

    while True:
        # Collect all uncaptured neighbors of compromised nodes
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

        # Normalize weights to probabilities
        total_weight = sum(edge_weights)
        if total_weight == 0:
            break
        probabilities = [w / total_weight for w in edge_weights]

        # Choose next node randomly based on probabilities
        chosen_idx = random.choices(range(len(neighbors)), weights=probabilities, k=1)[0]
        chosen_node = neighbors[chosen_idx]

        # Add to path and captured
        path.append(chosen_node)
        captured.add(chosen_node)

        # Check stopping conditions
        if chosen_node in honeypot_nodes or chosen_node == goal:
            break

    return path, captured

# Attacker's greedy attack with randomizer
def greedy_attack_priority_queue(graph, honeypot_nodes, goal):
    # 1. Tự động tìm tất cả các node có thuộc tính 'priority' == 1
    #    (Chúng ta dùng attrs.get('priority') để tránh lỗi nếu node không có thuộc tính)
    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    # 2. Xử lý trường hợp không tìm thấy node nào (để tránh lỗi)
    if not priority_1_nodes:
        # Bạn có thể chọn một hành vi dự phòng, ví dụ:
        # raise ValueError("Không tìm thấy node nào có priority = 1.")
        # Hoặc chọn từ tất cả các node (ngoại trừ 'Attacker' nếu có)
        all_nodes = [n for n in graph.nodes if n not in ["Attacker", goal] + honeypot_nodes]
        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        # 3. Chọn ngẫu nhiên một node từ danh sách priority 1 đã tìm thấy
        start_node = random.choice(priority_1_nodes)

    # --- THAY ĐỔI KẾT THÚC ---

    captured = {start_node}
    path = [start_node]

    pq = PriorityQueue()
    for neighbor in graph.successors("Attacker"):
        weight = max(graph["Attacker"][neighbor]['user'], graph["Attacker"][neighbor]['root'])
        randomizer = random.uniform(0, 1)  # Randomizer for tie-breaking
        pq.put((-weight, -randomizer, neighbor))  # Sort by -weight, -randomizer, neighbor

    while not pq.empty():
        neg_weight, neg_randomizer, to_node = pq.get()
        weight = -neg_weight
        randomizer = -neg_randomizer
        if to_node in honeypot_nodes:  # Stop at honeypot node
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
                    next_randomizer = random.uniform(0, 1)  # New randomizer for each edge
                    pq.put((-next_weight, -next_randomizer, next_node))
    return path, captured
