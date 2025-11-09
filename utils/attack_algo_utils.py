import math
import random
import numpy as np
from queue import PriorityQueue
import networkx as nx


# Attacker's greedy attack with randomizer
def global_weighted_random_attack(graph, honeypot_nodes, goal):
    """
    Đã cập nhật để xử lý 'goal' là một danh sách.
    """
    # Đảm bảo 'goal' luôn là một danh sách (an toàn nếu lỡ truyền vào None)
    if goal is None:
        goal = []

    # 1. Tự động tìm tất cả các node có thuộc tính 'priority' == 1
    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    # 2. Xử lý trường hợp không tìm thấy node nào
    if not priority_1_nodes:
        # <--- THAY ĐỔI: Loại trừ 'Attacker' VÀ tất cả các node trong list 'goal'
        all_nodes = [n for n in graph.nodes if n not in ["Attacker"] + goal + honeypot_nodes]

        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        # 3. Chọn ngẫu nhiên một node từ danh sách priority 1
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

        # <--- THAY ĐỔI: Kiểm tra xem 'chosen_node' có nằm TRONG danh sách 'goal' không
        if chosen_node in honeypot_nodes or chosen_node in goal:
            break

    return path, captured


# Attacker's greedy attack with randomizer
def greedy_attack_priority_queue(graph, honeypot_nodes, goal):
    """
    Đã cập nhật để xử lý 'goal' là một danh sách.
    CŨNG SỬA MỘT LỖI LOGIC: Bắt đầu hàng đợi (PQ) từ 'start_node' thay vì "Attacker".
    """
    # Đảm bảo 'goal' luôn là một danh sách (an toàn nếu lỡ truyền vào None)
    if goal is None:
        goal = []

    # 1. Tự động tìm tất cả các node có thuộc tính 'priority' == 1
    priority_1_nodes = [node for node, attrs in graph.nodes(data=True)
                        if attrs.get('priority') == 1]

    # 2. Xử lý trường hợp không tìm thấy node nào
    if not priority_1_nodes:
        # <--- THAY ĐỔI: Loại trừ 'Attacker' VÀ tất cả các node trong list 'goal'
        all_nodes = [n for n in graph.nodes if n not in ["Attacker"] + goal + honeypot_nodes]

        if not all_nodes:
            raise ValueError("Đồ thị không có node nào hợp lệ để bắt đầu.")
        start_node = random.choice(all_nodes)
        print(f"Cảnh báo: Không tìm thấy node priority 1. Bắt đầu ngẫu nhiên từ: {start_node}")
    else:
        # 3. Chọn ngẫu nhiên một node từ danh sách priority 1
        start_node = random.choice(priority_1_nodes)

    captured = {start_node}
    path = [start_node]

    pq = PriorityQueue()

    # <--- SỬA LỖI LOGIC: Bắt đầu từ 'start_node' đã tìm thấy
    #      (Code gốc của bạn bắt đầu từ "Attacker", làm cho logic 'start_node' ở trên bị vô nghĩa)
    for neighbor in graph.successors(start_node):
        # Đảm bảo kiểm tra các cạnh tồn tại trước khi truy cập
        if graph.has_edge(start_node, neighbor):
            weight = max(graph[start_node][neighbor]['user'], graph[start_node][neighbor]['root'])
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

            # <--- THAY ĐỔI: Kiểm tra xem 'to_node' có nằm TRONG danh sách 'goal' không
            if to_node in goal:
                break

            for next_node in graph.successors(to_node):
                if next_node not in captured and graph.has_edge(to_node, next_node):
                    next_weight = max(graph[to_node][next_node]['user'], graph[to_node][next_node]['root'])
                    next_randomizer = random.uniform(0, 1)
                    pq.put((-next_weight, -next_randomizer, next_node))

    return path, captured