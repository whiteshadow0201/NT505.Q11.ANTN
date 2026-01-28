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
import torch.nn.functional as F
import time
import os
import yaml
import networkx as nx
from networkx.readwrite import json_graph

def save_graph_env(experiment_id, G_nx, nfeats, efeats, node_order, base_dir='graphs'):
    """
    Saves the graph environment and related info to a .pth file for the Agent to use.

    Args:
        experiment_id (int/str): Experiment ID (subdirectory name).
        G_nx (networkx.Graph): Original NetworkX graph.
        nfeats (Tensor): Node features (original).
        efeats (Tensor): Edge features (original).
        node_order (list): List of node names in index order.
        base_dir (str): Parent directory containing graphs (default: 'graphs').
    """
    # 1. Create directory path
    save_path = os.path.join(base_dir, str(experiment_id))
    os.makedirs(save_path, exist_ok=True)
    print(f"Processing and saving data to: {save_path}")

    # 2. Create Name -> Index mapping
    node_map = {name: i for i, name in enumerate(node_order)}

    # 3. Pack data
    # Note: Key 'g1' is kept for backward compatibility with old Agent loading code
    env_data = {
        "G": G_nx,
        "nfeats": nfeats,
        "efeats": efeats,
        "node_order": node_order,
        "node_map": node_map,
    }

    # 4. Save environment file
    env_file_path = os.path.join(save_path, "graph_environment.pth")
    torch.save(env_data, env_file_path)

    print(f" >> Static Environment (env_data) saved to: {env_file_path}")

    return save_path

# --- HELPER FUNCTION: ASSIGN TIMESTAMP (MODIFIED) ---
def add_timestamp_to_edges(raw_edges_list):
    """
    Takes raw edge list, assigns timestamps at 0.1s intervals
    based on list order, and returns the sorted list.
    """
    processed_edges = []

    # Use current time as baseline
    start_time = time.time()

    # Use enumerate to get index i (0, 1, 2...)
    for i, (u, v, attrs) in enumerate(raw_edges_list):
        # Assign timestamp: each edge is exactly 0.1s apart
        # Edge 1: start_time
        # Edge 2: start_time + 0.1
        # Edge 3: start_time + 0.2 ...
        attrs['timestamp'] = start_time + (i * 0.1)

        processed_edges.append((u, v, attrs))

    # IMPORTANT: Sort list based on timestamp
    processed_edges.sort(key=lambda x: x[2]['timestamp'])
    return processed_edges


def build_dgl(nx_graph, sorted_edges, node_feat_keys, edge_feat_keys):
    """
    Creates a DGL graph while preserving the order of sorted_edges.
    """
    # 1. Create node mapping (using default NX order)
    node_list = list(nx_graph.nodes())
    node_map = {name: i for i, name in enumerate(node_list)}

    # 2. Create Source (src) and Destination (dst) ID lists based on sorted_edges
    src_ids = [node_map[u] for u, v, _ in sorted_edges]
    dst_ids = [node_map[v] for u, v, _ in sorted_edges]

    # 3. Create DGL Graph from ID Tensors (DGL will preserve this order)
    # [Image of DGL graph construction from tensors]
    g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))

    # 4. Assign Node Features
    if node_feat_keys:
        node_data = []
        for n in node_list:
            feats = [nx_graph.nodes[n][k] for k in node_feat_keys]
            node_data.append(feats)
        g.ndata['h'] = torch.tensor(node_data, dtype=torch.float32)

    # 5. Assign Edge Features (following sorted_edges order)
    if edge_feat_keys:
        edge_data = []
        for _, _, attrs in sorted_edges:
            feats = [attrs[k] for k in edge_feat_keys]
            edge_data.append(feats)
        g.edata['h'] = torch.tensor(edge_data, dtype=torch.float32)

    return g, node_list, node_map