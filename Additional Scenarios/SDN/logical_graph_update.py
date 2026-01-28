#!/usr/bin/env python

import networkx as nx
import pickle
import os
import re
import glob
import time
import requests
from datetime import datetime

# --- CONFIGURATION ---
GRAPH_FILE = "network_state.pth"
LOG_DIR_ROOT = "/var/log/snort"
TRIGGER_TIME_FILE = "trigger_time.txt"
POLLING_INTERVAL = 2

# Inference Server Configuration
INFERENCE_SERVER_IP = "10.233.244.149"
INFERENCE_SERVER_PORT = 36363
INFERENCE_TRIGGER_URL = f"http://{INFERENCE_SERVER_IP}:{INFERENCE_SERVER_PORT}/trigger_update"

# ============================================================
# PART 1: GRAPH MANAGEMENT & SNORT LOGIC
# ============================================================

def load_graph(filename):
    try:
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            G = pickle.load(f)
        return G
    except Exception:
        return None

def save_graph(G, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(G, f)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [+] Update saved to {filename}")
    except Exception as e:
        print(f"[!] File save error: {e}")

def get_ip_map(G):
    ip_map = {}
    for node, attrs in G.nodes(data=True):
        if 'ip' in attrs:
            ip_map[attrs['ip'].split('/')[0]] = node
        if 'private_ip' in attrs:
            ip_map[attrs['private_ip'].split('/')[0]] = node
    return ip_map

def parse_snort_logs_and_update(G, ip_map):
    """
    Returns: (set of affected nodes, flag indicating if SSH was found)
    """
    regex_pattern = r".*->\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)"
    updated_nodes = set()
    has_ssh = False  # <--- New flag: Marks if SSH keyword is found

    alert_files = glob.glob(os.path.join(LOG_DIR_ROOT, "*", "alert"))

    if not alert_files:
        return updated_nodes, False

    for log_file in alert_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Check if log line is related to SSH
                    if "SSH" in line or ":22" in line:
                        has_ssh = True  # <--- SSH sign found

                        match = re.search(regex_pattern, line)
                        if match:
                            dest_ip = match.group(1)
                            dest_port = match.group(2)

                            # Logic to update Node state in Graph
                            if dest_port == "22" and dest_ip in ip_map:
                                node_id = ip_map[dest_ip]
                                if G.nodes[node_id]['state'] == 0:
                                    G.nodes[node_id]['state'] = 1
                                    updated_nodes.add(node_id)
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 [SNORT] Attack detected on Node {node_id} ({dest_ip})")
        except Exception:
            pass

    return updated_nodes, has_ssh

# ============================================================
# PART 2: TRIGGER TO INFERENCE SERVER
# ============================================================

def notify_inference_server(message="UPDATE PLEASE!"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Sending to AI Server: '{message}' ...")
    try:
        response = requests.post(INFERENCE_TRIGGER_URL, data=message, timeout=5)
        if response.status_code == 200:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ AI Server response: {response.text}")
        else:
            print(f"[!] AI Server returned error: {response.status_code}")
    except Exception as e:
        print(f"[!] AI SERVER CONNECTION ERROR: {e}")

# ============================================================
# PART 3: MAIN LOOP
# ============================================================

def main_loop():
    print("\n-------------------------------------------")
    print(" >>> MONITOR AGENT STARTING...")
    print(" >>> MODE: Send when log changes AND contains 'SSH'")
    print("-------------------------------------------\n")

    # Save timestamp of log files for comparison
    last_file_timestamps = {}

    while True:
        # 1. Load Graph
        G = load_graph(GRAPH_FILE)
        if G is None:
            time.sleep(POLLING_INTERVAL)
            continue

        ip_map = get_ip_map(G)

        # 2. CHECK IF FILE CHANGED?
        alert_files = glob.glob(os.path.join(LOG_DIR_ROOT, "*", "alert"))
        is_log_changed = False

        for f in alert_files:
            try:
                mtime = os.path.getmtime(f)
                # If new file or modification time is newer than last time
                if f not in last_file_timestamps or mtime > last_file_timestamps[f]:
                    is_log_changed = True
                    last_file_timestamps[f] = mtime
            except FileNotFoundError:
                pass

        # 3. Parse log to find SSH and update graph
        # This function now returns 2 values
        affected_nodes, has_ssh = parse_snort_logs_and_update(G, ip_map)

        # 4. Save graph if node is hacked (for display purposes)
        if affected_nodes:
            save_graph(G, GRAPH_FILE)

        # 5. DECISION SENDING LOGIC:
        # Condition: File changed (is_log_changed) AND Log contains SSH (has_ssh)
        if is_log_changed and has_ssh:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ CONDITION MET...")

            # ---> ADD THIS FILE WRITING SECTION <---
            try:
                with open(TRIGGER_TIME_FILE, "w") as f:
                    # Write Unix Timestamp (seconds.microseconds)
                    f.write(str(time.time()))
            except Exception as e:
                print(f"[!] Cannot write timestamp file: {e}")
            # --------------------------------

            msg = f"SSH ATTACK DETECTED. Targets: {list(affected_nodes)}" if affected_nodes else "SSH ACTIVITY DETECTED"
            notify_inference_server(msg)

        elif is_log_changed and not has_ssh:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ Log changed but NOT SSH -> Ignored.")

        else:
            # No changes
            print(".", end="", flush=True)

        time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[STOP] Stopping program.")