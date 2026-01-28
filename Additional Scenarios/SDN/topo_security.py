#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
from mininet.net import Mininet
from mininet.node import Host, RemoteController, OVSKernelSwitch, Node
from mininet.cli import CLI
from mininet.log import setLogLevel, info, error
from mininet.link import Link
import os
import pickle
import http.server
import socketserver
import threading
import json
import subprocess
import time
import requests
import shutil
from requests.auth import HTTPBasicAuth

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
NET_INSTANCE = None
SERVER_PORT = 8000
ODL_IP = '127.0.0.1'
ODL_PORT = 8181
ODL_USER = 'admin'
ODL_PASS = 'admin'
SNORT_CONF_PATH = "/etc/snort/snort.conf"
SNORT_LOG_ROOT = "/var/log/snort"
GRAPH_STATE_FILE = os.path.abspath("network_state.pth")

# HONEYPOT POOL MANAGEMENT (REDUCED TO 3)
NUM_HONEYPOTS = 3 
HONEYPOT_POOL = []      
VICTIM_REGISTRY = {}    
ACTIVE_ASSIGNMENTS = {} 

# ==========================================
# HELPER: CLEAN SNORT LOGS
# ==========================================
def clean_snort_logs():
    """Helper function to remove all Snort logs"""
    if os.path.exists(SNORT_LOG_ROOT):
        info(f"\n*** [CLEANUP] Removing Snort logs in {SNORT_LOG_ROOT}... ***\n")
        os.system(f"rm -rf {SNORT_LOG_ROOT}/*")

# ==========================================
# 1. ODL HANDLER
# ==========================================
class ODLHandler:
    def __init__(self, ip, port, user, password):
        self.base_url = f"http://{ip}:{port}/restconf/config/opendaylight-inventory:nodes"
        self.auth = HTTPBasicAuth(user, password)
        self.headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    def _get_node_id(self, dpid_str):
        return f"openflow:{int(dpid_str, 16)}"

    def enable_basic_switching(self, switch_dpid):
        node_id = self._get_node_id(switch_dpid)
        self._push_flow(node_id, "default-ipv4", 10, match={"ethernet-match": {"ethernet-type": {"type": 2048}}},
                        action_type="NORMAL")
        self._push_flow(node_id, "default-arp", 100, match={"ethernet-match": {"ethernet-type": {"type": 2054}}},
                        action_type="FLOOD")

    def block_global_access_to_honeypot(self, switch_dpid, hp_ip, hp_name):
        node_id = self._get_node_id(switch_dpid)
        self._push_flow(node_id, f"deny-all-{hp_name}", 200,
                        match={"ethernet-match": {"ethernet-type": {"type": 2048}}, "ipv4-destination": f"{hp_ip}/32"},
                        action_type="DROP")

    def allow_owner_access(self, switch_dpid, owner_ip, hp_ip, hp_name):
        node_id = self._get_node_id(switch_dpid)
        self._push_flow(node_id, f"allow-owner-{hp_name}", 300,
                        match={"ethernet-match": {"ethernet-type": {"type": 2048}}, "ipv4-source": f"{owner_ip}/32",
                               "ipv4-destination": f"{hp_ip}/32"},
                        action_type="NORMAL")

    def _push_flow(self, node_id, flow_id, priority, match, action_type):
        url = f"{self.base_url}/node/{node_id}/table/0/flow/{flow_id}"
        action = [{"order": 0, "output-action": {"output-node-connector": "FLOOD"}}] if action_type == "FLOOD" else (
            [{"order": 0, "output-action": {"output-node-connector": "NORMAL"}}] if action_type == "NORMAL" else [])
        flow_data = {"flow": [
            {"id": flow_id, "table_id": 0, "priority": priority, "hard-timeout": 0, "idle-timeout": 0, "match": match,
             "instructions": {"instruction": [{"order": 0, "apply-actions": {"action": action}}]}}]}
        try:
            requests.put(url, auth=self.auth, headers=self.headers, data=json.dumps(flow_data))
        except:
            pass

odl_client = ODLHandler(ODL_IP, ODL_PORT, ODL_USER, ODL_PASS)

# ==========================================
# 2. LOGIC "RE-ALLOCATION"
# ==========================================
class LinuxRouter(Node):
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        self.cmd('sysctl -w net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl -w net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()

def activate_honeypot_logic(target_nodes):
    global NET_INSTANCE, HONEYPOT_POOL, VICTIM_REGISTRY, ACTIVE_ASSIGNMENTS
    if NET_INSTANCE is None: return False
    
    targets_to_process = target_nodes[:NUM_HONEYPOTS]
    info(f"\n[!!!] BATCH REQUEST: {targets_to_process}\n")

    for i, new_victim_id in enumerate(targets_to_process):
        hp_entry = HONEYPOT_POOL[i]
        hp_node = hp_entry['node']
        
        # If already protecting the right target, skip
        if hp_entry['assigned_to'] == new_victim_id:
            info(f"   [-] {hp_node.name} already protecting {new_victim_id}. Skipping.\n")
            continue

        # --- STEP 1: ABSOLUTE DETACH ---
        # Whether active or not, clean just to be safe
        if hp_entry['active'] and hp_entry['assigned_to']:
            old_switch = hp_entry['switch']
            info(f"   [<--] DETACHING {hp_node.name} from OLD switch {old_switch.name}...\n")
            try:
                NET_INSTANCE.delLinkBetween(hp_node, old_switch)
            except:
                pass # Ignore errors, let the code below handle it

        # Force delete interface on virtual Host (most important)
        # This command ensures hp1-eth0 disappears completely
        hp_node.cmd(f"ip link set {hp_node.name}-eth0 down")
        hp_node.cmd(f"ip link delete {hp_node.name}-eth0")
        
        # Clear management info
        if hp_entry['assigned_to'] in ACTIVE_ASSIGNMENTS:
            del ACTIVE_ASSIGNMENTS[hp_entry['assigned_to']]
        
        hp_entry['active'] = False
        hp_entry['assigned_to'] = None
        hp_entry['switch'] = None

        # --- STEP 2: DEPLOY (ATTACH) ---
        if new_victim_id not in VICTIM_REGISTRY:
            error(f"   [X] Unknown victim ID {new_victim_id}\n")
            continue
            
        vic_info = VICTIM_REGISTRY[new_victim_id]
        target_switch = vic_info['switch']
        target_ip = vic_info['hp_target_ip']
        gw_ip = vic_info['gw_ip']
        owner_ip = vic_info['owner_ip']

        info(f"   [-->] ATTACHING {hp_node.name} to NEW victim {new_victim_id} (@ {target_switch.name})\n")
        
        try:
            # Create new Link
            link = NET_INSTANCE.addLink(hp_node, target_switch)
            # Attach to OVS
            target_switch.attach(link.intf2)
            
            # Network configuration
            hp_node.setIP(target_ip, prefixLen=24, intf=link.intf1)
            hp_node.cmd(f"route add default gw {gw_ip}")
            
            # Reset ARP cache to avoid retaining old MAC
            hp_node.cmd("ip neigh flush all")
            hp_node.cmd(f"arping -U -c 1 -I {link.intf1.name} {target_ip}")

            # ODL Update
            odl_client.block_global_access_to_honeypot(target_switch.dpid, target_ip, hp_node.name)
            odl_client.allow_owner_access(target_switch.dpid, owner_ip, target_ip, hp_node.name)

            # Update status to success
            hp_entry['active'] = True
            hp_entry['assigned_to'] = new_victim_id
            hp_entry['switch'] = target_switch
            ACTIVE_ASSIGNMENTS[new_victim_id] = hp_node.name
            
        except Exception as e:
            error(f"   [!!!] ERROR ATTACHING: {e}\n")
            # If error, try to clean again to avoid future breakage
            hp_node.cmd(f"ip link delete {hp_node.name}-eth0")

    return True

def start_control_server(port, server_dir=None):
    class ControlHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            if "GET" in args[0]: return
            super().log_message(format, *args)

        def do_POST(self):
            if self.path == '/deploy_honeypots':
                try:
                    length = int(self.headers['Content-Length'])
                    data = json.loads(self.rfile.read(length).decode('utf-8'))
                    # Receive target list from API
                    activate_honeypot_logic(data.get('targets', []))
                    self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
                except Exception as e:
                    print(e)
                    self.send_response(500); self.end_headers()
            else:
                self.send_response(404); self.end_headers()

    def run_server():
        if server_dir: os.chdir(server_dir)
        socketserver.TCPServer.allow_reuse_address = True
        socketserver.TCPServer(("", port), ControlHandler).serve_forever()

    t = threading.Thread(target=run_server); t.daemon = True; t.start()

# ==========================================
# 3. SNORT FUNCTION & MAIN
# ==========================================
def start_host_based_snort(net, G):
    info("\n*** [SNORT] Starting Local Snort on each Victim Node ***\n")
    os.system("killall snort 2>/dev/null")
    time.sleep(1)
    if not os.path.exists(SNORT_CONF_PATH): return

    victim_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'victim']
    for node_id in victim_nodes:
        host = net.get(node_id)
        log_dir = f"{SNORT_LOG_ROOT}/{node_id}"
        if os.path.exists(log_dir): shutil.rmtree(log_dir)
        os.makedirs(log_dir, mode=0o777)
        interface = f"{node_id}-eth0"
        
        # [MODIFIED] Add 'tcp port 22' at the end to catch SSH only, ignore Ping (ICMP)
        cmd = f"snort -D -i {interface} -A fast -l {log_dir} -c {SNORT_CONF_PATH} -q tcp port 22"
        
        info(f"   + Node {node_id}: Snort listening on {interface} (SSH only)\n")
        host.cmd(cmd)

def build_and_export_simple_graph(G_logic, filename):
    G_simple = nx.DiGraph()
    weighted_edges_data = [
        ("Pad", "Host 1", {"user": 0.6, "root": 0.48}), ("Pad", "Host 2", {"user": 0.32, "root": 0.32}),
        ("Pad", "Host 3", {"user": 0.32, "root": 0.32}), ("Pad", "Web Server", {"user": 0.8, "root": 0.6}),
        ("Host 1", "Pad", {"user": 0.6, "root": 0.6}), ("Host 1", "Web Server", {"user": 0.8, "root": 0.6}),
        ("Host 1", "Host 2", {"user": 0.32, "root": 0.32}), ("Host 1", "Host 3", {"user": 0.32, "root": 0.32}),
        ("Host 2", "Host 3", {"user": 0.8, "root": 0.8}), ("Host 2", "File Server", {"user": 0.8, "root": 0.6}),
        ("Host 2", "Data Server", {"user": 0.8, "root": 0.6}), ("Host 3", "Host 2", {"user": 0.8, "root": 0.8}),
        ("Host 3", "File Server", {"user": 0.8, "root": 0.6}), ("Host 3", "Data Server", {"user": 0.8, "root": 0.6}),
        ("Web Server", "File Server", {"user": 0.8, "root": 0.04}), ("Web Server", "Data Server", {"user": 0.8, "root": 0.04}),
        ("File Server", "Data Server", {"user": 0.8, "root": 0.04}), ("Data Server", "File Server", {"user": 0.6, "root": 0.02}),
    ]
    label_to_id = {d['label']: n for n, d in G_logic.nodes(data=True)}
    for n in [x for x, y in G_logic.nodes(data=True) if y.get('node_type') == 'victim']:
        d = G_logic.nodes[n]
        G_simple.add_node(n, label=d['label'], state=0, priority=d.get('priority', 1), ip=d['ip'])
    
    nodes_list = list(G_simple.nodes())
    for u_l, v_l, dat in weighted_edges_data:
        u, v = label_to_id.get(u_l), label_to_id.get(v_l)
        if u in nodes_list and v in nodes_list: G_simple.add_edge(u, v, **dat)
    
    with open(filename, 'wb') as f: pickle.dump(G_simple, f)
    start_control_server(SERVER_PORT, os.path.dirname(filename))

def NetworkX_SDN_Builder():
    global NET_INSTANCE, HONEYPOT_POOL, VICTIM_REGISTRY
    G = nx.DiGraph()
    switches_conf = {'s_inet': '10.0.0.1', 's_dmz': '172.16.1.1', 's_sub1': '192.168.1.1', 'fw2': '192.168.2.1', 'fw3': '192.168.3.1'}

    nodes_data = [
        {"name": "Attacker", "mn_id": "att", "ip": "10.0.0.10/24", "switch": "s_inet", "node_type": "attacker", "priority": 0},
        {"name": "Web Server", "mn_id": "web", "ip": "172.16.1.10/24", "switch": "s_dmz", "node_type": "victim", "priority": 1},
        {"name": "Pad", "mn_id": "pad", "ip": "192.168.1.10/24", "switch": "s_sub1", "node_type": "victim", "priority": 1},
        {"name": "Host 1", "mn_id": "h1", "ip": "192.168.1.11/24", "switch": "s_sub1", "node_type": "victim", "priority": 1},
        {"name": "Host 3", "mn_id": "h3", "ip": "192.168.2.11/24", "switch": "fw2", "node_type": "victim", "priority": 0},
        {"name": "Host 2", "mn_id": "h2", "ip": "192.168.2.10/24", "switch": "fw2", "node_type": "victim", "priority": 0},
        {"name": "File Server", "mn_id": "fil", "ip": "192.168.3.11/24", "switch": "fw3", "node_type": "victim", "priority": 0},
        {"name": "Data Server", "mn_id": "dat", "ip": "192.168.3.12/24", "switch": "fw3", "node_type": "victim", "priority": 2},
    ]

    for n in nodes_data:
        G.add_node(n['mn_id'], label=n['name'], ip=n['ip'], switch=n['switch'], node_type=n['node_type'], priority=n['priority'], gw=switches_conf[n['switch']])

    build_and_export_simple_graph(G, GRAPH_STATE_FILE)

    net = Mininet(controller=None, switch=OVSKernelSwitch, autoSetMacs=True)
    NET_INSTANCE = net
    c0 = net.addController('c0', controller=RemoteController, ip=ODL_IP, port=6653)
    router = net.addHost('r0', cls=LinuxRouter, ip='10.0.0.1/24')

    mn_switches = {}
    sw_names = ['s_inet', 's_dmz', 's_sub1', 'fw2', 'fw3']
    for i, sw in enumerate(sw_names):
        dpid = "{:016x}".format(i + 1)
        if sw == 's_sub1': dpid = "0000000000000010"
        if sw == 's_dmz': dpid = "0000000000000020"
        mn_switches[sw] = net.addSwitch(sw, dpid=dpid, protocols="OpenFlow13")
        net.addLink(router, mn_switches[sw])

    # 1. Create Victim
    for node_id in G.nodes():
        d = G.nodes[node_id]
        h = net.addHost(node_id, ip=d['ip'], defaultRoute=f"via {d['gw']}")
        net.addLink(h, mn_switches[d['switch']])
        
        if d['node_type'] == 'victim':
            parts = d['ip'].split('/')[0].split('.')
            hp_target_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{int(parts[3]) + 100}"
            VICTIM_REGISTRY[node_id] = {
                'switch': mn_switches[d['switch']],
                'hp_target_ip': hp_target_ip,
                'gw_ip': d['gw'],
                'owner_ip': d['ip'].split('/')[0]
            }

    # 2. Create Pool with size 3
    info(f"\n*** Creating {NUM_HONEYPOTS} Floating Honeypots (Recyclable) ***\n")
    for i in range(1, NUM_HONEYPOTS + 1):
        hp_name = f"hp{i}"
        hp = net.addHost(hp_name, ip=f"10.99.99.{i}/24")
        HONEYPOT_POOL.append({
            'name': hp_name,
            'node': hp,
            'active': False,
            'assigned_to': None,
            'switch': None
        })

    net.build(); c0.start()
    for sw in mn_switches.values(): sw.start([c0])

    router.cmd("ifconfig r0-eth0 10.0.0.1/24 up")
    router.cmd("ifconfig r0-eth1 172.16.1.1/24 up")
    router.cmd("ifconfig r0-eth2 192.168.1.1/24 up")
    router.cmd("ifconfig r0-eth3 192.168.2.1/24 up")
    router.cmd("ifconfig r0-eth4 192.168.3.1/24 up")
    
    time.sleep(5)
    for sw in mn_switches.values(): odl_client.enable_basic_switching(sw.dpid)
    
    router.cmd("iptables -F; iptables -P FORWARD DROP")
    router.cmd("iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")
    router.cmd("iptables -A FORWARD -s 10.0.0.10 -d 172.16.1.0/24 -j ACCEPT")
    router.cmd("iptables -A FORWARD -s 10.0.0.10 -d 192.168.1.0/24 -j ACCEPT")
    router.cmd("iptables -A FORWARD -s 172.16.0.0/12 -j ACCEPT")
    router.cmd("iptables -A FORWARD -s 192.168.0.0/16 -j ACCEPT")

    for hp_entry in HONEYPOT_POOL:
        hp_entry['node'].cmd("python3 -m http.server 80 &")
    for n in [x for x, y in G.nodes(data=True) if y.get('node_type') == 'victim']:
        net.get(n).cmd("mkdir -p /var/run/sshd; /usr/sbin/sshd -D &")

    start_host_based_snort(net, G)
    info("*** Ready. Send POST to /deploy_honeypots with {'targets': ['h1', 'h2', 'h3']} to test. ***\n")
    
    # ----------------------------------------------------
    # RUN CLI AND CLEANUP AFTER EXIT
    # ----------------------------------------------------
    try:
        CLI(net)
    finally:
        net.stop()
        # [NEW] CALL LOG CLEANUP ON EXIT
        clean_snort_logs()

if __name__ == '__main__':
    setLogLevel('info')
    # Clean logs at startup (to ensure cleanliness)
    clean_snort_logs()
    
    if not os.path.exists(SNORT_LOG_ROOT):
        os.makedirs(SNORT_LOG_ROOT, exist_ok=True)
        
    subprocess.run(["mn", "-c"])
    NetworkX_SDN_Builder()