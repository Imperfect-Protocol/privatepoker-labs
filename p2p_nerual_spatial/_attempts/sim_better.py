import heapq
import hashlib
import numpy as np
import random
import time
import networkx as nx

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. TIMER & EVENT-DRIVEN ENVIRONMENT CLASSES
# ==========================================
class Timer:
    def __init__(self, dispatch_time, callback, work_data):
        self.dispatch_time = dispatch_time
        self.callback = callback
        self.work_data = work_data
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        return self.callback, self.work_data

    def __lt__(self, other):
        return self.dispatch_time < other.dispatch_time

class EventEnvironment:
    def __init__(self):
        self.now = 0.0
        self.queue = []

    def schedule(self, timer):
        heapq.heappush(self.queue, timer)

    def run(self, condition_func):
        while self.queue and condition_func():
            timer = heapq.heappop(self.queue)
            if not timer.cancelled:
                self.now = timer.dispatch_time
                timer.callback(timer.work_data)

# Global message tracking counters
total_messages_sent = 0
total_beacons_sent = 0
total_announces_sent = 0

# ==========================================
# 2. MESSAGE, CONNECTION & ADAPTER CLASSES
# ==========================================
class Message:
    def __init__(self, source, dest, data, sending_time, received_time):
        self.source = source
        self.dest = dest
        self.data = data
        self.sending_time = sending_time
        self.received_time = received_time

class Connection:
    def __init__(self, env, source_addr, dest_addr, source_adapter, dest_adapter):
        self.env = env
        self.source_addr = source_addr
        self.dest_addr = dest_addr
        self.source_adapter = source_adapter
        self.dest_adapter = dest_adapter

    def send(self, message_data, delay):
        global total_messages_sent, total_beacons_sent, total_announces_sent
        sending_time = self.env.now
        received_time = sending_time + delay
        
        msg_type = message_data.get('type', 'generic')
        total_messages_sent += 1
        if msg_type == 'beacon':
            total_beacons_sent += 1
        elif msg_type == 'announce':
            total_announces_sent += 1

        msg = Message(
            source=self.source_addr,
            dest=self.dest_addr,
            data=message_data,
            sending_time=sending_time,
            received_time=received_time
        )

        def deliver(data):
            self.dest_adapter.enqueue_incoming(msg)

        self.env.schedule(Timer(received_time, deliver, None))

class Adapter:
    def __init__(self, node_id, env, node_ref):
        self.node_id = node_id
        self.env = env
        self.node_ref = node_ref
        self.connections = {}
        self.inbox = []

    def add_connection(self, dest_id, connection):
        self.connections[dest_id] = connection

    def send(self, dest_id, message, delay=0.015):
        if dest_id in self.connections:
            self.connections[dest_id].send(message, delay)

    def enqueue_incoming(self, message):
        self.inbox.append(message)
        self.node_ref.handle_incoming_message(message)

# ==========================================
# 3. SPATIAL & NODE LOGIC
# ==========================================
def hash_to_2d_point(node_id):
    h = hashlib.sha256(str(node_id).encode()).hexdigest()
    x = (int(h[:8], 16) % 10000) / 100.0
    y = (int(h[8:16], 16) % 10000) / 100.0
    return np.array([x, y])

def cartesian_to_polar(origin, point):
    delta = point - origin
    dist = np.linalg.norm(delta)
    bearing = np.arctan2(delta[1], delta[0]) % (2 * np.pi)
    return np.array([bearing, dist])

def sigmoid_stunning_delay(msg_count, k=0.1, mid=20):
    """Sigmoidal stunning duration: starts at 50ms (0.05s) and reaches 500ms (0.50s)."""
    min_delay = 0.050
    max_delay = 0.500
    return min_delay + (max_delay - min_delay) / (1.0 + np.exp(-k * (msg_count - mid)))

class P2PNode:
    def __init__(self, node_id, env, node_dict_ref):
        self.id = node_id
        self.env = env
        self.node_dict_ref = node_dict_ref
        self.pos = hash_to_2d_point(node_id)
        self.adapter = Adapter(node_id, env, self)
        
        # Spatial-Aware Attention Tensor: Source_Cluster(8) x Target_Cluster(8) x Direction_Sector(8)
        self.tensor = np.ones((8, 8, 8), dtype=np.float32) * 0.1
        
        # Stunning & Timer State
        self.current_timer = None
        self.stun_started_at = None
        self.msg_count = 0
        
        # Schedule the first periodic beacon input tick
        self.schedule_next_beacon_tick()

    def do_coalescing(self, prev_data, new_work_data):
        """Coalesce work data from cancelled timer and new incoming work data."""
        print("do_coalescing")
        if prev_data is None:
            return new_work_data

        combined_messages = prev_data.get('messages', []) + new_work_data.get('messages', [])
        combined_announces = prev_data.get('announces', []) + new_work_data.get('announces', [])
        
        tensor_prev = prev_data.get('tensor_accum', np.zeros((8, 8), dtype=np.int32))
        tensor_new = new_work_data.get('tensor_accum', np.zeros((8, 8), dtype=np.int32))
        combined_tensor = tensor_prev + tensor_new

        is_beacon_tick = prev_data.get('is_beacon_tick', False) or new_work_data.get('is_beacon_tick', False)
        count = len(combined_messages) + len(combined_announces) + int(np.sum(combined_tensor))
        
        return {
            'messages': combined_messages,
            'announces': combined_announces,
            'tensor_accum': combined_tensor,
            'is_beacon_tick': is_beacon_tick,
            'count': max(1, count)
        }

    def schedule(self, new_work_callback, new_work_data):
        """Manages timer scheduling, stunning windows, and data coalescing."""
        current_time = self.env.now
        print(f"schedule")

        if self.current_timer is None or self.current_timer.cancelled:
            self.stun_started_at = current_time
            coalesced_data = new_work_data
        else:
            prev_work, prev_data = self.current_timer.cancel()
            coalesced_data = self.do_coalescing(prev_data, new_work_data)

        self.msg_count = coalesced_data.get('count', 1)
        stun_delay = sigmoid_stunning_delay(self.msg_count)
        dispatch_time = self.stun_started_at + stun_delay

        timer = Timer(dispatch_time, new_work_callback, coalesced_data)
        self.current_timer = timer
        self.env.schedule(timer)

    def schedule_next_beacon_tick(self):
        jitter = random.uniform(0.8, 1.2)
        interval = 1.0 * jitter
        print(f"schedule next beacon tick")

        def beacon_tick_callback(dispatch_time):
            # Treat beacon emission as an input work item
            beacon_work_data = {
                'messages': [],
                'announces': [],
                'tensor_accum': np.zeros((8, 8), dtype=np.int32),
                'is_beacon_tick': True,
                'count': 1
            }
            self.schedule(self.execute_node_action, beacon_work_data)
            self.schedule_next_beacon_tick()

        self.env.schedule(Timer(self.env.now + interval, beacon_tick_callback, None))

    def handle_incoming_message(self, message):
        """Pure accumulation of input messages; leaves execution to timer dispatch."""
        msg_type = message.data.get('type', 'beacon')
        print(f"handle incoming message from {message.source}")
        
        src_pos = hash_to_2d_point(message.source)
        polar = cartesian_to_polar(self.pos, src_pos)
        u_bin = int((polar[0] / (2 * np.pi)) * 8) % 8
        
        tensor_accum = np.zeros((8, 8), dtype=np.int32)
        tensor_accum[u_bin, u_bin] += 1

        work_data = {
            'messages': [message] if msg_type == 'beacon' else [],
            'announces': [message] if msg_type == 'announce' else [],
            'tensor_accum': tensor_accum,
            'is_beacon_tick': False,
            'count': 1
        }

        self.schedule(self.execute_node_action, work_data)

    def send_announce(self, target_id, node_dict):
        print("send announce")
        announce_work_data = {
            'messages': [],
            'announces': [{'target': target_id, 'node_dict': node_dict}],
            'tensor_accum': np.zeros((8, 8), dtype=np.int32),
            'is_beacon_tick': False,
            'count': 1
        }
        self.schedule(self.execute_node_action, announce_work_data)

    def execute_node_action(self, work_data):
        """Executed only upon actual timer expiration with fully coalesced work data."""
        print("execute node action")
        # 1. Hebbian Learning on accumulated messages
        for msg in work_data.get('messages', []):
            src_pos = hash_to_2d_point(msg.source)
            polar = cartesian_to_polar(self.pos, src_pos)
            src_bin = int((polar[0] / (2 * np.pi)) * 8) % 8
            
            reward = 0.15
            self.tensor[src_bin, src_bin, src_bin] += reward
            self.tensor[src_bin, src_bin, src_bin] = min(5.0, self.tensor[src_bin, src_bin, src_bin])

        # 2. Process announces
        for ann in work_data.get('announces', []):
            target_id = ann['target']
            node_dict = ann['node_dict']
            if self.adapter.connections:
                tgt_pos = node_dict[target_id].pos
                scores = []
                for dest_id in self.adapter.connections:
                    nbr_pos = node_dict[dest_id].pos
                    nbr_polar = cartesian_to_polar(self.pos, nbr_pos)
                    nbr_bin = int((nbr_polar[0] / (2 * np.pi)) * 8) % 8
                    tgt_polar = cartesian_to_polar(self.pos, tgt_pos)
                    tgt_bin = int((tgt_polar[0] / (2 * np.pi)) * 8) % 8
                    score = self.tensor[nbr_bin, tgt_bin, nbr_bin]
                    scores.append((score, dest_id))
                
                scores.sort(key=lambda x: x[0], reverse=True)
                best_neighbors = [dest_id for _, dest_id in scores[:2]]
                for nbr_id in best_neighbors:
                    payload = {'type': 'announce', 'target': target_id}
                    self.adapter.send(nbr_id, payload, delay=0.015)

        # 3. Publish coalesced tensors out if beacon tick or accumulated tensor input exists
        tensor_accum = work_data.get('tensor_accum', np.zeros((8, 8), dtype=np.int32))
        is_beacon_tick = work_data.get('is_beacon_tick', False)
        
        if is_beacon_tick or np.sum(tensor_accum) > 0:
            for dest_id in list(self.adapter.connections.keys()):
                payload = {
                    'type': 'beacon',
                    'tensor_frame': tensor_accum.copy()
                }
                self.adapter.send(dest_id, payload, delay=0.02)

        # 4. Reset timer & state
        self.current_timer = None
        self.stun_started_at = None
        self.msg_count = 0

# ==========================================
# 4. SIMULATION ORCHESTRATION & ANALYSIS
# ==========================================
env = EventEnvironment()
nodes = {}
next_node_id = 0
total_edges = 0
G = nx.Graph()

situation_weights = [0.5, 0.3, 0.2]

def create_node():
    global next_node_id
    nid = next_node_id
    next_node_id += 1
    node = P2PNode(nid, env, nodes)
    nodes[nid] = node
    G.add_node(nid)
    return node

def connect_nodes(id1, id2):
    global total_edges
    if id1 != id2 and id2 not in nodes[id1].adapter.connections:
        conn1 = Connection(env, id1, id2, nodes[id1].adapter, nodes[id2].adapter)
        conn2 = Connection(env, id2, id1, nodes[id2].adapter, nodes[id1].adapter)
        nodes[id1].adapter.add_connection(id2, conn1)
        nodes[id2].adapter.add_connection(id1, conn2)
        G.add_edge(id1, id2)
        total_edges += 1

# Bootstrap initial pair
u_init = create_node()
v_init = create_node()
connect_nodes(u_init.id, v_init.id)

def network_growth_callback(dispatch_time):
    if len(nodes) >= 1000:
        return

    situation = random.choices([1, 2, 3], weights=situation_weights, k=1)[0]
    n_keys = list(nodes.keys())

    if situation == 1 and len(n_keys) >= 2:
        u = random.choice(n_keys)
        candidates = [v for v in n_keys if v != u and v not in nodes[u].adapter.connections]
        if candidates:
            v = random.choice(candidates)
            connect_nodes(u, v)
        else:
            v = create_node()
            connect_nodes(u, v.id)
    elif situation == 2:
        u = random.choice(n_keys)
        v = create_node()
        connect_nodes(u, v.id)
    elif situation == 3:
        u = create_node()
        if len(nodes) < 1000:
            v = create_node()
            connect_nodes(u.id, v.id)

    if len(nodes) >= 10 and random.random() < 0.4:
        src = random.choice(n_keys)
        tgt = random.choice(n_keys)
        if src != tgt:
            nodes[src].send_announce(tgt, nodes)

    next_join_delay = random.uniform(0.05, 0.2)
    env.schedule(Timer(env.now + next_join_delay, network_growth_callback, None))

# Kick off growth loop
env.schedule(Timer(env.now + 0.1, network_growth_callback, None))

print("Starting event-driven P2P simulation up to 1,000 nodes...")
start_wall_time = time.time()
env.run(lambda: len(nodes) < 1000)

# Final network analysis metrics
is_connected = nx.is_connected(G)
num_components = nx.number_connected_components(G)
component_sizes = [len(c) for c in nx.connected_components(G)]

print("="*65)
print(" EVENT-DRIVEN P2P SIMULATION METRICS (1,000 NODES)")
print("="*65)
print(f"Situation Weights:              {', '.join(f'{100*w:.1f}%' for w in situation_weights)}")
print("="*65)
print(f"Total Nodes Reached:            {len(nodes):,}")
print(f"Total Edges Created:            {total_edges:,}")
print(f"Simulation Dispatch Time:       {env.now:.2f} seconds")
print(f"Wall Clock Execution Time:      {time.time() - start_wall_time:.2f} seconds")
print("-" * 65)
print("MESSAGE OVERHEAD BREAKDOWN")
print(f"Total Messages Sent:            {total_messages_sent:,}")
print(f"  └─ Coalesced Beacons Sent:    {total_beacons_sent:,}")
print(f"  └─ Targeted Announces Sent:   {total_announces_sent:,}")
print("-" * 65)
print("FINAL NETWORK CONNECTIVITY ANALYSIS")
print(f"Is Entire Network Connected?    {is_connected}")
print(f"Number of Connected Components: {num_components}")
print(f"Largest Component Size:         {max(component_sizes):,} nodes ({max(component_sizes)/len(nodes)*100:.1f}%)")
print("="*65)