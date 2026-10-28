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
        # NO flags!

    def __lt__(self, other):
        return self.dispatch_time < other.dispatch_time

class EventEnvironment:
    def __init__(self):
        self.now = 0.0
        self.queue = []

    def schedule(self, timer):
        heapq.heappush(self.queue, timer)

    def cancel(self, timer):
        """Cancel timer by removing it directly from the priority queue."""
        if timer in self.queue:
            self.queue.remove(timer)
            heapq.heapify(self.queue)
            return timer.work_data
        return None

    def run(self, condition_func):
        while self.queue and condition_func():
            timer = heapq.heappop(self.queue)
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
        if msg_type == 'beacon' or message_data.get('beacon_flag'):
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
        self.incoming_messages = []
        
        # Rule 1 & 6: Node once connected schedules a Timer to send a beacon
        self.schedule_beacon_timer()

    def schedule_beacon_timer(self):
        jitter = random.uniform(0.8, 1.2)
        delay = 1.0 * jitter
        work_data = {'type': 'beacon_tick'}
        
        dispatch_time = self.env.now + delay
        self.current_timer = Timer(dispatch_time, self.on_timer_dispatch, work_data)
        self.env.schedule(self.current_timer)

    def schedule_new_timer(self, work_data):
        msg_count = len(self.incoming_messages)
        delay = sigmoid_stunning_delay(msg_count)
        dispatch_time = self.stun_started_at + delay

        self.current_timer = Timer(dispatch_time, self.on_timer_dispatch, work_data)
        self.env.schedule(self.current_timer)

    def handle_incoming_message(self, message):
        # Rule 2: when node receives message it cancels current Timer by removing from queue, extracts work_data, and schedules new Timer
        prev_work_data = None
        if self.current_timer is not None:
            prev_work_data = self.env.cancel(self.current_timer)

        # Rule 3: stunning uses stun_started_at set at the first time input message was received
        if self.stun_started_at is None:
            self.stun_started_at = self.env.now

        self.incoming_messages.append(message)

        coalesced_messages = []
        if prev_work_data and 'messages' in prev_work_data:
            coalesced_messages.extend(prev_work_data['messages'])
        coalesced_messages.append(message)

        new_work_data = {
            'type': 'incoming_batch',
            'messages': coalesced_messages
        }
        self.schedule_new_timer(new_work_data)

    def send_announce(self, target_id, node_dict):
        prev_work_data = None
        if self.current_timer is not None:
            prev_work_data = self.env.cancel(self.current_timer)

        if self.stun_started_at is None:
            self.stun_started_at = self.env.now

        coalesced_announces = []
        if prev_work_data and 'announces' in prev_work_data:
            coalesced_announces.extend(prev_work_data['announces'])
        coalesced_announces.append({'target': target_id, 'node_dict': node_dict})

        new_work_data = {
            'type': 'incoming_batch',
            'messages': prev_work_data.get('messages', []) if prev_work_data else [],
            'announces': coalesced_announces
        }
        self.schedule_new_timer(new_work_data)

    def on_timer_dispatch(self, work_data):
        # Rule 3: stun_started_at reset to None once Timer runs
        self.stun_started_at = None
        self.incoming_messages.clear()
        self.current_timer = None

        messages = work_data.get('messages', [])
        announces = work_data.get('announces', [])
        is_beacon = work_data.get('type') == 'beacon_tick'

        print(f"timer dispatch messages={len(messages)} announces={len(announces)} is_beacon={is_beacon}")

        # Hebbian learning update from incoming messages
        tensor_accum = np.zeros((8, 8), dtype=np.int32)
        for msg in messages:
            src_pos = hash_to_2d_point(msg.source)
            polar = cartesian_to_polar(self.pos, src_pos)
            src_bin = int((polar[0] / (2 * np.pi)) * 8) % 8
            self.tensor[src_bin, src_bin, src_bin] += 0.15
            self.tensor[src_bin, src_bin, src_bin] = min(5.0, self.tensor[src_bin, src_bin, src_bin])
            tensor_accum[src_bin, src_bin] += 1

        # Rule 4: work ran by Timer should coalesce data from all separate messages into single message for each neighbour
        if self.adapter.connections:
            for nbr_id in self.adapter.connections:
                payload = {
                    'type': 'coalesced_frame',
                    'tensor_frame': tensor_accum.copy(),
                    'beacon_flag': is_beacon,
                    'announces': announces
                }
                self.adapter.send(nbr_id, payload, delay=0.02)

        # Rule 5 & 6: Last thing in timer should be to schedule new Timer with beacon
        self.schedule_beacon_timer()

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

def network_growth_callback(w_data):
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

    # Trigger periodic targeted announcement test
    if len(nodes) >= 10 and random.random() < 0.4:
        src = random.choice(n_keys)
        tgt = random.choice(n_keys)
        if src != tgt:
            nodes[src].send_announce(tgt, nodes)

    env.schedule(Timer(env.now + random.uniform(0.05, 0.2), network_growth_callback, None))

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