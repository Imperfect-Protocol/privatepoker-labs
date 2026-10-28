import hashlib
import numpy as np
import random
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Vector/Spatial helper functions
def hash_to_2d_point(node_id):
    """Project a 20-byte node ID / string into a deterministic 2D continuous space [0, 100]^2."""
    h = hashlib.sha256(str(node_id).encode()).hexdigest()
    x = (int(h[:8], 16) % 10000) / 100.0
    y = (int(h[8:16], 16) % 10000) / 100.0
    return np.array([x, y])

def cartesian_to_polar(origin, point):
    """Calculate relative polar coordinates (bearing u in radians [0, 2pi), distance v) from origin to point."""
    delta = point - origin
    dist = np.linalg.norm(delta)
    bearing = np.arctan2(delta[1], delta[0]) % (2 * np.pi)
    return np.array([bearing, dist])

def sigmoid_stunning_delay_ms(msg_count, k=0.1, mid=20):
    """Sigmoidal stunning duration function: starts at 50ms and reaches 500ms as input message volume scales."""
    min_delay = 0.050 # 50 ms
    max_delay = 0.500 # 500 ms
    val = min_delay + (max_delay - min_delay) / (1.0 + np.exp(-k * (msg_count - mid)))
    return val

# Node representation in P2P graph with Stunning & Message Coalescing
class P2PNode:
    def __init__(self, node_id, spatial_pos):
        self.id = node_id
        self.pos = spatial_pos
        self.neighbors = set()
        
        # Spatial-Aware Attention Tensor: Source_Cluster(8) x Target_Cluster(8) x Direction_Sector(8)
        self.tensor = np.ones((8, 8, 8), dtype=np.float32) * 0.1
        self.beacon_jitter_counter = random.uniform(0.8, 1.2)
        
        # Incoming Message Accumulator & Stunning State
        self.incoming_msg_buffer = [] # Holds received signals during stunning period
        self.is_stunned = False
        self.stun_until_time = 0.0
        self.stunning_delay_ms = 50.0 # Starts at 50ms
        self.total_messages_received_in_cycle = 0

        # Outgoing Coalesced Tensor Buffer: Embedding U (8) x Embedding V (8) x Message Count
        self.coalesced_tensor_out = np.zeros((8, 8), dtype=np.int32)
        self.coalesced_publishes_sent = 0

    def add_neighbor(self, nbr_id):
        self.neighbors.add(nbr_id)

    def receive_signal(self, src_pos, tgt_pos, nbr_pos, signal_type='beacon', sim_clock=0.0):
        """Accumulate message input into tensor and trigger proportional sigmoidal stunning."""
        self.incoming_msg_buffer.append((src_pos, tgt_pos, nbr_pos, signal_type))
        self.total_messages_received_in_cycle += 1
        
        # Map input into local coalesced outgoing tensor buffer: Embedding U x Embedding V x Message Count
        u_bin = int((cartesian_to_polar(self.pos, src_pos)[0] / (2 * np.pi)) * 8) % 8
        v_bin = int((cartesian_to_polar(self.pos, tgt_pos)[0] / (2 * np.pi)) * 8) % 8
        self.coalesced_tensor_out[u_bin, v_bin] += 1

        # Calculate sigmoidal stunning delay (50ms to 500ms) proportional to total accumulated inputs
        delay_sec = sigmoid_stunning_delay_ms(self.total_messages_received_in_cycle)
        self.stunning_delay_ms = delay_sec * 1000.0
        
        # Set/Extend stunning period
        self.is_stunned = True
        self.stun_until_time = max(self.stun_until_time, sim_clock + delay_sec)

    def process_accumulated_inputs(self):
        """Once stunning period expires, process accumulated inputs into main spatial tensor."""
        for src_pos, tgt_pos, nbr_pos, sig_type in self.incoming_msg_buffer:
            reward = 0.15 if sig_type == 'beacon' else 0.25
            self.update_tensor_hebbian(src_pos, tgt_pos, nbr_pos, reward=reward)
            
        self.incoming_msg_buffer.clear()

    def publish_coalesced_tensor(self, node_dict):
        """Publish accumulated messages coalesced as a tensor frame to neighbors upon stunning expiration."""
        if not self.neighbors or np.sum(self.coalesced_tensor_out) == 0:
            return 0

        publishes = 0
        for nbr_id in self.neighbors:
            nbr_node = node_dict[nbr_id]
            nbr_node.update_tensor_hebbian(self.pos, nbr_node.pos, self.pos, reward=0.05 * np.sum(self.coalesced_tensor_out))
            publishes += 1
            
        self.coalesced_publishes_sent += publishes
        self.coalesced_tensor_out.fill(0)
        self.total_messages_received_in_cycle = 0
        self.is_stunned = False
        return publishes

    def select_k_best_gossip_neighbors(self, src_pos, tgt_pos, node_dict, k=2):
        if not self.neighbors:
            return []
            
        src_polar = cartesian_to_polar(self.pos, src_pos)
        tgt_polar = cartesian_to_polar(self.pos, tgt_pos)
        
        src_bin = int((src_polar[0] / (2 * np.pi)) * 8) % 8
        tgt_bin = int((tgt_polar[0] / (2 * np.pi)) * 8) % 8
        
        nbr_scores = []
        for nbr_id in self.neighbors:
            nbr_node = node_dict[nbr_id]
            nbr_polar = cartesian_to_polar(self.pos, nbr_node.pos)
            nbr_dir_bin = int((nbr_polar[0] / (2 * np.pi)) * 8) % 8
            
            score = self.tensor[src_bin, tgt_bin, nbr_dir_bin]
            nbr_scores.append((score, nbr_id))
            
        nbr_scores.sort(key=lambda x: x[0], reverse=True)
        return [nbr_id for _, nbr_id in nbr_scores[:k]]

    def update_tensor_hebbian(self, src_pos, tgt_pos, nbr_pos, reward=0.2):
        src_bin = int((cartesian_to_polar(self.pos, src_pos)[0] / (2 * np.pi)) * 8) % 8
        tgt_bin = int((cartesian_to_polar(self.pos, tgt_pos)[0] / (2 * np.pi)) * 8) % 8
        nbr_bin = int((cartesian_to_polar(self.pos, nbr_pos)[0] / (2 * np.pi)) * 8) % 8
        
        self.tensor[src_bin, tgt_bin, nbr_bin] += reward
        self.tensor[src_bin, tgt_bin, nbr_bin] = min(5.0, self.tensor[src_bin, tgt_bin, nbr_bin])

nodes = {}
next_node_id = 0

total_beacons_generated = 0
total_announces_sent = 0
total_coalesced_tensor_publishes = 0
total_edges_created = 0
total_joins = 0
burst_events = 0
sim_clock = 0.0

def create_new_node():
    global next_node_id
    node_id = next_node_id
    next_node_id += 1
    pos = hash_to_2d_point(node_id)
    node = P2PNode(node_id, pos)
    nodes[node_id] = node
    return node_id

def connect_nodes(id1, id2):
    global total_edges_created
    if id1 != id2 and id2 not in nodes[id1].neighbors:
        nodes[id1].add_neighbor(id2)
        nodes[id2].add_neighbor(id1)
        total_edges_created += 1
        return True
    return False

start_real_time = time.time()
beacon_cycles = 0

u_init = create_new_node()
v_init = create_new_node()
connect_nodes(u_init, v_init)

stunning_delays_recorded = []

while len(nodes) < 1000:
    is_burst = random.random() < 0.15
    num_events = random.randint(3, 8) if is_burst else 1
    if is_burst:
        burst_events += 1

    for _ in range(num_events):
        if len(nodes) >= 1000:
            break

        situation = random.choice([1, 2, 3])
        total_joins += 1

        if situation == 1:
            n_keys = list(nodes.keys())
            u = random.choice(n_keys)
            candidates = [v for v in n_keys if v != u and v not in nodes[u].neighbors]
            if candidates:
                v = random.choice(candidates)
                connect_nodes(u, v)
            else:
                v = create_new_node()
                connect_nodes(u, v)
        elif situation == 2:
            u = random.choice(list(nodes.keys()))
            v = create_new_node()
            connect_nodes(u, v)
        elif situation == 3:
            u = create_new_node()
            if len(nodes) < 1000:
                v = create_new_node()
                connect_nodes(u, v)

    beacon_epochs = random.randint(3, 5)
    for epoch in range(beacon_epochs):
        beacon_cycles += 1
        for node_id, node in list(nodes.items()):
            total_beacons_generated += 1
            node.receive_signal(node.pos, node.pos, node.pos, 'beacon', sim_clock)
            for nbr_id in list(node.neighbors):
                nbr_node = nodes[nbr_id]
                total_beacons_generated += 1
                nbr_node.receive_signal(node.pos, nbr_node.pos, node.pos, 'beacon', sim_clock)

        sim_clock += 0.600

        for node in list(nodes.values()):
            stunning_delays_recorded.append(node.stunning_delay_ms)
            if sim_clock >= node.stun_until_time:
                node.process_accumulated_inputs()
                pub_count = node.publish_coalesced_tensor(nodes)
                total_coalesced_tensor_publishes += pub_count

    if len(nodes) >= 10:
        for _ in range(5):
            src_id = random.choice(list(nodes.keys()))
            tgt_id = random.choice(list(nodes.keys()))
            if src_id != tgt_id:
                best_k = nodes[src_id].select_k_best_gossip_neighbors(nodes[src_id].pos, nodes[tgt_id].pos, nodes, k=2)
                for nbr_id in best_k:
                    total_announces_sent += 1
                    nodes[nbr_id].receive_signal(nodes[src_id].pos, nodes[tgt_id].pos, nodes[src_id].pos, 'announce', sim_clock)

exec_real_time = round(time.time() - start_real_time, 2)
degrees = [len(n.neighbors) for n in nodes.values()]
avg_degree = np.mean(degrees)

print("="*65)
print("  NEURAL HAAR SPATIAL TENSOR SIMULATION (STUNNING & COALESCING)")
print("="*65)
print(f"Total Nodes Reached:                   {len(nodes)}")
print(f"Total Graph Edges Created:             {total_edges_created}")
print(f"Total Node Join Events:                {total_joins}")
print(f"Total Join Burst Events:               {burst_events}")
print(f"Total Beacon Cycles Executed:         {beacon_cycles}")
print(f"Total Raw Inputs Accumulated:          {total_beacons_generated:,}")
print(f"Total Coalesced Tensor Publishes:      {total_coalesced_tensor_publishes:,}")
print(f"Messages Saved via Tensor Coalescing:  {(1 - total_coalesced_tensor_publishes/total_beacons_generated)*100:.2f}%")
print(f"Simulation Execution Time:             {exec_real_time} seconds")
print("-" * 65)
print("SIGMOIDAL STUNNING DYNAMICS")
print(f"Minimum Observed Stunning Delay:       {np.min(stunning_delays_recorded):.2f} ms")
print(f"Average Observed Stunning Delay:       {np.mean(stunning_delays_recorded):.2f} ms")
print(f"Maximum Observed Stunning Delay:       {np.max(stunning_delays_recorded):.2f} ms")
print("-" * 65)
print("COALESCED TENSOR FRAME STRUCTURE")
sample_node = nodes[0]
print(f"Tensor Payload Format:                 Embedding U (8) x Embedding V (8) x MsgCount")
print(f"Sample Node 0 Coalesced Tensor Frame: {sample_node.coalesced_tensor_out}")
print("="*65)
