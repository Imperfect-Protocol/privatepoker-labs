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

def encode_polar_embedding(bearing, dist, num_bins=8):
    """Represent polar position as a compact spatial embedding vector (direction cone bins + scaled distance)."""
    bin_idx = int((bearing / (2 * np.pi)) * num_bins) % num_bins
    emb = np.zeros(num_bins + 1)
    emb[bin_idx] = 1.0
    emb[-1] = np.log1p(dist) # Log distance
    return emb

def blend_embeddings(emb_prev, emb_new, alpha=0.3):
    """EMA path-blending of trajectory embeddings."""
    return (1.0 - alpha) * emb_prev + alpha * emb_new

# Node representation in P2P graph
class P2PNode:
    def __init__(self, node_id, spatial_pos):
        self.id = node_id
        self.pos = spatial_pos
        self.neighbors = set()
        
        # Spatial-Aware Attention Tensor: Source_Cluster(8) x Target_Cluster(8) x Direction_Sector(8)
        # Initializes uniformly and updates via Hebbian self-organization
        self.tensor = np.ones((8, 8, 8), dtype=np.float32) * 0.1
        self.beacon_jitter_counter = random.uniform(0.8, 1.2)

    def add_neighbor(self, nbr_id):
        self.neighbors.add(nbr_id)

    def select_k_best_gossip_neighbors(self, src_pos, tgt_pos, node_dict, k=2):
        """Infer best k neighbors to route announcements using the self-learned spatial tensor."""
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
            
            # Tensor lookup for spatial directional alignment score
            score = self.tensor[src_bin, tgt_bin, nbr_dir_bin]
            # Soft-max with spatial distance component
            nbr_scores.append((score, nbr_id))
            
        nbr_scores.sort(key=lambda x: x[0], reverse=True)
        return [nbr_id for _, nbr_id in nbr_scores[:k]]

    def update_tensor_hebbian(self, src_pos, tgt_pos, nbr_pos, reward=0.2):
        """Hebbian spatial learning update upon receiving beacon/announcement path signal."""
        src_bin = int((cartesian_to_polar(self.pos, src_pos)[0] / (2 * np.pi)) * 8) % 8
        tgt_bin = int((cartesian_to_polar(self.pos, tgt_pos)[0] / (2 * np.pi)) * 8) % 8
        nbr_bin = int((cartesian_to_polar(self.pos, nbr_pos)[0] / (2 * np.pi)) * 8) % 8
        
        # Hebbian boost to active directional path sector
        self.tensor[src_bin, tgt_bin, nbr_bin] += reward
        # Soft max / normalization cap to prevent explosion
        self.tensor[src_bin, tgt_bin, nbr_bin] = min(5.0, self.tensor[src_bin, tgt_bin, nbr_bin])

print("Simulation core classes successfully initialized.")



nodes = {}
next_node_id = 0

total_beacons_sent = 0
total_announces_sent = 0
total_edges_created = 0
total_joins = 0
burst_events = 0

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

start_time = time.time()
beacon_cycles = 0

# Bootstrap graph with initial pair
u_init = create_new_node()
v_init = create_new_node()
connect_nodes(u_init, v_init)

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

    # Intermittent Beacon & Hebbian Learning Phase
    beacon_epochs = random.randint(3, 5)
    for epoch in range(beacon_epochs):
        beacon_cycles += 1
        for node_id, node in list(nodes.items()):
            total_beacons_sent += 1
            node.update_tensor_hebbian(node.pos, node.pos, node.pos, reward=0.05)
            
            for nbr_id in list(node.neighbors):
                nbr_node = nodes[nbr_id]
                total_beacons_sent += 1
                nbr_node.update_tensor_hebbian(node.pos, nbr_node.pos, node.pos, reward=0.15)
                
                for hop2_nbr_id in list(nbr_node.neighbors):
                    if hop2_nbr_id != node_id:
                        nodes[hop2_nbr_id].update_tensor_hebbian(node.pos, nodes[hop2_nbr_id].pos, nbr_node.pos, reward=0.10)

    # Targeted K-Best Gossip Announces
    if len(nodes) >= 10:
        for _ in range(5):
            src_id = random.choice(list(nodes.keys()))
            tgt_id = random.choice(list(nodes.keys()))
            if src_id != tgt_id:
                best_k = nodes[src_id].select_k_best_gossip_neighbors(nodes[src_id].pos, nodes[tgt_id].pos, nodes, k=2)
                total_announces_sent += len(best_k)

exec_time = round(time.time() - start_time, 2)

degrees = [len(n.neighbors) for n in nodes.values()]
avg_degree = np.mean(degrees)
max_degree = np.max(degrees)
min_degree = np.min(degrees)

print("="*60)
print("      NEURAL HAAR SPATIAL TENSOR SIMULATION RESULTS")
print("="*60)
print(f"Total Nodes Reached:               {len(nodes)}")
print(f"Total Graph Edges Created:         {total_edges_created}")
print(f"Total Node Join Events:            {total_joins}")
print(f"Total Join Burst Events:           {burst_events}")
print(f"Total Beacon Cycles Executed:     {beacon_cycles}")
print(f"Total Beacon Messages Sent:       {total_beacons_sent:,}")
print(f"Total Targeted Gossip Announces:  {total_announces_sent:,}")
print(f"Simulation Execution Time:         {exec_time} seconds")
print("-" * 60)
print("NETWORK TOPOLOGY METRICS")
print(f"Average Node Degree (Neighbors):   {avg_degree:.2f}")
print(f"Max Node Degree:                   {max_degree}")
print(f"Min Node Degree:                   {min_degree}")
print("-" * 60)
print("SPATIAL TENSOR & HEBBIAN LEARNING STATUS")
sample_node = nodes[0]
activated_tensor_cells = np.sum(sample_node.tensor > 0.1)
print(f"Sample Node 0 Tensor Shape:        {sample_node.tensor.shape} (Source x Target x Direction)")
print(f"Activated Spatial Cone Cells:     {activated_tensor_cells} / {sample_node.tensor.size}")
print(f"Highest Learned Cone Weight:      {np.max(sample_node.tensor):.4f}")
print("="*60)