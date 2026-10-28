# Let's write a pure discrete-event simulation (DES) version of the protocol using a priority queue (heapq).
# We will implement:
# 1. Priority queue event loop (no real-world clock, simulated time = event dispatch time).
# 2. Connection / Adapter structure with send(message, delay) and recv() -> metadata.
# 3. Timer events for node stunning and periodic jittered beacons.
# 4. Full 1,000 node growth with sigmoidal stunning, coalesced tensor frames, and Hebbian updates.

import heapq
import hashlib
import numpy as np
import random

random.seed(42)
np.random.seed(42)

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

def sigmoid_stunning_delay_ms(msg_count, k=0.1, mid=20):
    min_delay = 50.0  # ms
    max_delay = 500.0 # ms
    val = min_delay + (max_delay - min_delay) / (1.0 + np.exp(-k * (msg_count - mid)))
    return val # in ms

class EventQueue:
    def __init__(self):
        self.pq = []
        self.counter = 0 # tie-breaker for heap stability

    def push(self, dispatch_time, event_type, data):
        # dispatch_time in milliseconds
        heapq.heappush(self.pq, (dispatch_time, self.counter, event_type, data))
        self.counter += 1

    def pop(self):
        if not self.pq:
            return None, None, None
        t, _, event_type, data = heapq.heappop(self.pq)
        return t, event_type, data

    def __len__(self):
        return len(self.pq)

class ConnectionAdapter:
    def __init__(self, src_addr, dest_addr, event_queue):
        self.src_addr = src_addr
        self.dest_addr = dest_addr
        self.eq = event_queue
        self.last_sending_time = 0.0
        self.last_received_time = 0.0

    def send(self, message, delay_ms, current_time_ms):
        self.last_sending_time = current_time_ms
        self.last_received_time = current_time_ms + delay_ms
        
        event_data = {
            'connection': self,
            'message': message,
            'from': self.src_addr,
            'to': self.dest_addr,
            'sending_time': self.last_sending_time,
            'received_time': self.last_received_time
        }
        # Push to priority queue with dispatch time = received time
        self.eq.push(self.last_received_time, 'MESSAGE_DELIVERY', event_data)

    def recv(self, event_data):
        return (
            event_data['message'],
            event_data['from'],
            event_data['to'],
            event_data['sending_time'],
            event_data['received_time']
        )

class P2PNode:
    def __init__(self, node_id, spatial_pos):
        self.id = node_id
        self.pos = spatial_pos
        self.neighbors = {} # nbr_id -> ConnectionAdapter
        
        # Spatial-Aware Attention Tensor: Source(8) x Target(8) x Direction(8)
        self.tensor = np.ones((8, 8, 8), dtype=np.float32) * 0.1
        self.beacon_jitter_offset = random.uniform(800.0, 1200.0) # ms interval jitter base
        
        # Incoming Buffer & Stunning State
        self.incoming_buffer = []
        self.is_stunned = False
        self.stun_until_time = 0.0
        self.total_msgs_in_cycle = 0

        # Outgoing Coalesced Tensor Frame: Embedding U (8) x Embedding V (8) x Count
        self.coalesced_out = np.zeros((8, 8), dtype=np.int32)
        self.coalesced_publishes_sent = 0

    def add_neighbor(self, nbr_id, adapter):
        self.neighbors[nbr_id] = adapter

    def receive_incoming_message(self, message, from_addr, to_addr, sending_time, received_time, current_time_ms, eq):
        msg_type = message.get('type', 'beacon')
        src_pos = message.get('src_pos', self.pos)
        tgt_pos = message.get('tgt_pos', self.pos)
        
        self.incoming_buffer.append((src_pos, tgt_pos, from_addr, msg_type))
        self.total_msgs_in_cycle += 1

        # Update coalesced output buffer
        u_bin = int((cartesian_to_polar(self.pos, src_pos)[0] / (2 * np.pi)) * 8) % 8
        v_bin = int((cartesian_to_polar(self.pos, tgt_pos)[0] / (2 * np.pi)) * 8) % 8
        self.coalesced_out[u_bin, v_bin] += 1

        # Sigmoidal stunning delay (50ms to 500ms)
        stun_duration_ms = sigmoid_stunning_delay_ms(self.total_msgs_in_cycle)
        self.is_stunned = True
        new_stun_expiry = current_time_ms + stun_duration_ms
        
        if new_stun_expiry > self.stun_until_time:
            self.stun_until_time = new_stun_expiry
            # Schedule timer event for stunning expiration if not already scheduled
            eq.push(self.stun_until_time, 'STUN_EXPIRED', {'node_id': self.id})

    def process_stunning_expiration(self, current_time_ms, node_dict, eq):
        if not self.is_stunned or current_time_ms < self.stun_until_time:
            return

        # 1. Process accumulated inputs into Hebbian tensor
        for src_pos, tgt_pos, from_addr, msg_type in self.incoming_buffer:
            nbr_pos = node_dict[from_addr].pos if from_addr in node_dict else self.pos
            reward = 0.15 if msg_type == 'beacon' else 0.25
            self.update_tensor_hebbian(src_pos, tgt_pos, nbr_pos, reward=reward)
        self.incoming_buffer.clear()

        # 2. Publish coalesced tensor frame to neighbors
        if np.sum(self.coalesced_out) > 0:
            payload = {
                'type': 'coalesced_tensor',
                'src_pos': self.pos,
                'tgt_pos': self.pos,
                'tensor_frame': self.coalesced_out.copy()
            }
            for nbr_id, adapter in self.neighbors.items():
                adapter.send(payload, delay_ms=random.randint(15, 35), current_time_ms=current_time_ms)
                self.coalesced_publishes_sent += 1

            self.coalesced_out.fill(0)

        # Reset state
        self.total_msgs_in_cycle = 0
        self.is_stunned = False

    def update_tensor_hebbian(self, src_pos, tgt_pos, nbr_pos, reward=0.2):
        src_bin = int((cartesian_to_polar(self.pos, src_pos)[0] / (2 * np.pi)) * 8) % 8
        tgt_bin = int((cartesian_to_polar(self.pos, tgt_pos)[0] / (2 * np.pi)) * 8) % 8
        nbr_bin = int((cartesian_to_polar(self.pos, nbr_pos)[0] / (2 * np.pi)) * 8) % 8
        
        self.tensor[src_bin, tgt_bin, nbr_bin] += reward
        self.tensor[src_bin, tgt_bin, nbr_bin] = min(5.0, self.tensor[src_bin, tgt_bin, nbr_bin])

    def select_k_best_gossip_neighbors(self, src_pos, tgt_pos, k=2):
        if not self.neighbors:
            return []
        src_polar = cartesian_to_polar(self.pos, src_pos)
        tgt_polar = cartesian_to_polar(self.pos, tgt_pos)
        src_bin = int((src_polar[0] / (2 * np.pi)) * 8) % 8
        tgt_bin = int((tgt_polar[0] / (2 * np.pi)) * 8) % 8
        
        nbr_scores = []
        for nbr_id, adapter in self.neighbors.items():
            nbr_node = nodes[nbr_id]
            nbr_polar = cartesian_to_polar(self.pos, nbr_node.pos)
            nbr_dir_bin = int((nbr_polar[0] / (2 * np.pi)) * 8) % 8
            score = self.tensor[src_bin, tgt_bin, nbr_dir_bin]
            nbr_scores.append((score, nbr_id))
            
        nbr_scores.sort(key=lambda x: x[0], reverse=True)
        return [nbr_id for _, nbr_id in nbr_scores[:k]]

print("Discrete-Event Simulation core initialized.")