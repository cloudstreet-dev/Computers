# Chapter 14: Distributed Systems

## Introduction

Distributed systems consist of multiple autonomous computers that communicate through a network to achieve a common goal. They appear to users as a single coherent system despite being composed of independent nodes. This chapter explores the fundamental concepts, algorithms, and challenges of distributed computing, from consensus protocols to modern cloud architectures.

## 14.1 Distributed System Fundamentals

### Characteristics and Challenges

**Key Characteristics:**
- Concurrency: Multiple activities happen simultaneously
- No global clock: Nodes have independent clocks
- Independent failures: Components can fail independently
- Message passing: Communication through network messages

**Fundamental Challenges:**

```python
class DistributedSystemChallenges:
    """Eight fallacies of distributed computing"""
    
    def network_is_reliable(self):
        # Networks fail, packets drop, connections timeout
        try:
            response = send_message(node, message)
        except NetworkException:
            # Handle network failures
            implement_retry_logic()
    
    def latency_is_zero(self):
        # Network communication takes time
        start = time.time()
        response = remote_call()
        latency = time.time() - start  # Can be significant
    
    def bandwidth_is_infinite(self):
        # Network capacity is limited
        if message_size > MAX_PRACTICAL_SIZE:
            chunk_and_send(message)
    
    def network_is_secure(self):
        # Networks can be compromised
        encrypted_msg = encrypt(message)
        signed_msg = sign(encrypted_msg)
        send(signed_msg)
    
    def topology_doesnt_change(self):
        # Network topology is dynamic
        maintain_routing_table()
        handle_node_joins_and_leaves()
    
    def there_is_one_administrator(self):
        # Multiple administrative domains
        handle_different_policies()
        negotiate_protocols()
    
    def transport_cost_is_zero(self):
        # Network communication has costs
        minimize_data_transfer()
        use_caching()
    
    def network_is_homogeneous(self):
        # Different protocols and formats
        implement_protocol_translation()
        handle_data_serialization()
```

### System Models

```python
# Synchronous vs Asynchronous Systems
class SystemModel:
    def __init__(self, model_type):
        self.model_type = model_type
    
    def synchronous_model(self):
        """
        - Known upper bound on message delay
        - Known clock drift rate
        - Known processing time bounds
        """
        MAX_MESSAGE_DELAY = 100  # ms
        MAX_CLOCK_DRIFT = 0.01   # 1%
        MAX_PROCESSING_TIME = 50  # ms
        
        # Can use timeouts reliably
        timeout = 2 * MAX_MESSAGE_DELAY + MAX_PROCESSING_TIME
        
    def asynchronous_model(self):
        """
        - No bounds on message delay
        - No bounds on clock drift
        - No bounds on processing time
        """
        # Cannot use timeouts for failure detection
        # Must use other mechanisms (heartbeats, gossip)
        
    def partially_synchronous_model(self):
        """
        - Bounds exist but are unknown
        - Or bounds hold eventually
        """
        # Adaptive timeout mechanisms
        timeout = estimate_timeout()
        if timeout_exceeded:
            timeout *= 2  # Exponential backoff
```

## 14.2 Time and Ordering

### Logical Clocks

```python
class LamportClock:
    """Lamport's logical clock for causal ordering"""
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.counter = 0
    
    def local_event(self):
        self.counter += 1
        return self.counter
    
    def send_message(self, message):
        self.counter += 1
        message['timestamp'] = self.counter
        return message
    
    def receive_message(self, message):
        self.counter = max(self.counter, message['timestamp']) + 1
        return self.counter
    
    def happens_before(self, event1, event2):
        """Check if event1 → event2"""
        return event1['timestamp'] < event2['timestamp']

class VectorClock:
    """Vector clocks for detecting causality"""
    
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.vector = [0] * num_nodes
    
    def local_event(self):
        self.vector[self.node_id] += 1
        return self.vector.copy()
    
    def send_message(self, message):
        self.vector[self.node_id] += 1
        message['vector_clock'] = self.vector.copy()
        return message
    
    def receive_message(self, message):
        # Update vector clock
        for i in range(len(self.vector)):
            self.vector[i] = max(self.vector[i], message['vector_clock'][i])
        self.vector[self.node_id] += 1
        return self.vector.copy()
    
    def compare(self, vc1, vc2):
        """Compare two vector clocks"""
        less = False
        greater = False
        
        for i in range(len(vc1)):
            if vc1[i] < vc2[i]:
                less = True
            elif vc1[i] > vc2[i]:
                greater = True
        
        if less and not greater:
            return 'BEFORE'  # vc1 → vc2
        elif greater and not less:
            return 'AFTER'   # vc2 → vc1
        elif not less and not greater:
            return 'EQUAL'   # vc1 = vc2
        else:
            return 'CONCURRENT'  # vc1 || vc2
```

### Physical Clock Synchronization

```python
class NTPClient:
    """Network Time Protocol synchronization"""
    
    def synchronize(self, server):
        # Record local time before request
        t1 = self.local_time()
        
        # Send request to server
        request = {'type': 'TIME_REQUEST', 'client_time': t1}
        send_to_server(request)
        
        # Server records receive and send times
        # t2 = server receive time
        # t3 = server send time
        
        # Receive response
        response = receive_from_server()
        t4 = self.local_time()
        
        t2 = response['server_receive_time']
        t3 = response['server_send_time']
        
        # Calculate offset and round-trip delay
        delay = ((t4 - t1) - (t3 - t2)) / 2
        offset = ((t2 - t1) + (t3 - t4)) / 2
        
        # Adjust local clock
        self.adjust_clock(offset)
        
        return offset, delay

class PTPClock:
    """Precision Time Protocol for high accuracy"""
    
    def __init__(self):
        self.master_offset = 0
        self.path_delay = 0
    
    def sync_phase(self, master):
        # Master sends Sync message
        sync_msg = master.send_sync()
        t1 = sync_msg['timestamp']
        
        # Slave receives Sync
        t2 = self.receive_timestamp()
        
        # Master sends Follow_up with precise t1
        follow_up = master.send_follow_up(t1)
        
        # Calculate offset
        self.master_offset = t2 - t1 - self.path_delay
    
    def delay_phase(self, master):
        # Slave sends Delay_req
        t3 = self.send_delay_req_timestamp()
        
        # Master receives and responds
        t4 = master.receive_timestamp()
        delay_resp = master.send_delay_resp(t4)
        
        # Calculate path delay
        self.path_delay = ((t2 - t1) + (t4 - t3)) / 2
```

## 14.3 Distributed Coordination

### Mutual Exclusion

```python
class CentralizedMutex:
    """Centralized mutual exclusion with coordinator"""
    
    def __init__(self, is_coordinator=False):
        self.is_coordinator = is_coordinator
        self.token_holder = None
        self.request_queue = []
    
    def request_critical_section(self):
        if self.is_coordinator:
            # Coordinator logic
            if self.token_holder is None:
                return True  # Grant immediately
            else:
                self.request_queue.append(requester)
                return False  # Queue request
        else:
            # Client logic
            send_to_coordinator('REQUEST')
            response = wait_for_response()
            return response == 'GRANTED'
    
    def release_critical_section(self):
        if self.is_coordinator:
            self.token_holder = None
            if self.request_queue:
                next_holder = self.request_queue.pop(0)
                send_to_node(next_holder, 'GRANTED')
                self.token_holder = next_holder
        else:
            send_to_coordinator('RELEASE')

class TokenRingMutex:
    """Token-based mutual exclusion"""
    
    def __init__(self, node_id, ring_size):
        self.node_id = node_id
        self.next_node = (node_id + 1) % ring_size
        self.has_token = (node_id == 0)  # Node 0 starts with token
        self.wants_token = False
    
    def request_critical_section(self):
        self.wants_token = True
        while not self.has_token:
            time.sleep(0.01)  # Wait for token
    
    def release_critical_section(self):
        self.wants_token = False
        self.pass_token()
    
    def pass_token(self):
        if not self.wants_token:
            self.has_token = False
            send_to_node(self.next_node, 'TOKEN')
    
    def receive_token(self):
        self.has_token = True
        if not self.wants_token:
            self.pass_token()

class RicartAgrawalaMutex:
    """Distributed mutual exclusion without coordinator"""
    
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.timestamp = 0
        self.request_timestamp = 0
        self.replies_received = 0
        self.deferred_replies = []
        self.in_critical_section = False
    
    def request_critical_section(self):
        self.timestamp += 1
        self.request_timestamp = self.timestamp
        self.replies_received = 0
        
        # Send request to all other nodes
        for i in range(self.num_nodes):
            if i != self.node_id:
                send_to_node(i, {
                    'type': 'REQUEST',
                    'timestamp': self.request_timestamp,
                    'node_id': self.node_id
                })
        
        # Wait for replies from all nodes
        while self.replies_received < self.num_nodes - 1:
            time.sleep(0.01)
        
        self.in_critical_section = True
    
    def handle_request(self, request):
        request_ts = request['timestamp']
        request_node = request['node_id']
        
        if not self.in_critical_section and \
           (self.request_timestamp == 0 or 
            request_ts < self.request_timestamp or
            (request_ts == self.request_timestamp and request_node < self.node_id)):
            # Reply immediately
            send_to_node(request_node, {'type': 'REPLY'})
        else:
            # Defer reply
            self.deferred_replies.append(request_node)
    
    def release_critical_section(self):
        self.in_critical_section = False
        self.request_timestamp = 0
        
        # Send deferred replies
        for node in self.deferred_replies:
            send_to_node(node, {'type': 'REPLY'})
        self.deferred_replies = []
```

### Leader Election

```python
class BullyAlgorithm:
    """Bully algorithm for leader election"""
    
    def __init__(self, node_id, node_ids):
        self.node_id = node_id
        self.node_ids = node_ids
        self.leader = max(node_ids)  # Initially highest ID
        self.election_in_progress = False
    
    def start_election(self):
        self.election_in_progress = True
        higher_nodes = [n for n in self.node_ids if n > self.node_id]
        
        if not higher_nodes:
            # This node has highest ID
            self.become_leader()
        else:
            # Send election message to higher nodes
            responses = []
            for node in higher_nodes:
                response = send_with_timeout(node, 'ELECTION', timeout=T)
                if response:
                    responses.append(response)
            
            if not responses:
                # No higher node responded
                self.become_leader()
            else:
                # Wait for coordinator message
                self.wait_for_coordinator()
    
    def handle_election_message(self, from_node):
        if from_node < self.node_id:
            # Respond and start own election
            send_to_node(from_node, 'OK')
            if not self.election_in_progress:
                self.start_election()
    
    def become_leader(self):
        self.leader = self.node_id
        # Announce to all nodes
        for node in self.node_ids:
            if node != self.node_id:
                send_to_node(node, {'type': 'COORDINATOR', 'leader': self.node_id})

class RingElection:
    """Ring-based leader election"""
    
    def __init__(self, node_id, ring_size):
        self.node_id = node_id
        self.next_node = (node_id + 1) % ring_size
        self.leader = None
        self.participant = False
    
    def start_election(self):
        self.participant = True
        send_to_node(self.next_node, {
            'type': 'ELECTION',
            'candidates': [self.node_id]
        })
    
    def handle_election_message(self, message):
        candidates = message['candidates']
        
        if self.node_id in candidates:
            # Message completed circle
            leader = max(candidates)
            self.announce_leader(leader)
        else:
            # Forward message
            if not self.participant:
                candidates.append(self.node_id)
                self.participant = True
            
            send_to_node(self.next_node, {
                'type': 'ELECTION',
                'candidates': candidates
            })
    
    def announce_leader(self, leader):
        self.leader = leader
        send_to_node(self.next_node, {
            'type': 'COORDINATOR',
            'leader': leader
        })
```

## 14.4 Consensus Protocols

### Byzantine Generals Problem

```python
class ByzantineGeneral:
    """Byzantine fault-tolerant consensus"""
    
    def __init__(self, general_id, is_commander=False):
        self.general_id = general_id
        self.is_commander = is_commander
        self.received_values = {}
    
    def byzantine_agreement(self, value, generals, faulty_count):
        """
        Achieves consensus with f faulty nodes if total nodes >= 3f + 1
        """
        n = len(generals)
        f = faulty_count
        
        if n < 3 * f + 1:
            raise ValueError("Need at least 3f+1 nodes for f failures")
        
        if self.is_commander:
            # Commander sends value to all lieutenants
            for general in generals:
                if general != self.general_id:
                    send_to_general(general, value)
        else:
            # Lieutenant receives and exchanges values
            self.om_algorithm(value, f, generals)
    
    def om_algorithm(self, value, m, generals):
        """Oral Messages algorithm OM(m)"""
        if m == 0:
            # Base case: use received value
            return value
        else:
            # Receive values from commander/sender
            values = []
            
            # Each lieutenant sends value to others
            for general in generals:
                if general != self.general_id:
                    received = receive_from_general(general)
                    # Recursively run OM(m-1)
                    result = self.om_algorithm(received, m-1, 
                                              [g for g in generals if g != general])
                    values.append(result)
            
            # Take majority
            return self.majority(values)
    
    def majority(self, values):
        from collections import Counter
        counts = Counter(values)
        return counts.most_common(1)[0][0]
```

### Paxos

```python
class PaxosNode:
    """Basic Paxos consensus protocol"""
    
    def __init__(self, node_id, nodes):
        self.node_id = node_id
        self.nodes = nodes
        
        # Proposer state
        self.proposal_number = 0
        self.proposed_value = None
        
        # Acceptor state
        self.promised_proposal = None
        self.accepted_proposal = None
        self.accepted_value = None
    
    # Proposer methods
    def propose(self, value):
        # Phase 1a: Prepare
        self.proposal_number += 1
        n = (self.proposal_number, self.node_id)  # Unique proposal number
        
        promises = []
        for node in self.nodes:
            promise = self.send_prepare(node, n)
            if promise:
                promises.append(promise)
        
        # Need majority of promises
        if len(promises) > len(self.nodes) // 2:
            # Phase 2a: Accept
            # Choose value from highest numbered accepted proposal
            # or use own value if none accepted
            accepted_proposals = [p for p in promises if p['accepted_value']]
            
            if accepted_proposals:
                highest = max(accepted_proposals, 
                            key=lambda p: p['accepted_proposal'])
                chosen_value = highest['accepted_value']
            else:
                chosen_value = value
            
            # Send accept requests
            accepts = []
            for node in self.nodes:
                accepted = self.send_accept(node, n, chosen_value)
                if accepted:
                    accepts.append(accepted)
            
            if len(accepts) > len(self.nodes) // 2:
                # Value chosen!
                return chosen_value
        
        return None  # Failed to achieve consensus
    
    # Acceptor methods
    def handle_prepare(self, proposal_number):
        if self.promised_proposal is None or \
           proposal_number > self.promised_proposal:
            # Promise not to accept proposals numbered less than n
            self.promised_proposal = proposal_number
            
            return {
                'promise': True,
                'accepted_proposal': self.accepted_proposal,
                'accepted_value': self.accepted_value
            }
        else:
            return {'promise': False}
    
    def handle_accept(self, proposal_number, value):
        if self.promised_proposal is None or \
           proposal_number >= self.promised_proposal:
            # Accept the proposal
            self.promised_proposal = proposal_number
            self.accepted_proposal = proposal_number
            self.accepted_value = value
            
            return {'accepted': True}
        else:
            return {'accepted': False}
```

### Raft

```python
class RaftNode:
    """Raft consensus protocol - easier to understand than Paxos"""
    
    def __init__(self, node_id, nodes):
        self.node_id = node_id
        self.nodes = nodes
        
        # Persistent state
        self.current_term = 0
        self.voted_for = None
        self.log = []
        
        # Volatile state
        self.state = 'FOLLOWER'
        self.leader_id = None
        
        # Leader state
        self.next_index = {}
        self.match_index = {}
        
        # Election timeout
        self.reset_election_timeout()
    
    def reset_election_timeout(self):
        # Random timeout between 150-300ms
        self.election_timeout = random.uniform(0.15, 0.3)
        self.last_heartbeat = time.time()
    
    def start_election(self):
        self.state = 'CANDIDATE'
        self.current_term += 1
        self.voted_for = self.node_id
        votes = 1  # Vote for self
        
        # Request votes from other nodes
        for node in self.nodes:
            if node != self.node_id:
                vote = self.request_vote(node)
                if vote:
                    votes += 1
        
        if votes > len(self.nodes) // 2:
            self.become_leader()
        else:
            self.state = 'FOLLOWER'
    
    def request_vote(self, node):
        last_log_index = len(self.log) - 1
        last_log_term = self.log[last_log_index]['term'] if self.log else 0
        
        request = {
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': last_log_index,
            'last_log_term': last_log_term
        }
        
        response = send_to_node(node, {'type': 'VOTE_REQUEST', 'data': request})
        
        if response['term'] > self.current_term:
            self.current_term = response['term']
            self.state = 'FOLLOWER'
            self.voted_for = None
            return False
        
        return response['vote_granted']
    
    def handle_vote_request(self, request):
        if request['term'] < self.current_term:
            return {'term': self.current_term, 'vote_granted': False}
        
        if request['term'] > self.current_term:
            self.current_term = request['term']
            self.state = 'FOLLOWER'
            self.voted_for = None
        
        # Check if candidate's log is at least as up-to-date
        last_log_index = len(self.log) - 1
        last_log_term = self.log[last_log_index]['term'] if self.log else 0
        
        log_ok = (request['last_log_term'] > last_log_term or
                 (request['last_log_term'] == last_log_term and
                  request['last_log_index'] >= last_log_index))
        
        if (self.voted_for is None or self.voted_for == request['candidate_id']) and log_ok:
            self.voted_for = request['candidate_id']
            self.reset_election_timeout()
            return {'term': self.current_term, 'vote_granted': True}
        
        return {'term': self.current_term, 'vote_granted': False}
    
    def become_leader(self):
        self.state = 'LEADER'
        self.leader_id = self.node_id
        
        # Initialize leader state
        for node in self.nodes:
            self.next_index[node] = len(self.log)
            self.match_index[node] = 0
        
        # Send heartbeats
        self.send_heartbeats()
    
    def append_entries(self, entries):
        """Leader appends new entries and replicates"""
        if self.state != 'LEADER':
            return False
        
        # Append to own log
        for entry in entries:
            entry['term'] = self.current_term
            self.log.append(entry)
        
        # Replicate to followers
        successes = 1  # Count self
        
        for node in self.nodes:
            if node != self.node_id:
                success = self.replicate_to_follower(node)
                if success:
                    successes += 1
        
        # Commit if majority replicates
        if successes > len(self.nodes) // 2:
            self.commit_index = len(self.log) - 1
            return True
        
        return False
```

## 14.5 Distributed Storage

### Consistent Hashing

```python
class ConsistentHash:
    """Consistent hashing for distributed storage"""
    
    def __init__(self, nodes=None, virtual_nodes=150):
        self.nodes = nodes or []
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self._build_ring()
    
    def _hash(self, key):
        return hashlib.md5(key.encode()).hexdigest()
    
    def _build_ring(self):
        for node in self.nodes:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = node
    
    def add_node(self, node):
        """Add node and rebalance data"""
        self.nodes.append(node)
        
        # Add virtual nodes to ring
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
        
        # Trigger data rebalancing
        self.rebalance_after_add(node)
    
    def remove_node(self, node):
        """Remove node and rebalance data"""
        if node not in self.nodes:
            return
        
        self.nodes.remove(node)
        
        # Find data that needs to be moved
        affected_keys = []
        for hash_value, n in list(self.ring.items()):
            if n == node:
                affected_keys.append(hash_value)
                del self.ring[hash_value]
        
        # Trigger data rebalancing
        self.rebalance_after_remove(node, affected_keys)
    
    def get_node(self, key):
        """Find node responsible for key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find first node clockwise from hash
        sorted_hashes = sorted(self.ring.keys())
        for ring_hash in sorted_hashes:
            if ring_hash >= hash_value:
                return self.ring[ring_hash]
        
        # Wrap around
        return self.ring[sorted_hashes[0]]
    
    def get_preference_list(self, key, n=3):
        """Get N nodes for replication"""
        if not self.ring:
            return []
        
        hash_value = self._hash(key)
        sorted_hashes = sorted(self.ring.keys())
        
        # Find starting position
        start_idx = 0
        for i, ring_hash in enumerate(sorted_hashes):
            if ring_hash >= hash_value:
                start_idx = i
                break
        
        # Collect N unique nodes
        preference_list = []
        seen_nodes = set()
        
        for i in range(len(sorted_hashes)):
            idx = (start_idx + i) % len(sorted_hashes)
            node = self.ring[sorted_hashes[idx]]
            
            if node not in seen_nodes:
                preference_list.append(node)
                seen_nodes.add(node)
                
                if len(preference_list) == n:
                    break
        
        return preference_list
```

### Distributed Hash Table (DHT)

```python
class ChordNode:
    """Chord DHT implementation"""
    
    def __init__(self, node_id, m=160):  # m-bit identifier space
        self.node_id = node_id
        self.m = m
        self.finger_table = [None] * m
        self.predecessor = None
        self.successor = None
        self.data = {}
    
    def find_successor(self, key):
        """Find node responsible for key"""
        if self.in_range(key, self.node_id, self.successor.node_id, inclusive_right=True):
            return self.successor
        else:
            # Find closest preceding node
            n = self.closest_preceding_node(key)
            return n.find_successor(key)
    
    def closest_preceding_node(self, key):
        """Find closest node that precedes key"""
        for i in range(self.m - 1, -1, -1):
            if self.finger_table[i] and \
               self.in_range(self.finger_table[i].node_id, self.node_id, key):
                return self.finger_table[i]
        return self
    
    def join(self, known_node=None):
        """Join Chord ring"""
        if known_node:
            self.predecessor = None
            self.successor = known_node.find_successor(self.node_id)
            self.update_finger_table()
            self.move_keys()
        else:
            # First node in ring
            self.predecessor = self
            self.successor = self
            for i in range(self.m):
                self.finger_table[i] = self
    
    def update_finger_table(self):
        """Update finger table entries"""
        for i in range(self.m):
            start = (self.node_id + 2**i) % (2**self.m)
            self.finger_table[i] = self.find_successor(start)
    
    def stabilize(self):
        """Periodic stabilization for ring maintenance"""
        x = self.successor.predecessor
        if x and self.in_range(x.node_id, self.node_id, self.successor.node_id):
            self.successor = x
        self.successor.notify(self)
    
    def notify(self, node):
        """Handle notification from potential predecessor"""
        if not self.predecessor or \
           self.in_range(node.node_id, self.predecessor.node_id, self.node_id):
            self.predecessor = node
            self.move_keys()
    
    def move_keys(self):
        """Transfer keys to responsible nodes"""
        if not self.predecessor:
            return
        
        keys_to_move = []
        for key in self.data:
            if not self.in_range(key, self.predecessor.node_id, self.node_id, 
                                inclusive_right=True):
                keys_to_move.append(key)
        
        for key in keys_to_move:
            responsible_node = self.find_successor(key)
            responsible_node.store(key, self.data[key])
            del self.data[key]
    
    def in_range(self, value, start, end, inclusive_right=False):
        """Check if value is in range on ring"""
        if start == end:
            return True
        elif start < end:
            if inclusive_right:
                return start < value <= end
            else:
                return start < value < end
        else:  # Wrap around
            if inclusive_right:
                return value > start or value <= end
            else:
                return value > start or value < end
```

## 14.6 MapReduce and Stream Processing

### MapReduce Framework

```python
class MapReduceJob:
    """Simple MapReduce implementation"""
    
    def __init__(self, map_func, reduce_func, num_workers=4):
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.num_workers = num_workers
    
    def run(self, input_data):
        # Split input into chunks
        chunks = self.split_input(input_data, self.num_workers)
        
        # Map phase
        map_results = []
        with multiprocessing.Pool(self.num_workers) as pool:
            map_results = pool.map(self.map_worker, chunks)
        
        # Shuffle and sort
        intermediate = self.shuffle_and_sort(map_results)
        
        # Reduce phase
        reduce_inputs = self.group_by_key(intermediate)
        with multiprocessing.Pool(self.num_workers) as pool:
            results = pool.map(self.reduce_worker, reduce_inputs.items())
        
        return dict(results)
    
    def map_worker(self, chunk):
        """Execute map function on chunk"""
        results = []
        for item in chunk:
            # Map function emits (key, value) pairs
            for key, value in self.map_func(item):
                results.append((key, value))
        return results
    
    def reduce_worker(self, key_values):
        """Execute reduce function on grouped values"""
        key, values = key_values
        result = self.reduce_func(key, values)
        return (key, result)
    
    def shuffle_and_sort(self, map_results):
        """Combine and sort map outputs"""
        combined = []
        for result in map_results:
            combined.extend(result)
        return sorted(combined, key=lambda x: x[0])
    
    def group_by_key(self, sorted_pairs):
        """Group values by key"""
        grouped = {}
        for key, value in sorted_pairs:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)
        return grouped

# Example: Word count
def word_count_map(document):
    """Map function for word count"""
    for word in document.split():
        yield (word.lower(), 1)

def word_count_reduce(word, counts):
    """Reduce function for word count"""
    return sum(counts)

# Usage
job = MapReduceJob(word_count_map, word_count_reduce)
result = job.run(documents)
```

### Stream Processing

```python
class StreamProcessor:
    """Distributed stream processing"""
    
    def __init__(self):
        self.operators = []
        self.sources = []
        self.sinks = []
    
    def add_source(self, source):
        self.sources.append(source)
        return self
    
    def map(self, func):
        self.operators.append(('MAP', func))
        return self
    
    def filter(self, func):
        self.operators.append(('FILTER', func))
        return self
    
    def window(self, size, slide=None):
        self.operators.append(('WINDOW', {'size': size, 'slide': slide or size}))
        return self
    
    def aggregate(self, func, window_size):
        self.operators.append(('AGGREGATE', {'func': func, 'window': window_size}))
        return self
    
    def add_sink(self, sink):
        self.sinks.append(sink)
        return self
    
    def run(self):
        """Execute stream processing pipeline"""
        for source in self.sources:
            stream = source.get_stream()
            
            for op_type, op_data in self.operators:
                if op_type == 'MAP':
                    stream = self.apply_map(stream, op_data)
                elif op_type == 'FILTER':
                    stream = self.apply_filter(stream, op_data)
                elif op_type == 'WINDOW':
                    stream = self.apply_window(stream, op_data)
                elif op_type == 'AGGREGATE':
                    stream = self.apply_aggregate(stream, op_data)
            
            for sink in self.sinks:
                sink.write(stream)

class EventTimeWindow:
    """Event-time windowing for out-of-order events"""
    
    def __init__(self, window_size, watermark_delay):
        self.window_size = window_size
        self.watermark_delay = watermark_delay
        self.windows = {}
        self.watermark = 0
    
    def process_event(self, event):
        event_time = event['timestamp']
        window_start = (event_time // self.window_size) * self.window_size
        
        if window_start not in self.windows:
            self.windows[window_start] = []
        
        self.windows[window_start].append(event)
        
        # Update watermark
        self.watermark = max(self.watermark, event_time - self.watermark_delay)
        
        # Emit completed windows
        completed = []
        for window_start, events in list(self.windows.items()):
            if window_start + self.window_size <= self.watermark:
                completed.append((window_start, events))
                del self.windows[window_start]
        
        return completed
```

## 14.7 Distributed Transactions

### Two-Phase Commit (2PC)

```python
class TwoPhaseCommitCoordinator:
    """2PC coordinator for distributed transactions"""
    
    def __init__(self, participants):
        self.participants = participants
        self.transaction_log = []
    
    def execute_transaction(self, transaction):
        transaction_id = generate_transaction_id()
        
        # Phase 1: Voting phase
        self.log('BEGIN', transaction_id)
        
        votes = []
        for participant in self.participants:
            vote = self.request_vote(participant, transaction)
            votes.append(vote)
            
            if vote == 'NO':
                # Any NO vote aborts transaction
                self.abort_transaction(transaction_id)
                return False
        
        # All voted YES
        self.log('COMMIT', transaction_id)
        
        # Phase 2: Commit phase
        for participant in self.participants:
            self.send_decision(participant, 'COMMIT', transaction_id)
        
        self.log('END', transaction_id)
        return True
    
    def abort_transaction(self, transaction_id):
        self.log('ABORT', transaction_id)
        
        for participant in self.participants:
            self.send_decision(participant, 'ABORT', transaction_id)
        
        self.log('END', transaction_id)
    
    def handle_participant_failure(self, participant, transaction_id):
        """Handle participant failure during 2PC"""
        # Check transaction state
        state = self.get_transaction_state(transaction_id)
        
        if state == 'VOTING':
            # Abort if failure during voting
            self.abort_transaction(transaction_id)
        elif state == 'COMMITTED':
            # Retry commit message
            self.send_decision(participant, 'COMMIT', transaction_id)
        elif state == 'ABORTED':
            # Retry abort message
            self.send_decision(participant, 'ABORT', transaction_id)

class TwoPhaseCommitParticipant:
    """2PC participant"""
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.prepared_transactions = {}
    
    def handle_prepare(self, transaction):
        # Validate and prepare transaction
        if self.can_commit(transaction):
            # Write prepare record to log
            self.log('PREPARE', transaction.id)
            self.prepared_transactions[transaction.id] = transaction
            return 'YES'
        else:
            return 'NO'
    
    def handle_commit(self, transaction_id):
        if transaction_id in self.prepared_transactions:
            transaction = self.prepared_transactions[transaction_id]
            self.commit_transaction(transaction)
            del self.prepared_transactions[transaction_id]
            self.log('COMMIT', transaction_id)
    
    def handle_abort(self, transaction_id):
        if transaction_id in self.prepared_transactions:
            transaction = self.prepared_transactions[transaction_id]
            self.abort_transaction(transaction)
            del self.prepared_transactions[transaction_id]
            self.log('ABORT', transaction_id)
```

## 14.8 Cloud Computing Patterns

### Service Discovery

```python
class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services = {}  # service_name -> [(instance_id, endpoint, metadata)]
        self.health_checks = {}
    
    def register(self, service_name, instance_id, endpoint, metadata=None):
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append({
            'instance_id': instance_id,
            'endpoint': endpoint,
            'metadata': metadata or {},
            'health': 'HEALTHY',
            'last_heartbeat': time.time()
        })
        
        # Start health checking
        self.start_health_check(service_name, instance_id)
    
    def deregister(self, service_name, instance_id):
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name]
                if s['instance_id'] != instance_id
            ]
    
    def discover(self, service_name, criteria=None):
        """Discover service instances"""
        if service_name not in self.services:
            return []
        
        instances = self.services[service_name]
        
        # Filter by criteria
        if criteria:
            instances = [i for i in instances if self.matches_criteria(i, criteria)]
        
        # Return only healthy instances
        return [i for i in instances if i['health'] == 'HEALTHY']
    
    def health_check(self, service_name, instance_id):
        """Periodic health check"""
        for instance in self.services.get(service_name, []):
            if instance['instance_id'] == instance_id:
                # Check heartbeat timeout
                if time.time() - instance['last_heartbeat'] > 30:
                    instance['health'] = 'UNHEALTHY'
                else:
                    # Perform active health check
                    if self.ping_endpoint(instance['endpoint']):
                        instance['health'] = 'HEALTHY'
                    else:
                        instance['health'] = 'UNHEALTHY'

class LoadBalancer:
    """Client-side load balancing"""
    
    def __init__(self, service_registry):
        self.registry = service_registry
        self.round_robin_counters = {}
    
    def get_instance(self, service_name, strategy='ROUND_ROBIN'):
        instances = self.registry.discover(service_name)
        
        if not instances:
            return None
        
        if strategy == 'ROUND_ROBIN':
            if service_name not in self.round_robin_counters:
                self.round_robin_counters[service_name] = 0
            
            idx = self.round_robin_counters[service_name]
            instance = instances[idx % len(instances)]
            self.round_robin_counters[service_name] += 1
            
            return instance
            
        elif strategy == 'RANDOM':
            return random.choice(instances)
            
        elif strategy == 'LEAST_CONNECTIONS':
            # Requires tracking active connections
            return min(instances, key=lambda i: i.get('connections', 0))
```

## Exercises

1. Implement a distributed key-value store with:
   - Consistent hashing for partitioning
   - Replication for fault tolerance
   - Eventual consistency

2. Create a vector clock implementation that:
   - Tracks causality
   - Detects concurrent events
   - Merges vector clocks correctly

3. Build a simple Raft consensus implementation:
   - Leader election
   - Log replication
   - Safety properties

4. Design a distributed mutex using:
   - Lamport's algorithm
   - Maekawa's algorithm
   - Compare performance

5. Implement a gossip protocol for:
   - Membership management
   - Failure detection
   - Information dissemination

6. Create a MapReduce job for:
   - Inverted index creation
   - PageRank calculation
   - Log analysis

7. Build a distributed transaction coordinator:
   - Two-phase commit
   - Failure handling
   - Recovery mechanism

8. Implement a DHT with:
   - Chord routing
   - Finger table maintenance
   - Key migration

9. Design a stream processing system:
   - Window operations
   - Watermark handling
   - Exactly-once semantics

10. Create a service mesh with:
    - Service discovery
    - Load balancing
    - Circuit breaking

## Summary

This chapter covered distributed systems fundamentals:

- Time and ordering in distributed systems require logical clocks
- Coordination protocols enable distributed mutual exclusion and leader election
- Consensus algorithms like Paxos and Raft provide agreement despite failures
- Distributed storage uses consistent hashing and DHTs for scalability
- MapReduce and stream processing handle large-scale data processing
- Distributed transactions maintain consistency across nodes
- Cloud patterns enable scalable, resilient services

Distributed systems are essential for building scalable, reliable applications that span multiple machines. Understanding these concepts is crucial for modern system design and cloud computing.