# Chapter 13: Computer Networks

## Introduction

Computer networks enable communication between devices, forming the backbone of the internet and modern computing. This chapter explores networking fundamentals, from physical transmission to application protocols, examining how data travels across networks reliably and securely. Understanding networking is essential for building distributed systems, web applications, and secure communications.

## 13.1 Network Fundamentals

### Network Types and Topologies

**Network Classifications by Scale:**
- PAN (Personal Area Network): ~10m (Bluetooth, USB)
- LAN (Local Area Network): ~1km (Ethernet, WiFi)
- MAN (Metropolitan Area Network): ~10km (City-wide)
- WAN (Wide Area Network): ~1000km+ (Internet)

**Network Topologies:**
```
Bus:        Star:         Ring:         Mesh:
A-B-C-D     A   B        A---B         A---B
             \ /          |   |         |\ /|
              C           D---C         | X |
             / \                        |/ \|
            D   E                       C---D
```

### OSI Model

The 7-layer reference model:

```
Layer 7: Application    | HTTP, FTP, SMTP, DNS
Layer 6: Presentation   | SSL/TLS, Compression, Encryption
Layer 5: Session        | NetBIOS, SQL, RPC
Layer 4: Transport      | TCP, UDP
Layer 3: Network        | IP, ICMP, Routing
Layer 2: Data Link      | Ethernet, WiFi, PPP
Layer 1: Physical       | Cables, Radio, Fiber
```

### TCP/IP Model

The practical 4-layer model:

```
Application Layer  | HTTP, FTP, DNS, SMTP
Transport Layer    | TCP, UDP
Internet Layer     | IP, ICMP, ARP
Network Access     | Ethernet, WiFi
```

## 13.2 Physical and Data Link Layers

### Physical Transmission

```python
# Signal encoding examples

def manchester_encode(bits):
    """Manchester encoding - transition in middle of bit period"""
    encoded = []
    for bit in bits:
        if bit == 0:
            encoded.extend([1, 0])  # High-to-low for 0
        else:
            encoded.extend([0, 1])  # Low-to-high for 1
    return encoded

def nrzi_encode(bits):
    """Non-Return to Zero Inverted - transition for 1, no change for 0"""
    current_level = 0
    encoded = []
    
    for bit in bits:
        if bit == 1:
            current_level = 1 - current_level  # Toggle
        encoded.append(current_level)
    
    return encoded

# Error detection
def calculate_crc(data, polynomial):
    """Cyclic Redundancy Check calculation"""
    # Convert to binary representation
    data_bits = ''.join(format(byte, '08b') for byte in data)
    
    # Append zeros for CRC
    crc_bits = len(polynomial) - 1
    padded = data_bits + '0' * crc_bits
    
    # Perform polynomial division
    padded = list(padded)
    polynomial = list(polynomial)
    
    for i in range(len(data_bits)):
        if padded[i] == '1':
            for j in range(len(polynomial)):
                padded[i + j] = str(int(padded[i + j]) ^ int(polynomial[j]))
    
    return ''.join(padded[-crc_bits:])
```

### Ethernet

```c
// Ethernet frame structure
struct ethernet_frame {
    uint8_t preamble[7];      // 7 bytes of 0xAA
    uint8_t sfd;              // Start frame delimiter 0xAB
    uint8_t dest_mac[6];      // Destination MAC address
    uint8_t src_mac[6];       // Source MAC address
    uint16_t ethertype;       // Protocol type (0x0800 for IP)
    uint8_t payload[46-1500]; // 46-1500 bytes
    uint32_t fcs;             // Frame check sequence (CRC)
};

// CSMA/CD - Carrier Sense Multiple Access with Collision Detection
void ethernet_transmit(struct ethernet_frame* frame) {
    while (1) {
        // 1. Carrier sense - check if medium is idle
        if (!is_medium_busy()) {
            start_transmission(frame);
            
            // 2. Collision detection during transmission
            if (collision_detected()) {
                send_jam_signal();
                
                // 3. Exponential backoff
                int attempts = 0;
                while (attempts < 16) {
                    int k = min(attempts, 10);
                    int r = random(0, (1 << k) - 1);
                    wait_time = r * SLOT_TIME;
                    wait(wait_time);
                    attempts++;
                    break;  // Retry transmission
                }
                
                if (attempts >= 16) {
                    return ERROR_TOO_MANY_COLLISIONS;
                }
            } else {
                return SUCCESS;  // Transmission complete
            }
        }
        
        // Wait for medium to become idle
        wait_for_idle();
    }
}
```

### WiFi (802.11)

```python
# CSMA/CA - Collision Avoidance for wireless
class WiFiStation:
    def __init__(self, address):
        self.address = address
        self.nav = 0  # Network Allocation Vector
        self.backoff_counter = 0
    
    def transmit(self, data, destination):
        # 1. Check if medium is idle
        if self.is_medium_idle():
            # 2. Wait for DIFS (Distributed Inter-Frame Space)
            self.wait(DIFS)
            
            # 3. Random backoff
            if self.backoff_counter == 0:
                self.backoff_counter = random.randint(0, CW_MIN)
            
            while self.backoff_counter > 0:
                if self.is_medium_idle():
                    self.backoff_counter -= 1
                    self.wait(SLOT_TIME)
                else:
                    # Freeze backoff counter
                    self.wait_for_idle()
                    self.wait(DIFS)
            
            # 4. Optional RTS/CTS for hidden terminal problem
            if len(data) > RTS_THRESHOLD:
                self.send_rts(destination)
                cts = self.wait_for_cts()
                if not cts:
                    return False
            
            # 5. Transmit data
            self.send_data(data, destination)
            
            # 6. Wait for ACK
            ack = self.wait_for_ack()
            return ack is not None
        
        return False
```

## 13.3 Network Layer

### IP Addressing

```python
class IPv4Address:
    def __init__(self, address):
        if isinstance(address, str):
            self.octets = [int(o) for o in address.split('.')]
        else:
            self.octets = [(address >> (24 - i*8)) & 0xFF for i in range(4)]
        self.value = sum(o << (24 - i*8) for i, o in enumerate(self.octets))
    
    def __str__(self):
        return '.'.join(str(o) for o in self.octets)
    
    def is_private(self):
        return (self.octets[0] == 10 or
                (self.octets[0] == 172 and 16 <= self.octets[1] <= 31) or
                (self.octets[0] == 192 and self.octets[1] == 168))
    
    def subnet_mask(self, prefix_length):
        mask = (0xFFFFFFFF << (32 - prefix_length)) & 0xFFFFFFFF
        return IPv4Address(mask)
    
    def network_address(self, prefix_length):
        mask = self.subnet_mask(prefix_length)
        return IPv4Address(self.value & mask.value)

class IPv6Address:
    def __init__(self, address):
        if isinstance(address, str):
            # Handle :: expansion
            parts = address.split('::')
            if len(parts) == 2:
                left = parts[0].split(':') if parts[0] else []
                right = parts[1].split(':') if parts[1] else []
                missing = 8 - len(left) - len(right)
                groups = left + ['0'] * missing + right
            else:
                groups = address.split(':')
            
            self.groups = [int(g, 16) for g in groups]
        else:
            self.groups = [(address >> (112 - i*16)) & 0xFFFF for i in range(8)]
    
    def __str__(self):
        # Compress longest sequence of zeros
        hex_groups = [f'{g:x}' for g in self.groups]
        result = ':'.join(hex_groups)
        
        # Find longest sequence of zeros
        import re
        result = re.sub(r'\b:?(?:0+:)+0+\b', '::', result, count=1)
        return result
```

### IP Packet Structure

```c
// IPv4 header
struct ipv4_header {
    uint8_t version_ihl;      // Version (4 bits) + Header length (4 bits)
    uint8_t tos;              // Type of service
    uint16_t total_length;    // Total packet length
    uint16_t identification;  // Fragment ID
    uint16_t flags_fragment;  // Flags (3 bits) + Fragment offset (13 bits)
    uint8_t ttl;              // Time to live
    uint8_t protocol;         // Next protocol (TCP=6, UDP=17)
    uint16_t checksum;        // Header checksum
    uint32_t src_addr;        // Source IP
    uint32_t dest_addr;       // Destination IP
    // Options (variable length)
};

// IPv6 header
struct ipv6_header {
    uint32_t version_class_label;  // Version + Traffic class + Flow label
    uint16_t payload_length;       // Payload length
    uint8_t next_header;           // Next header type
    uint8_t hop_limit;             // Hop limit (like TTL)
    uint8_t src_addr[16];          // Source IPv6 address
    uint8_t dest_addr[16];         // Destination IPv6 address
};
```

### Routing

```python
class RoutingTable:
    def __init__(self):
        self.entries = []
    
    def add_route(self, network, prefix_len, next_hop, interface, metric=1):
        self.entries.append({
            'network': network,
            'prefix_len': prefix_len,
            'next_hop': next_hop,
            'interface': interface,
            'metric': metric
        })
        # Sort by prefix length (longest prefix first)
        self.entries.sort(key=lambda x: x['prefix_len'], reverse=True)
    
    def lookup(self, destination):
        dest_addr = IPv4Address(destination)
        
        for entry in self.entries:
            network = IPv4Address(entry['network'])
            if dest_addr.network_address(entry['prefix_len']).value == \
               network.network_address(entry['prefix_len']).value:
                return entry
        
        return None  # No route found

# Distance Vector Routing (RIP-like)
class DistanceVectorRouter:
    def __init__(self, router_id):
        self.router_id = router_id
        self.routing_table = {}  # destination -> (next_hop, distance)
        self.neighbors = {}
    
    def update_routes(self, neighbor, their_routes):
        updated = False
        
        for dest, distance in their_routes.items():
            if dest == self.router_id:
                continue  # Skip route to self
            
            new_distance = distance + 1  # Add hop through neighbor
            
            if dest not in self.routing_table or \
               new_distance < self.routing_table[dest][1]:
                self.routing_table[dest] = (neighbor, new_distance)
                updated = True
        
        return updated
    
    def send_updates(self):
        # Send routing table to all neighbors
        for neighbor in self.neighbors:
            # Implement split horizon with poison reverse
            routes_to_send = {}
            for dest, (next_hop, distance) in self.routing_table.items():
                if next_hop == neighbor:
                    # Poison reverse
                    routes_to_send[dest] = float('inf')
                else:
                    routes_to_send[dest] = distance
            
            self.neighbors[neighbor].update_routes(self.router_id, routes_to_send)

# Link State Routing (OSPF-like)
class LinkStateRouter:
    def __init__(self, router_id):
        self.router_id = router_id
        self.link_state_db = {}  # Complete network topology
        self.neighbors = {}
        self.sequence_number = 0
    
    def flood_lsa(self):
        """Flood Link State Advertisement"""
        self.sequence_number += 1
        lsa = {
            'router_id': self.router_id,
            'sequence': self.sequence_number,
            'neighbors': {n: cost for n, cost in self.neighbors.items()}
        }
        
        # Send to all neighbors
        for neighbor in self.neighbors:
            neighbor.receive_lsa(lsa)
    
    def receive_lsa(self, lsa):
        router_id = lsa['router_id']
        
        # Check if newer than stored version
        if router_id not in self.link_state_db or \
           lsa['sequence'] > self.link_state_db[router_id]['sequence']:
            
            self.link_state_db[router_id] = lsa
            
            # Forward to other neighbors (flooding)
            for neighbor in self.neighbors:
                if neighbor.router_id != lsa['router_id']:
                    neighbor.receive_lsa(lsa)
            
            # Recalculate routes using Dijkstra
            self.calculate_shortest_paths()
    
    def calculate_shortest_paths(self):
        # Implement Dijkstra's algorithm on link_state_db
        pass
```

## 13.4 Transport Layer

### TCP (Transmission Control Protocol)

```python
class TCPConnection:
    def __init__(self):
        self.state = 'CLOSED'
        self.seq_num = random.randint(0, 2**32 - 1)
        self.ack_num = 0
        self.send_window = 65535
        self.recv_window = 65535
        self.congestion_window = 1  # Start with 1 MSS
        self.ssthresh = 65535
        self.rtt_estimator = RTTEstimator()
    
    def three_way_handshake_client(self, server_addr):
        # 1. Send SYN
        syn_packet = self.create_packet(flags='SYN', seq=self.seq_num)
        self.send(syn_packet, server_addr)
        self.state = 'SYN_SENT'
        
        # 2. Receive SYN-ACK
        syn_ack = self.receive()
        if syn_ack.flags == 'SYN,ACK':
            self.ack_num = syn_ack.seq + 1
            self.state = 'SYN_RECEIVED'
            
            # 3. Send ACK
            ack_packet = self.create_packet(
                flags='ACK',
                seq=self.seq_num + 1,
                ack=self.ack_num
            )
            self.send(ack_packet, server_addr)
            self.state = 'ESTABLISHED'
            return True
        
        return False
    
    def congestion_control(self, event):
        if event == 'ACK_RECEIVED':
            if self.congestion_window < self.ssthresh:
                # Slow start
                self.congestion_window *= 2
            else:
                # Congestion avoidance
                self.congestion_window += 1 / self.congestion_window
        
        elif event == 'TIMEOUT':
            # Timeout - severe congestion
            self.ssthresh = self.congestion_window / 2
            self.congestion_window = 1
        
        elif event == 'TRIPLE_DUP_ACK':
            # Fast recovery
            self.ssthresh = self.congestion_window / 2
            self.congestion_window = self.ssthresh + 3
    
    def sliding_window_send(self, data):
        window_size = min(self.send_window, self.congestion_window)
        sent_unacked = []
        
        while data:
            # Send up to window size
            while len(sent_unacked) < window_size and data:
                segment_size = min(MSS, len(data))
                segment = data[:segment_size]
                data = data[segment_size:]
                
                packet = self.create_packet(
                    data=segment,
                    seq=self.seq_num
                )
                self.send(packet)
                sent_unacked.append({
                    'seq': self.seq_num,
                    'data': segment,
                    'timestamp': time.time()
                })
                self.seq_num += len(segment)
            
            # Wait for ACKs
            ack = self.receive_ack()
            if ack:
                # Remove acknowledged segments
                sent_unacked = [s for s in sent_unacked if s['seq'] >= ack.ack_num]
                
                # Update RTT estimate
                for segment in sent_unacked:
                    if segment['seq'] < ack.ack_num:
                        rtt = time.time() - segment['timestamp']
                        self.rtt_estimator.update(rtt)
                
                self.congestion_control('ACK_RECEIVED')

class RTTEstimator:
    def __init__(self):
        self.srtt = 0  # Smoothed RTT
        self.rttvar = 0  # RTT variance
        self.rto = 1.0  # Retransmission timeout
        self.alpha = 0.125
        self.beta = 0.25
    
    def update(self, measured_rtt):
        if self.srtt == 0:
            self.srtt = measured_rtt
            self.rttvar = measured_rtt / 2
        else:
            self.rttvar = (1 - self.beta) * self.rttvar + \
                         self.beta * abs(self.srtt - measured_rtt)
            self.srtt = (1 - self.alpha) * self.srtt + \
                       self.alpha * measured_rtt
        
        self.rto = self.srtt + 4 * self.rttvar
```

### UDP (User Datagram Protocol)

```c
struct udp_header {
    uint16_t src_port;     // Source port
    uint16_t dest_port;    // Destination port
    uint16_t length;       // Length of header + data
    uint16_t checksum;     // Optional checksum
};

// Simple UDP implementation
void udp_send(int socket, const void* data, size_t len, 
              struct sockaddr_in* dest) {
    struct udp_packet {
        struct udp_header header;
        uint8_t data[];
    } *packet;
    
    packet = malloc(sizeof(struct udp_header) + len);
    
    packet->header.src_port = htons(get_local_port(socket));
    packet->header.dest_port = htons(dest->sin_port);
    packet->header.length = htons(sizeof(struct udp_header) + len);
    packet->header.checksum = 0;  // Optional
    
    memcpy(packet->data, data, len);
    
    // Calculate checksum if needed
    packet->header.checksum = udp_checksum(packet, dest->sin_addr);
    
    // Send via IP layer
    ip_send((uint8_t*)packet, sizeof(struct udp_header) + len,
            IPPROTO_UDP, dest->sin_addr);
    
    free(packet);
}
```

## 13.5 Application Layer Protocols

### HTTP/HTTPS

```python
# Simple HTTP server
class HTTPServer:
    def __init__(self, port=80):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', port))
        self.socket.listen(5)
    
    def handle_request(self, client_socket):
        request = client_socket.recv(4096).decode()
        lines = request.split('\r\n')
        
        # Parse request line
        method, path, version = lines[0].split()
        
        # Parse headers
        headers = {}
        for line in lines[1:]:
            if not line:
                break
            key, value = line.split(': ', 1)
            headers[key] = value
        
        # Route request
        if method == 'GET':
            response = self.handle_get(path, headers)
        elif method == 'POST':
            body = lines[-1]  # Simplified
            response = self.handle_post(path, headers, body)
        else:
            response = self.create_response(405, 'Method Not Allowed')
        
        client_socket.send(response.encode())
        client_socket.close()
    
    def create_response(self, status_code, status_text, body='', headers=None):
        response = f'HTTP/1.1 {status_code} {status_text}\r\n'
        
        headers = headers or {}
        headers['Content-Length'] = len(body)
        headers['Connection'] = 'close'
        
        for key, value in headers.items():
            response += f'{key}: {value}\r\n'
        
        response += '\r\n' + body
        return response

# HTTPS with TLS
class HTTPSServer(HTTPServer):
    def __init__(self, port=443, cert_file='server.crt', key_file='server.key'):
        super().__init__(port)
        
        # Wrap socket with TLS
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(cert_file, key_file)
        self.socket = context.wrap_socket(self.socket, server_side=True)
```

### DNS (Domain Name System)

```python
class DNSResolver:
    def __init__(self):
        self.cache = {}  # Simple cache
        self.root_servers = [
            '198.41.0.4',   # a.root-servers.net
            '199.9.14.201',  # b.root-servers.net
            # ... more root servers
        ]
    
    def resolve(self, domain, record_type='A'):
        # Check cache
        cache_key = (domain, record_type)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() < entry['expires']:
                return entry['value']
        
        # Recursive resolution
        return self.recursive_resolve(domain, record_type, self.root_servers)
    
    def recursive_resolve(self, domain, record_type, nameservers):
        for ns in nameservers:
            response = self.query_dns_server(ns, domain, record_type)
            
            if response['answer']:
                # Found answer
                result = response['answer'][0]['data']
                
                # Cache result
                self.cache[(domain, record_type)] = {
                    'value': result,
                    'expires': time.time() + response['answer'][0]['ttl']
                }
                
                return result
            
            elif response['authority']:
                # Get nameservers for next level
                next_ns = []
                for auth in response['authority']:
                    if auth['type'] == 'NS':
                        # Resolve nameserver address if needed
                        ns_addr = self.resolve(auth['data'], 'A')
                        next_ns.append(ns_addr)
                
                if next_ns:
                    return self.recursive_resolve(domain, record_type, next_ns)
        
        return None
    
    def query_dns_server(self, server, domain, record_type):
        # Build DNS query packet
        query = self.build_dns_query(domain, record_type)
        
        # Send UDP packet
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        sock.sendto(query, (server, 53))
        
        # Receive response
        response, _ = sock.recvfrom(512)
        sock.close()
        
        return self.parse_dns_response(response)

# DNS message structure
class DNSMessage:
    def __init__(self):
        self.id = random.randint(0, 65535)
        self.flags = 0x0100  # Standard query
        self.questions = []
        self.answers = []
        self.authority = []
        self.additional = []
    
    def to_bytes(self):
        data = struct.pack('>HHHHHH',
            self.id,
            self.flags,
            len(self.questions),
            len(self.answers),
            len(self.authority),
            len(self.additional)
        )
        
        # Add questions
        for q in self.questions:
            # Encode domain name
            for label in q['name'].split('.'):
                data += bytes([len(label)]) + label.encode()
            data += b'\x00'  # Root label
            
            data += struct.pack('>HH', q['type'], q['class'])
        
        return data
```

### WebSocket

```python
class WebSocketServer:
    def __init__(self, port=8080):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('', port))
        self.server.listen(5)
        self.clients = []
    
    def handshake(self, client_socket):
        request = client_socket.recv(1024).decode()
        headers = self.parse_headers(request)
        
        # Check for WebSocket upgrade
        if headers.get('Upgrade') != 'websocket':
            return False
        
        # Generate accept key
        key = headers['Sec-WebSocket-Key']
        accept_key = self.generate_accept_key(key)
        
        # Send handshake response
        response = (
            'HTTP/1.1 101 Switching Protocols\r\n'
            'Upgrade: websocket\r\n'
            'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Accept: {accept_key}\r\n'
            '\r\n'
        )
        
        client_socket.send(response.encode())
        return True
    
    def generate_accept_key(self, key):
        GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        combined = key + GUID
        sha1 = hashlib.sha1(combined.encode()).digest()
        return base64.b64encode(sha1).decode()
    
    def decode_frame(self, data):
        if len(data) < 2:
            return None
        
        # Parse frame header
        byte1, byte2 = data[0], data[1]
        
        fin = (byte1 >> 7) & 1
        opcode = byte1 & 0x0F
        masked = (byte2 >> 7) & 1
        payload_len = byte2 & 0x7F
        
        offset = 2
        
        # Extended payload length
        if payload_len == 126:
            payload_len = struct.unpack('>H', data[offset:offset+2])[0]
            offset += 2
        elif payload_len == 127:
            payload_len = struct.unpack('>Q', data[offset:offset+8])[0]
            offset += 8
        
        # Masking key
        if masked:
            mask = data[offset:offset+4]
            offset += 4
        
        # Extract payload
        payload = data[offset:offset+payload_len]
        
        # Unmask payload
        if masked:
            payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
        
        return {
            'fin': fin,
            'opcode': opcode,
            'payload': payload
        }
    
    def encode_frame(self, payload, opcode=0x1):
        frame = bytearray()
        
        # FIN = 1, opcode
        frame.append(0x80 | opcode)
        
        # Payload length
        payload_len = len(payload)
        if payload_len < 126:
            frame.append(payload_len)
        elif payload_len < 65536:
            frame.append(126)
            frame.extend(struct.pack('>H', payload_len))
        else:
            frame.append(127)
            frame.extend(struct.pack('>Q', payload_len))
        
        # Payload (no masking for server->client)
        frame.extend(payload)
        
        return bytes(frame)
```

## 13.6 Network Security

### Firewalls and NAT

```python
class Firewall:
    def __init__(self):
        self.rules = []
        self.connection_table = {}  # Stateful tracking
    
    def add_rule(self, action, protocol=None, src_ip=None, src_port=None,
                 dst_ip=None, dst_port=None, direction='INBOUND'):
        self.rules.append({
            'action': action,  # ALLOW or DENY
            'protocol': protocol,
            'src_ip': src_ip,
            'src_port': src_port,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'direction': direction
        })
    
    def filter_packet(self, packet):
        # Check stateful connection table first
        conn_key = self.get_connection_key(packet)
        if conn_key in self.connection_table:
            # Established connection
            self.connection_table[conn_key]['last_seen'] = time.time()
            return 'ALLOW'
        
        # Check rules
        for rule in self.rules:
            if self.match_rule(packet, rule):
                if rule['action'] == 'ALLOW' and packet.flags & SYN:
                    # New connection, add to table
                    self.connection_table[conn_key] = {
                        'state': 'ESTABLISHED',
                        'last_seen': time.time()
                    }
                
                return rule['action']
        
        # Default deny
        return 'DENY'
    
    def match_rule(self, packet, rule):
        if rule['protocol'] and packet.protocol != rule['protocol']:
            return False
        
        if rule['src_ip'] and not self.ip_match(packet.src_ip, rule['src_ip']):
            return False
        
        if rule['dst_ip'] and not self.ip_match(packet.dst_ip, rule['dst_ip']):
            return False
        
        # ... check ports
        
        return True

class NAT:
    def __init__(self, public_ip):
        self.public_ip = public_ip
        self.translation_table = {}
        self.port_pool = list(range(49152, 65536))  # Dynamic ports
    
    def outbound_nat(self, packet):
        # Source NAT for outbound packets
        key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
        
        if key not in self.translation_table:
            # Allocate new public port
            public_port = self.port_pool.pop(0)
            self.translation_table[key] = {
                'public_port': public_port,
                'last_seen': time.time()
            }
            
            # Reverse mapping for inbound
            reverse_key = (packet.dst_ip, packet.dst_port, self.public_ip, public_port)
            self.translation_table[reverse_key] = {
                'private_ip': packet.src_ip,
                'private_port': packet.src_port,
                'last_seen': time.time()
            }
        
        # Modify packet
        packet.src_ip = self.public_ip
        packet.src_port = self.translation_table[key]['public_port']
        
        return packet
    
    def inbound_nat(self, packet):
        # Destination NAT for inbound packets
        key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
        
        if key in self.translation_table:
            entry = self.translation_table[key]
            packet.dst_ip = entry['private_ip']
            packet.dst_port = entry['private_port']
            entry['last_seen'] = time.time()
            return packet
        
        return None  # No translation, drop packet
```

### VPN and Tunneling

```python
class VPNTunnel:
    def __init__(self, local_ip, remote_ip, shared_key):
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.cipher = AES.new(shared_key, AES.MODE_GCM)
        self.tunnel_interface = self.create_tunnel_interface()
    
    def encapsulate(self, packet):
        # Encrypt original packet
        nonce = os.urandom(12)
        cipher = AES.new(self.shared_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(packet)
        
        # Create outer packet
        outer_packet = IPPacket()
        outer_packet.src_ip = self.local_ip
        outer_packet.dst_ip = self.remote_ip
        outer_packet.protocol = IPPROTO_ESP  # IPSec ESP
        outer_packet.payload = nonce + tag + ciphertext
        
        return outer_packet
    
    def decapsulate(self, outer_packet):
        if outer_packet.protocol != IPPROTO_ESP:
            return None
        
        # Extract encrypted data
        payload = outer_packet.payload
        nonce = payload[:12]
        tag = payload[12:28]
        ciphertext = payload[28:]
        
        # Decrypt
        cipher = AES.new(self.shared_key, AES.MODE_GCM, nonce=nonce)
        try:
            inner_packet = cipher.decrypt_and_verify(ciphertext, tag)
            return inner_packet
        except ValueError:
            return None  # Authentication failed
```

## 13.7 Network Programming

### Socket Programming

```c
// TCP Server
int create_tcp_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }
    
    // Allow reuse of address
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }
    
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }
    
    return server_fd;
}

// Non-blocking I/O with select
void multiplex_server(int server_fd) {
    fd_set master_set, read_set;
    int max_fd = server_fd;
    
    FD_ZERO(&master_set);
    FD_SET(server_fd, &master_set);
    
    while (1) {
        read_set = master_set;
        
        if (select(max_fd + 1, &read_set, NULL, NULL, NULL) < 0) {
            perror("select");
            break;
        }
        
        for (int fd = 0; fd <= max_fd; fd++) {
            if (!FD_ISSET(fd, &read_set)) continue;
            
            if (fd == server_fd) {
                // New connection
                struct sockaddr_in client_addr;
                socklen_t addr_len = sizeof(client_addr);
                
                int client_fd = accept(server_fd, 
                                     (struct sockaddr*)&client_addr,
                                     &addr_len);
                
                if (client_fd >= 0) {
                    FD_SET(client_fd, &master_set);
                    if (client_fd > max_fd) max_fd = client_fd;
                }
            } else {
                // Data from client
                char buffer[1024];
                int bytes = recv(fd, buffer, sizeof(buffer), 0);
                
                if (bytes <= 0) {
                    // Connection closed
                    close(fd);
                    FD_CLR(fd, &master_set);
                } else {
                    // Echo back
                    send(fd, buffer, bytes, 0);
                }
            }
        }
    }
}
```

## Exercises

1. Implement a simple chat application using TCP sockets with multiple clients.

2. Create a basic HTTP/1.1 web server that:
   - Serves static files
   - Handles GET and POST requests
   - Implements keep-alive connections

3. Build a DNS resolver that:
   - Performs recursive resolution
   - Caches responses
   - Handles multiple record types

4. Implement a reliable data transfer protocol over UDP with:
   - Sequence numbers
   - Acknowledgments
   - Retransmission

5. Create a network packet sniffer that:
   - Captures packets
   - Parses protocol headers
   - Filters by criteria

6. Design a simple routing protocol:
   - Distance vector or link state
   - Handle topology changes
   - Prevent routing loops

7. Implement a basic firewall with:
   - Stateful packet filtering
   - Rule management
   - Logging

8. Build a WebSocket echo server supporting:
   - Handshake
   - Frame encoding/decoding
   - Ping/pong

9. Create a bandwidth throttling system:
   - Token bucket algorithm
   - Per-connection limits
   - Fair queuing

10. Implement a simple VPN:
    - Packet encapsulation
    - Encryption
    - Tunnel management

## Summary

This chapter covered computer networking fundamentals:

- Network layers provide modular communication architecture
- Physical and data link layers handle direct communication
- Network layer enables internetworking through IP
- Transport layer provides reliable (TCP) or fast (UDP) delivery
- Application protocols enable user services
- Security mechanisms protect network communications
- Socket programming enables network applications

Understanding networking is essential for building distributed systems, web applications, and secure communications. Networks form the foundation of the internet and modern computing infrastructure.