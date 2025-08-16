# Chapter 18: Computer Security

## Introduction

Computer security encompasses the protection of computer systems and data from theft, damage, disruption, and unauthorized access. As our dependence on digital systems grows, security becomes increasingly critical. This chapter explores security principles, common vulnerabilities and attacks, defensive techniques, access control mechanisms, network security, application security, and incident response strategies that form the foundation of modern cybersecurity.

## 18.1 Security Fundamentals

### Security Principles

```python
class SecurityPrinciples:
    """Core security concepts and principles"""
    
    def __init__(self):
        self.cia_triad = {
            'confidentiality': 'Ensuring information is accessible only to authorized parties',
            'integrity': 'Maintaining accuracy and completeness of data',
            'availability': 'Ensuring authorized users have reliable access to resources'
        }
        
        self.additional_principles = {
            'authentication': 'Verifying identity of users or systems',
            'authorization': 'Granting appropriate access rights',
            'non_repudiation': 'Preventing denial of actions',
            'accountability': 'Tracking and logging actions'
        }
    
    def defense_in_depth(self):
        """Layered security approach"""
        layers = [
            {
                'layer': 'Physical',
                'controls': ['Locks', 'Guards', 'Cameras', 'Biometric access'],
                'threats': ['Theft', 'Tampering', 'Environmental damage']
            },
            {
                'layer': 'Network',
                'controls': ['Firewalls', 'IDS/IPS', 'VPN', 'Segmentation'],
                'threats': ['Intrusion', 'DoS', 'Man-in-the-middle']
            },
            {
                'layer': 'Host',
                'controls': ['Antivirus', 'Host firewall', 'Patching', 'Hardening'],
                'threats': ['Malware', 'Privilege escalation', 'Rootkits']
            },
            {
                'layer': 'Application',
                'controls': ['Input validation', 'Authentication', 'Encryption'],
                'threats': ['Injection', 'XSS', 'Buffer overflow']
            },
            {
                'layer': 'Data',
                'controls': ['Encryption', 'Access controls', 'Backup'],
                'threats': ['Data breach', 'Data loss', 'Data corruption']
            }
        ]
        return layers
    
    def principle_of_least_privilege(self, user, resource):
        """Grant minimum necessary permissions"""
        required_permissions = self.determine_minimum_permissions(user, resource)
        
        access_policy = {
            'user': user,
            'resource': resource,
            'permissions': required_permissions,
            'duration': 'temporary',  # Time-limited access
            'conditions': ['business_hours', 'from_corporate_network']
        }
        
        return access_policy
    
    def separation_of_duties(self, critical_operation):
        """Require multiple parties for critical operations"""
        if critical_operation == 'wire_transfer':
            return {
                'initiation': 'finance_clerk',
                'approval': 'finance_manager',
                'execution': 'treasury_officer',
                'audit': 'internal_auditor'
            }
```

### Threat Modeling

```python
class ThreatModel:
    def __init__(self, system):
        self.system = system
        self.assets = []
        self.threats = []
        self.vulnerabilities = []
        self.risks = []
    
    def identify_assets(self):
        """Identify valuable assets to protect"""
        self.assets = [
            {'name': 'Customer Database', 'value': 'critical', 'type': 'data'},
            {'name': 'Payment Processing System', 'value': 'critical', 'type': 'service'},
            {'name': 'Source Code', 'value': 'high', 'type': 'intellectual_property'},
            {'name': 'Employee Credentials', 'value': 'high', 'type': 'authentication'},
            {'name': 'Public Website', 'value': 'medium', 'type': 'reputation'}
        ]
        return self.assets
    
    def stride_analysis(self):
        """STRIDE threat categorization"""
        stride_threats = {
            'Spoofing': {
                'description': 'Impersonating another user or system',
                'examples': ['Fake login pages', 'IP spoofing', 'Email spoofing'],
                'mitigations': ['Strong authentication', 'Digital signatures', 'Certificates']
            },
            'Tampering': {
                'description': 'Unauthorized modification of data',
                'examples': ['SQL injection', 'Man-in-the-middle', 'Code injection'],
                'mitigations': ['Input validation', 'Integrity checks', 'Access controls']
            },
            'Repudiation': {
                'description': 'Denying actions performed',
                'examples': ['Deleting logs', 'Denying transactions', 'False claims'],
                'mitigations': ['Audit logging', 'Digital signatures', 'Timestamps']
            },
            'Information Disclosure': {
                'description': 'Exposing information to unauthorized parties',
                'examples': ['Data breaches', 'Information leakage', 'Side channels'],
                'mitigations': ['Encryption', 'Access controls', 'Data masking']
            },
            'Denial of Service': {
                'description': 'Making resources unavailable',
                'examples': ['DDoS attacks', 'Resource exhaustion', 'Crash attacks'],
                'mitigations': ['Rate limiting', 'Load balancing', 'Redundancy']
            },
            'Elevation of Privilege': {
                'description': 'Gaining unauthorized permissions',
                'examples': ['Buffer overflow', 'Privilege escalation', 'Backdoors'],
                'mitigations': ['Least privilege', 'Input validation', 'Secure coding']
            }
        }
        return stride_threats
    
    def calculate_risk(self, threat, vulnerability, impact, likelihood):
        """Calculate risk score"""
        risk_score = impact * likelihood
        
        risk = {
            'threat': threat,
            'vulnerability': vulnerability,
            'impact': impact,
            'likelihood': likelihood,
            'risk_score': risk_score,
            'risk_level': self.categorize_risk(risk_score)
        }
        
        return risk
    
    def categorize_risk(self, score):
        if score >= 20:
            return 'critical'
        elif score >= 12:
            return 'high'
        elif score >= 6:
            return 'medium'
        else:
            return 'low'
    
    def dread_scoring(self, vulnerability):
        """DREAD risk assessment model"""
        scores = {
            'damage_potential': 8,  # 0-10: How bad if exploited?
            'reproducibility': 9,   # 0-10: How easy to reproduce?
            'exploitability': 7,    # 0-10: How easy to exploit?
            'affected_users': 9,    # 0-10: How many users affected?
            'discoverability': 8    # 0-10: How easy to discover?
        }
        
        total_score = sum(scores.values())
        average_score = total_score / len(scores)
        
        return {
            'scores': scores,
            'total': total_score,
            'average': average_score,
            'risk_level': self.categorize_risk(average_score * 5)
        }
```

## 18.2 Common Vulnerabilities and Attacks

### Software Vulnerabilities

```python
class VulnerabilityExamples:
    def buffer_overflow_vulnerable(self, user_input):
        """Example of buffer overflow vulnerability"""
        # VULNERABLE CODE - DO NOT USE
        buffer = bytearray(256)
        
        # No bounds checking - vulnerable!
        for i, byte in enumerate(user_input):
            buffer[i] = byte  # Can write beyond buffer
        
        return buffer
    
    def buffer_overflow_secure(self, user_input):
        """Secure version with bounds checking"""
        buffer = bytearray(256)
        max_size = len(buffer)
        
        # Proper bounds checking
        for i, byte in enumerate(user_input[:max_size]):
            buffer[i] = byte
        
        return buffer
    
    def sql_injection_vulnerable(self, username, password):
        """Example of SQL injection vulnerability"""
        # VULNERABLE CODE - DO NOT USE
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        # Input: username = "admin' --" bypasses password check!
        return query
    
    def sql_injection_secure(self, username, password):
        """Secure version using parameterized queries"""
        query = "SELECT * FROM users WHERE username = ? AND password = ?"
        params = (username, password)
        # Database driver handles escaping
        return query, params
    
    def xss_vulnerable(self, user_comment):
        """Example of XSS vulnerability"""
        # VULNERABLE CODE - DO NOT USE
        html = f"<div class='comment'>{user_comment}</div>"
        # Input: <script>alert('XSS')</script> executes!
        return html
    
    def xss_secure(self, user_comment):
        """Secure version with HTML escaping"""
        import html
        escaped_comment = html.escape(user_comment)
        safe_html = f"<div class='comment'>{escaped_comment}</div>"
        return safe_html
    
    def path_traversal_vulnerable(self, filename):
        """Example of path traversal vulnerability"""
        # VULNERABLE CODE - DO NOT USE
        file_path = f"/var/www/uploads/{filename}"
        # Input: "../../../etc/passwd" accesses system files!
        with open(file_path, 'r') as f:
            return f.read()
    
    def path_traversal_secure(self, filename):
        """Secure version with path validation"""
        import os
        
        base_dir = "/var/www/uploads"
        # Resolve to absolute path and check if within base directory
        requested_path = os.path.abspath(os.path.join(base_dir, filename))
        
        if not requested_path.startswith(base_dir):
            raise ValueError("Invalid file path")
        
        with open(requested_path, 'r') as f:
            return f.read()

class CommonAttacks:
    def demonstrate_timing_attack(self, input_password, stored_password):
        """Timing attack on string comparison"""
        # VULNERABLE: Early return reveals password length/content
        if len(input_password) != len(stored_password):
            return False
        
        for i in range(len(input_password)):
            if input_password[i] != stored_password[i]:
                return False  # Early return leaks information
        
        return True
    
    def constant_time_compare(self, input_password, stored_password):
        """Constant-time comparison to prevent timing attacks"""
        if len(input_password) != len(stored_password):
            return False
        
        result = 0
        for x, y in zip(input_password, stored_password):
            result |= ord(x) ^ ord(y)
        
        return result == 0
    
    def race_condition_vulnerable(self, filename, content):
        """TOCTOU (Time-of-check to time-of-use) vulnerability"""
        # VULNERABLE CODE
        import os
        
        # Check if file exists
        if os.path.exists(filename):
            # Race condition: file could be changed between check and use!
            with open(filename, 'w') as f:
                f.write(content)
    
    def race_condition_secure(self, filename, content):
        """Secure version using atomic operations"""
        import os
        import tempfile
        
        # Create temporary file atomically
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename))
        
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            
            # Atomic rename
            os.replace(temp_path, filename)
        except:
            os.unlink(temp_path)
            raise
```

### Network Attacks

```python
class NetworkAttacks:
    def port_scanner(self, target_ip, start_port=1, end_port=1000):
        """Basic port scanning implementation"""
        import socket
        
        open_ports = []
        
        for port in range(start_port, end_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            
            result = sock.connect_ex((target_ip, port))
            if result == 0:
                open_ports.append(port)
            
            sock.close()
        
        return open_ports
    
    def syn_flood_attack(self, target_ip, target_port):
        """SYN flood DoS attack demonstration"""
        # EDUCATIONAL PURPOSE ONLY - DO NOT USE MALICIOUSLY
        import socket
        import struct
        
        def create_syn_packet(src_ip, dst_ip, dst_port):
            # IP header
            ip_header = struct.pack('!BBHHHBBH4s4s',
                69,  # Version and IHL
                0,   # Type of service
                40,  # Total length
                54321,  # ID
                0,   # Flags and fragment offset
                255, # TTL
                socket.IPPROTO_TCP,  # Protocol
                0,   # Checksum
                socket.inet_aton(src_ip),
                socket.inet_aton(dst_ip)
            )
            
            # TCP header with SYN flag
            tcp_header = struct.pack('!HHLLBBHHH',
                12345,  # Source port
                dst_port,  # Destination port
                0,      # Sequence number
                0,      # Acknowledgment number
                80,     # Header length and reserved
                2,      # Flags (SYN)
                1024,   # Window size
                0,      # Checksum
                0       # Urgent pointer
            )
            
            return ip_header + tcp_header
    
    def arp_spoofing(self, target_ip, gateway_ip):
        """ARP spoofing attack demonstration"""
        # EDUCATIONAL PURPOSE ONLY
        import scapy.all as scapy
        
        def get_mac(ip):
            arp_request = scapy.ARP(pdst=ip)
            broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
            response = scapy.srp(broadcast/arp_request, timeout=1, verbose=False)[0]
            return response[0][1].hwsrc
        
        def spoof_arp(target_ip, spoof_ip):
            target_mac = get_mac(target_ip)
            packet = scapy.ARP(op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip)
            scapy.send(packet, verbose=False)
    
    def dns_spoofing_detector(self):
        """Detect DNS spoofing attempts"""
        import socket
        
        def resolve_domain(domain, dns_server):
            # Query specific DNS server
            resolver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            resolver.settimeout(2)
            
            # Construct DNS query (simplified)
            query = self.build_dns_query(domain)
            resolver.sendto(query, (dns_server, 53))
            
            response, _ = resolver.recvfrom(512)
            return self.parse_dns_response(response)
        
        def detect_spoofing(domain):
            # Query multiple DNS servers
            dns_servers = ['8.8.8.8', '1.1.1.1', '9.9.9.9']
            results = []
            
            for server in dns_servers:
                try:
                    ip = resolve_domain(domain, server)
                    results.append(ip)
                except:
                    pass
            
            # Check for inconsistencies
            if len(set(results)) > 1:
                return True, results  # Possible spoofing detected
            
            return False, results
```

## 18.3 Access Control

### Authentication Systems

```python
class AuthenticationSystem:
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.failed_attempts = {}
    
    def multi_factor_authentication(self, username, password, otp_code):
        """Implement multi-factor authentication"""
        # Factor 1: Something you know (password)
        if not self.verify_password(username, password):
            return False, "Invalid password"
        
        # Factor 2: Something you have (OTP token)
        if not self.verify_otp(username, otp_code):
            return False, "Invalid OTP code"
        
        # Optional Factor 3: Something you are (biometric)
        # if not self.verify_biometric(username, biometric_data):
        #     return False, "Biometric verification failed"
        
        # Create session
        session_token = self.create_session(username)
        return True, session_token
    
    def verify_password(self, username, password):
        """Verify password with secure hashing"""
        import hashlib
        import hmac
        
        if username not in self.users:
            # Prevent timing attack - still hash even if user doesn't exist
            fake_hash = hashlib.pbkdf2_hmac('sha256', b'fake', b'salt', 100000)
            return False
        
        stored_hash = self.users[username]['password_hash']
        salt = self.users[username]['salt']
        
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                           password.encode(), 
                                           salt, 
                                           100000)
        
        return hmac.compare_digest(password_hash, stored_hash)
    
    def verify_otp(self, username, otp_code):
        """Verify TOTP (Time-based One-Time Password)"""
        import hmac
        import hashlib
        import time
        import struct
        
        secret = self.users[username].get('otp_secret')
        if not secret:
            return False
        
        # Get current time window
        counter = int(time.time()) // 30
        
        # Check current and adjacent windows for clock skew
        for window in [counter - 1, counter, counter + 1]:
            # Generate TOTP
            msg = struct.pack('>Q', window)
            hmac_digest = hmac.new(secret, msg, hashlib.sha1).digest()
            
            offset = hmac_digest[-1] & 0x0f
            truncated = struct.unpack('>I', hmac_digest[offset:offset + 4])[0]
            truncated &= 0x7fffffff
            
            otp = truncated % 1000000
            
            if str(otp).zfill(6) == otp_code:
                return True
        
        return False
    
    def implement_oauth2_flow(self):
        """OAuth 2.0 authorization flow"""
        class OAuth2Server:
            def __init__(self):
                self.clients = {}
                self.auth_codes = {}
                self.access_tokens = {}
            
            def authorize(self, client_id, redirect_uri, scope, state):
                """Authorization endpoint"""
                if client_id not in self.clients:
                    return None, "Invalid client"
                
                # User authentication and consent would happen here
                auth_code = self.generate_auth_code()
                
                self.auth_codes[auth_code] = {
                    'client_id': client_id,
                    'redirect_uri': redirect_uri,
                    'scope': scope,
                    'expires': time.time() + 600  # 10 minutes
                }
                
                return auth_code, state
            
            def token(self, grant_type, code, client_id, client_secret):
                """Token endpoint"""
                if grant_type != 'authorization_code':
                    return None, "Unsupported grant type"
                
                if code not in self.auth_codes:
                    return None, "Invalid authorization code"
                
                auth_data = self.auth_codes[code]
                
                if auth_data['client_id'] != client_id:
                    return None, "Client mismatch"
                
                if time.time() > auth_data['expires']:
                    return None, "Authorization code expired"
                
                # Verify client credentials
                if not self.verify_client(client_id, client_secret):
                    return None, "Invalid client credentials"
                
                # Generate access token
                access_token = self.generate_access_token()
                refresh_token = self.generate_refresh_token()
                
                self.access_tokens[access_token] = {
                    'client_id': client_id,
                    'scope': auth_data['scope'],
                    'expires': time.time() + 3600  # 1 hour
                }
                
                # Clean up used auth code
                del self.auth_codes[code]
                
                return {
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'token_type': 'Bearer',
                    'expires_in': 3600
                }

class AuthorizationSystem:
    def __init__(self):
        self.roles = {}
        self.permissions = {}
        self.role_hierarchy = {}
    
    def rbac_implementation(self):
        """Role-Based Access Control"""
        # Define roles
        self.roles = {
            'admin': ['read', 'write', 'delete', 'admin'],
            'editor': ['read', 'write'],
            'viewer': ['read']
        }
        
        # Role hierarchy
        self.role_hierarchy = {
            'admin': ['editor', 'viewer'],
            'editor': ['viewer'],
            'viewer': []
        }
        
        def check_permission(user_role, required_permission):
            # Check direct permissions
            if required_permission in self.roles.get(user_role, []):
                return True
            
            # Check inherited permissions
            for inherited_role in self.role_hierarchy.get(user_role, []):
                if check_permission(inherited_role, required_permission):
                    return True
            
            return False
        
        return check_permission
    
    def abac_implementation(self):
        """Attribute-Based Access Control"""
        def evaluate_policy(subject_attrs, resource_attrs, action, environment_attrs):
            policies = [
                {
                    'description': 'Employees can read their own records',
                    'condition': lambda s, r, a, e: (
                        s.get('role') == 'employee' and
                        a == 'read' and
                        s.get('id') == r.get('owner_id')
                    )
                },
                {
                    'description': 'Managers can read team records',
                    'condition': lambda s, r, a, e: (
                        s.get('role') == 'manager' and
                        a == 'read' and
                        r.get('department') == s.get('department')
                    )
                },
                {
                    'description': 'No access outside business hours',
                    'condition': lambda s, r, a, e: (
                        e.get('time_of_day') >= 9 and
                        e.get('time_of_day') <= 17
                    )
                }
            ]
            
            for policy in policies:
                if policy['condition'](subject_attrs, resource_attrs, action, environment_attrs):
                    return True, policy['description']
            
            return False, "Access denied"
        
        return evaluate_policy
```

## 18.4 Network Security

### Firewall Implementation

```python
class Firewall:
    def __init__(self):
        self.rules = []
        self.default_policy = 'DENY'
        self.connection_table = {}  # For stateful inspection
    
    def add_rule(self, rule):
        """Add firewall rule"""
        self.rules.append({
            'priority': rule.get('priority', 1000),
            'action': rule['action'],  # ALLOW or DENY
            'protocol': rule.get('protocol', 'any'),
            'src_ip': rule.get('src_ip', 'any'),
            'src_port': rule.get('src_port', 'any'),
            'dst_ip': rule.get('dst_ip', 'any'),
            'dst_port': rule.get('dst_port', 'any'),
            'state': rule.get('state', 'any')  # NEW, ESTABLISHED, RELATED
        })
        
        # Sort rules by priority
        self.rules.sort(key=lambda x: x['priority'])
    
    def filter_packet(self, packet):
        """Filter network packet based on rules"""
        # Check connection tracking table
        connection_key = self.get_connection_key(packet)
        
        if connection_key in self.connection_table:
            # Established connection
            if packet['flags'] & 0x04:  # RST flag
                del self.connection_table[connection_key]
            return 'ALLOW'
        
        # Check rules
        for rule in self.rules:
            if self.match_rule(packet, rule):
                if rule['action'] == 'ALLOW' and packet['flags'] & 0x02:  # SYN flag
                    # Track new connection
                    self.connection_table[connection_key] = {
                        'state': 'SYN_SENT',
                        'timestamp': time.time()
                    }
                
                return rule['action']
        
        return self.default_policy
    
    def match_rule(self, packet, rule):
        """Check if packet matches rule"""
        if rule['protocol'] != 'any' and packet['protocol'] != rule['protocol']:
            return False
        
        if rule['src_ip'] != 'any' and not self.ip_match(packet['src_ip'], rule['src_ip']):
            return False
        
        if rule['src_port'] != 'any' and packet.get('src_port') != rule['src_port']:
            return False
        
        if rule['dst_ip'] != 'any' and not self.ip_match(packet['dst_ip'], rule['dst_ip']):
            return False
        
        if rule['dst_port'] != 'any' and packet.get('dst_port') != rule['dst_port']:
            return False
        
        return True
    
    def ip_match(self, ip, rule_ip):
        """Check if IP matches rule (supports CIDR)"""
        if '/' in rule_ip:
            # CIDR notation
            import ipaddress
            network = ipaddress.ip_network(rule_ip)
            return ipaddress.ip_address(ip) in network
        
        return ip == rule_ip

class IntrusionDetectionSystem:
    def __init__(self):
        self.signatures = []
        self.anomaly_baseline = {}
        self.alerts = []
    
    def signature_based_detection(self, packet):
        """Detect known attack patterns"""
        signatures = [
            {
                'name': 'SQL Injection Attempt',
                'pattern': r"(\bunion\b.*\bselect\b|\bselect\b.*\bfrom\b.*\bwhere\b)",
                'severity': 'high'
            },
            {
                'name': 'XSS Attempt',
                'pattern': r"<script[^>]*>.*?</script>",
                'severity': 'medium'
            },
            {
                'name': 'Directory Traversal',
                'pattern': r"\.\./\.\./|\.\.\\\.\.\\",
                'severity': 'medium'
            },
            {
                'name': 'Command Injection',
                'pattern': r";\s*(ls|cat|wget|curl|bash|sh)\s",
                'severity': 'high'
            }
        ]
        
        import re
        
        payload = packet.get('payload', '').decode('utf-8', errors='ignore')
        
        for signature in signatures:
            if re.search(signature['pattern'], payload, re.IGNORECASE):
                self.raise_alert({
                    'type': 'signature',
                    'name': signature['name'],
                    'severity': signature['severity'],
                    'packet': packet,
                    'timestamp': time.time()
                })
                return True
        
        return False
    
    def anomaly_based_detection(self, traffic_stats):
        """Detect anomalous behavior"""
        # Build baseline if not exists
        if not self.anomaly_baseline:
            self.anomaly_baseline = {
                'avg_packet_size': 500,
                'avg_packets_per_second': 100,
                'avg_unique_ips': 50,
                'avg_port_scan_attempts': 5
            }
        
        # Calculate deviations
        deviations = {}
        
        for metric, baseline_value in self.anomaly_baseline.items():
            current_value = traffic_stats.get(metric, 0)
            deviation = abs(current_value - baseline_value) / baseline_value
            
            if deviation > 2.0:  # More than 200% deviation
                deviations[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'deviation_percent': deviation * 100
                }
        
        if deviations:
            self.raise_alert({
                'type': 'anomaly',
                'deviations': deviations,
                'severity': 'medium' if len(deviations) < 3 else 'high',
                'timestamp': time.time()
            })
            return True
        
        return False
    
    def raise_alert(self, alert):
        """Generate security alert"""
        self.alerts.append(alert)
        
        # Send notifications based on severity
        if alert['severity'] == 'high':
            self.send_immediate_notification(alert)
        elif alert['severity'] == 'medium':
            self.log_alert(alert)
```

## 18.5 Application Security

### Secure Coding Practices

```python
class SecureCoding:
    def input_validation(self, user_input, input_type):
        """Comprehensive input validation"""
        validators = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'alphanumeric': r'^[a-zA-Z0-9]+$',
            'integer': r'^-?\d+$',
            'url': r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
        
        import re
        
        if input_type in validators:
            pattern = validators[input_type]
            if not re.match(pattern, user_input):
                raise ValueError(f"Invalid {input_type} format")
        
        # Length check
        if len(user_input) > 1000:
            raise ValueError("Input too long")
        
        # Sanitize for different contexts
        sanitized = {
            'html': html.escape(user_input),
            'sql': user_input.replace("'", "''"),
            'shell': ''.join(c for c in user_input if c.isalnum() or c in '-_.'),
            'path': os.path.basename(user_input)
        }
        
        return sanitized
    
    def secure_random_token(self, length=32):
        """Generate cryptographically secure random token"""
        import secrets
        
        return secrets.token_urlsafe(length)
    
    def secure_session_management(self):
        """Implement secure session handling"""
        class SecureSession:
            def __init__(self):
                self.sessions = {}
                self.session_timeout = 1800  # 30 minutes
            
            def create_session(self, user_id):
                import secrets
                
                session_id = secrets.token_urlsafe(32)
                
                self.sessions[session_id] = {
                    'user_id': user_id,
                    'created': time.time(),
                    'last_activity': time.time(),
                    'ip_address': self.get_client_ip(),
                    'user_agent': self.get_user_agent(),
                    'csrf_token': secrets.token_urlsafe(32)
                }
                
                return session_id
            
            def validate_session(self, session_id):
                if session_id not in self.sessions:
                    return False, "Invalid session"
                
                session = self.sessions[session_id]
                
                # Check timeout
                if time.time() - session['last_activity'] > self.session_timeout:
                    del self.sessions[session_id]
                    return False, "Session expired"
                
                # Check IP address binding
                if session['ip_address'] != self.get_client_ip():
                    del self.sessions[session_id]
                    return False, "IP address mismatch"
                
                # Update last activity
                session['last_activity'] = time.time()
                
                return True, session
            
            def validate_csrf_token(self, session_id, csrf_token):
                if session_id not in self.sessions:
                    return False
                
                return self.sessions[session_id]['csrf_token'] == csrf_token
        
        return SecureSession()
    
    def secure_file_upload(self, file_data, filename):
        """Secure file upload handling"""
        import magic
        import hashlib
        
        # Validate file size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_data) > max_size:
            raise ValueError("File too large")
        
        # Check file type using magic bytes
        file_type = magic.from_buffer(file_data, mime=True)
        allowed_types = ['image/jpeg', 'image/png', 'application/pdf']
        
        if file_type not in allowed_types:
            raise ValueError(f"File type {file_type} not allowed")
        
        # Generate safe filename
        import os
        import uuid
        
        extension = os.path.splitext(filename)[1]
        safe_filename = f"{uuid.uuid4()}{extension}"
        
        # Scan for malware (simplified)
        if self.scan_for_malware(file_data):
            raise ValueError("Malware detected")
        
        # Store file outside web root
        storage_path = f"/secure/uploads/{safe_filename}"
        
        with open(storage_path, 'wb') as f:
            f.write(file_data)
        
        # Store metadata
        metadata = {
            'original_name': filename,
            'stored_name': safe_filename,
            'size': len(file_data),
            'type': file_type,
            'hash': hashlib.sha256(file_data).hexdigest(),
            'upload_time': time.time()
        }
        
        return metadata
    
    def scan_for_malware(self, file_data):
        """Basic malware scanning"""
        # Simplified signature-based detection
        malware_signatures = [
            b'MZ\x90\x00',  # PE executable
            b'\x7fELF',     # ELF executable
            b'<%eval',      # PHP eval
            b'<script>alert',  # JavaScript payload
        ]
        
        for signature in malware_signatures:
            if signature in file_data:
                return True
        
        return False
```

## 18.6 Incident Response

### Incident Detection and Response

```python
class IncidentResponse:
    def __init__(self):
        self.incidents = []
        self.response_team = []
        self.playbooks = {}
    
    def incident_lifecycle(self):
        """NIST incident response lifecycle"""
        phases = {
            'preparation': {
                'activities': [
                    'Establish incident response team',
                    'Create incident response plan',
                    'Deploy monitoring tools',
                    'Conduct training exercises'
                ],
                'tools': ['SIEM', 'IDS/IPS', 'Log aggregation', 'Forensic tools']
            },
            'detection_and_analysis': {
                'activities': [
                    'Monitor security events',
                    'Analyze alerts and indicators',
                    'Determine incident scope',
                    'Document findings'
                ],
                'indicators': ['Unusual network traffic', 'System crashes', 
                             'Unauthorized access', 'Data exfiltration']
            },
            'containment_eradication_recovery': {
                'containment': [
                    'Isolate affected systems',
                    'Block malicious IPs',
                    'Disable compromised accounts'
                ],
                'eradication': [
                    'Remove malware',
                    'Patch vulnerabilities',
                    'Reset credentials'
                ],
                'recovery': [
                    'Restore from backups',
                    'Rebuild systems',
                    'Verify system integrity'
                ]
            },
            'post_incident': {
                'activities': [
                    'Conduct lessons learned',
                    'Update response procedures',
                    'Improve security controls',
                    'Report to stakeholders'
                ]
            }
        }
        return phases
    
    def detect_incident(self, event):
        """Analyze event to determine if it's an incident"""
        incident_indicators = {
            'multiple_failed_logins': lambda e: e.get('failed_attempts', 0) > 5,
            'data_exfiltration': lambda e: e.get('outbound_data_mb', 0) > 1000,
            'privilege_escalation': lambda e: e.get('privilege_change', False),
            'malware_detection': lambda e: e.get('malware_found', False),
            'unauthorized_access': lambda e: e.get('access_denied', False)
        }
        
        for indicator_name, check_func in incident_indicators.items():
            if check_func(event):
                incident = {
                    'id': self.generate_incident_id(),
                    'type': indicator_name,
                    'severity': self.calculate_severity(event),
                    'timestamp': time.time(),
                    'status': 'detected',
                    'event_data': event
                }
                
                self.incidents.append(incident)
                self.trigger_response(incident)
                return incident
        
        return None
    
    def trigger_response(self, incident):
        """Initiate incident response based on playbook"""
        playbook = self.get_playbook(incident['type'])
        
        response = {
            'incident_id': incident['id'],
            'actions_taken': [],
            'start_time': time.time()
        }
        
        for step in playbook['steps']:
            try:
                result = self.execute_response_action(step)
                response['actions_taken'].append({
                    'action': step['action'],
                    'result': result,
                    'timestamp': time.time()
                })
            except Exception as e:
                response['actions_taken'].append({
                    'action': step['action'],
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        response['end_time'] = time.time()
        return response
    
    def execute_response_action(self, step):
        """Execute specific response action"""
        actions = {
            'isolate_system': self.isolate_system,
            'block_ip': self.block_ip_address,
            'disable_account': self.disable_user_account,
            'capture_memory': self.capture_memory_dump,
            'collect_logs': self.collect_forensic_logs,
            'notify_team': self.notify_response_team
        }
        
        action_func = actions.get(step['action'])
        if action_func:
            return action_func(step.get('parameters', {}))
        
        return f"Unknown action: {step['action']}"

class ForensicAnalysis:
    def __init__(self):
        self.evidence = []
        self.chain_of_custody = []
    
    def collect_evidence(self, system):
        """Collect digital evidence"""
        evidence_types = {
            'memory_dump': self.capture_memory,
            'disk_image': self.create_disk_image,
            'network_capture': self.capture_network_traffic,
            'log_files': self.collect_logs,
            'registry': self.export_registry,
            'browser_artifacts': self.collect_browser_data
        }
        
        collected = {}
        
        for evidence_type, collect_func in evidence_types.items():
            try:
                data = collect_func(system)
                hash_value = self.calculate_hash(data)
                
                evidence_item = {
                    'type': evidence_type,
                    'system': system,
                    'timestamp': time.time(),
                    'hash': hash_value,
                    'data': data
                }
                
                self.evidence.append(evidence_item)
                collected[evidence_type] = 'success'
                
                # Update chain of custody
                self.update_chain_of_custody(evidence_item)
                
            except Exception as e:
                collected[evidence_type] = f'failed: {str(e)}'
        
        return collected
    
    def analyze_timeline(self, evidence):
        """Create timeline of events"""
        events = []
        
        # Extract timestamps from various sources
        for item in evidence:
            if item['type'] == 'log_files':
                events.extend(self.parse_log_timestamps(item['data']))
            elif item['type'] == 'registry':
                events.extend(self.parse_registry_timestamps(item['data']))
            elif item['type'] == 'browser_artifacts':
                events.extend(self.parse_browser_timestamps(item['data']))
        
        # Sort events chronologically
        events.sort(key=lambda x: x['timestamp'])
        
        # Identify suspicious patterns
        suspicious_patterns = []
        
        for i in range(len(events) - 1):
            # Rapid succession of events
            if events[i+1]['timestamp'] - events[i]['timestamp'] < 1:
                if events[i]['type'] == 'file_creation':
                    suspicious_patterns.append({
                        'pattern': 'rapid_file_creation',
                        'events': [events[i], events[i+1]],
                        'severity': 'medium'
                    })
        
        return {
            'timeline': events,
            'suspicious_patterns': suspicious_patterns
        }
    
    def find_indicators_of_compromise(self, evidence):
        """Search for IOCs in evidence"""
        iocs = {
            'file_hashes': [
                'e3b0c44298fc1c149afbf4c8996fb92427ae41e4',  # Known malware
                '5d41402abc4b2a76b9719d911017c592'           # Suspicious file
            ],
            'ip_addresses': [
                '192.168.1.100',  # C&C server
                '10.0.0.50'       # Known attacker IP
            ],
            'domains': [
                'evil.com',
                'malware-c2.net'
            ],
            'registry_keys': [
                r'HKLM\Software\Microsoft\Windows\CurrentVersion\Run\Malware'
            ],
            'file_paths': [
                r'C:\Windows\Temp\evil.exe',
                r'/tmp/.hidden/backdoor.sh'
            ]
        }
        
        found_iocs = []
        
        for evidence_item in evidence:
            # Search for IOCs in evidence
            data_str = str(evidence_item['data'])
            
            for ioc_type, ioc_list in iocs.items():
                for ioc in ioc_list:
                    if ioc in data_str:
                        found_iocs.append({
                            'type': ioc_type,
                            'value': ioc,
                            'evidence_source': evidence_item['type'],
                            'confidence': 'high'
                        })
        
        return found_iocs
```

## Exercises

1. Implement a complete authentication system with:
   - Multi-factor authentication
   - Biometric support
   - Account lockout policies
   - Password complexity requirements

2. Build a web application firewall that:
   - Detects common attacks (SQLi, XSS, CSRF)
   - Implements rate limiting
   - Blocks malicious IPs
   - Provides detailed logging

3. Create a vulnerability scanner that:
   - Identifies open ports
   - Detects outdated software
   - Checks for misconfigurations
   - Generates remediation reports

4. Implement a SIEM system that:
   - Aggregates logs from multiple sources
   - Correlates security events
   - Generates alerts
   - Provides dashboards

5. Build a secure coding analyzer that:
   - Detects insecure patterns
   - Suggests fixes
   - Validates input handling
   - Checks cryptographic usage

6. Create an incident response platform that:
   - Manages incident lifecycle
   - Automates response actions
   - Tracks evidence
   - Generates reports

7. Implement a zero-trust architecture with:
   - Micro-segmentation
   - Continuous verification
   - Least privilege access
   - Encrypted communications

8. Build a security training platform that:
   - Simulates attacks
   - Teaches secure coding
   - Provides CTF challenges
   - Tracks progress

9. Create a compliance framework that:
   - Maps to standards (ISO 27001, NIST)
   - Performs gap analysis
   - Tracks controls
   - Generates audit reports

10. Implement a threat intelligence platform that:
    - Collects threat feeds
    - Correlates indicators
    - Provides early warning
    - Shares intelligence

## Summary

This chapter covered comprehensive computer security concepts:

- Security fundamentals establish core principles and threat models
- Common vulnerabilities and attacks demonstrate real-world risks
- Access control mechanisms protect resources from unauthorized use
- Network security defends against remote attacks
- Application security prevents software vulnerabilities
- Incident response manages security breaches effectively
- Forensic analysis investigates security incidents
- Security requires continuous vigilance and improvement

Computer security is an ongoing challenge requiring defense in depth, continuous monitoring, and rapid response to evolving threats. Understanding these concepts is essential for building and maintaining secure systems.