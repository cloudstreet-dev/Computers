# Chapter 20: System Security and Hardening

## Introduction

System security hardening is the process of securing a system by reducing its attack surface and vulnerability to threats. This final chapter brings together concepts from throughout the book to create a comprehensive approach to securing Unix/Linux systems. We'll cover security baselines, hardening techniques, compliance frameworks, penetration testing, security monitoring, and incident response procedures that form a complete security posture for modern systems.

## 20.1 Security Baselines and Standards

### Security Frameworks

```python
class SecurityFrameworks:
    def __init__(self):
        self.frameworks = {
            'CIS': 'Center for Internet Security Benchmarks',
            'NIST': 'National Institute of Standards and Technology',
            'PCI-DSS': 'Payment Card Industry Data Security Standard',
            'HIPAA': 'Health Insurance Portability and Accountability Act',
            'SOC2': 'Service Organization Control 2',
            'ISO27001': 'Information Security Management System'
        }
    
    def cis_benchmark_implementation(self):
        """Implement CIS benchmark controls"""
        controls = {
            'initial_setup': [
                'Filesystem configuration',
                'Software updates configuration',
                'Filesystem integrity checking',
                'Secure boot settings',
                'Additional process hardening',
                'Mandatory access controls',
                'Command line warning banners'
            ],
            'services': [
                'Disable unnecessary services',
                'Configure time synchronization',
                'Remove X Windows System',
                'Disable Avahi Server',
                'Disable CUPS',
                'Disable DHCP Server',
                'Configure LDAP client',
                'Disable NFS and RPC',
                'Disable DNS Server',
                'Disable FTP Server',
                'Disable HTTP Server',
                'Disable IMAP and POP3',
                'Disable Samba',
                'Disable HTTP Proxy',
                'Disable SNMP',
                'Configure Mail Transfer Agent',
                'Disable rsync service'
            ],
            'network': [
                'Network parameters (Host Only)',
                'Network parameters (Host and Router)',
                'IPv6 configuration',
                'TCP Wrappers',
                'Firewall configuration',
                'Wireless interfaces disabled'
            ],
            'logging_auditing': [
                'Configure system accounting',
                'Configure logging',
                'Ensure log rotation',
                'Configure auditd',
                'Configure system log permissions'
            ],
            'access_auth': [
                'Configure cron',
                'SSH Server configuration',
                'Configure PAM',
                'User accounts and environment',
                'Root login restrictions',
                'Ensure password requirements'
            ],
            'system_maintenance': [
                'System file permissions',
                'User and group settings',
                'Shadow password suite configuration',
                'Review for duplicate UIDs/GIDs',
                'Review user home directories'
            ]
        }
        return controls
    
    def nist_cybersecurity_framework(self):
        """NIST Cybersecurity Framework implementation"""
        framework = {
            'identify': {
                'asset_management': 'Inventory physical devices and systems',
                'business_environment': 'Understand organizational mission',
                'governance': 'Establish policies and procedures',
                'risk_assessment': 'Identify and document risks',
                'risk_management': 'Establish risk management processes'
            },
            'protect': {
                'access_control': 'Limit access to authorized users',
                'awareness_training': 'Train users on security',
                'data_security': 'Protect data at rest and in transit',
                'maintenance': 'Perform maintenance and repairs',
                'protective_technology': 'Implement security solutions'
            },
            'detect': {
                'anomalies_events': 'Detect anomalous activity',
                'continuous_monitoring': 'Monitor systems continuously',
                'detection_processes': 'Maintain detection processes'
            },
            'respond': {
                'response_planning': 'Execute response plan',
                'communications': 'Coordinate with stakeholders',
                'analysis': 'Analyze incident',
                'mitigation': 'Contain incident',
                'improvements': 'Learn from incidents'
            },
            'recover': {
                'recovery_planning': 'Execute recovery plan',
                'improvements': 'Incorporate lessons learned',
                'communications': 'Manage public relations'
            }
        }
        return framework

class ComplianceChecker:
    def __init__(self, standard):
        self.standard = standard
        self.checks = []
        self.results = {}
    
    def check_password_policy(self):
        """Check password policy compliance"""
        checks = {
            'min_length': self.check_password_length(),
            'complexity': self.check_password_complexity(),
            'history': self.check_password_history(),
            'age': self.check_password_age(),
            'lockout': self.check_account_lockout()
        }
        return checks
    
    def check_password_length(self):
        """Check minimum password length"""
        import subprocess
        
        try:
            result = subprocess.run(['grep', '^PASS_MIN_LEN', '/etc/login.defs'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                min_len = int(result.stdout.split()[1])
                return {'status': 'pass' if min_len >= 14 else 'fail',
                       'value': min_len,
                       'requirement': '>=14'}
        except:
            return {'status': 'error', 'message': 'Could not check password length'}
    
    def generate_compliance_report(self):
        """Generate compliance report"""
        report = {
            'standard': self.standard,
            'timestamp': time.time(),
            'system': platform.node(),
            'checks_performed': len(self.checks),
            'passed': sum(1 for r in self.results.values() if r['status'] == 'pass'),
            'failed': sum(1 for r in self.results.values() if r['status'] == 'fail'),
            'errors': sum(1 for r in self.results.values() if r['status'] == 'error'),
            'compliance_score': 0,
            'details': self.results
        }
        
        if report['checks_performed'] > 0:
            report['compliance_score'] = (report['passed'] / report['checks_performed']) * 100
        
        return report
```

## 20.2 System Hardening

### OS Hardening Script

```bash
#!/bin/bash

# Comprehensive system hardening script

set -euo pipefail

LOGFILE="/var/log/hardening.log"
BACKUP_DIR="/root/hardening-backup-$(date +%Y%m%d)"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# Create backup directory
create_backup() {
    mkdir -p "$BACKUP_DIR"
    log "Created backup directory: $BACKUP_DIR"
    
    # Backup critical files
    cp -a /etc/ssh "$BACKUP_DIR/"
    cp -a /etc/pam.d "$BACKUP_DIR/"
    cp /etc/sysctl.conf "$BACKUP_DIR/"
    cp /etc/fstab "$BACKUP_DIR/"
    cp /etc/login.defs "$BACKUP_DIR/"
    cp /etc/security/limits.conf "$BACKUP_DIR/"
}

# Kernel hardening
harden_kernel() {
    log "Hardening kernel parameters..."
    
    cat >> /etc/sysctl.d/99-hardening.conf <<EOF
# Kernel hardening parameters

# Network security
net.ipv4.tcp_syncookies = 1
net.ipv4.ip_forward = 0
net.ipv6.conf.all.forwarding = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_timestamps = 0

# Kernel security
kernel.randomize_va_space = 2
kernel.exec-shield = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1
kernel.panic = 60
kernel.panic_on_oops = 1
kernel.sysrq = 0
kernel.core_uses_pid = 1
kernel.dmesg_restrict = 1
kernel.kexec_load_disabled = 1
kernel.unprivileged_bpf_disabled = 1
kernel.unprivileged_userns_clone = 0

# File system security
fs.suid_dumpable = 0
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
EOF
    
    sysctl -p /etc/sysctl.d/99-hardening.conf
    log "Kernel parameters hardened"
}

# SSH hardening
harden_ssh() {
    log "Hardening SSH configuration..."
    
    # Backup original SSH config
    cp /etc/ssh/sshd_config "$BACKUP_DIR/sshd_config.orig"
    
    cat > /etc/ssh/sshd_config.d/99-hardening.conf <<EOF
# SSH Hardening Configuration

# Protocol and port
Port 22
Protocol 2
AddressFamily inet

# Host keys (use only secure algorithms)
HostKey /etc/ssh/ssh_host_ed25519_key
HostKey /etc/ssh/ssh_host_rsa_key

# Ciphers and keying
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org
HostKeyAlgorithms ssh-ed25519,rsa-sha2-512,rsa-sha2-256

# Authentication
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
KerberosAuthentication no
GSSAPIAuthentication no
UsePAM yes
AuthenticationMethods publickey
MaxAuthTries 3
MaxSessions 10

# User/group restrictions
AllowUsers sysadmin
DenyUsers root
AllowGroups ssh-users

# Security options
StrictModes yes
IgnoreRhosts yes
HostbasedAuthentication no
X11Forwarding no
PermitTunnel no
AllowAgentForwarding no
AllowTcpForwarding no
PermitUserEnvironment no

# Logging
SyslogFacility AUTH
LogLevel VERBOSE

# Session management
ClientAliveInterval 300
ClientAliveCountMax 2
LoginGraceTime 60
MaxStartups 10:30:60

# Banner
Banner /etc/ssh/banner

# Subsystems
Subsystem sftp internal-sftp
EOF
    
    # Create SSH banner
    cat > /etc/ssh/banner <<EOF
##############################################################
#                                                            #
#  Unauthorized access to this system is strictly prohibited #
#  All access attempts are logged and monitored             #
#  Violators will be prosecuted to the full extent of law   #
#                                                            #
##############################################################
EOF
    
    # Restart SSH service
    systemctl restart sshd
    log "SSH configuration hardened"
}

# File system hardening
harden_filesystem() {
    log "Hardening file system..."
    
    # Set secure mount options
    cp /etc/fstab "$BACKUP_DIR/fstab.orig"
    
    # Add nodev, nosuid, noexec to appropriate partitions
    sed -i 's/\(\/tmp.*defaults\)/\1,nodev,nosuid,noexec/' /etc/fstab
    sed -i 's/\(\/var\/tmp.*defaults\)/\1,nodev,nosuid,noexec/' /etc/fstab
    sed -i 's/\(\/home.*defaults\)/\1,nodev/' /etc/fstab
    
    # Create separate partition for /var/log if not exists
    if ! grep -q "/var/log" /etc/fstab; then
        log "Warning: /var/log should be on a separate partition"
    fi
    
    # Disable uncommon filesystems
    cat > /etc/modprobe.d/uncommon-fs.conf <<EOF
# Disable uncommon filesystems
install cramfs /bin/true
install freevxfs /bin/true
install jffs2 /bin/true
install hfs /bin/true
install hfsplus /bin/true
install squashfs /bin/true
install udf /bin/true
install vfat /bin/true
EOF
    
    # Disable uncommon network protocols
    cat > /etc/modprobe.d/uncommon-net.conf <<EOF
# Disable uncommon network protocols
install dccp /bin/true
install sctp /bin/true
install rds /bin/true
install tipc /bin/true
EOF
    
    # Set secure permissions on important files
    chmod 644 /etc/passwd
    chmod 640 /etc/shadow
    chmod 644 /etc/group
    chmod 640 /etc/gshadow
    chmod 600 /etc/ssh/sshd_config
    chmod 644 /etc/ssh/ssh_config
    chmod 400 /etc/crontab
    chmod 700 /etc/cron.d
    chmod 700 /etc/cron.daily
    chmod 700 /etc/cron.hourly
    chmod 700 /etc/cron.monthly
    chmod 700 /etc/cron.weekly
    
    log "File system hardened"
}

# Service hardening
harden_services() {
    log "Hardening system services..."
    
    # Disable unnecessary services
    services_to_disable=(
        "bluetooth"
        "cups"
        "avahi-daemon"
        "nfs-client"
        "rpcbind"
        "rsync"
        "xinetd"
    )
    
    for service in "${services_to_disable[@]}"; do
        if systemctl list-unit-files | grep -q "$service"; then
            systemctl stop "$service" 2>/dev/null || true
            systemctl disable "$service" 2>/dev/null || true
            log "Disabled service: $service"
        fi
    done
    
    # Remove unnecessary packages
    packages_to_remove=(
        "xorg-x11*"
        "telnet"
        "rsh-client"
        "rsh-redone-client"
        "nis"
        "ntpdate"
        "prelink"
        "talk"
        "rsync"
    )
    
    for package in "${packages_to_remove[@]}"; do
        if dpkg -l | grep -q "$package"; then
            apt-get remove --purge -y "$package" 2>/dev/null || true
            log "Removed package: $package"
        fi
    done
    
    log "Services hardened"
}

# User account hardening
harden_accounts() {
    log "Hardening user accounts..."
    
    # Set password requirements
    sed -i 's/^PASS_MAX_DAYS.*/PASS_MAX_DAYS   90/' /etc/login.defs
    sed -i 's/^PASS_MIN_DAYS.*/PASS_MIN_DAYS   7/' /etc/login.defs
    sed -i 's/^PASS_MIN_LEN.*/PASS_MIN_LEN    14/' /etc/login.defs
    sed -i 's/^PASS_WARN_AGE.*/PASS_WARN_AGE   14/' /etc/login.defs
    
    # Set default umask
    sed -i 's/^UMASK.*/UMASK           077/' /etc/login.defs
    
    # Disable unused accounts
    for user in games news uucp proxy www-data list irc gnats nobody; do
        if id "$user" &>/dev/null; then
            usermod -L "$user"
            usermod -s /usr/sbin/nologin "$user"
            log "Disabled account: $user"
        fi
    done
    
    # Ensure root account is secured
    passwd -l root
    
    # Set account lockout policy
    cat > /etc/pam.d/common-auth <<EOF
# Account lockout policy
auth required pam_tally2.so onerr=fail audit silent deny=5 unlock_time=900
auth required pam_unix.so nullok_secure
auth optional pam_cap.so
EOF
    
    # Remove empty password fields
    awk -F: '($2 == "" ) { print $1 }' /etc/shadow | while read user; do
        passwd -l "$user"
        log "Locked account with empty password: $user"
    done
    
    log "User accounts hardened"
}

# Auditing configuration
configure_auditing() {
    log "Configuring system auditing..."
    
    # Install and configure auditd
    apt-get install -y auditd audispd-plugins
    
    # Configure audit rules
    cat > /etc/audit/rules.d/hardening.rules <<EOF
# Delete all rules
-D

# Buffer Size
-b 8192

# Failure Mode
-f 1

# Ignore errors
-i

# System calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
-a always,exit -F arch=b64 -S clock_settime -k time-change
-a always,exit -F arch=b32 -S clock_settime -k time-change
-w /etc/localtime -p wa -k time-change

# User/group changes
-w /etc/group -p wa -k identity
-w /etc/passwd -p wa -k identity
-w /etc/gshadow -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/security/opasswd -p wa -k identity

# Network configuration
-a always,exit -F arch=b64 -S sethostname -S setdomainname -k network-change
-a always,exit -F arch=b32 -S sethostname -S setdomainname -k network-change
-w /etc/issue -p wa -k network-change
-w /etc/issue.net -p wa -k network-change
-w /etc/hosts -p wa -k network-change
-w /etc/network -p wa -k network-change

# Login/logout events
-w /var/log/faillog -p wa -k logins
-w /var/log/lastlog -p wa -k logins
-w /var/log/tallylog -p wa -k logins
-w /var/log/wtmp -p wa -k logins
-w /var/log/btmp -p wa -k logins

# Session initiation
-w /var/run/utmp -p wa -k session

# Unauthorized access attempts
-a always,exit -F arch=b64 -S open -S openat -F exit=-EACCES -F auid>=1000 -F auid!=4294967295 -k access
-a always,exit -F arch=b64 -S open -S openat -F exit=-EPERM -F auid>=1000 -F auid!=4294967295 -k access

# Privileged commands
-a always,exit -F path=/usr/bin/passwd -F perm=x -F auid>=1000 -F auid!=4294967295 -k privileged-passwd
-a always,exit -F path=/usr/bin/sudo -F perm=x -F auid>=1000 -F auid!=4294967295 -k privileged-sudo

# Make configuration immutable
-e 2
EOF
    
    # Load audit rules
    augenrules --load
    
    # Enable and start auditd
    systemctl enable auditd
    systemctl start auditd
    
    log "Auditing configured"
}

# Main execution
main() {
    log "Starting system hardening..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log "This script must be run as root"
        exit 1
    fi
    
    # Create backup
    create_backup
    
    # Run hardening functions
    harden_kernel
    harden_ssh
    harden_filesystem
    harden_services
    harden_accounts
    configure_auditing
    
    log "System hardening completed"
    log "Backup stored in: $BACKUP_DIR"
    log "Please review changes and test system functionality"
    log "Reboot recommended to apply all changes"
}

main "$@"
```

## 20.3 Security Monitoring

### Intrusion Detection System

```python
class IntrusionDetectionSystem:
    def __init__(self):
        self.alerts = []
        self.rules = []
        self.baseline = {}
        
    def monitor_file_integrity(self):
        """File integrity monitoring using AIDE/Tripwire approach"""
        import hashlib
        import os
        
        critical_files = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/group',
            '/etc/sudoers',
            '/etc/ssh/sshd_config',
            '/bin/ls',
            '/bin/ps',
            '/usr/bin/netstat',
            '/usr/bin/find',
            '/usr/bin/top'
        ]
        
        integrity_db = {}
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                # Calculate file hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                # Get file metadata
                stat = os.stat(file_path)
                
                integrity_db[file_path] = {
                    'hash': file_hash,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'mode': oct(stat.st_mode),
                    'uid': stat.st_uid,
                    'gid': stat.st_gid
                }
        
        return integrity_db
    
    def detect_rootkits(self):
        """Rootkit detection techniques"""
        checks = {
            'hidden_processes': self.check_hidden_processes(),
            'hidden_files': self.check_hidden_files(),
            'kernel_modules': self.check_kernel_modules(),
            'network_backdoors': self.check_network_backdoors(),
            'system_calls': self.check_system_call_table()
        }
        
        return checks
    
    def check_hidden_processes(self):
        """Detect hidden processes"""
        import subprocess
        
        # Get process list from /proc
        proc_pids = []
        for entry in os.listdir('/proc'):
            if entry.isdigit():
                proc_pids.append(int(entry))
        
        # Get process list from ps
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        ps_pids = []
        for line in ps_output.stdout.split('\n')[1:]:
            if line:
                ps_pids.append(int(line.split()[1]))
        
        # Find discrepancies
        hidden = set(proc_pids) - set(ps_pids)
        
        if hidden:
            return {'status': 'suspicious', 'hidden_pids': list(hidden)}
        
        return {'status': 'clean'}
    
    def monitor_network_connections(self):
        """Monitor network connections for suspicious activity"""
        import socket
        import struct
        
        suspicious_ports = [31337, 12345, 4444, 6666, 6667]  # Common backdoor ports
        suspicious_connections = []
        
        # Parse /proc/net/tcp for connections
        with open('/proc/net/tcp', 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
            for line in lines:
                fields = line.split()
                
                # Parse local and remote addresses
                local_addr = fields[1].split(':')
                remote_addr = fields[2].split(':')
                
                local_port = int(local_addr[1], 16)
                remote_port = int(remote_addr[1], 16)
                
                # Check for suspicious ports
                if local_port in suspicious_ports or remote_port in suspicious_ports:
                    suspicious_connections.append({
                        'local_port': local_port,
                        'remote_port': remote_port,
                        'state': fields[3]
                    })
        
        return suspicious_connections
    
    def behavioral_analysis(self):
        """Analyze system behavior for anomalies"""
        import psutil
        
        anomalies = []
        
        # Check for unusual process behavior
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                # High CPU usage by unknown process
                if proc.info['cpu_percent'] > 80:
                    if proc.info['name'] not in ['systemd', 'kernel', 'nginx', 'mysql']:
                        anomalies.append({
                            'type': 'high_cpu',
                            'process': proc.info['name'],
                            'pid': proc.info['pid'],
                            'cpu': proc.info['cpu_percent']
                        })
                
                # Suspicious process names
                suspicious_names = ['nc', 'ncat', 'socat', 'cryptominer', 'xmrig']
                if any(susp in proc.info['name'].lower() for susp in suspicious_names):
                    anomalies.append({
                        'type': 'suspicious_process',
                        'process': proc.info['name'],
                        'pid': proc.info['pid']
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return anomalies

class SecurityEventCorrelation:
    def __init__(self):
        self.events = []
        self.patterns = []
        self.incidents = []
    
    def correlate_events(self, new_event):
        """Correlate security events to detect attack patterns"""
        self.events.append(new_event)
        
        # Check for brute force attacks
        if self.detect_brute_force():
            self.create_incident('brute_force', 'high')
        
        # Check for privilege escalation
        if self.detect_privilege_escalation():
            self.create_incident('privilege_escalation', 'critical')
        
        # Check for data exfiltration
        if self.detect_data_exfiltration():
            self.create_incident('data_exfiltration', 'critical')
        
        # Check for lateral movement
        if self.detect_lateral_movement():
            self.create_incident('lateral_movement', 'high')
    
    def detect_brute_force(self):
        """Detect brute force attack patterns"""
        # Look for multiple failed login attempts
        failed_logins = [e for e in self.events[-100:] 
                        if e.get('type') == 'auth_failure']
        
        if len(failed_logins) > 10:
            # Check if from same source
            sources = [e.get('source_ip') for e in failed_logins]
            if sources.count(sources[0]) > 5:
                return True
        
        return False
    
    def detect_privilege_escalation(self):
        """Detect privilege escalation attempts"""
        recent_events = self.events[-50:]
        
        # Look for sudo usage followed by system file modifications
        sudo_events = [e for e in recent_events if 'sudo' in e.get('command', '')]
        file_mods = [e for e in recent_events if e.get('type') == 'file_modified']
        
        if sudo_events and file_mods:
            critical_files = ['/etc/passwd', '/etc/shadow', '/etc/sudoers']
            for file_event in file_mods:
                if file_event.get('path') in critical_files:
                    return True
        
        return False
    
    def detect_data_exfiltration(self):
        """Detect potential data exfiltration"""
        recent_events = self.events[-100:]
        
        # Look for large outbound data transfers
        network_events = [e for e in recent_events 
                         if e.get('type') == 'network_transfer']
        
        for event in network_events:
            if event.get('direction') == 'outbound' and event.get('bytes', 0) > 100000000:
                # More than 100MB transferred
                return True
        
        return False
```

## 20.4 Penetration Testing

### Security Assessment Tools

```python
class PenetrationTestingFramework:
    def __init__(self):
        self.target = None
        self.vulnerabilities = []
        self.exploits = []
    
    def vulnerability_scanner(self, target):
        """Comprehensive vulnerability scanning"""
        self.target = target
        scan_results = {
            'port_scan': self.port_scan(),
            'service_detection': self.service_detection(),
            'vulnerability_assessment': self.vulnerability_assessment(),
            'configuration_audit': self.configuration_audit()
        }
        return scan_results
    
    def port_scan(self):
        """TCP/UDP port scanning"""
        import socket
        
        open_ports = []
        common_ports = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 
                       445, 993, 995, 1723, 3306, 3389, 5900, 8080]
        
        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.target, port))
            
            if result == 0:
                open_ports.append({
                    'port': port,
                    'state': 'open',
                    'service': self.identify_service(port)
                })
            
            sock.close()
        
        return open_ports
    
    def identify_service(self, port):
        """Identify service running on port"""
        services = {
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            110: 'POP3',
            143: 'IMAP',
            443: 'HTTPS',
            445: 'SMB',
            3306: 'MySQL',
            3389: 'RDP',
            5432: 'PostgreSQL',
            8080: 'HTTP-Alt'
        }
        return services.get(port, 'Unknown')
    
    def check_weak_credentials(self):
        """Test for weak/default credentials"""
        weak_passwords = ['admin', 'password', '123456', 'password123', 
                         'admin123', 'root', 'toor', 'pass', 'test']
        
        default_creds = {
            'ssh': [('root', 'toor'), ('admin', 'admin')],
            'mysql': [('root', ''), ('root', 'root')],
            'postgresql': [('postgres', 'postgres')],
            'ftp': [('anonymous', ''), ('ftp', 'ftp')]
        }
        
        vulnerable_services = []
        
        # This is for demonstration - actual testing would attempt connections
        for service, creds in default_creds.items():
            for username, password in creds:
                # Simulated check
                if self.test_credential(service, username, password):
                    vulnerable_services.append({
                        'service': service,
                        'username': username,
                        'vulnerability': 'weak_credential'
                    })
        
        return vulnerable_services
    
    def web_application_testing(self):
        """Web application security testing"""
        tests = {
            'sql_injection': self.test_sql_injection(),
            'xss': self.test_xss(),
            'csrf': self.test_csrf(),
            'directory_traversal': self.test_directory_traversal(),
            'file_upload': self.test_file_upload(),
            'authentication': self.test_authentication(),
            'session_management': self.test_session_management(),
            'access_control': self.test_access_control()
        }
        return tests
    
    def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "admin' --",
            "' UNION SELECT NULL--",
            "' AND 1=0 UNION SELECT NULL--"
        ]
        
        vulnerable_endpoints = []
        
        # Test common endpoints
        endpoints = ['/login', '/search', '/product', '/user']
        
        for endpoint in endpoints:
            for payload in payloads:
                # Simulated test - actual implementation would make HTTP requests
                if self.is_vulnerable_to_sqli(endpoint, payload):
                    vulnerable_endpoints.append({
                        'endpoint': endpoint,
                        'payload': payload,
                        'type': 'sql_injection'
                    })
        
        return vulnerable_endpoints
    
    def generate_penetration_test_report(self):
        """Generate comprehensive penetration test report"""
        report = {
            'executive_summary': {
                'test_date': time.strftime('%Y-%m-%d'),
                'target': self.target,
                'risk_level': self.calculate_risk_level(),
                'critical_findings': len([v for v in self.vulnerabilities 
                                        if v.get('severity') == 'critical']),
                'high_findings': len([v for v in self.vulnerabilities 
                                    if v.get('severity') == 'high'])
            },
            'methodology': {
                'reconnaissance': 'Information gathering and enumeration',
                'scanning': 'Port scanning and service identification',
                'enumeration': 'Detailed service enumeration',
                'vulnerability_assessment': 'Automated and manual testing',
                'exploitation': 'Proof of concept exploitation',
                'post_exploitation': 'Impact assessment'
            },
            'findings': self.vulnerabilities,
            'recommendations': self.generate_recommendations(),
            'technical_details': {
                'tools_used': ['nmap', 'metasploit', 'burp suite', 'sqlmap'],
                'testing_period': '5 days',
                'scope': 'Full infrastructure and application testing'
            }
        }
        return report
```

## 20.5 Incident Response

### Incident Response Procedures

```bash
#!/bin/bash

# Incident response script

INCIDENT_DIR="/var/incident/$(date +%Y%m%d_%H%M%S)"
EVIDENCE_DIR="$INCIDENT_DIR/evidence"
LOGS_DIR="$INCIDENT_DIR/logs"

# Create incident response directory
initialize_incident() {
    mkdir -p "$EVIDENCE_DIR"
    mkdir -p "$LOGS_DIR"
    
    echo "Incident Response Initiated: $(date)" > "$INCIDENT_DIR/incident.log"
    echo "Incident ID: $(uuidgen)" >> "$INCIDENT_DIR/incident.log"
}

# Contain the incident
contain_incident() {
    local threat_type=$1
    
    case "$threat_type" in
        malware)
            # Isolate infected system
            iptables -I INPUT -j DROP
            iptables -I OUTPUT -j DROP
            # Allow only incident response connections
            iptables -I INPUT -s 10.0.0.100 -j ACCEPT
            iptables -I OUTPUT -d 10.0.0.100 -j ACCEPT
            ;;
            
        intrusion)
            # Block attacker IP
            local attacker_ip=$2
            iptables -I INPUT -s "$attacker_ip" -j DROP
            # Kill suspicious processes
            for pid in $(ps aux | grep -E 'nc|ncat|/tmp/' | awk '{print $2}'); do
                kill -9 "$pid" 2>/dev/null
            done
            ;;
            
        data_breach)
            # Disable external network access
            ip link set eth0 down
            # Stop web services
            systemctl stop nginx apache2 httpd
            ;;
    esac
    
    echo "Containment measures applied for: $threat_type" >> "$INCIDENT_DIR/incident.log"
}

# Collect evidence
collect_evidence() {
    echo "Collecting evidence..." >> "$INCIDENT_DIR/incident.log"
    
    # System information
    uname -a > "$EVIDENCE_DIR/system_info.txt"
    date >> "$EVIDENCE_DIR/system_info.txt"
    uptime >> "$EVIDENCE_DIR/system_info.txt"
    
    # Network connections
    netstat -antp > "$EVIDENCE_DIR/network_connections.txt"
    ss -tulpn >> "$EVIDENCE_DIR/network_connections.txt"
    
    # Process list
    ps auxwwf > "$EVIDENCE_DIR/process_list.txt"
    pstree -p >> "$EVIDENCE_DIR/process_list.txt"
    
    # Open files
    lsof > "$EVIDENCE_DIR/open_files.txt"
    
    # User sessions
    w > "$EVIDENCE_DIR/user_sessions.txt"
    last -50 >> "$EVIDENCE_DIR/user_sessions.txt"
    lastb -50 >> "$EVIDENCE_DIR/failed_logins.txt"
    
    # Memory dump (if available)
    if command -v LiME &> /dev/null; then
        insmod /path/to/lime.ko "path=$EVIDENCE_DIR/memory.dump format=lime"
    fi
    
    # File system timeline
    find / -type f -mtime -7 -ls > "$EVIDENCE_DIR/recently_modified_files.txt" 2>/dev/null
    
    # Copy critical logs
    cp -r /var/log/* "$LOGS_DIR/" 2>/dev/null
    
    # Hash evidence files
    find "$EVIDENCE_DIR" -type f -exec sha256sum {} \; > "$EVIDENCE_DIR/evidence_hashes.txt"
    
    echo "Evidence collection completed" >> "$INCIDENT_DIR/incident.log"
}

# Analyze the incident
analyze_incident() {
    echo "Analyzing incident..." >> "$INCIDENT_DIR/incident.log"
    
    # Check for suspicious processes
    echo "=== Suspicious Processes ===" >> "$INCIDENT_DIR/analysis.txt"
    ps aux | grep -E '/tmp/|/dev/shm/|nc -l|/bin/sh' >> "$INCIDENT_DIR/analysis.txt"
    
    # Check for unauthorized users
    echo "=== Unauthorized Users ===" >> "$INCIDENT_DIR/analysis.txt"
    awk -F: '$3 >= 1000 {print $1}' /etc/passwd | while read user; do
        if ! grep -q "^$user:" /etc/passwd.backup; then
            echo "New user found: $user" >> "$INCIDENT_DIR/analysis.txt"
        fi
    done
    
    # Check for modified system files
    echo "=== Modified System Files ===" >> "$INCIDENT_DIR/analysis.txt"
    debsums -c 2>/dev/null >> "$INCIDENT_DIR/analysis.txt" || \
    rpm -Va 2>/dev/null >> "$INCIDENT_DIR/analysis.txt"
    
    # Check for rootkit indicators
    echo "=== Rootkit Check ===" >> "$INCIDENT_DIR/analysis.txt"
    chkrootkit >> "$INCIDENT_DIR/analysis.txt" 2>&1
    
    # Timeline analysis
    echo "=== Timeline Analysis ===" >> "$INCIDENT_DIR/analysis.txt"
    grep -h "sudo\|su\|login\|sshd" /var/log/auth.log* | tail -100 >> "$INCIDENT_DIR/analysis.txt"
}

# Eradicate the threat
eradicate_threat() {
    echo "Eradicating threat..." >> "$INCIDENT_DIR/incident.log"
    
    # Remove malicious files
    if [ -f "$INCIDENT_DIR/malicious_files.txt" ]; then
        while read -r file; do
            rm -f "$file"
            echo "Removed: $file" >> "$INCIDENT_DIR/incident.log"
        done < "$INCIDENT_DIR/malicious_files.txt"
    fi
    
    # Reset compromised accounts
    if [ -f "$INCIDENT_DIR/compromised_accounts.txt" ]; then
        while read -r account; do
            passwd -l "$account"
            echo "Locked account: $account" >> "$INCIDENT_DIR/incident.log"
        done < "$INCIDENT_DIR/compromised_accounts.txt"
    fi
    
    # Clean crontabs
    for user in $(cut -f1 -d: /etc/passwd); do
        crontab -u "$user" -r 2>/dev/null
    done
    
    # Remove suspicious systemd services
    find /etc/systemd/system/ -name "*.service" -mtime -7 -exec rm {} \;
    systemctl daemon-reload
}

# Recovery procedures
recovery_procedures() {
    echo "Starting recovery..." >> "$INCIDENT_DIR/incident.log"
    
    # Restore from backup
    # rsync -av /backup/ /
    
    # Reset security configurations
    /usr/local/bin/harden_system.sh
    
    # Change all passwords
    echo "Password reset required for all accounts" >> "$INCIDENT_DIR/incident.log"
    
    # Regenerate SSH keys
    rm -f /etc/ssh/ssh_host_*
    ssh-keygen -A
    
    # Update all software
    apt-get update && apt-get upgrade -y
    
    echo "Recovery completed" >> "$INCIDENT_DIR/incident.log"
}

# Main incident response flow
main() {
    local incident_type=${1:-unknown}
    
    echo "=== INCIDENT RESPONSE INITIATED ==="
    echo "Incident Type: $incident_type"
    
    initialize_incident
    
    # Immediate containment
    contain_incident "$incident_type"
    
    # Evidence collection
    collect_evidence
    
    # Analysis
    analyze_incident
    
    # Eradication
    read -p "Proceed with threat eradication? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        eradicate_threat
    fi
    
    # Recovery
    read -p "Proceed with recovery procedures? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        recovery_procedures
    fi
    
    echo "=== INCIDENT RESPONSE COMPLETED ==="
    echo "Incident data stored in: $INCIDENT_DIR"
}

main "$@"
```

## 20.6 Security Automation

### Security Orchestration

```python
class SecurityOrchestration:
    def __init__(self):
        self.playbooks = {}
        self.integrations = {}
        self.workflows = []
    
    def automated_response_playbook(self, threat_type):
        """Automated incident response playbook"""
        playbooks = {
            'malware': {
                'detection': ['file_hash_check', 'behavior_analysis'],
                'containment': ['isolate_host', 'block_c2_communication'],
                'eradication': ['remove_malware', 'clean_registry'],
                'recovery': ['restore_from_backup', 'patch_vulnerabilities'],
                'lessons_learned': ['update_signatures', 'improve_controls']
            },
            'phishing': {
                'detection': ['email_analysis', 'url_reputation_check'],
                'containment': ['quarantine_email', 'block_sender'],
                'eradication': ['remove_emails', 'reset_credentials'],
                'recovery': ['user_training', 'update_filters'],
                'lessons_learned': ['update_awareness_training']
            },
            'ransomware': {
                'detection': ['file_encryption_detection', 'network_traffic_analysis'],
                'containment': ['network_isolation', 'disable_shares'],
                'eradication': ['identify_variant', 'remove_ransomware'],
                'recovery': ['restore_from_backup', 'decrypt_if_possible'],
                'lessons_learned': ['improve_backup_strategy', 'update_edr']
            }
        }
        
        return playbooks.get(threat_type, {})
    
    def security_automation_workflow(self):
        """Complete security automation workflow"""
        workflow = """
import time
import subprocess
from datetime import datetime

class SecurityAutomation:
    def __init__(self):
        self.alerts = []
        self.responses = []
        
    def monitor(self):
        '''Continuous security monitoring'''
        while True:
            # Check for security events
            events = self.collect_events()
            
            for event in events:
                # Analyze event
                threat_level = self.analyze_event(event)
                
                if threat_level > 0:
                    # Create alert
                    alert = self.create_alert(event, threat_level)
                    self.alerts.append(alert)
                    
                    # Determine response
                    response = self.determine_response(alert)
                    
                    # Execute response
                    if response['auto_execute']:
                        self.execute_response(response)
                    else:
                        self.request_approval(response)
            
            time.sleep(60)  # Check every minute
    
    def collect_events(self):
        '''Collect security events from various sources'''
        events = []
        
        # Collect from system logs
        events.extend(self.parse_auth_log())
        events.extend(self.parse_syslog())
        
        # Collect from IDS
        events.extend(self.get_ids_alerts())
        
        # Collect from file integrity monitor
        events.extend(self.check_file_integrity())
        
        return events
    
    def analyze_event(self, event):
        '''Analyze event severity'''
        severity_scores = {
            'failed_login': 1,
            'successful_login_unusual_time': 3,
            'privilege_escalation': 8,
            'file_modification': 5,
            'new_user_created': 7,
            'firewall_violation': 6,
            'malware_detected': 9,
            'data_exfiltration': 10
        }
        
        return severity_scores.get(event['type'], 0)
    
    def execute_response(self, response):
        '''Execute automated response'''
        print(f"Executing response: {response['action']}")
        
        if response['action'] == 'block_ip':
            subprocess.run(['iptables', '-I', 'INPUT', '-s', 
                          response['target'], '-j', 'DROP'])
        
        elif response['action'] == 'disable_account':
            subprocess.run(['usermod', '-L', response['target']])
        
        elif response['action'] == 'isolate_host':
            # Implement network isolation
            self.isolate_network(response['target'])
        
        elif response['action'] == 'snapshot_vm':
            # Take VM snapshot for forensics
            self.create_snapshot(response['target'])
        
        self.responses.append({
            'timestamp': datetime.now(),
            'action': response['action'],
            'target': response['target'],
            'result': 'executed'
        })

automation = SecurityAutomation()
automation.monitor()
"""
        return workflow
```

## Exercises

1. Implement a complete security hardening script that:
   - Performs CIS benchmark compliance
   - Configures all security controls
   - Validates configuration changes
   - Generates compliance report

2. Build a real-time intrusion detection system that:
   - Monitors system calls
   - Detects anomalous behavior
   - Correlates events
   - Triggers automated responses

3. Create a vulnerability assessment tool that:
   - Scans for known vulnerabilities
   - Checks configuration weaknesses
   - Tests for common exploits
   - Prioritizes remediation

4. Implement a security information event management (SIEM) system that:
   - Aggregates logs from multiple sources
   - Normalizes log formats
   - Detects attack patterns
   - Provides real-time dashboards

5. Build a penetration testing framework that:
   - Performs reconnaissance
   - Identifies vulnerabilities
   - Exploits weaknesses safely
   - Documents findings

6. Create an incident response platform that:
   - Manages incident lifecycle
   - Collects forensic evidence
   - Coordinates response teams
   - Generates post-incident reports

7. Implement a compliance automation system that:
   - Checks multiple standards
   - Tracks control implementation
   - Generates audit evidence
   - Maintains compliance documentation

8. Build a threat hunting platform that:
   - Proactively searches for threats
   - Analyzes indicators of compromise
   - Tracks threat actors
   - Shares threat intelligence

9. Create a security orchestration system that:
   - Integrates security tools
   - Automates workflows
   - Manages playbooks
   - Measures effectiveness

10. Implement a zero-trust security model that:
    - Verifies every transaction
    - Implements micro-segmentation
    - Manages identity and access
    - Monitors continuously

## Summary

This chapter covered comprehensive system security and hardening:

- Security baselines provide standards for secure configurations
- System hardening reduces attack surface and vulnerabilities
- Security monitoring detects threats and anomalous behavior
- Penetration testing identifies weaknesses before attackers
- Incident response manages security breaches effectively
- Security automation improves response time and consistency
- Compliance frameworks ensure regulatory requirements
- Defense in depth creates multiple security layers

System security requires continuous improvement, regular assessment, and rapid response to emerging threats. The combination of preventive, detective, and responsive controls creates a robust security posture that protects systems and data from evolving cyber threats.

## Conclusion

This completes our comprehensive journey through computer science, from the fundamental concepts of counting and number systems to the complex challenges of securing modern Unix/Linux systems. Throughout these twenty chapters, we've explored:

- The mathematical and logical foundations that underpin all computing
- How computers are built from simple components to complex systems
- The software layers that make computers useful and programmable
- Networking and distributed systems that connect the world
- Security principles that protect our digital infrastructure

The field of computer science continues to evolve rapidly, with new technologies, challenges, and opportunities emerging constantly. The foundational knowledge presented in this book provides the basis for understanding current systems and adapting to future innovations.

Remember that mastery comes through practice, experimentation, and continuous learning. Use this knowledge as a springboard for deeper exploration, practical application, and creative problem-solving in the ever-expanding world of computing.