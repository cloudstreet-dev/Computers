# Chapter 19: Unix/Linux System Administration

## Introduction

Unix and Linux systems power the majority of servers, cloud infrastructure, and embedded devices worldwide. System administration involves managing, maintaining, and securing these systems to ensure reliable operation. This chapter covers essential administration tasks including system configuration, user management, process control, package management, networking, storage administration, performance monitoring, automation, and troubleshooting techniques that form the foundation of professional system administration.

## 19.1 System Architecture and Boot Process

### Unix/Linux System Architecture

```bash
# System architecture layers
┌─────────────────────────────────────┐
│         User Applications           │
├─────────────────────────────────────┤
│      Shell and Utilities            │
├─────────────────────────────────────┤
│      System Libraries (glibc)       │
├─────────────────────────────────────┤
│    System Call Interface            │
├─────────────────────────────────────┤
│         Linux Kernel                │
│  - Process Management               │
│  - Memory Management                │
│  - File Systems                     │
│  - Device Drivers                   │
│  - Network Stack                    │
└─────────────────────────────────────┘
```

### Boot Process

```python
class LinuxBootProcess:
    def __init__(self):
        self.boot_stages = []
    
    def bios_uefi_stage(self):
        """BIOS/UEFI initialization"""
        stage = {
            'name': 'BIOS/UEFI',
            'tasks': [
                'POST (Power-On Self Test)',
                'Hardware initialization',
                'Boot device selection',
                'Load bootloader from MBR/EFI partition'
            ],
            'configuration': {
                'BIOS': '/dev/sda (MBR - first 512 bytes)',
                'UEFI': '/boot/efi/EFI/BOOT/bootx64.efi'
            }
        }
        return stage
    
    def bootloader_stage(self):
        """GRUB2 bootloader"""
        stage = {
            'name': 'GRUB2',
            'config_file': '/boot/grub/grub.cfg',
            'tasks': [
                'Display boot menu',
                'Load kernel image',
                'Load initial ramdisk',
                'Pass parameters to kernel'
            ],
            'example_entry': """
menuentry 'Ubuntu 20.04' {
    set root='hd0,msdos1'
    linux /boot/vmlinuz-5.4.0 root=/dev/sda1 ro quiet splash
    initrd /boot/initrd.img-5.4.0
}
"""
        }
        return stage
    
    def kernel_stage(self):
        """Linux kernel initialization"""
        stage = {
            'name': 'Kernel',
            'tasks': [
                'Decompress kernel image',
                'Initialize kernel subsystems',
                'Mount initial RAM filesystem',
                'Start init process (PID 1)'
            ],
            'kernel_parameters': {
                'root': 'Root filesystem device',
                'init': 'Init program path',
                'ro/rw': 'Mount root as read-only/read-write',
                'quiet': 'Suppress verbose messages',
                'single': 'Boot to single-user mode'
            }
        }
        return stage
    
    def init_stage(self):
        """Init system (systemd/SysV)"""
        stage = {
            'name': 'Init System',
            'systems': {
                'systemd': {
                    'pid': 1,
                    'config': '/etc/systemd/',
                    'default_target': 'multi-user.target',
                    'units': ['*.service', '*.socket', '*.target']
                },
                'sysv': {
                    'pid': 1,
                    'config': '/etc/inittab',
                    'runlevels': {
                        0: 'Halt',
                        1: 'Single user',
                        2: 'Multi-user (no network)',
                        3: 'Multi-user (with network)',
                        4: 'Unused',
                        5: 'Multi-user (GUI)',
                        6: 'Reboot'
                    }
                }
            }
        }
        return stage

# Systemd management
class SystemdManager:
    def manage_services(self):
        """Systemd service management commands"""
        commands = {
            'start': 'systemctl start service_name',
            'stop': 'systemctl stop service_name',
            'restart': 'systemctl restart service_name',
            'reload': 'systemctl reload service_name',
            'status': 'systemctl status service_name',
            'enable': 'systemctl enable service_name',
            'disable': 'systemctl disable service_name',
            'mask': 'systemctl mask service_name',
            'list': 'systemctl list-units --type=service'
        }
        return commands
    
    def create_service_unit(self, service_name):
        """Create systemd service unit file"""
        unit_file = f"""[Unit]
Description={service_name} Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=serviceuser
Group=servicegroup
WorkingDirectory=/opt/{service_name}
ExecStart=/opt/{service_name}/bin/start.sh
ExecStop=/opt/{service_name}/bin/stop.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
        
        # Save to /etc/systemd/system/{service_name}.service
        return unit_file
```

## 19.2 User and Group Management

### User Administration

```bash
#!/bin/bash

# User management functions

create_user() {
    local username=$1
    local fullname=$2
    local groups=$3
    
    # Create user with home directory
    useradd -m -c "$fullname" -s /bin/bash "$username"
    
    # Set password
    echo "$username:TempPass123!" | chpasswd
    
    # Force password change on first login
    passwd -e "$username"
    
    # Add to groups
    if [ -n "$groups" ]; then
        usermod -aG "$groups" "$username"
    fi
    
    # Set up SSH directory
    mkdir -p "/home/$username/.ssh"
    touch "/home/$username/.ssh/authorized_keys"
    chmod 700 "/home/$username/.ssh"
    chmod 600 "/home/$username/.ssh/authorized_keys"
    chown -R "$username:$username" "/home/$username/.ssh"
    
    echo "User $username created successfully"
}

# Bulk user creation from CSV
bulk_create_users() {
    local csv_file=$1
    
    while IFS=, read -r username fullname department groups
    do
        # Skip header
        [ "$username" = "username" ] && continue
        
        # Create user
        create_user "$username" "$fullname" "$groups"
        
        # Set department-specific settings
        case "$department" in
            "engineering")
                usermod -aG docker,developers "$username"
                ;;
            "finance")
                usermod -aG finance,reports "$username"
                ;;
            "admin")
                usermod -aG sudo,adm "$username"
                ;;
        esac
        
    done < "$csv_file"
}

# User account auditing
audit_users() {
    echo "=== User Account Audit ==="
    echo
    
    # Check for users with UID 0 (root privileges)
    echo "Users with UID 0:"
    awk -F: '$3 == 0 {print $1}' /etc/passwd
    echo
    
    # Check for users without passwords
    echo "Users without passwords:"
    awk -F: '$2 == "" || $2 == "!" {print $1}' /etc/shadow
    echo
    
    # Check for users with weak passwords
    echo "Checking password policy compliance..."
    for user in $(cut -d: -f1 /etc/passwd); do
        chage -l "$user" | grep -E "Password expires|Account expires"
    done
    echo
    
    # List users with sudo privileges
    echo "Users with sudo privileges:"
    grep -E "^[^#].*ALL=" /etc/sudoers /etc/sudoers.d/* 2>/dev/null
    echo
    
    # Find inactive users
    echo "Inactive users (no login for 90+ days):"
    lastlog -b 90 | tail -n +2
}
```

### PAM Configuration

```python
class PAMConfiguration:
    def __init__(self):
        self.pam_dir = '/etc/pam.d/'
        self.modules = []
    
    def configure_password_policy(self):
        """Configure password complexity requirements"""
        pam_password = """# /etc/pam.d/common-password
# Password quality requirements
password requisite pam_pwquality.so retry=3 minlen=12 \
    ucredit=-1 lcredit=-1 dcredit=-1 ocredit=-1 \
    maxrepeat=3 gecoscheck=1 reject_username \
    enforce_for_root

# Password history
password required pam_pwhistory.so remember=5 use_authtok

# Standard Unix authentication
password [success=1 default=ignore] pam_unix.so obscure use_authtok \
    try_first_pass sha512 rounds=5000

# Enable password aging
password optional pam_gnome_keyring.so
"""
        return pam_password
    
    def configure_account_lockout(self):
        """Configure account lockout policy"""
        pam_auth = """# /etc/pam.d/common-auth
# Account lockout after failed attempts
auth required pam_tally2.so onerr=fail audit silent deny=5 unlock_time=900

# Standard Unix authentication
auth [success=1 default=ignore] pam_unix.so nullok_secure

# Enable additional authentication methods
auth optional pam_cap.so
"""
        return pam_auth
    
    def configure_session_limits(self):
        """Configure session and resource limits"""
        pam_limits = """# /etc/security/limits.conf
# Domain    Type    Item         Value
# ----      ----    ----         -----

# Prevent fork bombs
*           hard    nproc        1000
*           soft    nproc        500

# Limit memory usage
*           hard    memlock      unlimited
*           soft    memlock      unlimited

# Core dump settings
*           soft    core         0
*           hard    core         unlimited

# File descriptor limits
*           soft    nofile       4096
*           hard    nofile       8192

# Priority settings
@audio      -       nice         -20
@audio      -       rtprio       95

# Specific user limits
oracle      soft    nproc        2047
oracle      hard    nproc        16384
oracle      soft    nofile       1024
oracle      hard    nofile       65536
"""
        return pam_limits
```

## 19.3 File System Management

### Disk and Partition Management

```bash
#!/bin/bash

# Disk management functions

# Create and format partition
create_partition() {
    local device=$1
    local size=$2
    local fstype=$3
    local mountpoint=$4
    
    # Create partition using parted
    parted -s "$device" mklabel gpt
    parted -s "$device" mkpart primary "$fstype" 1MiB "$size"
    
    # Get partition name
    partition="${device}1"
    
    # Format partition
    case "$fstype" in
        ext4)
            mkfs.ext4 -L "data" "$partition"
            ;;
        xfs)
            mkfs.xfs -L "data" "$partition"
            ;;
        btrfs)
            mkfs.btrfs -L "data" "$partition"
            ;;
    esac
    
    # Create mount point and mount
    mkdir -p "$mountpoint"
    mount "$partition" "$mountpoint"
    
    # Add to fstab for persistent mounting
    echo "UUID=$(blkid -s UUID -o value $partition) $mountpoint $fstype defaults 0 2" >> /etc/fstab
}

# LVM management
setup_lvm() {
    local disks=("$@")
    
    # Create physical volumes
    for disk in "${disks[@]}"; do
        pvcreate "$disk"
    done
    
    # Create volume group
    vgcreate datavg "${disks[@]}"
    
    # Create logical volumes
    lvcreate -L 10G -n apps datavg
    lvcreate -L 20G -n data datavg
    lvcreate -L 5G -n logs datavg
    
    # Format logical volumes
    mkfs.ext4 /dev/datavg/apps
    mkfs.ext4 /dev/datavg/data
    mkfs.ext4 /dev/datavg/logs
    
    # Mount logical volumes
    mkdir -p /apps /data /logs
    mount /dev/datavg/apps /apps
    mount /dev/datavg/data /data
    mount /dev/datavg/logs /logs
}

# RAID configuration
setup_raid() {
    local level=$1
    shift
    local devices=("$@")
    
    # Create RAID array
    mdadm --create /dev/md0 --level="$level" \
          --raid-devices="${#devices[@]}" "${devices[@]}"
    
    # Wait for sync
    while [ "$(cat /proc/mdstat | grep resync)" ]; do
        echo "RAID sync in progress..."
        sleep 10
    done
    
    # Save RAID configuration
    mdadm --detail --scan >> /etc/mdadm/mdadm.conf
    
    # Format and mount RAID array
    mkfs.ext4 /dev/md0
    mkdir -p /raid
    mount /dev/md0 /raid
    
    # Add to fstab
    echo "/dev/md0 /raid ext4 defaults 0 2" >> /etc/fstab
}

# File system monitoring
monitor_filesystems() {
    echo "=== File System Status ==="
    
    # Disk usage
    df -hT | grep -v tmpfs
    
    echo
    echo "=== Inode Usage ==="
    df -i | grep -v tmpfs
    
    echo
    echo "=== Mount Options ==="
    mount | grep -E "ext4|xfs|btrfs"
    
    echo
    echo "=== File System Errors ==="
    dmesg | grep -E "EXT4-fs error|XFS|BTRFS" | tail -10
    
    echo
    echo "=== Disk Health (SMART) ==="
    for disk in /dev/sd[a-z]; do
        if [ -b "$disk" ]; then
            echo "Disk: $disk"
            smartctl -H "$disk" | grep -E "SMART overall-health"
        fi
    done
}
```

### Quota Management

```python
class QuotaManager:
    def setup_quotas(self, filesystem):
        """Setup disk quotas on filesystem"""
        commands = [
            f"quotacheck -cug {filesystem}",  # Create quota files
            f"quotaon {filesystem}",           # Enable quotas
        ]
        
        # Set user quota
        def set_user_quota(username, soft_blocks, hard_blocks, soft_inodes, hard_inodes):
            cmd = f"setquota -u {username} {soft_blocks} {hard_blocks} {soft_inodes} {hard_inodes} {filesystem}"
            return cmd
        
        # Set group quota
        def set_group_quota(groupname, soft_blocks, hard_blocks, soft_inodes, hard_inodes):
            cmd = f"setquota -g {groupname} {soft_blocks} {hard_blocks} {soft_inodes} {hard_inodes} {filesystem}"
            return cmd
        
        # Example quotas (in KB)
        quotas = {
            'users': {
                'john': (1048576, 2097152, 10000, 20000),  # 1GB soft, 2GB hard
                'jane': (2097152, 4194304, 20000, 40000),  # 2GB soft, 4GB hard
            },
            'groups': {
                'developers': (10485760, 20971520, 100000, 200000),  # 10GB soft, 20GB hard
                'finance': (5242880, 10485760, 50000, 100000),       # 5GB soft, 10GB hard
            }
        }
        
        return commands, quotas
    
    def monitor_quota_usage(self):
        """Monitor quota usage"""
        reports = {
            'user_report': 'repquota -u /',
            'group_report': 'repquota -g /',
            'all_filesystems': 'repquota -a',
            'user_specific': 'quota -u username',
            'group_specific': 'quota -g groupname'
        }
        return reports
```

## 19.4 Process and Service Management

### Process Control

```bash
#!/bin/bash

# Process management utilities

# Advanced process monitoring
monitor_processes() {
    echo "=== System Process Overview ==="
    
    # Top CPU consumers
    echo "Top 10 CPU intensive processes:"
    ps aux --sort=-%cpu | head -11
    
    echo
    echo "Top 10 Memory intensive processes:"
    ps aux --sort=-%mem | head -11
    
    echo
    echo "Process tree for critical services:"
    for service in sshd nginx mysql postgresql; do
        if pgrep "$service" > /dev/null; then
            echo "--- $service ---"
            pstree -p $(pgrep -o "$service")
        fi
    done
    
    echo
    echo "Zombie processes:"
    ps aux | grep -E "Z|<defunct>"
    
    echo
    echo "Process limits:"
    cat /proc/sys/kernel/pid_max
    echo "Current process count: $(ps aux | wc -l)"
}

# Kill processes safely
safe_kill() {
    local process_name=$1
    local signal=${2:-TERM}
    
    # Find process PIDs
    pids=$(pgrep -f "$process_name")
    
    if [ -z "$pids" ]; then
        echo "No processes found matching: $process_name"
        return 1
    fi
    
    echo "Found processes: $pids"
    
    # Send signal
    for pid in $pids; do
        echo "Sending $signal to PID $pid"
        kill -$signal "$pid"
    done
    
    # Wait for graceful termination
    sleep 5
    
    # Check if processes still exist
    remaining=$(pgrep -f "$process_name")
    if [ -n "$remaining" ]; then
        echo "Processes still running: $remaining"
        echo "Send KILL signal? (y/n)"
        read -r answer
        if [ "$answer" = "y" ]; then
            kill -9 $remaining
        fi
    fi
}

# Service dependency management
check_service_dependencies() {
    local service=$1
    
    echo "=== Service: $service ==="
    
    # Systemd dependencies
    echo "Required by:"
    systemctl list-dependencies --reverse "$service" | head -20
    
    echo
    echo "Requires:"
    systemctl list-dependencies "$service" | head -20
    
    echo
    echo "Service status:"
    systemctl status "$service" --no-pager
    
    echo
    echo "Recent logs:"
    journalctl -u "$service" -n 20 --no-pager
}
```

### Cron and Task Scheduling

```python
class CronManager:
    def __init__(self):
        self.crontab_dir = '/var/spool/cron/crontabs/'
        self.system_cron = '/etc/crontab'
    
    def parse_cron_expression(self, expression):
        """Parse and validate cron expression"""
        fields = expression.split()
        
        if len(fields) < 5:
            raise ValueError("Invalid cron expression")
        
        cron_fields = {
            'minute': fields[0],      # 0-59
            'hour': fields[1],        # 0-23
            'day': fields[2],         # 1-31
            'month': fields[3],       # 1-12
            'weekday': fields[4],     # 0-7 (0 and 7 are Sunday)
            'command': ' '.join(fields[5:]) if len(fields) > 5 else ''
        }
        
        # Special strings
        special_strings = {
            '@reboot': 'Run at startup',
            '@yearly': '0 0 1 1 *',
            '@annually': '0 0 1 1 *',
            '@monthly': '0 0 1 * *',
            '@weekly': '0 0 * * 0',
            '@daily': '0 0 * * *',
            '@midnight': '0 0 * * *',
            '@hourly': '0 * * * *'
        }
        
        if expression.startswith('@'):
            return special_strings.get(expression.split()[0])
        
        return cron_fields
    
    def create_cron_job(self, user, schedule, command, description=""):
        """Create a new cron job"""
        cron_entry = f"""# {description}
{schedule} {command}
"""
        
        # Validate schedule
        try:
            self.parse_cron_expression(f"{schedule} {command}")
        except ValueError as e:
            return f"Error: {e}"
        
        # Add to user's crontab
        crontab_file = f"{self.crontab_dir}{user}"
        
        return cron_entry
    
    def system_cron_jobs(self):
        """System-wide cron configuration"""
        system_cron = """# /etc/crontab
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=root

# m h dom mon dow user  command
17 *    * * *   root    cd / && run-parts --report /etc/cron.hourly
25 6    * * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )
47 6    * * 7   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.weekly )
52 6    1 * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.monthly )

# Custom jobs
0 2 * * * root /usr/local/bin/backup.sh
0 */4 * * * root /usr/local/bin/update-logs.sh
30 3 * * 1 root /usr/local/bin/weekly-report.sh
"""
        return system_cron
    
    def monitor_cron_jobs(self):
        """Monitor and audit cron jobs"""
        audit_commands = {
            'list_user_crons': 'for user in $(cut -f1 -d: /etc/passwd); do \
                                echo "=== $user ==="; crontab -u $user -l 2>/dev/null; done',
            'check_cron_logs': 'grep CRON /var/log/syslog | tail -50',
            'failed_jobs': 'grep "CRON.*FAILED" /var/log/syslog',
            'next_execution': 'systemctl list-timers',
            'anacron_status': 'cat /var/spool/anacron/*'
        }
        return audit_commands
```

## 19.5 Network Configuration

### Network Interface Management

```bash
#!/bin/bash

# Network configuration and management

# Configure network interface
configure_interface() {
    local interface=$1
    local ip_address=$2
    local netmask=$3
    local gateway=$4
    local dns_servers=$5
    
    # NetworkManager configuration
    nmcli con add type ethernet \
        con-name "$interface" \
        ifname "$interface" \
        ip4 "$ip_address/$netmask" \
        gw4 "$gateway"
    
    # Set DNS servers
    nmcli con mod "$interface" ipv4.dns "$dns_servers"
    
    # Activate connection
    nmcli con up "$interface"
    
    # Alternative: netplan configuration (Ubuntu)
    cat > "/etc/netplan/01-$interface.yaml" <<EOF
network:
  version: 2
  ethernets:
    $interface:
      dhcp4: no
      addresses:
        - $ip_address/$netmask
      gateway4: $gateway
      nameservers:
        addresses: [$dns_servers]
EOF
    
    netplan apply
}

# Configure bonding/teaming
setup_network_bonding() {
    local bond_name=$1
    local mode=$2  # balance-rr, active-backup, balance-xor, etc.
    shift 2
    local interfaces=("$@")
    
    # Load bonding module
    modprobe bonding
    
    # Create bond interface
    ip link add "$bond_name" type bond mode "$mode"
    
    # Add slave interfaces
    for iface in "${interfaces[@]}"; do
        ip link set "$iface" down
        ip link set "$iface" master "$bond_name"
    done
    
    # Bring up bond interface
    ip link set "$bond_name" up
    
    # Persistent configuration
    cat > "/etc/sysconfig/network-scripts/ifcfg-$bond_name" <<EOF
DEVICE=$bond_name
TYPE=Bond
BONDING_MASTER=yes
BOOTPROTO=static
ONBOOT=yes
BONDING_OPTS="mode=$mode miimon=100"
EOF
    
    # Configure slave interfaces
    for iface in "${interfaces[@]}"; do
        cat > "/etc/sysconfig/network-scripts/ifcfg-$iface" <<EOF
DEVICE=$iface
TYPE=Ethernet
BOOTPROTO=none
ONBOOT=yes
MASTER=$bond_name
SLAVE=yes
EOF
    done
}

# Network diagnostics
network_diagnostics() {
    echo "=== Network Configuration ==="
    ip addr show
    
    echo
    echo "=== Routing Table ==="
    ip route show
    
    echo
    echo "=== Network Statistics ==="
    ss -s
    
    echo
    echo "=== Open Ports ==="
    ss -tulpn
    
    echo
    echo "=== Network Connections ==="
    ss -tupn | head -20
    
    echo
    echo "=== DNS Resolution ==="
    cat /etc/resolv.conf
    
    echo
    echo "=== Network Performance ==="
    for iface in $(ls /sys/class/net/ | grep -v lo); do
        echo "Interface: $iface"
        echo "  RX: $(cat /sys/class/net/$iface/statistics/rx_bytes) bytes"
        echo "  TX: $(cat /sys/class/net/$iface/statistics/tx_bytes) bytes"
        echo "  Errors: RX=$(cat /sys/class/net/$iface/statistics/rx_errors) TX=$(cat /sys/class/net/$iface/statistics/tx_errors)"
    done
}
```

### Firewall Configuration

```python
class FirewallManager:
    def __init__(self, firewall_type='iptables'):
        self.firewall_type = firewall_type
        self.rules = []
    
    def iptables_configuration(self):
        """Configure iptables firewall"""
        iptables_script = """#!/bin/bash
# Basic iptables firewall configuration

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow DNS
iptables -A INPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT -p tcp --dport 53 -j ACCEPT

# Allow ICMP (ping)
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT

# Log dropped packets
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-dropped: " --log-level 7

# Save rules
iptables-save > /etc/iptables/rules.v4
ip6tables-save > /etc/iptables/rules.v6
"""
        return iptables_script
    
    def firewalld_configuration(self):
        """Configure firewalld"""
        commands = {
            'zones': {
                'list': 'firewall-cmd --list-all-zones',
                'get_default': 'firewall-cmd --get-default-zone',
                'set_default': 'firewall-cmd --set-default-zone=public',
                'create': 'firewall-cmd --permanent --new-zone=custom'
            },
            'services': {
                'list': 'firewall-cmd --list-services',
                'add': 'firewall-cmd --permanent --add-service=http',
                'remove': 'firewall-cmd --permanent --remove-service=telnet',
                'custom': 'firewall-cmd --permanent --add-port=8080/tcp'
            },
            'rules': {
                'rich_rule': 'firewall-cmd --permanent --add-rich-rule=\'rule family="ipv4" source address="192.168.1.0/24" port port="22" protocol="tcp" accept\'',
                'masquerade': 'firewall-cmd --permanent --add-masquerade',
                'forward': 'firewall-cmd --permanent --add-forward-port=port=80:proto=tcp:toport=8080'
            },
            'apply': 'firewall-cmd --reload'
        }
        return commands
    
    def fail2ban_configuration(self):
        """Configure fail2ban for intrusion prevention"""
        jail_conf = """# /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
destemail = admin@example.com
action = %(action_mwl)s

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 2

[wordpress]
enabled = true
port = http,https
filter = wordpress
logpath = /var/log/nginx/access.log
maxretry = 2
"""
        return jail_conf
```

## 19.6 Package Management

### Package Manager Operations

```bash
#!/bin/bash

# Universal package management wrapper

pkg_manager() {
    local action=$1
    shift
    local packages=("$@")
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        PM="apt"
    elif command -v yum &> /dev/null; then
        PM="yum"
    elif command -v dnf &> /dev/null; then
        PM="dnf"
    elif command -v zypper &> /dev/null; then
        PM="zypper"
    elif command -v pacman &> /dev/null; then
        PM="pacman"
    else
        echo "No supported package manager found"
        return 1
    fi
    
    case "$action" in
        install)
            case "$PM" in
                apt) apt-get install -y "${packages[@]}" ;;
                yum|dnf) $PM install -y "${packages[@]}" ;;
                zypper) zypper install -y "${packages[@]}" ;;
                pacman) pacman -S --noconfirm "${packages[@]}" ;;
            esac
            ;;
        remove)
            case "$PM" in
                apt) apt-get remove -y "${packages[@]}" ;;
                yum|dnf) $PM remove -y "${packages[@]}" ;;
                zypper) zypper remove -y "${packages[@]}" ;;
                pacman) pacman -R --noconfirm "${packages[@]}" ;;
            esac
            ;;
        update)
            case "$PM" in
                apt) apt-get update && apt-get upgrade -y ;;
                yum|dnf) $PM update -y ;;
                zypper) zypper update -y ;;
                pacman) pacman -Syu --noconfirm ;;
            esac
            ;;
        search)
            case "$PM" in
                apt) apt-cache search "${packages[@]}" ;;
                yum|dnf) $PM search "${packages[@]}" ;;
                zypper) zypper search "${packages[@]}" ;;
                pacman) pacman -Ss "${packages[@]}" ;;
            esac
            ;;
        info)
            case "$PM" in
                apt) apt-cache show "${packages[@]}" ;;
                yum|dnf) $PM info "${packages[@]}" ;;
                zypper) zypper info "${packages[@]}" ;;
                pacman) pacman -Si "${packages[@]}" ;;
            esac
            ;;
    esac
}

# Repository management
manage_repositories() {
    local action=$1
    local repo=$2
    
    case "$PM" in
        apt)
            case "$action" in
                add)
                    add-apt-repository "$repo"
                    apt-get update
                    ;;
                remove)
                    add-apt-repository --remove "$repo"
                    ;;
                list)
                    grep -h ^deb /etc/apt/sources.list /etc/apt/sources.list.d/*
                    ;;
            esac
            ;;
        yum|dnf)
            case "$action" in
                add)
                    $PM config-manager --add-repo "$repo"
                    ;;
                remove)
                    $PM config-manager --disable "$repo"
                    ;;
                list)
                    $PM repolist
                    ;;
            esac
            ;;
    esac
}

# Package security updates
security_updates() {
    echo "=== Security Updates ==="
    
    case "$PM" in
        apt)
            # Check for security updates
            apt-get update
            apt-get upgrade -s | grep -i security
            
            # Unattended upgrades configuration
            cat > /etc/apt/apt.conf.d/50unattended-upgrades <<EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
            ;;
        yum|dnf)
            # Check for security updates
            $PM updateinfo list security
            
            # Install security updates only
            $PM update --security -y
            
            # Enable automatic updates
            $PM install -y dnf-automatic
            systemctl enable dnf-automatic.timer
            ;;
    esac
}
```

## 19.7 Performance Monitoring and Tuning

### System Performance Analysis

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'cpu_usage': 80,
            'memory_usage': 90,
            'disk_usage': 85,
            'load_average': 4.0
        }
    
    def collect_metrics(self):
        """Collect system performance metrics"""
        import psutil
        
        # CPU metrics
        self.metrics['cpu'] = {
            'usage_percent': psutil.cpu_percent(interval=1),
            'load_average': os.getloadavg(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()
        }
        
        # Memory metrics
        mem = psutil.virtual_memory()
        self.metrics['memory'] = {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used,
            'free': mem.free,
            'cached': mem.cached if hasattr(mem, 'cached') else 0
        }
        
        # Disk metrics
        self.metrics['disk'] = {}
        for partition in psutil.disk_partitions():
            usage = psutil.disk_usage(partition.mountpoint)
            self.metrics['disk'][partition.mountpoint] = {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': usage.percent
            }
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.metrics['network'] = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors_in': net_io.errin,
            'errors_out': net_io.errout
        }
        
        return self.metrics
    
    def performance_tuning_script(self):
        """System performance tuning"""
        tuning_script = """#!/bin/bash
# System performance tuning script

# Kernel parameters optimization
cat >> /etc/sysctl.conf <<EOF
# Network optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# File system optimizations
fs.file-max = 2097152
fs.suid_dumpable = 0

# Memory optimizations
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Security optimizations
kernel.randomize_va_space = 2
kernel.exec-shield = 1
EOF

sysctl -p

# I/O scheduler optimization
for disk in /sys/block/sd*/queue/scheduler; do
    echo noop > $disk
done

# CPU governor setting
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# Transparent Huge Pages
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Update limits
cat >> /etc/security/limits.conf <<EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF

echo "Performance tuning applied"
"""
        return tuning_script
```

## 19.8 Backup and Recovery

### Backup Strategies

```bash
#!/bin/bash

# Comprehensive backup script

perform_backup() {
    local backup_type=$1  # full, incremental, differential
    local source_dir=$2
    local backup_dir=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    case "$backup_type" in
        full)
            # Full backup using tar
            tar -czf "$backup_dir/full_backup_$timestamp.tar.gz" \
                --exclude='*.tmp' \
                --exclude='*.cache' \
                "$source_dir"
            
            # Create backup manifest
            tar -tzf "$backup_dir/full_backup_$timestamp.tar.gz" > \
                "$backup_dir/full_backup_$timestamp.manifest"
            ;;
            
        incremental)
            # Incremental backup using rsync
            rsync -avz --delete \
                --backup --backup-dir="$backup_dir/incremental_$timestamp" \
                --log-file="$backup_dir/incremental_$timestamp.log" \
                "$source_dir/" "$backup_dir/current/"
            ;;
            
        differential)
            # Find files modified since last full backup
            last_full=$(ls -t "$backup_dir"/full_backup_*.tar.gz 2>/dev/null | head -1)
            if [ -z "$last_full" ]; then
                echo "No full backup found. Performing full backup instead."
                perform_backup full "$source_dir" "$backup_dir"
                return
            fi
            
            # Create differential backup
            find "$source_dir" -newer "$last_full" -print0 | \
                tar -czf "$backup_dir/diff_backup_$timestamp.tar.gz" \
                --null -T -
            ;;
    esac
    
    # Verify backup integrity
    verify_backup "$backup_dir" "$timestamp"
    
    # Clean old backups
    cleanup_old_backups "$backup_dir" 30
}

# Database backup
backup_databases() {
    local backup_dir="/backup/databases"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    mkdir -p "$backup_dir"
    
    # MySQL/MariaDB backup
    if command -v mysqldump &> /dev/null; then
        mysqldump --all-databases --single-transaction --quick \
            --lock-tables=false > "$backup_dir/mysql_$timestamp.sql"
        gzip "$backup_dir/mysql_$timestamp.sql"
    fi
    
    # PostgreSQL backup
    if command -v pg_dumpall &> /dev/null; then
        sudo -u postgres pg_dumpall > "$backup_dir/postgresql_$timestamp.sql"
        gzip "$backup_dir/postgresql_$timestamp.sql"
    fi
    
    # MongoDB backup
    if command -v mongodump &> /dev/null; then
        mongodump --out "$backup_dir/mongodb_$timestamp"
        tar -czf "$backup_dir/mongodb_$timestamp.tar.gz" \
            "$backup_dir/mongodb_$timestamp"
        rm -rf "$backup_dir/mongodb_$timestamp"
    fi
}

# System recovery procedures
system_recovery() {
    local recovery_type=$1
    
    case "$recovery_type" in
        boot_repair)
            # Repair boot loader
            mount /dev/sda1 /mnt
            mount --bind /dev /mnt/dev
            mount --bind /proc /mnt/proc
            mount --bind /sys /mnt/sys
            
            chroot /mnt /bin/bash <<EOF
grub-install /dev/sda
update-grub
EOF
            
            umount /mnt/{dev,proc,sys}
            umount /mnt
            ;;
            
        filesystem_check)
            # Check and repair file systems
            for fs in $(findmnt -rno SOURCE,FSTYPE | grep -E 'ext4|xfs'); do
                device=$(echo $fs | cut -d' ' -f1)
                fstype=$(echo $fs | cut -d' ' -f2)
                
                case "$fstype" in
                    ext4)
                        e2fsck -p "$device"
                        ;;
                    xfs)
                        xfs_repair "$device"
                        ;;
                esac
            done
            ;;
            
        restore_backup)
            # Restore from backup
            local backup_file=$2
            local restore_dir=$3
            
            tar -xzf "$backup_file" -C "$restore_dir"
            ;;
    esac
}
```

## 19.9 Shell Scripting and Automation

### Advanced Shell Scripting

```bash
#!/bin/bash

# Advanced shell scripting techniques

# Error handling and logging
set -euo pipefail  # Exit on error, undefined variables, pipe failures
IFS=$'\n\t'       # Set secure IFS

# Logging functions
readonly LOG_FILE="/var/log/admin_script.log"
readonly SCRIPT_NAME=$(basename "$0")

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Trap signals and errors
trap 'error "Script failed at line $LINENO"' ERR
trap 'log "Script interrupted"' INT TERM

# Function library
source /usr/local/lib/admin_functions.sh || {
    error "Failed to source function library"
    exit 1
}

# Configuration management
load_config() {
    local config_file="${1:-/etc/admin/config.conf}"
    
    if [[ ! -f "$config_file" ]]; then
        error "Configuration file not found: $config_file"
        return 1
    fi
    
    # Source configuration with validation
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Validate and export variables
        if [[ "$key" =~ ^[A-Z_]+$ ]]; then
            export "$key=$value"
            log "Loaded config: $key"
        else
            error "Invalid configuration key: $key"
        fi
    done < "$config_file"
}

# Parallel execution
parallel_execute() {
    local max_jobs=${1:-4}
    shift
    local commands=("$@")
    
    for cmd in "${commands[@]}"; do
        while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
            sleep 1
        done
        
        {
            log "Starting: $cmd"
            eval "$cmd"
            log "Completed: $cmd"
        } &
    done
    
    wait
    log "All parallel jobs completed"
}

# Remote execution
remote_execute() {
    local host=$1
    local command=$2
    local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
    
    log "Executing on $host: $command"
    
    if ssh $ssh_opts "$host" "$command"; then
        log "Success on $host"
        return 0
    else
        error "Failed on $host"
        return 1
    fi
}

# Bulk operations on multiple servers
bulk_operation() {
    local operation=$1
    local servers_file="/etc/admin/servers.txt"
    
    if [[ ! -f "$servers_file" ]]; then
        error "Servers file not found"
        return 1
    fi
    
    while IFS= read -r server; do
        [[ "$server" =~ ^#.*$ ]] && continue
        [[ -z "$server" ]] && continue
        
        case "$operation" in
            update)
                remote_execute "$server" "sudo apt-get update && sudo apt-get upgrade -y"
                ;;
            check)
                remote_execute "$server" "uptime; df -h; free -m"
                ;;
            deploy)
                scp -r /deploy/* "$server:/opt/application/"
                remote_execute "$server" "sudo systemctl restart application"
                ;;
        esac
    done < "$servers_file"
}
```

### Ansible Automation

```python
class AnsibleAutomation:
    def __init__(self):
        self.inventory = '/etc/ansible/hosts'
        self.playbook_dir = '/etc/ansible/playbooks/'
    
    def create_playbook(self, name, tasks):
        """Create Ansible playbook"""
        playbook = f"""---
- name: {name}
  hosts: all
  become: yes
  gather_facts: yes
  
  vars:
    admin_email: admin@example.com
    backup_dir: /backup
    
  tasks:
"""
        
        for task in tasks:
            playbook += f"""
    - name: {task['name']}
      {task['module']}:
"""
            for key, value in task.get('params', {}).items():
                playbook += f"        {key}: {value}\n"
            
            if task.get('when'):
                playbook += f"      when: {task['when']}\n"
            
            if task.get('notify'):
                playbook += f"      notify: {task['notify']}\n"
        
        # Add handlers if needed
        playbook += """
  handlers:
    - name: restart apache
      service:
        name: apache2
        state: restarted
        
    - name: restart nginx
      service:
        name: nginx
        state: restarted
"""
        
        return playbook
    
    def system_configuration_playbook(self):
        """Complete system configuration playbook"""
        tasks = [
            {
                'name': 'Update package cache',
                'module': 'apt',
                'params': {
                    'update_cache': 'yes',
                    'cache_valid_time': 3600
                },
                'when': 'ansible_os_family == "Debian"'
            },
            {
                'name': 'Install essential packages',
                'module': 'package',
                'params': {
                    'name': ['vim', 'htop', 'git', 'curl', 'wget', 'net-tools'],
                    'state': 'present'
                }
            },
            {
                'name': 'Create admin user',
                'module': 'user',
                'params': {
                    'name': 'sysadmin',
                    'groups': 'sudo',
                    'shell': '/bin/bash',
                    'create_home': 'yes'
                }
            },
            {
                'name': 'Configure SSH',
                'module': 'lineinfile',
                'params': {
                    'path': '/etc/ssh/sshd_config',
                    'regexp': '^PermitRootLogin',
                    'line': 'PermitRootLogin no'
                },
                'notify': 'restart ssh'
            },
            {
                'name': 'Setup firewall',
                'module': 'ufw',
                'params': {
                    'rule': 'allow',
                    'port': '22',
                    'proto': 'tcp'
                }
            }
        ]
        
        return self.create_playbook('System Configuration', tasks)
```

## Exercises

1. Create a comprehensive system monitoring dashboard that:
   - Displays real-time metrics
   - Alerts on threshold violations
   - Logs historical data
   - Provides trend analysis

2. Build an automated backup system that:
   - Performs incremental backups
   - Verifies backup integrity
   - Manages retention policies
   - Tests restore procedures

3. Implement a user management system that:
   - Automates account creation
   - Enforces password policies
   - Manages group memberships
   - Audits user activities

4. Create a network configuration tool that:
   - Configures interfaces
   - Manages firewall rules
   - Sets up VPN connections
   - Monitors network performance

5. Build a package deployment system that:
   - Manages repositories
   - Handles dependencies
   - Performs rollbacks
   - Tracks versions

6. Implement a log management solution that:
   - Centralizes logs
   - Parses log formats
   - Generates alerts
   - Creates reports

7. Create a performance tuning toolkit that:
   - Analyzes bottlenecks
   - Optimizes kernel parameters
   - Tunes application settings
   - Benchmarks improvements

8. Build a security auditing system that:
   - Scans for vulnerabilities
   - Checks compliance
   - Monitors file integrity
   - Reports security events

9. Implement a disaster recovery plan that:
   - Documents procedures
   - Automates failover
   - Tests recovery
   - Maintains documentation

10. Create a configuration management system that:
    - Manages configuration files
    - Tracks changes
    - Deploys updates
    - Maintains consistency

## Summary

This chapter covered Unix/Linux system administration essentials:

- System architecture provides the foundation for administration
- User and group management controls system access
- File system management ensures data organization and availability
- Process and service management maintains system operations
- Network configuration enables connectivity and communication
- Package management keeps systems updated and functional
- Performance monitoring identifies and resolves bottlenecks
- Backup and recovery protects against data loss
- Automation reduces manual tasks and improves consistency

Effective system administration requires understanding these core concepts, continuous learning, and adapting to evolving technologies and requirements.