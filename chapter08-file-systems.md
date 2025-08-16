# Chapter 8: File Systems

## Introduction

File systems provide the crucial abstraction layer between raw storage devices and the organized, hierarchical data structures that users and applications interact with. They manage how data is stored, retrieved, organized, and protected on persistent storage devices. This chapter explores file system concepts, implementation techniques, and modern file system designs that handle everything from tiny embedded devices to massive distributed storage systems.

## 8.1 File System Concepts

### File Abstraction

A file is a named collection of related data:

```c
struct file {
    char name[256];           // File name
    size_t size;              // File size in bytes
    time_t created;           // Creation timestamp
    time_t modified;          // Last modification
    time_t accessed;          // Last access
    uid_t owner;              // Owner user ID
    gid_t group;              // Group ID
    mode_t permissions;       // Access permissions
    ino_t inode_number;       // Unique identifier
    nlink_t link_count;       // Number of hard links
    blkcnt_t blocks;          // Blocks allocated
    dev_t device;             // Device ID
};
```

### File Types

**Regular Files**: Ordinary data files
**Directories**: Special files containing file names and references
**Symbolic Links**: Files pointing to other files
**Device Files**: Interface to hardware devices
- Block devices: Random access (disks)
- Character devices: Sequential access (terminals)
**Pipes**: For inter-process communication
**Sockets**: Network communication endpoints

### File Operations

Basic file operations:

```c
// File operations
int fd = open("file.txt", O_RDWR | O_CREAT, 0644);
ssize_t bytes = read(fd, buffer, sizeof(buffer));
ssize_t written = write(fd, data, data_size);
off_t pos = lseek(fd, offset, SEEK_SET);
int result = close(fd);

// File management
int status = unlink("file.txt");           // Delete
int renamed = rename("old.txt", "new.txt"); // Rename
int linked = link("file.txt", "hardlink");  // Hard link
int symlinked = symlink("file.txt", "symlink"); // Symbolic link

// Metadata operations
struct stat st;
int stated = stat("file.txt", &st);
int chmoded = chmod("file.txt", 0755);
int chowned = chown("file.txt", uid, gid);
```

## 8.2 Directory Structure

### Directory Implementation

Directories map names to file metadata:

```c
// Simple directory entry
struct dirent {
    ino_t d_ino;              // Inode number
    off_t d_off;              // Offset to next dirent
    unsigned short d_reclen;   // Length of this record
    unsigned char d_type;      // File type
    char d_name[256];         // Filename
};

// Directory operations
DIR* dir = opendir("/home/user");
struct dirent* entry;
while ((entry = readdir(dir)) != NULL) {
    printf("%s (inode: %lu)\n", entry->d_name, entry->d_ino);
}
closedir(dir);
```

### Path Name Resolution

Converting path to inode:

```c
ino_t path_to_inode(const char* path) {
    if (path[0] == '/') {
        // Absolute path - start from root
        current_inode = ROOT_INODE;
        path++;
    } else {
        // Relative path - start from current directory
        current_inode = get_current_directory_inode();
    }
    
    char component[256];
    while (extract_next_component(&path, component)) {
        // Look up component in current directory
        current_inode = lookup_in_directory(current_inode, component);
        if (current_inode == INVALID_INODE) {
            return INVALID_INODE;  // Path not found
        }
        
        // Follow symbolic links if necessary
        if (is_symlink(current_inode)) {
            current_inode = follow_symlink(current_inode);
        }
    }
    
    return current_inode;
}
```

## 8.3 File System Layout

### Classic Unix File System Layout

```
+------------------+
| Boot Block       | Block 0
+------------------+
| Superblock       | Block 1
+------------------+
| Inode Bitmap     | Blocks 2-n
+------------------+
| Data Bitmap      | Blocks n+1-m
+------------------+
| Inode Table      | Blocks m+1-k
+------------------+
| Data Blocks      | Blocks k+1-end
+------------------+
```

### Superblock

Contains file system metadata:

```c
struct superblock {
    uint32_t magic_number;     // File system identifier
    uint32_t block_size;       // Block size in bytes
    uint64_t total_blocks;     // Total blocks in file system
    uint64_t free_blocks;      // Number of free blocks
    uint64_t total_inodes;     // Total inodes
    uint64_t free_inodes;      // Number of free inodes
    uint64_t first_data_block; // First data block number
    time_t mount_time;         // Last mount time
    time_t write_time;         // Last write time
    uint16_t mount_count;      // Number of mounts
    uint16_t max_mount_count;  // Force check after N mounts
    uint16_t state;            // Clean/dirty state
    char volume_name[64];      // Volume label
};
```

## 8.4 File Allocation Methods

### Contiguous Allocation

Files occupy consecutive blocks:

```c
struct contiguous_file {
    uint32_t start_block;
    uint32_t num_blocks;
};

// Simple but suffers from external fragmentation
// Good for read-only file systems (CD-ROM)
```

### Linked Allocation

Each block points to the next:

```c
struct linked_block {
    char data[BLOCK_SIZE - sizeof(uint32_t)];
    uint32_t next_block;  // Pointer to next block
};

// No external fragmentation but poor random access
// Used in FAT file system
```

### Indexed Allocation (inodes)

Index block contains pointers to data blocks:

```c
struct inode {
    mode_t mode;              // File type and permissions
    uid_t uid;                // Owner user ID
    gid_t gid;                // Owner group ID
    off_t size;               // File size
    time_t atime;             // Access time
    time_t mtime;             // Modification time
    time_t ctime;             // Change time
    uint32_t direct[12];      // Direct block pointers
    uint32_t indirect;        // Single indirect
    uint32_t double_indirect; // Double indirect
    uint32_t triple_indirect; // Triple indirect
};

// Calculate maximum file size (4KB blocks, 32-bit pointers)
// Direct: 12 × 4KB = 48KB
// Single indirect: (4KB/4) × 4KB = 4MB
// Double indirect: (4KB/4)² × 4KB = 4GB
// Triple indirect: (4KB/4)³ × 4KB = 4TB
```

### Extent-Based Allocation

Groups of contiguous blocks:

```c
struct extent {
    uint64_t start_block;
    uint64_t num_blocks;
};

struct extent_inode {
    struct extent extents[4];  // Inline extents
    uint64_t extent_tree_root;  // For larger files
};

// Used in modern file systems (ext4, Btrfs, XFS)
// Reduces metadata overhead
```

## 8.5 Free Space Management

### Bitmap

One bit per block:

```c
typedef struct {
    uint8_t* bitmap;
    size_t num_blocks;
} block_bitmap_t;

bool is_block_free(block_bitmap_t* bitmap, uint32_t block) {
    uint32_t byte = block / 8;
    uint32_t bit = block % 8;
    return !(bitmap->bitmap[byte] & (1 << bit));
}

void allocate_block(block_bitmap_t* bitmap, uint32_t block) {
    uint32_t byte = block / 8;
    uint32_t bit = block % 8;
    bitmap->bitmap[byte] |= (1 << bit);
}

void free_block(block_bitmap_t* bitmap, uint32_t block) {
    uint32_t byte = block / 8;
    uint32_t bit = block % 8;
    bitmap->bitmap[byte] &= ~(1 << bit);
}

// Find contiguous free blocks
uint32_t find_free_blocks(block_bitmap_t* bitmap, uint32_t count) {
    uint32_t consecutive = 0;
    uint32_t start = 0;
    
    for (uint32_t i = 0; i < bitmap->num_blocks; i++) {
        if (is_block_free(bitmap, i)) {
            if (consecutive == 0) start = i;
            consecutive++;
            if (consecutive == count) return start;
        } else {
            consecutive = 0;
        }
    }
    return INVALID_BLOCK;
}
```

### Free List

Linked list of free blocks:

```c
struct free_list {
    uint32_t next_free;
    // Rest of block can store more free block numbers
    uint32_t free_blocks[(BLOCK_SIZE - 4) / 4];
};

uint32_t allocate_from_free_list(struct superblock* sb) {
    if (sb->free_list_head == INVALID_BLOCK) {
        return INVALID_BLOCK;  // No free blocks
    }
    
    uint32_t allocated = sb->free_list_head;
    struct free_list* fl = read_block(allocated);
    sb->free_list_head = fl->next_free;
    sb->free_blocks--;
    
    return allocated;
}
```

## 8.6 Directory Implementation

### Linear List

Simple but slow for large directories:

```c
struct linear_directory {
    struct {
        char name[256];
        ino_t inode;
    } entries[MAX_ENTRIES];
    int num_entries;
};

ino_t linear_lookup(struct linear_directory* dir, const char* name) {
    for (int i = 0; i < dir->num_entries; i++) {
        if (strcmp(dir->entries[i].name, name) == 0) {
            return dir->entries[i].inode;
        }
    }
    return INVALID_INODE;
}
```

### Hash Table

Fast lookup with hash function:

```c
struct hash_directory {
    struct hash_entry {
        char name[256];
        ino_t inode;
        struct hash_entry* next;  // Collision chain
    } *buckets[HASH_SIZE];
};

uint32_t hash_function(const char* name) {
    uint32_t hash = 5381;
    int c;
    while ((c = *name++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % HASH_SIZE;
}

ino_t hash_lookup(struct hash_directory* dir, const char* name) {
    uint32_t bucket = hash_function(name);
    struct hash_entry* entry = dir->buckets[bucket];
    
    while (entry) {
        if (strcmp(entry->name, name) == 0) {
            return entry->inode;
        }
        entry = entry->next;
    }
    return INVALID_INODE;
}
```

### B-Tree

Balanced tree for very large directories:

```c
struct btree_node {
    int num_keys;
    char keys[MAX_KEYS][256];
    ino_t values[MAX_KEYS];
    struct btree_node* children[MAX_KEYS + 1];
    bool is_leaf;
};

ino_t btree_lookup(struct btree_node* root, const char* name) {
    int i = 0;
    while (i < root->num_keys && strcmp(name, root->keys[i]) > 0) {
        i++;
    }
    
    if (i < root->num_keys && strcmp(name, root->keys[i]) == 0) {
        return root->values[i];
    }
    
    if (root->is_leaf) {
        return INVALID_INODE;
    }
    
    return btree_lookup(root->children[i], name);
}
```

## 8.7 File System Implementation

### Virtual File System (VFS)

Abstraction layer for multiple file systems:

```c
struct vfs_operations {
    struct inode* (*lookup)(struct inode* dir, const char* name);
    int (*create)(struct inode* dir, const char* name, mode_t mode);
    int (*mkdir)(struct inode* dir, const char* name, mode_t mode);
    int (*rmdir)(struct inode* dir, const char* name);
    int (*rename)(struct inode* old_dir, const char* old_name,
                  struct inode* new_dir, const char* new_name);
    int (*link)(struct inode* old, struct inode* dir, const char* name);
    int (*unlink)(struct inode* dir, const char* name);
    int (*symlink)(const char* target, struct inode* dir, const char* name);
};

struct file_operations {
    ssize_t (*read)(struct file* file, char* buf, size_t count, loff_t* pos);
    ssize_t (*write)(struct file* file, const char* buf, size_t count, loff_t* pos);
    int (*open)(struct inode* inode, struct file* file);
    int (*release)(struct inode* inode, struct file* file);
    int (*fsync)(struct file* file);
    loff_t (*llseek)(struct file* file, loff_t offset, int whence);
};

struct file_system_type {
    const char* name;
    int (*mount)(struct file_system_type* fs, int flags,
                 const char* dev, void* data, struct vfsmount* mnt);
    void (*kill_sb)(struct super_block* sb);
    struct module* owner;
    struct file_system_type* next;
};
```

### Buffer Cache

Cache frequently accessed blocks:

```c
struct buffer_head {
    sector_t block_number;        // Block number on device
    size_t size;                  // Block size
    char* data;                   // Actual data
    struct block_device* bdev;    // Block device
    
    atomic_t count;               // Reference count
    unsigned long state;          // Buffer state flags
    
    struct list_head lru;         // LRU list
    struct hlist_node hash;       // Hash table link
    
    void (*end_io)(struct buffer_head* bh, int uptodate);
};

#define BH_Uptodate  0  // Buffer contains valid data
#define BH_Dirty     1  // Buffer modified
#define BH_Lock      2  // Buffer locked
#define BH_Req       3  // I/O requested
#define BH_Mapped    4  // Buffer mapped to disk

struct buffer_head* get_block_buffer(dev_t dev, sector_t block) {
    struct buffer_head* bh;
    
    // Check hash table
    bh = lookup_buffer(dev, block);
    if (bh) {
        atomic_inc(&bh->count);
        return bh;
    }
    
    // Not in cache, allocate new buffer
    bh = alloc_buffer_head();
    bh->block_number = block;
    bh->bdev = get_block_device(dev);
    
    // Read from disk
    submit_bh(READ, bh);
    wait_on_buffer(bh);
    
    // Add to cache
    insert_buffer_hash(bh);
    add_to_lru(bh);
    
    return bh;
}
```

## 8.8 Modern File Systems

### ext4 (Fourth Extended File System)

Linux's primary file system:

```c
struct ext4_super_block {
    __le32 s_inodes_count;        // Total inodes
    __le32 s_blocks_count_lo;     // Total blocks (low 32 bits)
    __le32 s_r_blocks_count_lo;   // Reserved blocks
    __le32 s_free_blocks_count_lo; // Free blocks
    __le32 s_free_inodes_count;   // Free inodes
    __le32 s_first_data_block;    // First data block
    __le32 s_log_block_size;      // Block size = 1024 << s_log_block_size
    __le32 s_log_cluster_size;    // Cluster size
    __le32 s_blocks_per_group;    // Blocks per group
    __le32 s_clusters_per_group;  // Clusters per group
    __le32 s_inodes_per_group;    // Inodes per group
    __le32 s_mtime;               // Mount time
    __le32 s_wtime;               // Write time
    __le16 s_mnt_count;           // Mount count
    __le16 s_max_mnt_count;       // Max mount count
    __le16 s_magic;               // Magic signature (0xEF53)
    __le16 s_state;               // File system state
    // ... many more fields
};

// Features
// - Extents for efficient large file support
// - Delayed allocation for better performance
// - Journal checksumming
// - Fast fsck
// - Larger file system support (1 EiB)
```

### Btrfs (B-tree File System)

Copy-on-write file system:

```c
// B-tree based, everything is a tree
struct btrfs_key {
    __u64 objectid;  // Object identifier
    __u8 type;       // Item type
    __u64 offset;    // Offset or other info
};

// Copy-on-write operation
void cow_write_block(struct btree_block* block) {
    // Allocate new block
    struct btree_block* new_block = allocate_block();
    
    // Copy data
    memcpy(new_block, block, BLOCK_SIZE);
    
    // Modify new block
    modify_block(new_block);
    
    // Update parent pointer
    update_parent_pointer(block->parent, block, new_block);
    
    // Old block becomes part of snapshot
}

// Features:
// - Snapshots and clones
// - Built-in RAID
// - Self-healing with checksums
// - Online defragmentation
// - Compression
```

### ZFS (Zettabyte File System)

Advanced file system with volume management:

```c
// Copy-on-write with transactional semantics
// Merkle tree for data integrity
struct zfs_block_pointer {
    dva_t dva[3];        // Data virtual addresses (triple redundancy)
    uint64_t birth_txg;  // Transaction group when created
    uint64_t fill_count; // Number of non-zero blocks
    uint8_t checksum[32]; // 256-bit checksum
    uint8_t compress;    // Compression algorithm
    uint8_t type;        // Block type
    uint8_t level;       // Indirection level
};

// Features:
// - 128-bit addressing
// - End-to-end data integrity
// - Snapshots and clones
// - Built-in compression and deduplication
// - RAID-Z (enhanced RAID)
// - Self-healing
```

### NTFS (New Technology File System)

Windows file system:

```c
struct ntfs_mft_record {
    char signature[4];      // "FILE"
    uint16_t usa_offset;    // Update sequence array offset
    uint16_t usa_count;     // Update sequence array count
    uint64_t lsn;          // Log file sequence number
    uint16_t sequence;     // Sequence number
    uint16_t link_count;   // Hard link count
    uint16_t attrs_offset; // First attribute offset
    uint16_t flags;        // Flags
    uint32_t bytes_used;   // Bytes used in record
    uint32_t bytes_allocated; // Bytes allocated for record
    uint64_t base_mft_record; // Base MFT record
    uint16_t next_attr_id; // Next attribute ID
};

// Everything is a file, including metadata
// $MFT - Master File Table
// $MFTMirr - MFT mirror
// $LogFile - Transaction log
// $Volume - Volume information
// $AttrDef - Attribute definitions
// $Bitmap - Allocation bitmap
// $Boot - Boot sector
// $BadClus - Bad clusters
// $Secure - Security descriptors
// $UpCase - Unicode uppercase table
```

## 8.9 File System Performance

### Read-Ahead and Write-Behind

```c
// Read-ahead for sequential access
void read_ahead(struct file* file, loff_t offset, size_t count) {
    // Detect sequential pattern
    if (offset == file->last_read_end) {
        file->sequential_count++;
        
        if (file->sequential_count > SEQUENTIAL_THRESHOLD) {
            // Read ahead next blocks
            size_t ahead_size = min(READ_AHEAD_SIZE, 
                                   file->size - offset - count);
            
            for (size_t i = 0; i < ahead_size; i += BLOCK_SIZE) {
                struct buffer_head* bh = get_block_async(
                    file->inode, 
                    (offset + count + i) / BLOCK_SIZE
                );
                mark_buffer_readahead(bh);
            }
        }
    } else {
        file->sequential_count = 0;
    }
    
    file->last_read_end = offset + count;
}

// Write-behind for better performance
void write_behind(struct buffer_cache* cache) {
    struct buffer_head* bh;
    int nr_written = 0;
    
    list_for_each_entry(bh, &cache->dirty_list, list) {
        if (!buffer_locked(bh) && buffer_dirty(bh)) {
            // Write dirty buffer asynchronously
            submit_bh(WRITE, bh);
            nr_written++;
            
            if (nr_written >= MAX_WRITEBACK) {
                break;  // Limit per iteration
            }
        }
    }
}
```

### Disk Scheduling Integration

```c
// Elevator algorithm for disk I/O
struct request_queue {
    struct list_head queue;
    sector_t last_sector;
    int direction;  // 0 = down, 1 = up
};

void elevator_add_request(struct request_queue* q, struct request* req) {
    struct request* pos;
    
    // Find insertion position
    if (q->direction == 1) {  // Moving up
        // Insert in ascending order
        list_for_each_entry(pos, &q->queue, list) {
            if (pos->sector > req->sector) {
                list_add_tail(&req->list, &pos->list);
                return;
            }
        }
    } else {  // Moving down
        // Insert in descending order
        list_for_each_entry_reverse(pos, &q->queue, list) {
            if (pos->sector < req->sector) {
                list_add(&req->list, &pos->list);
                return;
            }
        }
    }
    
    // Add to end if no position found
    list_add_tail(&req->list, &q->queue);
}
```

## 8.10 File System Consistency

### Journaling

Write-ahead logging for crash recovery:

```c
struct journal_transaction {
    tid_t tid;                    // Transaction ID
    enum { RUNNING, COMMIT, COMMITTED } state;
    struct list_head buffers;     // Modified buffers
    struct list_head checkpoint;  // Checkpoint list
};

// Journal write sequence
void journal_commit_transaction(struct journal* j, struct transaction* t) {
    // 1. Write descriptor block
    write_journal_descriptor(j, t);
    
    // 2. Write data/metadata blocks to journal
    struct buffer_head* bh;
    list_for_each_entry(bh, &t->buffers, list) {
        write_journal_block(j, bh);
    }
    
    // 3. Write commit block
    write_journal_commit(j, t);
    
    // 4. Write blocks to final location
    list_for_each_entry(bh, &t->buffers, list) {
        write_buffer_to_disk(bh);
    }
    
    // 5. Update journal superblock
    update_journal_superblock(j, t->tid);
}

// Recovery after crash
void journal_recover(struct journal* j) {
    tid_t last_committed = read_journal_superblock(j);
    
    // Scan journal for committed transactions
    for (tid_t tid = last_committed + 1; ; tid++) {
        struct transaction* t = read_transaction(j, tid);
        if (!t) break;
        
        if (t->state == COMMITTED) {
            // Replay transaction
            replay_transaction(t);
            last_committed = tid;
        } else {
            // Incomplete transaction, discard
            break;
        }
    }
    
    update_journal_superblock(j, last_committed);
}
```

### Copy-on-Write (COW)

Never overwrite data in place:

```c
// COW block update
block_t cow_update_block(struct cow_fs* fs, block_t old_block) {
    // Allocate new block
    block_t new_block = allocate_block(fs);
    
    // Copy old data
    copy_block(old_block, new_block);
    
    // Modify new block
    modify_block(new_block);
    
    // Update pointer in parent
    update_parent_pointer(old_block, new_block);
    
    // Old block available for snapshot
    return new_block;
}

// Snapshot creation (instant)
void create_snapshot(struct cow_fs* fs, const char* name) {
    struct snapshot* snap = allocate_snapshot();
    snap->root = fs->current_root;  // Just copy root pointer
    snap->timestamp = current_time();
    snap->name = strdup(name);
    
    add_snapshot(fs, snap);
}
```

### File System Check (fsck)

```c
void fsck_check_consistency(struct filesystem* fs) {
    // Phase 1: Check blocks and sizes
    for (each inode) {
        check_inode_blocks(inode);
        check_inode_size(inode);
        mark_blocks_used(inode);
    }
    
    // Phase 2: Check directory structure
    check_root_directory();
    for (each directory) {
        check_directory_entries(directory);
        check_parent_pointers(directory);
    }
    
    // Phase 3: Check connectivity
    mark_reachable_inodes();
    find_orphaned_inodes();
    
    // Phase 4: Check reference counts
    for (each inode) {
        verify_link_count(inode);
    }
    
    // Phase 5: Check free space
    verify_free_block_bitmap();
    verify_free_inode_bitmap();
    
    // Phase 6: Salvage orphaned files
    reconnect_orphans_to_lost_found();
}
```

## 8.11 Network File Systems

### NFS (Network File System)

```c
// NFS client operations
struct nfs_client_ops {
    int (*lookup)(struct nfs_server* server, struct nfs_fh* dir,
                  const char* name, struct nfs_fh* result);
    int (*read)(struct nfs_server* server, struct nfs_fh* file,
                loff_t offset, size_t count, void* buffer);
    int (*write)(struct nfs_server* server, struct nfs_fh* file,
                 loff_t offset, size_t count, const void* buffer);
    int (*getattr)(struct nfs_server* server, struct nfs_fh* file,
                   struct nfs_fattr* attr);
};

// RPC call example
int nfs_read_rpc(struct nfs_server* server, struct nfs_fh* fh,
                 loff_t offset, size_t count, void* buffer) {
    struct rpc_message msg = {
        .procedure = NFSPROC_READ,
        .args = {fh, offset, count},
        .result = buffer
    };
    
    return rpc_call(server->client, &msg);
}

// Client-side caching
struct nfs_cache_entry {
    struct nfs_fh file_handle;
    loff_t offset;
    size_t size;
    void* data;
    time_t timestamp;
    bool dirty;
};
```

### Distributed File Systems

```c
// Distributed file system with replication
struct dfs_file {
    char name[256];
    size_t size;
    int replication_factor;
    struct {
        char server[256];
        uint64_t block_id;
    } blocks[MAX_BLOCKS];
};

// Read with failover
ssize_t dfs_read(struct dfs_file* file, void* buffer, 
                 size_t size, loff_t offset) {
    int block_index = offset / BLOCK_SIZE;
    int block_offset = offset % BLOCK_SIZE;
    
    // Try primary replica
    for (int replica = 0; replica < file->replication_factor; replica++) {
        struct dfs_server* server = connect_to_server(
            file->blocks[block_index].server
        );
        
        if (server) {
            ssize_t result = server_read(
                server,
                file->blocks[block_index].block_id,
                buffer,
                size,
                block_offset
            );
            
            if (result >= 0) {
                return result;  // Success
            }
        }
        
        // Try next replica
    }
    
    return -EIO;  // All replicas failed
}
```

## 8.12 Special-Purpose File Systems

### Flash File Systems

Optimized for NAND flash characteristics:

```c
// Log-structured for flash
struct flash_fs {
    uint32_t current_segment;
    uint32_t segment_size;
    uint32_t erase_count[MAX_SEGMENTS];
    
    // Wear leveling
    uint32_t get_next_segment() {
        // Find segment with lowest erase count
        uint32_t min_erases = UINT32_MAX;
        uint32_t best_segment = 0;
        
        for (uint32_t i = 0; i < MAX_SEGMENTS; i++) {
            if (erase_count[i] < min_erases) {
                min_erases = erase_count[i];
                best_segment = i;
            }
        }
        
        return best_segment;
    }
    
    // Garbage collection
    void garbage_collect() {
        // Find segment with most dead data
        uint32_t victim = find_victim_segment();
        
        // Copy live data to new segment
        copy_live_data(victim, current_segment);
        
        // Erase victim segment
        erase_segment(victim);
        erase_count[victim]++;
    }
};
```

### In-Memory File Systems

```c
// tmpfs implementation
struct tmpfs_inode {
    struct inode vfs_inode;
    union {
        struct {
            struct page** pages;  // For regular files
            size_t num_pages;
        };
        struct {
            struct tmpfs_dirent* entries;  // For directories
            size_t num_entries;
        };
    };
};

void* tmpfs_read_page(struct tmpfs_inode* inode, size_t page_num) {
    if (page_num >= inode->num_pages) {
        return NULL;
    }
    
    struct page* page = inode->pages[page_num];
    if (!page) {
        // Allocate on demand
        page = alloc_page(GFP_KERNEL);
        inode->pages[page_num] = page;
    }
    
    return page_address(page);
}
```

## 8.13 File System Security

### Access Control Lists (ACLs)

```c
struct posix_acl_entry {
    short tag;      // ACL_USER, ACL_GROUP, etc.
    unsigned short perm;  // R, W, X permissions
    union {
        uid_t uid;
        gid_t gid;
    };
};

struct posix_acl {
    atomic_t refcount;
    unsigned int count;
    struct posix_acl_entry entries[];
};

int check_acl_permission(struct inode* inode, int mask, uid_t uid, gid_t gid) {
    struct posix_acl* acl = get_acl(inode);
    
    // Check user entry
    for (int i = 0; i < acl->count; i++) {
        if (acl->entries[i].tag == ACL_USER && 
            acl->entries[i].uid == uid) {
            return (acl->entries[i].perm & mask) == mask;
        }
    }
    
    // Check group entries
    for (int i = 0; i < acl->count; i++) {
        if (acl->entries[i].tag == ACL_GROUP && 
            user_in_group(uid, acl->entries[i].gid)) {
            if ((acl->entries[i].perm & mask) == mask) {
                return 1;
            }
        }
    }
    
    // Check others
    for (int i = 0; i < acl->count; i++) {
        if (acl->entries[i].tag == ACL_OTHER) {
            return (acl->entries[i].perm & mask) == mask;
        }
    }
    
    return 0;
}
```

### Encryption

```c
// File-level encryption
struct encrypted_file {
    uint8_t nonce[12];          // Unique per file
    uint8_t encrypted_key[32];  // File encryption key
    uint8_t signature[64];      // HMAC signature
};

ssize_t encrypted_read(struct file* file, void* buffer, 
                       size_t count, loff_t offset) {
    // Read encrypted data
    void* encrypted = kmalloc(count, GFP_KERNEL);
    ssize_t bytes = vfs_read(file, encrypted, count, offset);
    
    if (bytes > 0) {
        // Decrypt data
        struct crypto_context* ctx = get_file_crypto_context(file);
        decrypt_data(ctx, encrypted, buffer, bytes, offset);
    }
    
    kfree(encrypted);
    return bytes;
}
```

## Exercises

1. Implement a simple file system with:
   - File creation/deletion
   - Directory support
   - Block allocation using bitmap
   - Basic inode structure

2. Write a program that:
   - Creates a memory-mapped file
   - Implements a B-tree index in the file
   - Supports concurrent access

3. Design a log-structured file system for SSDs that:
   - Minimizes write amplification
   - Implements wear leveling
   - Handles garbage collection

4. Implement file system caching with:
   - LRU replacement
   - Read-ahead detection
   - Write-back with periodic flush

5. Create a simple FUSE (Filesystem in Userspace) file system that:
   - Stores files in a database
   - Supports basic operations
   - Implements permissions

6. Write a program demonstrating:
   - Hard links vs symbolic links
   - File locking (advisory and mandatory)
   - Extended attributes

7. Implement a basic journaling system:
   - Write-ahead logging
   - Transaction commit
   - Recovery after crash

8. Design a versioning file system that:
   - Keeps file history
   - Supports snapshots
   - Implements copy-on-write

9. Create a simple distributed file system with:
   - File replication
   - Consistency protocol
   - Failure handling

10. Write an fsck utility that:
    - Checks inode consistency
    - Verifies directory structure
    - Repairs common problems

## Summary

This chapter explored file system design and implementation:

- File systems provide organized, persistent storage abstractions
- Multiple allocation strategies trade off between fragmentation and performance
- Directory structures range from simple lists to sophisticated B-trees
- Modern file systems use journaling or copy-on-write for consistency
- Virtual file system layers enable multiple file system types
- Caching and buffering are critical for performance
- Network and distributed file systems enable remote storage access
- Special-purpose file systems optimize for specific hardware or use cases
- Security features include access controls and encryption

File systems are essential for organizing and persisting data, forming the foundation for data storage in modern computing systems. The next chapter will explore programming fundamentals, building on our system-level understanding to examine how high-level programs are constructed and executed.