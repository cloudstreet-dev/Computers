# Chapter 6: Memory Management

## Introduction

Memory management is one of the most critical functions of an operating system. It involves allocating memory to processes, protecting processes from each other, and providing the abstraction of a large, contiguous address space to each process despite physical memory limitations. This chapter explores memory management techniques from simple schemes to sophisticated virtual memory systems used in modern operating systems.

## 6.1 Memory Hierarchy and Abstractions

### Memory Hierarchy Review

```
Register         1 cycle        ~1 KB
L1 Cache        2-4 cycles      32-64 KB
L2 Cache        10-20 cycles    256 KB - 1 MB
L3 Cache        30-50 cycles    8-32 MB
Main Memory     100-300 cycles  8-64 GB
SSD             10,000 cycles   256 GB - 4 TB
HDD             1M+ cycles      1-20 TB
```

### Address Spaces

**Physical Address Space**
- Actual hardware memory addresses
- Limited by installed RAM
- Directly accessed by memory controller

**Logical/Virtual Address Space**
- Addresses used by programs
- Can be larger than physical memory
- Translated to physical addresses by MMU

**Address Binding**
- Compile time: Absolute addresses
- Load time: Relocatable addresses
- Execution time: Virtual addresses (most flexible)

## 6.2 Memory Management Requirements

### Relocation
Programs should run regardless of physical location:
```c
// Program thinks it's at address 0x1000
int data = *(int*)0x1000;
// MMU translates to actual physical address
```

### Protection
Prevent processes from accessing each other's memory:
```c
// Process A tries to access Process B's memory
char *ptr = (char*)0x50000;  // B's address
*ptr = 0;  // Should cause segmentation fault
```

### Sharing
Allow controlled sharing when needed:
- Shared libraries
- Inter-process communication
- Copy-on-write pages

### Logical Organization
Support modular programs:
- Code segment
- Data segment
- Stack segment
- Heap segment

### Physical Organization
Manage hierarchy efficiently:
- Main memory
- Secondary storage
- Memory-mapped files

## 6.3 Memory Allocation Schemes

### Fixed Partitioning

Divide memory into fixed-size partitions:

```
+----------------+ 0 KB
|     OS         |
+----------------+ 256 KB
|   Partition 1  | (256 KB)
+----------------+ 512 KB
|   Partition 2  | (512 KB)
+----------------+ 1024 KB
|   Partition 3  | (1 MB)
+----------------+ 2048 KB
|   Partition 4  | (2 MB)
+----------------+ 4096 KB
```

Problems:
- Internal fragmentation
- Fixed maximum process size
- Limited multiprogramming degree

### Variable Partitioning

Allocate exactly what's needed:

```
Initial:          After allocations:     After P2 terminates:
+--------+        +--------+             +--------+
|   OS   |        |   OS   |             |   OS   |
+--------+        +--------+             +--------+
|        |        | P1-2MB |             | P1-2MB |
|  Free  |        +--------+             +--------+
|  10MB  |        | P2-3MB |             |  Free  |
|        |        +--------+             |  3MB   | <- Hole
|        |        | P3-1MB |             +--------+
+--------+        +--------+             | P3-1MB |
                  |  Free  |             +--------+
                  |  4MB   |             |  Free  |
                  +--------+             |  4MB   |
                                        +--------+
```

### Dynamic Storage Allocation

**Allocation Strategies**

First Fit:
```c
block* first_fit(size_t size) {
    for (block* b = free_list; b != NULL; b = b->next) {
        if (b->size >= size) {
            return b;  // First block that fits
        }
    }
    return NULL;
}
```

Best Fit:
```c
block* best_fit(size_t size) {
    block* best = NULL;
    size_t best_size = SIZE_MAX;
    
    for (block* b = free_list; b != NULL; b = b->next) {
        if (b->size >= size && b->size < best_size) {
            best = b;
            best_size = b->size;
        }
    }
    return best;
}
```

Worst Fit:
```c
block* worst_fit(size_t size) {
    block* worst = NULL;
    size_t worst_size = 0;
    
    for (block* b = free_list; b != NULL; b = b->next) {
        if (b->size >= size && b->size > worst_size) {
            worst = b;
            worst_size = b->size;
        }
    }
    return worst;
}
```

### Fragmentation

**Internal Fragmentation**
- Wasted space within allocated blocks
- Fixed partitioning problem

**External Fragmentation**
- Free memory scattered in small pieces
- Variable partitioning problem

**Compaction**
```
Before:                    After:
+--------+                +--------+
| P1-2MB |                | P1-2MB |
+--------+                +--------+
| Free-1 |                | P3-1MB |
+--------+                +--------+
| P3-1MB |                | P4-3MB |
+--------+                +--------+
| Free-2 |     =====>     | Free   |
+--------+                | 5MB    |
| P4-3MB |                |        |
+--------+                |        |
| Free-3 |                +--------+
+--------+
```

## 6.4 Paging

Divide memory into fixed-size pages (logical) and frames (physical):

### Basic Paging

```
Logical Address:        Physical Memory:
Page 0 ----+           +--------+ Frame 0
Page 1 ----|---------> | Page 2 |
Page 2 ----|----+      +--------+ Frame 1
Page 3 ----|----|----> | Page 0 |
           |    |      +--------+ Frame 2
           |    +----> | Page 3 |
           |           +--------+ Frame 3
           +---------> | Page 1 |
                       +--------+ Frame 4
                       | Free   |
                       +--------+
```

### Address Translation

Logical address = (page number, offset)

```
32-bit logical address with 4KB pages:
+----------------+-------------+
| Page Number    | Page Offset |
| (20 bits)      | (12 bits)   |
+----------------+-------------+

Page number: Indexes page table
Page offset: Offset within page
```

### Page Table

Maps logical pages to physical frames:

```c
struct page_table_entry {
    unsigned int frame_number : 20;  // Physical frame
    unsigned int present : 1;        // Page in memory?
    unsigned int writable : 1;       // Write permission
    unsigned int user : 1;           // User accessible?
    unsigned int accessed : 1;       // Recently accessed?
    unsigned int dirty : 1;          // Modified?
    unsigned int reserved : 7;       // OS-specific
};

// Address translation
physical_addr = page_table[page_number].frame_number * PAGE_SIZE + offset;
```

### Translation Lookaside Buffer (TLB)

Cache for page table entries:

```
TLB Hit:
Logical Address → TLB → Physical Address (fast)

TLB Miss:
Logical Address → TLB (miss) → Page Table → Update TLB → Physical Address
```

TLB Management:
```c
struct tlb_entry {
    unsigned int page_number;
    unsigned int frame_number;
    unsigned int valid;
    unsigned int asid;  // Address Space ID
};

// TLB lookup
tlb_entry* tlb_lookup(unsigned int page) {
    for (int i = 0; i < TLB_SIZE; i++) {
        if (tlb[i].valid && tlb[i].page_number == page) {
            return &tlb[i];  // TLB hit
        }
    }
    return NULL;  // TLB miss
}
```

## 6.5 Segmentation

Divide memory into logical segments:

### Segment Table

```c
struct segment_descriptor {
    unsigned int base;      // Starting address
    unsigned int limit;     // Segment size
    unsigned int access;    // R/W/X permissions
};

// Segments for a process
segment_descriptor segments[] = {
    {0x10000, 0x2000, READ|EXECUTE},  // Code
    {0x20000, 0x3000, READ|WRITE},    // Data
    {0x30000, 0x1000, READ|WRITE},    // Stack
    {0x40000, 0x5000, READ|WRITE}     // Heap
};
```

### Segmentation with Paging

Combine benefits of both:

```
Logical Address:
+----------+---------+--------+
| Segment  | Page    | Offset |
+----------+---------+--------+
     ↓         ↓         ↓
  Segment   Page in   Offset
   Table    Segment   in Page
     ↓         ↓         ↓
   Base +   Frame  +  Offset = Physical Address
```

## 6.6 Virtual Memory

Allows execution of processes larger than physical memory:

### Demand Paging

Load pages only when needed:

```c
// Page fault handler
void page_fault_handler(unsigned int virtual_addr) {
    unsigned int page = virtual_addr / PAGE_SIZE;
    
    if (!is_valid_page(page)) {
        // Segmentation fault
        terminate_process();
        return;
    }
    
    // Allocate physical frame
    unsigned int frame = allocate_frame();
    if (frame == INVALID_FRAME) {
        // Need to evict a page
        frame = evict_page();
    }
    
    // Load page from disk
    load_page_from_disk(page, frame);
    
    // Update page table
    page_table[page].frame_number = frame;
    page_table[page].present = 1;
}
```

### Page Replacement Algorithms

**FIFO (First-In, First-Out)**
```c
typedef struct {
    int pages[MAX_FRAMES];
    int front, rear;
} fifo_queue;

int fifo_replace(fifo_queue* q) {
    int victim = q->pages[q->front];
    q->front = (q->front + 1) % MAX_FRAMES;
    return victim;
}
```

**LRU (Least Recently Used)**
```c
typedef struct {
    int page_number;
    unsigned long timestamp;
} lru_entry;

int lru_replace(lru_entry* entries, int n) {
    int victim = 0;
    unsigned long oldest = entries[0].timestamp;
    
    for (int i = 1; i < n; i++) {
        if (entries[i].timestamp < oldest) {
            oldest = entries[i].timestamp;
            victim = i;
        }
    }
    return victim;
}
```

**Clock (Second Chance)**
```c
typedef struct {
    int page_number;
    int reference_bit;
} clock_entry;

int clock_replace(clock_entry* entries, int n, int* hand) {
    while (1) {
        if (entries[*hand].reference_bit == 0) {
            int victim = *hand;
            *hand = (*hand + 1) % n;
            return victim;
        }
        entries[*hand].reference_bit = 0;
        *hand = (*hand + 1) % n;
    }
}
```

**Optimal (theoretical)**
- Replace page that won't be used for longest time
- Requires future knowledge
- Used as benchmark

### Page Fault Rate Analysis

Belady's Anomaly: More frames can cause more page faults (FIFO)

```
Reference string: 1,2,3,4,1,2,5,1,2,3,4,5
3 frames: 9 page faults
4 frames: 10 page faults (!)
```

Effective Access Time:
```
EAT = (1 - p) × memory_access_time + p × page_fault_time
where p = page fault rate

Example:
Memory access = 100 ns
Page fault service = 10 ms = 10,000,000 ns
p = 0.001 (1 in 1000)

EAT = 0.999 × 100 + 0.001 × 10,000,000
    = 99.9 + 10,000
    = 10,099.9 ns (100× slower!)
```

## 6.7 Working Set and Thrashing

### Working Set Model

Set of pages actively used by process:

```c
struct working_set {
    int pages[MAX_PAGES];
    int size;
    int window_size;  // Time window Δ
};

void update_working_set(working_set* ws, int page, int time) {
    // Remove old pages outside window
    remove_old_pages(ws, time - ws->window_size);
    
    // Add new page if not present
    if (!in_working_set(ws, page)) {
        add_page(ws, page);
    }
}
```

### Thrashing

When system spends more time paging than executing:

```
Degree of Multiprogramming →
↑
CPU Utilization
|     /-\
|    /   \  ← Thrashing
|   /     \
|  /       \___
| /
+----------------→
```

Prevention:
- Working set strategy
- Page fault frequency control
- Suspend processes when thrashing detected

## 6.8 Memory-Mapped Files

Map files directly into address space:

```c
#include <sys/mman.h>

// Map file into memory
int fd = open("data.bin", O_RDWR);
struct stat sb;
fstat(fd, &sb);

void* mapped = mmap(NULL, sb.st_size, 
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);

// Access file as memory
char* data = (char*)mapped;
data[100] = 'X';  // Writes to file

// Cleanup
munmap(mapped, sb.st_size);
close(fd);
```

Benefits:
- No explicit read/write calls
- Efficient for random access
- Automatic caching
- Shared memory between processes

## 6.9 Kernel Memory Management

### Buddy System

Allocate memory in powers of 2:

```
Initial: 1024 KB free

Request 70 KB:
1024 → 512 + 512
512 → 256 + 256
256 → 128 + 128
Allocate 128 KB (internal fragmentation: 58 KB)

Memory layout:
[128-used][128][256][512]
```

### Slab Allocator

Cache frequently allocated objects:

```c
struct kmem_cache {
    char name[32];           // Cache name
    size_t object_size;      // Size of each object
    void* (*ctor)(void*);    // Constructor
    void* (*dtor)(void*);    // Destructor
    struct slab* slabs_full;
    struct slab* slabs_partial;
    struct slab* slabs_empty;
};

// Allocate object
void* kmem_cache_alloc(struct kmem_cache* cache) {
    struct slab* slab = cache->slabs_partial;
    if (!slab) {
        slab = cache->slabs_empty;
        if (!slab) {
            slab = allocate_new_slab(cache);
        }
    }
    return allocate_from_slab(slab);
}
```

## 6.10 Memory Protection

### Protection Bits

Page table entry protection:

```c
#define PAGE_PRESENT  0x001
#define PAGE_WRITE    0x002
#define PAGE_USER     0x004
#define PAGE_ACCESSED 0x020
#define PAGE_DIRTY    0x040
#define PAGE_EXECUTE  0x100  // NX bit

// Check permissions
bool can_access(pte_t pte, int mode) {
    if (!(pte & PAGE_PRESENT)) return false;
    if (mode & MODE_WRITE && !(pte & PAGE_WRITE)) return false;
    if (mode & MODE_USER && !(pte & PAGE_USER)) return false;
    if (mode & MODE_EXECUTE && (pte & PAGE_EXECUTE)) return false;
    return true;
}
```

### Address Space Layout Randomization (ASLR)

Randomize memory layout to prevent exploits:

```c
// Random stack base
stack_base = STACK_TOP - (random() & STACK_RANDOM_MASK);

// Random heap base
heap_base = HEAP_START + (random() & HEAP_RANDOM_MASK);

// Random mmap base
mmap_base = MMAP_START + (random() & MMAP_RANDOM_MASK);
```

## 6.11 Multi-level Page Tables

Reduce page table size:

### Two-Level Page Table

```
Linear Address (32-bit):
+---------+---------+----------+
| Dir(10) | Page(10)| Offset(12)|
+---------+---------+----------+
     |         |          |
     v         |          |
Page Directory |          |
     |         |          |
     v         v          |
  Page Table Entry        |
     |                    |
     v                    v
Physical Frame + Offset = Physical Address
```

### Four-Level Page Table (x86-64)

```
Virtual Address (48-bit used of 64):
+-----+-----+-----+-----+-----+----------+
|PML4 | PDPT| PD  | PT  |Offset(12 bits)|
|(9)  | (9) | (9) | (9) |                |
+-----+-----+-----+-----+-----+----------+

Each level: 512 entries × 8 bytes = 4KB page
```

Implementation:
```c
// Walk page table hierarchy
uint64_t translate_address(uint64_t vaddr) {
    uint64_t pml4_idx = (vaddr >> 39) & 0x1FF;
    uint64_t pdpt_idx = (vaddr >> 30) & 0x1FF;
    uint64_t pd_idx = (vaddr >> 21) & 0x1FF;
    uint64_t pt_idx = (vaddr >> 12) & 0x1FF;
    uint64_t offset = vaddr & 0xFFF;
    
    pml4_entry = pml4_table[pml4_idx];
    if (!present(pml4_entry)) page_fault();
    
    pdpt_entry = pdpt_table[pdpt_idx];
    if (!present(pdpt_entry)) page_fault();
    
    pd_entry = pd_table[pd_idx];
    if (!present(pd_entry)) page_fault();
    
    pt_entry = pt_table[pt_idx];
    if (!present(pt_entry)) page_fault();
    
    return (pt_entry & FRAME_MASK) | offset;
}
```

## 6.12 Copy-on-Write (COW)

Defer copying until modification:

```c
// Fork implementation with COW
pid_t fork_cow() {
    pid_t child = create_process();
    
    // Share all pages, mark read-only
    for (each page in parent) {
        share_page(parent, child, page);
        mark_readonly(page);
        set_cow_flag(page);
    }
    
    return child;
}

// Page fault handler for COW
void cow_fault_handler(void* addr) {
    page_t* page = get_page(addr);
    
    if (is_cow(page)) {
        // Allocate new page
        page_t* new_page = allocate_page();
        
        // Copy contents
        memcpy(new_page, page, PAGE_SIZE);
        
        // Update page table
        update_page_table(addr, new_page);
        mark_writable(new_page);
        
        // Decrease reference count
        if (--page->ref_count == 1) {
            // Last reference, make writable
            mark_writable(page);
            clear_cow_flag(page);
        }
    }
}
```

## 6.13 Large Pages

Reduce TLB misses with larger pages:

```c
// Huge pages (2MB or 1GB on x86-64)
void* mmap_huge(size_t size) {
    return mmap(NULL, size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                -1, 0);
}

// Transparent Huge Pages (THP)
int enable_thp() {
    return madvise(addr, length, MADV_HUGEPAGE);
}
```

Benefits:
- Fewer TLB entries needed
- Reduced page table overhead
- Better for large, contiguous allocations

Drawbacks:
- Internal fragmentation
- Slower page faults
- Memory waste for small allocations

## 6.14 NUMA (Non-Uniform Memory Access)

Memory locality in multi-socket systems:

```c
// NUMA-aware allocation
#include <numa.h>

void numa_example() {
    // Bind to NUMA node
    numa_set_preferred(0);  // Prefer node 0
    
    // Allocate on specific node
    void* mem = numa_alloc_onnode(size, 1);  // Node 1
    
    // Check memory policy
    int mode;
    unsigned long nodemask;
    get_mempolicy(&mode, &nodemask, MAX_NODES, addr, MPOL_F_ADDR);
}
```

## 6.15 Memory Management in Practice

### Linux Memory Management

```c
// Linux process memory layout
/*
0xFFFFFFFFFFFFFFFF +-----------------+
                   |     Kernel      |
0xFFFF800000000000 +-----------------+
                   |                 |
                   |   User Stack    |
                   |       ↓         |
                   |                 |
                   |       ↑         |
                   |   Memory Map    |
                   |    (mmap)       |
                   |                 |
                   |       ↑         |
                   |      Heap       |
                   +-----------------+
                   |      BSS        |
                   +-----------------+
                   |      Data       |
                   +-----------------+
                   |      Text       |
0x0000000000400000 +-----------------+
                   |    Reserved     |
0x0000000000000000 +-----------------+
*/
```

### Memory Allocation Library (malloc)

Simple implementation:
```c
typedef struct block {
    size_t size;
    struct block* next;
    int free;
} block_t;

void* simple_malloc(size_t size) {
    block_t* block = find_free_block(size);
    
    if (!block) {
        // Request more memory from OS
        block = sbrk(size + sizeof(block_t));
        block->size = size;
        block->free = 0;
        block->next = NULL;
    } else {
        // Split block if too large
        if (block->size > size + sizeof(block_t)) {
            split_block(block, size);
        }
        block->free = 0;
    }
    
    return (void*)(block + 1);
}

void simple_free(void* ptr) {
    if (!ptr) return;
    
    block_t* block = ((block_t*)ptr) - 1;
    block->free = 1;
    
    // Coalesce adjacent free blocks
    coalesce_free_blocks();
}
```

## Exercises

1. Calculate the page table size for:
   - 32-bit address space, 4KB pages
   - 64-bit address space, 4KB pages
   - Why are multi-level page tables necessary?

2. Simulate page replacement algorithms:
   - Given reference string: 7,0,1,2,0,3,0,4,2,3,0,3,2
   - 3 frames available
   - Compare FIFO, LRU, and Optimal

3. Implement a simple memory allocator with:
   - First-fit allocation
   - Coalescing of free blocks
   - Splitting of large blocks

4. Calculate effective access time:
   - TLB hit rate: 95%
   - TLB access: 10 ns
   - Memory access: 100 ns
   - Page fault rate: 0.1%
   - Disk access: 10 ms

5. Design a page table structure for:
   - 40-bit virtual addresses
   - 8KB pages
   - Minimize memory overhead

6. Write a program demonstrating:
   - Memory-mapped file I/O
   - Shared memory between processes
   - Copy-on-write behavior

7. Analyze working set:
   - Given page reference string and window size
   - Calculate working set at each time
   - Determine minimum frames needed

8. Implement buddy system allocator:
   - Power-of-2 allocations
   - Coalescing of buddies
   - Internal fragmentation analysis

9. Compare segmentation vs paging:
   - Advantages and disadvantages
   - Use cases for each
   - Modern system choices

10. Measure and optimize TLB performance:
    - Write code with good/bad locality
    - Measure TLB misses
    - Optimize for huge pages

## Summary

This chapter explored memory management in depth:

- Memory allocation schemes from fixed partitions to dynamic allocation
- Paging provides flexible memory management with fixed-size units
- Virtual memory enables programs larger than physical memory
- Page replacement algorithms manage limited physical memory
- Multi-level page tables reduce memory overhead
- TLBs accelerate address translation
- Copy-on-write optimizes process creation
- Protection mechanisms ensure process isolation
- Modern systems use sophisticated techniques like NUMA and huge pages

Memory management is crucial for system performance and security. The techniques discussed form the foundation of modern operating systems, enabling efficient and secure execution of multiple processes. The next chapter will explore process management and concurrency in detail, building on the memory management concepts to understand how operating systems coordinate multiple executing programs.