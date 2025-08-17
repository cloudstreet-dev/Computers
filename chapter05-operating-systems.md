# Chapter 5: Operating System Concepts

## Introduction

An operating system (OS) is the fundamental software layer that manages computer hardware and provides services to applications. It acts as an intermediary between users and hardware, transforming raw computing resources into a convenient, efficient, and secure computing environment. This chapter explores the core concepts of operating systems, their evolution, and the essential services they provide.

### Why Operating Systems Matter

Without an OS, every program would need to:
- Directly control hardware (complex and error-prone)
- Manage its own memory (inefficient and insecure)
- Handle all I/O operations (reinventing the wheel)
- Coordinate with other programs (chaos)

The OS solves these problems by providing abstractions, resource management, and standardized interfaces.

## 5.1 What is an Operating System?

An operating system serves multiple roles:

### Resource Manager
- Allocates CPU time among processes
- Manages memory allocation
- Controls I/O device access
- Provides file system organization

### Extended Machine
- Hides hardware complexity
- Provides abstractions (files, processes, sockets)
- Offers standard interfaces (system calls)

### Control Program
- Prevents errors and improper use
- Ensures security and protection
- Manages user authentication and authorization

### Platform
- Provides environment for application execution
- Offers libraries and services
- Maintains backward compatibility

## 5.2 Evolution of Operating Systems

### Batch Systems (1950s)
- Jobs submitted on punch cards
- Sequential execution
- No user interaction during execution
- Simple monitor programs
- **Problem solved**: Manual operation overhead
- **New problem**: CPU idle during I/O

### Multiprogramming (1960s)
- Multiple jobs in memory simultaneously
- CPU switches between jobs when one waits for I/O
- Better resource utilization (CPU rarely idle)
- Job scheduling algorithms emerge
- **Problem solved**: CPU idle time
- **New problem**: How to share resources fairly?

### Time-Sharing (1960s-1970s)
- Interactive computing (users get immediate feedback)
- Multiple users simultaneously via terminals
- Quick response time through rapid context switching
- Virtual terminals give illusion of dedicated machine
- **Examples**: CTSS, Multics, early Unix
- **Innovation**: Made computing accessible to non-specialists

### Personal Computers (1980s)
- Single-user systems
- Graphical interfaces
- MS-DOS, early Windows, Mac OS

### Modern Systems (1990s-present)
- Multitasking and multiuser
- Network and distributed systems
- Mobile and embedded OS
- Cloud and container platforms

## 5.3 Operating System Architecture

### Monolithic Kernel

All OS services run in kernel space:

```
+------------------------+
|    User Applications   |
+------------------------+
|      System Calls      |
+========================+ Kernel/User boundary
|                        |
|    Monolithic Kernel   |
|  - Process Management  |
|  - Memory Management   |
|  - File Systems        |
|  - Device Drivers      |
|  - Network Stack       |
|                        |
+------------------------+
|        Hardware        |
+------------------------+
```

Examples: Linux, traditional Unix

Advantages:
- Fast (no context switches between services)
- Simple design
- Direct hardware access

Disadvantages:
- Large, complex codebase (millions of lines)
- Single failure can crash entire system (no isolation)
- Difficult to maintain and extend
- Security vulnerabilities affect entire kernel

### Microkernel

Minimal kernel with services in user space:

```
+------------------------+
|  User Applications     |
+------------------------+
| File    | Network      |
| Server  | Server       |
+------------------------+
|      System Calls      |
+========================+
|     Microkernel        |
|  - IPC                 |
|  - Basic Memory Mgmt   |
|  - Basic Scheduling    |
+------------------------+
|        Hardware        |
+------------------------+
```

Examples: MINIX, QNX, L4

Advantages:
- Modular and extensible
- Better fault isolation
- Easier to port

Disadvantages:
- Performance overhead (IPC)
- Complex communication

### Hybrid Kernel

Combination of monolithic and microkernel:

Examples: Windows NT, macOS

- Core services in kernel
- Some services in user space
- Loadable kernel modules

### Layered Architecture

OS organized as hierarchy of layers:

```
Layer N:   User Interface
Layer N-1: User Programs
Layer N-2: I/O Management
Layer N-3: Message Interpreter
Layer N-4: Memory Management
Layer N-5: CPU Scheduling
Layer 0:   Hardware
```

Each layer only uses services of lower layers.

## 5.4 System Calls

System calls provide the interface between applications and the OS kernel:

### Categories of System Calls

**Process Control**
```c
fork()      // Create new process
exec()      // Execute program
exit()      // Terminate process
wait()      // Wait for child process
getpid()    // Get process ID
nice()      // Change priority
```

**File Management**
```c
open()      // Open file
read()      // Read from file
write()     // Write to file
close()     // Close file
seek()      // Change file position
stat()      // Get file information
```

**Device Management**
```c
ioctl()     // Device control
read()      // Read from device
write()     // Write to device
```

**Information Maintenance**
```c
time()      // Get system time
getpid()    // Get process ID
getuid()    // Get user ID
uname()     // Get system information
```

**Communication**
```c
socket()    // Create network endpoint
send()      // Send message
receive()   // Receive message
pipe()      // Create pipe
shmget()    // Get shared memory
```

### System Call Implementation

1. **User program** calls library function (e.g., fopen())
2. **Library** prepares parameters and executes trap instruction
3. **Trap** switches CPU to kernel mode (privileged execution)
4. **Kernel** validates parameters and executes system call handler
5. **Return** to user mode with results or error code

**Performance note**: System calls are expensive (~1000x slower than function calls) due to:
- Mode switching overhead
- Parameter validation
- Cache effects
- TLB flushes

Example: Reading a file
```c
// User code
char buffer[100];
int fd = open("file.txt", O_RDONLY);
if (fd < 0) {
    perror("open failed");
    exit(1);
}
int bytes = read(fd, buffer, 100);
close(fd);

// What happens under the hood:
// 1. open() library function called
// 2. Parameters placed in registers (filename ptr, flags)
// 3. System call number placed in register (e.g., rax on x86-64)
// 4. Trap instruction executed (int 0x80 or syscall)
// 5. CPU switches to kernel mode
// 6. Kernel validates parameters (file exists? permissions?)
// 7. Kernel opens file, creates file descriptor
// 8. Return to user mode with fd or error code
// 9. read() follows similar process
```

## 5.5 Process Management

A process is a program in execution, the unit of work in an OS.

### Process States

```
        admit
  New ---------> Ready
                  ↑ ↓  dispatch
      interrupt   | |
    (time slice)  | ↓
                Running
                  ↓ |
            exit  | | I/O or event wait
                  ↓ ↓
  Terminated    Waiting
                  ↑ |
                  +-+ I/O or event completion
```

States explained:
- **New**: Process being created
- **Ready**: Waiting for CPU
- **Running**: Currently executing
- **Waiting**: Waiting for I/O or event
- **Terminated**: Finished execution

### Process Control Block (PCB)

Data structure containing process information:

```c
struct PCB {
    int pid;                  // Process ID
    enum state process_state; // Current state
    int program_counter;      // Next instruction
    int cpu_registers[16];    // Register values
    int priority;             // Scheduling priority
    struct memory_info mem;   // Memory management
    struct file files[MAX];   // Open files
    struct PCB *parent;       // Parent process
    struct PCB *children;     // Child processes
    int cpu_time_used;        // Accounting
    // ... more fields
};
```

### Process Creation

**Fork-Exec Model (Unix/Linux)**
```c
pid_t pid = fork();       // Create copy of current process
if (pid == 0) {
    // Child process
    exec("/bin/ls", ...); // Replace with new program
} else {
    // Parent process
    wait(&status);        // Wait for child
}
```

**CreateProcess (Windows)**
```c
CreateProcess(
    "C:\\Program.exe",    // Program path
    NULL,                 // Command line
    NULL,                 // Process attributes
    NULL,                 // Thread attributes
    FALSE,                // Inherit handles
    0,                    // Creation flags
    NULL,                 // Environment
    NULL,                 // Current directory
    &si,                  // Startup info
    &pi                   // Process info
);
```

### Inter-Process Communication (IPC)

**Shared Memory**
```c
// Process 1: Create shared memory
int shmid = shmget(key, size, IPC_CREAT | 0666);
char *shared = shmat(shmid, NULL, 0);
strcpy(shared, "Hello");

// Process 2: Access shared memory
int shmid = shmget(key, size, 0666);
char *shared = shmat(shmid, NULL, 0);
printf("%s\n", shared);  // Prints "Hello"
```

**Message Passing**
```c
// Pipes
int pipefd[2];
pipe(pipefd);
if (fork() == 0) {
    close(pipefd[0]);     // Close read end
    write(pipefd[1], "Hello", 5);
} else {
    close(pipefd[1]);     // Close write end
    read(pipefd[0], buffer, 5);
}
```

**Signals**
```c
// Send signal
kill(pid, SIGTERM);

// Handle signal
signal(SIGTERM, handler_function);
```

## 5.6 CPU Scheduling

The scheduler determines which process runs when.

### Scheduling Criteria

- **CPU Utilization**: Keep CPU busy
- **Throughput**: Processes completed per time
- **Turnaround Time**: Total time from submission to completion
- **Waiting Time**: Time spent in ready queue
- **Response Time**: Time to first response

### Scheduling Algorithms

**First-Come, First-Served (FCFS)**
```
Process  Arrival  Burst  Wait  Turnaround
P1       0        24     0     24
P2       1        3      23    26
P3       2        3      25    28
Average waiting time: 16
```

**Shortest Job First (SJF)**
```
Process  Arrival  Burst  Wait  Turnaround
P2       1        3      0     3
P3       2        3      2     5
P1       0        24     5     29
Average waiting time: 2.33
```

**Round Robin (RR)**
Time quantum = 4
```
Time:    0-4   4-7   7-10  10-14 14-18 18-22 22-26 26-30
Process: P1    P2    P3    P1    P1    P1    P1    P1
```

**Priority Scheduling**
```c
struct process {
    int pid;
    int priority;  // Lower value = higher priority
    int burst_time;
};
// Schedule process with lowest priority value
```

**Multilevel Queue**
Different queues for different process types:
- System processes (highest priority)
- Interactive processes
- Batch processes (lowest priority)

### Context Switching

Saving and restoring process state:

```c
context_switch(old_process, new_process) {
    // Save old process state
    save_registers(old_process->pcb);
    save_program_counter(old_process->pcb);
    
    // Load new process state
    load_registers(new_process->pcb);
    load_program_counter(new_process->pcb);
    
    // Update page tables
    switch_address_space(new_process);
}
```

Overhead includes:
- Saving/restoring registers
- Switching memory maps
- Flushing caches
- Updating scheduling structures

## 5.7 Threads

Threads are lightweight processes sharing the same address space.

### Thread vs Process

**Process has:**
- Independent address space
- Own resources (files, signals)
- Heavy creation/switching

**Thread has:**
- Shared address space
- Shared resources
- Lightweight creation/switching

### Thread Implementation

**User-Level Threads**
- Managed by user library
- Fast context switch
- No kernel involvement
- Can't utilize multiple CPUs

**Kernel-Level Threads**
- Managed by kernel
- Can run on multiple CPUs
- Slower context switch
- True parallelism

**Hybrid (M:N Model)**
- M user threads mapped to N kernel threads
- Balance of performance and functionality

### POSIX Threads (pthreads)

```c
#include <pthread.h>

void* thread_function(void* arg) {
    int* value = (int*)arg;
    printf("Thread received: %d\n", *value);
    return NULL;
}

int main() {
    pthread_t thread;
    int argument = 42;
    
    // Create thread
    pthread_create(&thread, NULL, thread_function, &argument);
    
    // Wait for thread
    pthread_join(thread, NULL);
    
    return 0;
}
```

## 5.8 Deadlocks

Deadlock occurs when processes are waiting for resources held by each other.

### Deadlock Conditions (Coffman Conditions)

All four must hold simultaneously:

1. **Mutual Exclusion**: Resources can't be shared
2. **Hold and Wait**: Process holding resources can request more
3. **No Preemption**: Resources can't be forcibly taken
4. **Circular Wait**: Circular chain of processes waiting

### Resource Allocation Graph

```
    P1 -----> R1
    ↑         ↓
    |         |
    R2 <----- P2

P1 holds R2, wants R1
P2 holds R1, wants R2
=> Deadlock!
```

### Deadlock Handling

**Prevention**: Ensure at least one condition can't hold
- Mutual exclusion: Make resources sharable
- Hold and wait: Request all resources at once
- No preemption: Allow preemption
- Circular wait: Order resource requests

**Avoidance**: Banker's Algorithm
```c
// Check if state is safe before granting request
bool is_safe(available, max, allocation) {
    work = available;
    finish = [false, false, ...];
    
    while (exists unfinished process with need <= work) {
        work += allocation[process];
        finish[process] = true;
    }
    
    return all(finish);  // Safe if all can finish
}
```

**Detection**: Periodically check for cycles

**Recovery**: 
- Process termination
- Resource preemption
- Rollback to checkpoint

## 5.9 Synchronization

Coordinating access to shared resources.

### Critical Section Problem

```c
// Multiple processes executing:
while (true) {
    // Entry section
    enter_critical_section();
    
    // Critical section
    access_shared_resource();
    
    // Exit section
    leave_critical_section();
    
    // Remainder section
    do_other_work();
}
```

Requirements:
1. **Mutual Exclusion**: Only one process in critical section
2. **Progress**: No unnecessary delays
3. **Bounded Waiting**: Finite wait time

### Synchronization Primitives

**Mutex (Mutual Exclusion)**
```c
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_lock(&mutex);
// Critical section
shared_variable++;
pthread_mutex_unlock(&mutex);
```

**Semaphore**
```c
sem_t semaphore;
sem_init(&semaphore, 0, 1);  // Binary semaphore

sem_wait(&semaphore);         // P operation (decrement)
// Critical section
sem_post(&semaphore);         // V operation (increment)
```

**Condition Variable**
```c
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Wait for condition
pthread_mutex_lock(&mutex);
while (!condition) {
    pthread_cond_wait(&cond, &mutex);
}
// Condition met
pthread_mutex_unlock(&mutex);

// Signal condition
pthread_mutex_lock(&mutex);
condition = true;
pthread_cond_signal(&cond);
pthread_mutex_unlock(&mutex);
```

### Classic Synchronization Problems

**Producer-Consumer**
```c
// Bounded buffer with semaphores
sem_t empty, full, mutex;
sem_init(&empty, 0, BUFFER_SIZE);  // Empty slots
sem_init(&full, 0, 0);             // Full slots
sem_init(&mutex, 0, 1);            // Mutual exclusion

// Producer
sem_wait(&empty);
sem_wait(&mutex);
add_to_buffer(item);
sem_post(&mutex);
sem_post(&full);

// Consumer
sem_wait(&full);
sem_wait(&mutex);
item = remove_from_buffer();
sem_post(&mutex);
sem_post(&empty);
```

**Readers-Writers**
```c
int readers = 0;
sem_t mutex, write_lock;

// Reader
sem_wait(&mutex);
readers++;
if (readers == 1) sem_wait(&write_lock);
sem_post(&mutex);

read_data();

sem_wait(&mutex);
readers--;
if (readers == 0) sem_post(&write_lock);
sem_post(&mutex);

// Writer
sem_wait(&write_lock);
write_data();
sem_post(&write_lock);
```

**Dining Philosophers**
```c
// Deadlock-free solution using resource ordering
philosopher(int id) {
    while (true) {
        think();
        
        // Pick up lower-numbered fork first
        int first = min(id, (id + 1) % 5);
        int second = max(id, (id + 1) % 5);
        
        pick_up_fork(first);
        pick_up_fork(second);
        
        eat();
        
        put_down_fork(first);
        put_down_fork(second);
    }
}
```

## 5.10 Device Management

### I/O Hardware

**Device Types**
- Block devices: Disk drives (random access)
- Character devices: Keyboards, printers (sequential)
- Network devices: Network interfaces

**Device Controllers**
- Hardware interface between device and bus
- Contains registers for control and data
- May have local buffer

### Device Drivers

Software modules that manage devices:

```c
struct device_driver {
    int (*open)(struct device *dev);
    int (*close)(struct device *dev);
    int (*read)(struct device *dev, char *buf, size_t len);
    int (*write)(struct device *dev, char *buf, size_t len);
    int (*ioctl)(struct device *dev, int cmd, void *arg);
};
```

### I/O Techniques

**Programmed I/O**
```c
while (!device_ready()) {
    // Busy wait
}
transfer_data();
```

**Interrupt-Driven I/O**
```c
// Interrupt handler
void io_interrupt_handler() {
    if (operation_complete()) {
        wake_up_waiting_process();
    } else {
        continue_operation();
    }
}
```

**Direct Memory Access (DMA)**
```c
setup_dma(source, destination, count);
start_dma();
// CPU free to do other work
// Interrupt when complete
```

### I/O Scheduling

**Disk Scheduling Algorithms**

FCFS (First-Come, First-Served):
```
Request queue: 98, 183, 37, 122, 14, 124, 65, 67
Head at: 53
Order: 53 → 98 → 183 → 37 → 122 → 14 → 124 → 65 → 67
Total movement: 640 cylinders
```

SCAN (Elevator):
```
Same queue, head at 53 moving toward 0
Order: 53 → 37 → 14 → 0 → 65 → 67 → 98 → 122 → 124 → 183
Total movement: 236 cylinders
```

## 5.11 Protection and Security

### Protection Domains

Define what resources a process can access:

```
Domain D1: {<file1, read>, <file2, read-write>, <printer, write>}
Domain D2: {<file2, read>, <file3, read-write>}
```

### Access Control

**Access Control Lists (ACL)**
```
File: report.txt
  User Alice: read, write
  User Bob: read
  Group Staff: read
  Others: none
```

**Capability Lists**
```
Process P1 capabilities:
  File1: read
  File2: read, write
  Network: send, receive
```

### User Authentication

Methods:
- Something you know (password)
- Something you have (token, smartcard)
- Something you are (biometrics)

Password storage:
```c
// Never store plaintext!
// Use salted hashes
salt = generate_random_salt();
hash = hash_function(password + salt);
store(username, salt, hash);
```

### Security Threats

**Malware Types**
- Virus: Attaches to programs
- Worm: Self-replicating
- Trojan: Hidden in legitimate software
- Rootkit: Hides presence

**Attack Vectors**
- Buffer overflow
- SQL injection
- Privilege escalation
- Social engineering

## 5.12 Modern OS Features

### Virtualization

**Virtual Machines**
- Type 1 Hypervisor: Runs on bare metal
- Type 2 Hypervisor: Runs on host OS

**Containers**
- OS-level virtualization
- Shared kernel
- Lightweight isolation

### Real-Time Systems

**Hard Real-Time**
- Deadlines must be met
- Predictable scheduling
- Examples: Flight control, medical devices

**Soft Real-Time**
- Deadlines important but not critical
- Best-effort scheduling
- Examples: Video streaming, gaming

### Distributed Operating Systems

Features:
- Process migration
- Distributed file systems
- Network transparency
- Fault tolerance

## Exercises

1. Implement a simple shell that can:
   - Execute commands
   - Handle pipes
   - Support background processes

2. Write a program demonstrating:
   - Process creation with fork()
   - IPC using pipes
   - Signal handling

3. Implement scheduling algorithms:
   - FCFS
   - SJF
   - Round Robin
   Compare average waiting times

4. Create a solution for producer-consumer using:
   - Semaphores
   - Monitors
   - Message passing

5. Detect deadlock in a system with:
   - 3 resource types
   - 5 processes
   - Given allocation and request matrices

6. Design a simple device driver interface for a virtual device.

7. Implement a basic file system with:
   - File creation/deletion
   - Directory support
   - Simple allocation strategy

8. Write a program demonstrating race conditions and fix with:
   - Mutex
   - Semaphore
   - Atomic operations

9. Calculate disk seek time for different scheduling algorithms given a request queue.

10. Design an access control system with:
    - Users and groups
    - File permissions
    - Access validation

## Summary

This chapter covered fundamental operating system concepts:

- Operating systems manage hardware resources and provide abstractions
- Processes are the basic unit of execution with defined states and transitions
- CPU scheduling algorithms balance various performance criteria
- Threads provide lightweight concurrency within processes
- Synchronization primitives coordinate access to shared resources
- Deadlocks can occur with concurrent resource allocation
- Device management handles I/O operations efficiently
- Protection and security mechanisms ensure system integrity

Operating systems form the foundation of modern computing, enabling multiple programs to share resources efficiently and securely. The next chapter will dive deeper into memory management, exploring how operating systems provide the illusion of large, private address spaces to each process.