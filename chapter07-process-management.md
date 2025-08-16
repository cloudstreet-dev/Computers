# Chapter 7: Process Management and Concurrency

## Introduction

Modern computers execute multiple programs simultaneously, creating the illusion that each program has exclusive access to system resources. This chapter delves deep into process management and concurrency, exploring how operating systems create, schedule, and coordinate processes and threads. We'll examine synchronization mechanisms, concurrent programming patterns, and the challenges of writing correct concurrent programs.

## 7.1 Processes and Threads Revisited

### Process Anatomy

A process consists of:

```c
struct process {
    // Identification
    pid_t pid;                    // Process ID
    pid_t ppid;                   // Parent process ID
    uid_t uid;                    // User ID
    gid_t gid;                    // Group ID
    
    // CPU State
    struct cpu_registers regs;    // Register values
    unsigned long pc;             // Program counter
    unsigned long sp;             // Stack pointer
    
    // Memory
    struct mm_struct *mm;         // Memory descriptor
    unsigned long brk;            // Heap boundary
    
    // Files
    struct files_struct *files;   // Open file descriptors
    mode_t umask;                 // File creation mask
    
    // Signals
    struct signal_struct *signal; // Signal handlers
    sigset_t blocked;             // Blocked signals
    
    // Scheduling
    int priority;                 // Static priority
    int nice;                     // Nice value
    unsigned long time_slice;     // Remaining time
    enum state state;             // Process state
    
    // Statistics
    clock_t utime;                // User CPU time
    clock_t stime;                // System CPU time
    time_t start_time;            // Process start time
};
```

### Thread Implementation Models

**1:1 Model (Kernel-Level Threads)**
```
User Thread 1 ←→ Kernel Thread 1
User Thread 2 ←→ Kernel Thread 2
User Thread 3 ←→ Kernel Thread 3
```

**N:1 Model (User-Level Threads)**
```
User Thread 1 ─┐
User Thread 2 ─┼→ Kernel Thread
User Thread 3 ─┘
```

**M:N Model (Hybrid)**
```
User Thread 1 ─┐    ┌→ Kernel Thread 1
User Thread 2 ─┼────┤
User Thread 3 ─┘    └→ Kernel Thread 2
```

### Thread-Local Storage (TLS)

Each thread gets private storage:

```c
// Thread-local variable
__thread int tls_counter = 0;

void* thread_function(void* arg) {
    tls_counter++;  // Each thread has its own copy
    printf("Thread %ld: counter = %d\n", 
           pthread_self(), tls_counter);
    return NULL;
}
```

## 7.2 Process Creation and Termination

### Fork-Exec Pattern (Unix/Linux)

```c
#include <unistd.h>
#include <sys/wait.h>

void create_process() {
    pid_t pid = fork();
    
    if (pid < 0) {
        // Fork failed
        perror("fork");
        exit(1);
    } else if (pid == 0) {
        // Child process
        printf("Child PID: %d\n", getpid());
        
        // Replace process image
        char* args[] = {"/bin/ls", "-l", NULL};
        execvp(args[0], args);
        
        // Only reached if exec fails
        perror("exec");
        exit(1);
    } else {
        // Parent process
        printf("Parent PID: %d, Child PID: %d\n", 
               getpid(), pid);
        
        // Wait for child
        int status;
        waitpid(pid, &status, 0);
        
        if (WIFEXITED(status)) {
            printf("Child exited with status %d\n", 
                   WEXITSTATUS(status));
        }
    }
}
```

### Process Termination

```c
// Normal termination
exit(0);           // Clean termination
_exit(0);          // Immediate termination (no cleanup)
return 0;          // From main()

// Abnormal termination
abort();           // Generate SIGABRT
raise(SIGKILL);    // Send signal to self
// Unhandled signal
// Unhandled exception
```

### Zombie and Orphan Processes

```c
// Creating a zombie
if (fork() == 0) {
    // Child exits immediately
    exit(0);
}
// Parent doesn't wait() - child becomes zombie
sleep(60);

// Creating an orphan
if (fork() == 0) {
    // Child continues
    sleep(60);
    printf("Orphan adopted by init\n");
}
// Parent exits - child becomes orphan
exit(0);
```

## 7.3 CPU Scheduling Deep Dive

### Scheduling Classes and Policies

**Linux Scheduling Classes**
```c
#define SCHED_NORMAL    0  // CFS, normal processes
#define SCHED_FIFO      1  // Real-time, FIFO
#define SCHED_RR        2  // Real-time, round-robin
#define SCHED_BATCH     3  // Batch processing
#define SCHED_IDLE      5  // Very low priority
#define SCHED_DEADLINE  6  // Deadline scheduling

// Set scheduling policy
struct sched_param param;
param.sched_priority = 50;
sched_setscheduler(pid, SCHED_FIFO, &param);
```

### Completely Fair Scheduler (CFS)

Linux's default scheduler:

```c
struct cfs_rq {
    struct rb_root_cached tasks_timeline;  // Red-black tree
    u64 min_vruntime;                      // Minimum virtual runtime
    unsigned int nr_running;               // Number of tasks
};

struct sched_entity {
    u64 vruntime;           // Virtual runtime
    u64 sum_exec_runtime;   // Total execution time
    u64 prev_sum_exec_runtime;
    struct rb_node run_node;  // Red-black tree node
};

// Virtual runtime calculation
vruntime += delta_exec * (NICE_0_LOAD / load_weight);

// Pick next task (leftmost in red-black tree)
struct task_struct* pick_next_task_cfs() {
    struct rb_node* leftmost = rb_first_cached(&cfs_rq->tasks_timeline);
    if (!leftmost) return NULL;
    
    return rb_entry(leftmost, struct sched_entity, run_node);
}
```

### Real-Time Scheduling

**Rate Monotonic Scheduling**
```c
// Shorter period = higher priority
struct task {
    int period;      // Task period
    int execution;   // Execution time
    int deadline;    // Relative deadline
};

// Schedulability test
bool is_schedulable_rms(struct task* tasks, int n) {
    double utilization = 0;
    for (int i = 0; i < n; i++) {
        utilization += (double)tasks[i].execution / tasks[i].period;
    }
    double bound = n * (pow(2, 1.0/n) - 1);
    return utilization <= bound;
}
```

**Earliest Deadline First (EDF)**
```c
// Dynamic priority based on absolute deadline
struct edf_task {
    time_t absolute_deadline;
    int remaining_execution;
};

struct edf_task* pick_next_edf(struct edf_task* tasks, int n) {
    struct edf_task* earliest = &tasks[0];
    for (int i = 1; i < n; i++) {
        if (tasks[i].absolute_deadline < earliest->absolute_deadline) {
            earliest = &tasks[i];
        }
    }
    return earliest;
}
```

### Multiprocessor Scheduling

**CPU Affinity**
```c
#include <sched.h>

// Set CPU affinity
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);  // Run on CPU 0
CPU_SET(2, &cpuset);  // Also allow CPU 2

sched_setaffinity(0, sizeof(cpuset), &cpuset);
```

**Load Balancing**
```c
// Simplified load balancing
void load_balance(struct cpu* cpus, int num_cpus) {
    for (int i = 0; i < num_cpus; i++) {
        if (cpus[i].load > IMBALANCE_THRESHOLD) {
            // Find least loaded CPU
            int target = find_least_loaded(cpus, num_cpus);
            
            // Migrate tasks
            migrate_tasks(&cpus[i], &cpus[target]);
        }
    }
}
```

## 7.4 Synchronization Mechanisms

### Atomic Operations

```c
#include <stdatomic.h>

// Atomic types and operations
atomic_int counter = 0;
atomic_fetch_add(&counter, 1);  // Atomic increment

// Compare and swap
int expected = 0;
int desired = 1;
atomic_compare_exchange_strong(&counter, &expected, desired);

// Memory ordering
atomic_store_explicit(&counter, 42, memory_order_release);
int value = atomic_load_explicit(&counter, memory_order_acquire);
```

### Spinlocks

```c
typedef struct {
    atomic_flag flag;
} spinlock_t;

void spin_lock(spinlock_t* lock) {
    while (atomic_flag_test_and_set(&lock->flag)) {
        // Spin
        #ifdef __x86_64__
        __builtin_ia32_pause();  // CPU pause instruction
        #endif
    }
}

void spin_unlock(spinlock_t* lock) {
    atomic_flag_clear(&lock->flag);
}

// Ticket spinlock for fairness
typedef struct {
    atomic_uint next_ticket;
    atomic_uint now_serving;
} ticket_lock_t;

void ticket_lock(ticket_lock_t* lock) {
    unsigned int my_ticket = atomic_fetch_add(&lock->next_ticket, 1);
    while (atomic_load(&lock->now_serving) != my_ticket) {
        // Spin
    }
}

void ticket_unlock(ticket_lock_t* lock) {
    atomic_fetch_add(&lock->now_serving, 1);
}
```

### Read-Write Locks

```c
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t read_cond;
    pthread_cond_t write_cond;
    int readers;
    int writers;
    int write_waiters;
} rwlock_t;

void read_lock(rwlock_t* lock) {
    pthread_mutex_lock(&lock->mutex);
    while (lock->writers > 0 || lock->write_waiters > 0) {
        pthread_cond_wait(&lock->read_cond, &lock->mutex);
    }
    lock->readers++;
    pthread_mutex_unlock(&lock->mutex);
}

void read_unlock(rwlock_t* lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->readers--;
    if (lock->readers == 0) {
        pthread_cond_signal(&lock->write_cond);
    }
    pthread_mutex_unlock(&lock->mutex);
}

void write_lock(rwlock_t* lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->write_waiters++;
    while (lock->readers > 0 || lock->writers > 0) {
        pthread_cond_wait(&lock->write_cond, &lock->mutex);
    }
    lock->write_waiters--;
    lock->writers++;
    pthread_mutex_unlock(&lock->mutex);
}

void write_unlock(rwlock_t* lock) {
    pthread_mutex_lock(&lock->mutex);
    lock->writers--;
    pthread_cond_broadcast(&lock->read_cond);
    pthread_cond_signal(&lock->write_cond);
    pthread_mutex_unlock(&lock->mutex);
}
```

### Barriers

```c
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int waiting;
    int generation;
} barrier_t;

void barrier_wait(barrier_t* barrier) {
    pthread_mutex_lock(&barrier->mutex);
    
    int my_generation = barrier->generation;
    barrier->waiting++;
    
    if (barrier->waiting == barrier->count) {
        // Last thread to arrive
        barrier->generation++;
        barrier->waiting = 0;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        // Wait for others
        while (my_generation == barrier->generation) {
            pthread_cond_wait(&barrier->cond, &barrier->mutex);
        }
    }
    
    pthread_mutex_unlock(&barrier->mutex);
}
```

## 7.5 Lock-Free Programming

### Lock-Free Stack

```c
typedef struct node {
    void* data;
    struct node* next;
} node_t;

typedef struct {
    atomic_uintptr_t head;
    atomic_uint aba_counter;
} lock_free_stack_t;

void push(lock_free_stack_t* stack, void* data) {
    node_t* new_node = malloc(sizeof(node_t));
    new_node->data = data;
    
    uintptr_t head;
    do {
        head = atomic_load(&stack->head);
        new_node->next = (node_t*)head;
    } while (!atomic_compare_exchange_weak(&stack->head, 
                                          &head, 
                                          (uintptr_t)new_node));
}

void* pop(lock_free_stack_t* stack) {
    node_t* head;
    node_t* next;
    
    do {
        head = (node_t*)atomic_load(&stack->head);
        if (head == NULL) return NULL;
        
        next = head->next;
    } while (!atomic_compare_exchange_weak(&stack->head,
                                          (uintptr_t*)&head,
                                          (uintptr_t)next));
    
    void* data = head->data;
    free(head);
    return data;
}
```

### Lock-Free Queue (Michael & Scott)

```c
typedef struct {
    atomic_uintptr_t head;
    atomic_uintptr_t tail;
} lock_free_queue_t;

void enqueue(lock_free_queue_t* queue, void* data) {
    node_t* new_node = malloc(sizeof(node_t));
    new_node->data = data;
    new_node->next = NULL;
    
    node_t* tail;
    node_t* next;
    
    while (1) {
        tail = (node_t*)atomic_load(&queue->tail);
        next = tail->next;
        
        if (tail == (node_t*)atomic_load(&queue->tail)) {
            if (next == NULL) {
                if (atomic_compare_exchange_weak((atomic_uintptr_t*)&tail->next,
                                                (uintptr_t*)&next,
                                                (uintptr_t)new_node)) {
                    break;
                }
            } else {
                atomic_compare_exchange_weak(&queue->tail,
                                            (uintptr_t*)&tail,
                                            (uintptr_t)next);
            }
        }
    }
    
    atomic_compare_exchange_weak(&queue->tail,
                                (uintptr_t*)&tail,
                                (uintptr_t)new_node);
}
```

## 7.6 Concurrent Programming Patterns

### Producer-Consumer with Circular Buffer

```c
typedef struct {
    void* buffer[BUFFER_SIZE];
    atomic_uint head;
    atomic_uint tail;
    sem_t items;
    sem_t spaces;
    pthread_mutex_t mutex;
} circular_buffer_t;

void produce(circular_buffer_t* buf, void* item) {
    sem_wait(&buf->spaces);
    pthread_mutex_lock(&buf->mutex);
    
    unsigned int head = buf->head;
    buf->buffer[head] = item;
    buf->head = (head + 1) % BUFFER_SIZE;
    
    pthread_mutex_unlock(&buf->mutex);
    sem_post(&buf->items);
}

void* consume(circular_buffer_t* buf) {
    sem_wait(&buf->items);
    pthread_mutex_lock(&buf->mutex);
    
    unsigned int tail = buf->tail;
    void* item = buf->buffer[tail];
    buf->tail = (tail + 1) % BUFFER_SIZE;
    
    pthread_mutex_unlock(&buf->mutex);
    sem_post(&buf->spaces);
    
    return item;
}
```

### Thread Pool

```c
typedef struct {
    pthread_t* threads;
    int num_threads;
    
    void (*task_queue)[MAX_TASKS];
    int queue_head;
    int queue_tail;
    
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_not_full;
    
    atomic_bool shutdown;
} thread_pool_t;

void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    
    while (!atomic_load(&pool->shutdown)) {
        pthread_mutex_lock(&pool->queue_mutex);
        
        while (pool->queue_head == pool->queue_tail && 
               !atomic_load(&pool->shutdown)) {
            pthread_cond_wait(&pool->queue_not_empty, 
                            &pool->queue_mutex);
        }
        
        if (atomic_load(&pool->shutdown)) {
            pthread_mutex_unlock(&pool->queue_mutex);
            break;
        }
        
        void (*task)() = pool->task_queue[pool->queue_tail];
        pool->queue_tail = (pool->queue_tail + 1) % MAX_TASKS;
        
        pthread_cond_signal(&pool->queue_not_full);
        pthread_mutex_unlock(&pool->queue_mutex);
        
        task();  // Execute task
    }
    
    return NULL;
}

void submit_task(thread_pool_t* pool, void (*task)()) {
    pthread_mutex_lock(&pool->queue_mutex);
    
    while ((pool->queue_head + 1) % MAX_TASKS == pool->queue_tail) {
        pthread_cond_wait(&pool->queue_not_full, &pool->queue_mutex);
    }
    
    pool->task_queue[pool->queue_head] = task;
    pool->queue_head = (pool->queue_head + 1) % MAX_TASKS;
    
    pthread_cond_signal(&pool->queue_not_empty);
    pthread_mutex_unlock(&pool->queue_mutex);
}
```

### Fork-Join Parallelism

```c
typedef struct {
    void (*task)(void*, int, int);
    void* data;
    int start;
    int end;
    int threshold;
} fork_join_task_t;

void fork_join_execute(fork_join_task_t* task) {
    if (task->end - task->start <= task->threshold) {
        // Execute sequentially
        task->task(task->data, task->start, task->end);
    } else {
        // Fork
        int mid = (task->start + task->end) / 2;
        
        fork_join_task_t left = {
            .task = task->task,
            .data = task->data,
            .start = task->start,
            .end = mid,
            .threshold = task->threshold
        };
        
        fork_join_task_t right = {
            .task = task->task,
            .data = task->data,
            .start = mid,
            .end = task->end,
            .threshold = task->threshold
        };
        
        // Execute in parallel
        pthread_t thread;
        pthread_create(&thread, NULL, 
                      (void*)fork_join_execute, &left);
        fork_join_execute(&right);
        
        // Join
        pthread_join(thread, NULL);
    }
}
```

## 7.7 Deadlock Prevention and Detection

### Banker's Algorithm Implementation

```c
typedef struct {
    int** max;           // Maximum demand
    int** allocation;    // Current allocation
    int** need;          // Remaining need
    int* available;      // Available resources
    int num_processes;
    int num_resources;
} banker_state_t;

bool is_safe_state(banker_state_t* state) {
    int* work = malloc(state->num_resources * sizeof(int));
    bool* finish = calloc(state->num_processes, sizeof(bool));
    
    // Initialize work = available
    memcpy(work, state->available, 
           state->num_resources * sizeof(int));
    
    int count = 0;
    while (count < state->num_processes) {
        bool found = false;
        
        for (int i = 0; i < state->num_processes; i++) {
            if (!finish[i]) {
                // Check if need[i] <= work
                bool can_allocate = true;
                for (int j = 0; j < state->num_resources; j++) {
                    if (state->need[i][j] > work[j]) {
                        can_allocate = false;
                        break;
                    }
                }
                
                if (can_allocate) {
                    // Process can complete
                    for (int j = 0; j < state->num_resources; j++) {
                        work[j] += state->allocation[i][j];
                    }
                    finish[i] = true;
                    found = true;
                    count++;
                }
            }
        }
        
        if (!found) {
            // No process can proceed - unsafe
            free(work);
            free(finish);
            return false;
        }
    }
    
    free(work);
    free(finish);
    return true;
}

bool request_resources(banker_state_t* state, int process, 
                       int* request) {
    // Check if request <= need
    for (int i = 0; i < state->num_resources; i++) {
        if (request[i] > state->need[process][i]) {
            return false;  // Exceeds maximum claim
        }
    }
    
    // Check if request <= available
    for (int i = 0; i < state->num_resources; i++) {
        if (request[i] > state->available[i]) {
            return false;  // Must wait
        }
    }
    
    // Tentatively allocate
    for (int i = 0; i < state->num_resources; i++) {
        state->available[i] -= request[i];
        state->allocation[process][i] += request[i];
        state->need[process][i] -= request[i];
    }
    
    // Check if state is safe
    if (is_safe_state(state)) {
        return true;  // Grant request
    } else {
        // Rollback
        for (int i = 0; i < state->num_resources; i++) {
            state->available[i] += request[i];
            state->allocation[process][i] -= request[i];
            state->need[process][i] += request[i];
        }
        return false;  // Deny request
    }
}
```

### Wait-For Graph

```c
typedef struct {
    int** adj_matrix;     // Adjacency matrix
    int num_processes;
} wait_for_graph_t;

bool has_cycle_util(wait_for_graph_t* graph, int v, 
                   bool* visited, bool* rec_stack) {
    visited[v] = true;
    rec_stack[v] = true;
    
    for (int i = 0; i < graph->num_processes; i++) {
        if (graph->adj_matrix[v][i]) {
            if (!visited[i]) {
                if (has_cycle_util(graph, i, visited, rec_stack)) {
                    return true;
                }
            } else if (rec_stack[i]) {
                return true;  // Back edge found
            }
        }
    }
    
    rec_stack[v] = false;
    return false;
}

bool detect_deadlock(wait_for_graph_t* graph) {
    bool* visited = calloc(graph->num_processes, sizeof(bool));
    bool* rec_stack = calloc(graph->num_processes, sizeof(bool));
    
    for (int i = 0; i < graph->num_processes; i++) {
        if (!visited[i]) {
            if (has_cycle_util(graph, i, visited, rec_stack)) {
                free(visited);
                free(rec_stack);
                return true;  // Deadlock detected
            }
        }
    }
    
    free(visited);
    free(rec_stack);
    return false;
}
```

## 7.8 Inter-Process Communication (IPC)

### Shared Memory with Synchronization

```c
#include <sys/shm.h>
#include <sys/sem.h>

typedef struct {
    int counter;
    char buffer[1024];
} shared_data_t;

// Semaphore operations
void sem_wait(int semid, int sem_num) {
    struct sembuf op = {sem_num, -1, 0};
    semop(semid, &op, 1);
}

void sem_signal(int semid, int sem_num) {
    struct sembuf op = {sem_num, 1, 0};
    semop(semid, &op, 1);
}

void shared_memory_example() {
    // Create shared memory
    int shmid = shmget(IPC_PRIVATE, sizeof(shared_data_t), 
                       IPC_CREAT | 0666);
    shared_data_t* data = shmat(shmid, NULL, 0);
    
    // Create semaphore
    int semid = semget(IPC_PRIVATE, 1, IPC_CREAT | 0666);
    semctl(semid, 0, SETVAL, 1);  // Initialize to 1
    
    if (fork() == 0) {
        // Child process
        for (int i = 0; i < 1000; i++) {
            sem_wait(semid, 0);
            data->counter++;
            sem_signal(semid, 0);
        }
        exit(0);
    } else {
        // Parent process
        for (int i = 0; i < 1000; i++) {
            sem_wait(semid, 0);
            data->counter++;
            sem_signal(semid, 0);
        }
        
        wait(NULL);
        printf("Final counter: %d\n", data->counter);
        
        // Cleanup
        shmdt(data);
        shmctl(shmid, IPC_RMID, NULL);
        semctl(semid, 0, IPC_RMID);
    }
}
```

### Message Queues

```c
#include <mqueue.h>

typedef struct {
    int type;
    char data[256];
} message_t;

void message_queue_example() {
    mqd_t mq;
    struct mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = 10,
        .mq_msgsize = sizeof(message_t),
        .mq_curmsgs = 0
    };
    
    // Create message queue
    mq = mq_open("/myqueue", O_CREAT | O_RDWR, 0666, &attr);
    
    if (fork() == 0) {
        // Child: Send messages
        message_t msg = {.type = 1, .data = "Hello from child"};
        mq_send(mq, (char*)&msg, sizeof(msg), 0);
        exit(0);
    } else {
        // Parent: Receive messages
        message_t msg;
        unsigned int priority;
        
        mq_receive(mq, (char*)&msg, sizeof(msg), &priority);
        printf("Received: %s\n", msg.data);
        
        wait(NULL);
        mq_close(mq);
        mq_unlink("/myqueue");
    }
}
```

### Unix Domain Sockets

```c
#include <sys/socket.h>
#include <sys/un.h>

void unix_socket_example() {
    int sockfd;
    struct sockaddr_un addr;
    
    if (fork() == 0) {
        // Client
        sleep(1);  // Let server start
        
        sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
        
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, "/tmp/mysocket");
        
        connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));
        
        char* message = "Hello, Server!";
        send(sockfd, message, strlen(message), 0);
        
        close(sockfd);
        exit(0);
    } else {
        // Server
        sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
        
        addr.sun_family = AF_UNIX;
        strcpy(addr.sun_path, "/tmp/mysocket");
        unlink(addr.sun_path);  // Remove if exists
        
        bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
        listen(sockfd, 5);
        
        int client = accept(sockfd, NULL, NULL);
        
        char buffer[256];
        int n = recv(client, buffer, sizeof(buffer)-1, 0);
        buffer[n] = '\0';
        printf("Server received: %s\n", buffer);
        
        close(client);
        close(sockfd);
        unlink(addr.sun_path);
        
        wait(NULL);
    }
}
```

## 7.9 Signal Handling

### Signal Management

```c
#include <signal.h>

// Signal handler
void signal_handler(int signum) {
    printf("Caught signal %d\n", signum);
    
    if (signum == SIGINT) {
        printf("Interrupt signal received\n");
    } else if (signum == SIGTERM) {
        printf("Termination signal received\n");
        exit(0);
    }
}

// Sigaction for more control
void setup_signal_handlers() {
    struct sigaction sa;
    
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;  // Restart interrupted system calls
    
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    
    // Block signals during critical section
    sigset_t blocked;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGINT);
    
    sigprocmask(SIG_BLOCK, &blocked, NULL);
    // Critical section
    sigprocmask(SIG_UNBLOCK, &blocked, NULL);
}

// Real-time signals
void realtime_signal_example() {
    struct sigaction sa;
    
    sa.sa_sigaction = realtime_handler;
    sa.sa_flags = SA_SIGINFO;  // Extended info
    
    sigaction(SIGRTMIN, &sa, NULL);
    
    // Send with data
    union sigval value;
    value.sival_int = 42;
    sigqueue(getpid(), SIGRTMIN, value);
}

void realtime_handler(int sig, siginfo_t* info, void* context) {
    printf("Received RT signal with value: %d\n", 
           info->si_value.sival_int);
}
```

## 7.10 Performance and Scalability

### Cache-Aware Programming

```c
// False sharing problem
struct bad_counter {
    atomic_int count1;  // Same cache line
    atomic_int count2;  // Causes contention
};

// Solution: Padding
struct good_counter {
    atomic_int count1;
    char padding[64 - sizeof(atomic_int)];  // Cache line size
    atomic_int count2;
};

// Cache-friendly data access
void matrix_multiply_cache_friendly(double** a, double** b, 
                                   double** c, int n) {
    int block_size = 64 / sizeof(double);  // Fit in cache line
    
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                // Block multiplication
                for (int i = ii; i < min(ii + block_size, n); i++) {
                    for (int j = jj; j < min(jj + block_size, n); j++) {
                        double sum = c[i][j];
                        for (int k = kk; k < min(kk + block_size, n); k++) {
                            sum += a[i][k] * b[k][j];
                        }
                        c[i][j] = sum;
                    }
                }
            }
        }
    }
}
```

### NUMA-Aware Memory Allocation

```c
#include <numa.h>

void numa_aware_allocation() {
    int num_nodes = numa_num_configured_nodes();
    
    // Allocate on specific node
    void* local_memory = numa_alloc_onnode(1024 * 1024, 0);
    
    // Bind thread to CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);  // CPU 0 on node 0
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    
    // Interleaved allocation
    void* interleaved = numa_alloc_interleaved(1024 * 1024);
    
    // Set memory policy
    unsigned long nodemask = 1 << 0;  // Node 0
    set_mempolicy(MPOL_BIND, &nodemask, num_nodes);
}
```

## 7.11 Modern Concurrency Features

### C11 Atomics and Memory Model

```c
#include <stdatomic.h>

// Memory ordering
atomic_int x = 0;
atomic_int y = 0;

void producer() {
    atomic_store_explicit(&x, 42, memory_order_relaxed);
    atomic_store_explicit(&y, 1, memory_order_release);
}

void consumer() {
    while (atomic_load_explicit(&y, memory_order_acquire) == 0);
    int value = atomic_load_explicit(&x, memory_order_relaxed);
    // value guaranteed to be 42
}

// Atomic operations
void atomic_examples() {
    atomic_int counter = ATOMIC_VAR_INIT(0);
    
    // Various operations
    atomic_fetch_add(&counter, 1);
    atomic_fetch_sub(&counter, 1);
    atomic_fetch_or(&counter, 0x0F);
    atomic_fetch_and(&counter, 0xF0);
    
    // Compare and exchange
    int expected = 0;
    int desired = 1;
    bool success = atomic_compare_exchange_strong(&counter, 
                                                  &expected, 
                                                  desired);
}
```

### Transactional Memory (GCC Extension)

```c
// Software transactional memory
__transaction_atomic {
    // All operations here are atomic
    shared_var1++;
    shared_var2 = shared_var1 * 2;
    if (shared_var3 > 100) {
        shared_var3 = 0;
    }
}

// Hardware transactional memory (Intel TSX)
unsigned status = _xbegin();
if (status == _XBEGIN_STARTED) {
    // Transaction
    shared_data++;
    _xend();
} else {
    // Fallback to lock
    pthread_mutex_lock(&mutex);
    shared_data++;
    pthread_mutex_unlock(&mutex);
}
```

## Exercises

1. Implement a thread-safe bounded buffer using:
   - Mutex and condition variables
   - Semaphores
   - Lock-free techniques

2. Write a program that creates 10 threads, each incrementing a shared counter 1 million times. Compare:
   - No synchronization (race condition)
   - Mutex protection
   - Atomic operations
   - Thread-local storage with final merge

3. Implement the Dining Philosophers problem with:
   - Resource ordering
   - Banker's algorithm
   - Chandy/Misra solution

4. Create a thread pool that:
   - Accepts tasks dynamically
   - Scales workers based on load
   - Handles task priorities

5. Write a program demonstrating:
   - Priority inversion
   - Priority inheritance solution

6. Implement a readers-writers solution that:
   - Prevents writer starvation
   - Maximizes concurrency
   - Is fair to both readers and writers

7. Create a lock-free SPSC (single producer, single consumer) queue.

8. Write a program that detects and recovers from deadlock using:
   - Timeout-based detection
   - Wait-for graph analysis

9. Implement parallel merge sort using:
   - Fork-join pattern
   - Thread pool
   Compare performance with sequential version

10. Create a simple user-level thread library with:
    - Thread creation/destruction
    - Cooperative scheduling
    - Basic synchronization

## Summary

This chapter explored process management and concurrency in depth:

- Processes and threads are the fundamental units of execution
- CPU scheduling algorithms balance fairness, throughput, and response time
- Synchronization primitives coordinate access to shared resources
- Lock-free programming offers performance benefits but is complex
- IPC mechanisms enable communication between processes
- Deadlock prevention and detection ensure system progress
- Modern hardware requires cache-aware and NUMA-aware programming
- Concurrent programming patterns simplify complex synchronization

Understanding concurrency is essential for modern software development, as multi-core processors are ubiquitous and distributed systems are increasingly common. The next chapter will explore file systems, examining how operating systems organize and manage persistent storage.