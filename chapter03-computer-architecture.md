# Chapter 3: Computer Architecture Fundamentals

## Introduction

Computer architecture bridges the gap between the digital logic circuits we explored in the previous chapter and the software that runs on modern computers. It defines how a computer's components work together to execute programs. This chapter examines the fundamental concepts of computer organization, from the basic von Neumann architecture to modern processor designs with multiple cores, caches, and sophisticated instruction execution strategies.

### Real-World Impact

Architectural decisions affect everything from battery life in mobile devices to the performance of data centers:
- **Cache design** determines whether your game runs at 30 or 60 FPS
- **Pipeline depth** affects both performance and power consumption
- **Branch prediction** can make a 50% difference in real-world performance
- **Memory bandwidth** often limits AI and scientific computing more than CPU speed

## 3.1 The von Neumann Architecture

In 1945, John von Neumann described a computer architecture that remains the foundation of most modern computers. The key innovation was the stored-program concept: both data and instructions are stored in the same memory.

### Components of von Neumann Architecture

1. **Central Processing Unit (CPU)**
   - Arithmetic Logic Unit (ALU): Performs calculations
   - Control Unit (CU): Interprets instructions and controls operations
   - Registers: Fast storage within the CPU

2. **Memory**
   - Stores both programs and data
   - Addressable by location
   - Random access capability

3. **Input/Output System**
   - Interfaces with external devices
   - Transfers data to/from memory

4. **System Bus**
   - Data bus: Carries data
   - Address bus: Specifies memory locations
   - Control bus: Carries control signals

### The von Neumann Bottleneck

The single bus connecting CPU and memory creates a bottleneck:
- CPU often waits for memory access (CPU: 1 ns, RAM: 100 ns)
- Instructions and data compete for bus bandwidth
- Modern architectures use caches and parallel buses to mitigate this

**Real-world example**: A modern CPU can execute 4+ instructions per cycle at 4 GHz (16 billion ops/sec), but RAM delivers only ~20 GB/s. Without caches, the CPU would be idle 99% of the time!

## 3.2 The Central Processing Unit (CPU)

### CPU Components

**Arithmetic Logic Unit (ALU)**
```
Inputs:  Operand A, Operand B, Operation Code
Outputs: Result, Status Flags (Zero, Negative, Carry, Overflow)

Operations:
- Arithmetic: ADD, SUB, MUL, DIV
- Logical: AND, OR, XOR, NOT
- Shift: SHL, SHR, ROL, ROR
- Compare: CMP, TEST
```

**Control Unit**
- Instruction decoder: Interprets instruction opcodes
- Sequencer: Generates control signals
- Timing generator: Synchronizes operations

**Registers**

General Purpose:
- Used for data manipulation
- Typically 8-32 registers in modern CPUs
- Examples: EAX, EBX, ECX, EDX (x86)

Special Purpose:
- Program Counter (PC): Next instruction address
- Stack Pointer (SP): Top of stack
- Instruction Register (IR): Current instruction
- Status Register (FLAGS): Condition codes
- Memory Address Register (MAR): Address for memory access
- Memory Data Register (MDR): Data for memory transfer

### The Fetch-Decode-Execute Cycle

The fundamental CPU operation cycle:

1. **Fetch**
   ```
   MAR ← PC
   MDR ← Memory[MAR]
   IR ← MDR
   PC ← PC + instruction_length
   ```

2. **Decode**
   - Control unit decodes instruction in IR
   - Identifies operation and operands
   - Generates control signals

3. **Execute**
   - Perform the operation
   - May involve:
     - ALU operations
     - Memory access
     - Register updates
     - I/O operations

4. **Store** (if needed)
   - Write results back to memory or registers

### CPU Performance Metrics

**Clock Speed**: Cycles per second (Hz)
- Modern CPUs: 2-5 GHz typically
- Note: Higher clock ≠ better performance (Pentium 4 vs. Core 2)

**Instructions Per Cycle (IPC)**: Average instructions completed per clock cycle
- Varies by workload and architecture
- Modern CPUs: 2-4 IPC common
- Superscalar processors can exceed 1 IPC

**MIPS/FLOPS**: Millions of Instructions/Floating-point Operations Per Second
- Misleading metric: different ISAs have different instruction complexity

**CPI (Cycles Per Instruction)**: Inverse of IPC
```
Execution Time = Instruction Count × CPI × Clock Period

Amdahl's Law: Speedup = 1 / ((1 - P) + P/S)
where P = parallel portion, S = speedup of that portion
```

## 3.3 Memory Hierarchy

Memory systems balance speed, capacity, and cost through a hierarchy:

### Hierarchy Levels

1. **Registers**
   - Size: ~1 KB
   - Access: 0 cycles (same cycle as instruction)
   - Technology: Flip-flops
   - Cost: ~$1000/MB equivalent

2. **L1 Cache**
   - Size: 32-64 KB per core
   - Access: 2-4 cycles
   - Split: I-cache (instructions) and D-cache (data)

3. **L2 Cache**
   - Size: 256 KB - 1 MB per core
   - Access: 10-20 cycles
   - Usually unified (instructions and data)

4. **L3 Cache**
   - Size: 8-32 MB shared
   - Access: 30-50 cycles
   - Shared among cores

5. **Main Memory (RAM)**
   - Size: 8-64 GB typical
   - Access: 100-300 cycles (~50-100 ns)
   - Technology: DRAM (requires refresh)
   - Bandwidth: DDR4-3200 = 25.6 GB/s theoretical

6. **Secondary Storage (SSD/HDD)**
   - Size: 256 GB - several TB
   - Access: 10,000+ cycles (SSD: ~100 μs), millions (HDD: ~10 ms)
   - Persistent storage
   - SSD: 500 MB/s - 7 GB/s (NVMe)
   - HDD: 100-200 MB/s sequential

### Cache Organization

**Direct Mapped Cache**
- Each memory block maps to exactly one cache line
- Simple but can cause conflicts

```
Cache Line = (Memory Address) mod (Number of Cache Lines)
```

**Set-Associative Cache**
- Memory blocks map to a set of cache lines
- n-way: each set has n lines
- Balance between complexity and performance

**Fully Associative Cache**
- Any memory block can go in any cache line
- Maximum flexibility but complex

### Cache Replacement Policies

When cache is full, which line to evict?

- **LRU (Least Recently Used)**: Replace oldest accessed
  - Good for temporal locality
  - Complex to implement perfectly (often use pseudo-LRU)
- **FIFO (First In First Out)**: Replace oldest loaded
  - Simple but ignores access patterns
- **Random**: Simple but unpredictable
  - Surprisingly effective, avoids pathological cases
- **LFU (Least Frequently Used)**: Replace least accessed
  - Can get stuck on old frequently-used data

### Cache Coherence

Multiple cores with separate caches must maintain consistency:

**MESI Protocol** states:
- Modified: Only this cache has valid, modified data
- Exclusive: Only this cache has valid, unmodified data
- Shared: Multiple caches have valid, unmodified data
- Invalid: Cache line is invalid

## 3.4 Instruction Set Architecture (ISA)

The ISA defines the interface between hardware and software:

### CISC vs RISC

**CISC (Complex Instruction Set Computer)**
- Examples: x86, VAX
- Features:
  - Variable-length instructions (1-15 bytes in x86)
  - Complex addressing modes
  - Many specialized instructions (>1000 in modern x86)
  - Microcode implementation
- Philosophy: "Let hardware do complex operations"

**RISC (Reduced Instruction Set Computer)**
- Examples: ARM, RISC-V, MIPS
- Features:
  - Fixed-length instructions (32 bits typical)
  - Simple addressing modes
  - Load/store architecture (only load/store access memory)
  - Hardware implementation (no microcode)
- Philosophy: "Keep hardware simple, let compiler optimize"

**Modern reality**: x86 CPUs translate CISC to RISC-like micro-ops internally!

### Instruction Types

**Data Transfer**
```
LOAD  R1, [address]    ; Load from memory
STORE [address], R1    ; Store to memory
MOVE  R1, R2          ; Register to register
```

**Arithmetic/Logic**
```
ADD  R1, R2, R3       ; R1 = R2 + R3
SUB  R1, R2, R3       ; R1 = R2 - R3
AND  R1, R2, R3       ; R1 = R2 & R3
SHL  R1, R2, #3       ; R1 = R2 << 3
```

**Control Flow**
```
JMP  label            ; Unconditional jump
JZ   label            ; Jump if zero
CALL procedure        ; Call subroutine
RET                   ; Return from subroutine
```

### Addressing Modes

1. **Immediate**: Operand in instruction
   ```
   ADD R1, #5        ; R1 = R1 + 5
   ```

2. **Register**: Operand in register
   ```
   ADD R1, R2        ; R1 = R1 + R2
   ```

3. **Direct**: Address in instruction
   ```
   LOAD R1, [1000]   ; R1 = Memory[1000]
   ```

4. **Indirect**: Address in register
   ```
   LOAD R1, [R2]     ; R1 = Memory[R2]
   ```

5. **Indexed**: Base + offset
   ```
   LOAD R1, [R2+100] ; R1 = Memory[R2+100]
   ```

6. **PC-relative**: Relative to program counter
   ```
   JMP PC+50         ; Jump forward 50 bytes
   ```

## 3.5 Pipelining

Pipelining overlaps instruction execution stages for increased throughput:

### Basic 5-Stage Pipeline

```
Time →
Inst 1: IF | ID | EX | MEM | WB |
Inst 2:    | IF | ID | EX  | MEM | WB |
Inst 3:       | IF | ID | EX  | MEM | WB |
Inst 4:          | IF | ID | EX  | MEM | WB |
```

Stages:
1. **IF (Instruction Fetch)**: Fetch from memory
2. **ID (Instruction Decode)**: Decode and read registers
3. **EX (Execute)**: Perform ALU operation
4. **MEM (Memory)**: Access memory if needed
5. **WB (Write Back)**: Write result to register

### Pipeline Hazards

**Structural Hazards**
- Hardware resource conflicts
- Solution: Duplicate resources or stall

**Data Hazards**
- Instruction needs result from previous instruction
```
ADD R1, R2, R3    ; Produces R1
SUB R4, R1, R5    ; Needs R1 (hazard!)
```
Solutions:
- Forwarding/bypassing: Route result directly
- Stalling: Insert pipeline bubbles

**Control Hazards**
- Branch instructions change program flow
- Solutions:
  - Branch prediction
  - Delayed branches
  - Speculative execution

### Superscalar Execution

Execute multiple instructions per cycle:

```
Dual-issue pipeline:
Time →
Inst 1: IF | ID | EX | MEM | WB |
Inst 2: IF | ID | EX | MEM | WB |
Inst 3:    | IF | ID | EX  | MEM | WB |
Inst 4:    | IF | ID | EX  | MEM | WB |
```

Requirements:
- Multiple execution units
- Dependency checking
- Instruction reordering capability

## 3.6 Advanced CPU Features

### Out-of-Order Execution

Execute instructions in optimal order, not program order:

1. **Instruction Fetch**: Get instructions in order
2. **Dispatch**: Send to reservation stations
3. **Execute**: When operands ready
4. **Commit**: Retire in program order

Benefits:
- Hide memory latency
- Better resource utilization
- Increased ILP (Instruction Level Parallelism)

### Branch Prediction

Predict branch outcomes to avoid pipeline stalls:

**Static Prediction**
- Always taken/not taken
- Backward taken, forward not taken

**Dynamic Prediction**
- Branch History Table (BHT)
- Two-bit saturating counter:
```
Strongly Taken ← Weakly Taken ← Weakly Not Taken ← Strongly Not Taken
```

**Advanced Predictors**
- Two-level adaptive
- Tournament predictor
- Neural branch prediction

### Speculative Execution

Execute instructions before knowing if they're needed:

1. Predict branch outcome
2. Execute predicted path
3. If correct: commit results
4. If wrong: flush pipeline, restart

Security implications:
- Spectre/Meltdown vulnerabilities
- Side-channel attacks

### SIMD (Single Instruction, Multiple Data)

Process multiple data elements with one instruction:

```
Regular scalar:
for (i = 0; i < 4; i++)
    c[i] = a[i] + b[i];

SIMD vector:
c[0:3] = a[0:3] + b[0:3];  // One instruction
```

Examples:
- x86: MMX, SSE, AVX
- ARM: NEON, SVE
- Applications: Graphics, scientific computing, machine learning

## 3.7 Multicore and Parallel Architectures

### Multicore Processors

Multiple CPU cores on single chip:

**Private vs Shared Resources**
- Private: L1/L2 cache, registers
- Shared: L3 cache, memory controller, I/O

**Cache Coherence Protocols**
- Snooping: Broadcast-based
- Directory: Scalable for many cores

### Hardware Multithreading

**Coarse-grained**: Switch threads on long stalls
**Fine-grained**: Switch every cycle
**Simultaneous (SMT)**: Multiple threads per cycle

Intel Hyper-Threading example:
- 2 logical processors per physical core
- Share execution units
- Independent architectural state

### GPU Architecture

Massively parallel for graphics and compute:

**Streaming Multiprocessors (SMs)**
- Many simple cores (32-128)
- SIMT execution (Single Instruction, Multiple Threads)
- Shared memory and caches

**Memory Hierarchy**
- Global memory: Large, slow
- Shared memory: Fast, per-SM
- Registers: Very fast, per-thread

## 3.8 Input/Output Systems

### I/O Methods

**Programmed I/O**
- CPU actively transfers data
- Simple but inefficient

```
while (device_not_ready) {
    // Poll status
}
transfer_data();
```

**Interrupt-Driven I/O**
- Device signals completion
- CPU can do other work

**Direct Memory Access (DMA)**
- Dedicated controller transfers data
- CPU only sets up transfer

### Bus Systems

**System Bus Types**
- Data bus: Carries data
- Address bus: Specifies locations
- Control bus: Control signals

**Bus Protocols**
- Synchronous: Clock-based timing
- Asynchronous: Handshaking

**Modern Bus Standards**
- PCIe: High-speed serial, point-to-point
- USB: Universal serial bus
- SATA: Storage devices
- NVMe: High-speed SSDs

### Interrupts

**Types**
- Hardware: From devices
- Software: System calls
- Exceptions: Errors, page faults

**Interrupt Handling**
1. Save current state
2. Identify interrupt source
3. Execute interrupt handler
4. Restore state and resume

**Interrupt Priority**
- Multiple levels
- Higher priority can interrupt lower
- Maskable vs non-maskable

## 3.9 Performance Optimization

### Amdahl's Law

Maximum speedup from parallelization:

```
Speedup = 1 / (S + P/N)

Where:
S = Serial fraction
P = Parallel fraction (1-S)
N = Number of processors
```

Example: 90% parallel, 10% serial, 10 processors:
```
Speedup = 1 / (0.1 + 0.9/10) = 5.26×
```

### Performance Bottlenecks

**Memory Wall**
- CPU speed grows faster than memory
- Solutions: Caches, prefetching, memory parallelism

**Power Wall**
- Power consumption limits frequency
- Solutions: Multiple cores, power gating, DVFS

**ILP Wall**
- Limited instruction-level parallelism
- Solutions: Thread-level parallelism, specialized units

### Benchmarking

**Types of Benchmarks**
- Synthetic: Dhrystone, Whetstone
- Application: SPEC CPU, PARSEC
- Microbenchmarks: Cache latency, bandwidth

**Metrics**
- Throughput: Work per time
- Latency: Time per operation
- Energy efficiency: Performance per watt

## 3.10 Modern Architecture Examples

### x86-64 Architecture

**Features**
- CISC ISA with RISC micro-ops
- 16 general-purpose 64-bit registers
- Out-of-order superscalar execution
- SIMD extensions (SSE, AVX)

**Memory Model**
- 48-bit virtual addresses (256 TB)
- 4-level page tables
- Multiple page sizes (4KB, 2MB, 1GB)

### ARM Architecture

**Features**
- RISC ISA
- Fixed 32-bit instructions (AArch32) or 32-bit/16-bit (Thumb)
- 31 general-purpose 64-bit registers (AArch64)
- Energy-efficient design

**big.LITTLE**
- Heterogeneous multiprocessing
- High-performance and efficiency cores
- Dynamic task migration

### RISC-V Architecture

**Features**
- Open-source ISA
- Modular design (base + extensions)
- 32 integer registers
- Simple, regular encoding

**Extensions**
- I: Base integer
- M: Multiplication/division
- A: Atomic operations
- F/D: Floating-point
- C: Compressed instructions

## Exercises

1. Calculate the CPI for a program with:
   - 40% ALU instructions (1 cycle)
   - 30% load instructions (2 cycles)
   - 20% store instructions (2 cycles)
   - 10% branch instructions (3 cycles)

2. Design a 4-way set-associative cache with:
   - 64 KB total size
   - 64-byte blocks
   - Calculate tag, index, and offset bits for 32-bit addresses

3. Show the pipeline execution for:
   ```
   LOAD  R1, [100]
   ADD   R2, R1, R3
   STORE [200], R2
   ```
   Identify hazards and solutions.

4. Calculate speedup using Amdahl's Law:
   - 75% parallel code
   - 8, 16, and 32 processors

5. Design a simple 8-bit CPU with:
   - 4 general-purpose registers
   - Basic ALU operations
   - 256 bytes of memory

6. Compare instruction count for CISC vs RISC:
   - Array copy operation
   - Matrix multiplication

7. Explain how cache coherence is maintained when:
   - CPU 1 writes to address X
   - CPU 2 reads from address X

8. Calculate effective memory access time:
   - L1 hit rate: 95%, latency: 2 cycles
   - L2 hit rate: 80%, latency: 20 cycles
   - Memory latency: 200 cycles

9. Design a branch predictor state machine with 90% accuracy goal.

10. Compare GPU vs CPU for:
    - Matrix multiplication
    - Database queries
    - Web server

## Summary

This chapter explored the fundamental concepts of computer architecture:

- The von Neumann architecture provides the basic organizational model
- CPUs execute instructions through fetch-decode-execute cycles
- Memory hierarchies balance speed and capacity
- Pipelining and parallelism increase performance
- Modern processors use sophisticated techniques like out-of-order execution
- Multiple cores and specialized processors handle parallel workloads
- I/O systems connect computers to the external world

These architectural concepts form the foundation for understanding how software runs on hardware. The next chapter will explore assembly language programming, showing how to write programs that directly interact with the processor architecture we've just studied.