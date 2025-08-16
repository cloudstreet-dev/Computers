# Chapter 4: Assembly Language and Machine Code

## Introduction

Assembly language is the lowest-level programming language that remains human-readable. It provides a symbolic representation of machine code, with a one-to-one correspondence between assembly instructions and machine instructions. Understanding assembly language reveals how programs actually execute on hardware, provides insights into compiler optimizations, and enables system-level programming. This chapter explores assembly language programming, the translation to machine code, and the intimate relationship between software and hardware.

## 4.1 From High-Level to Machine Code

The journey from source code to execution:

```
High-Level Language (C, Python, Java)
         ↓ Compiler/Interpreter
Assembly Language
         ↓ Assembler
Machine Code (Binary)
         ↓ Loader
Memory (Execution)
```

### Why Learn Assembly?

1. **Performance Optimization**: Critical code sections
2. **System Programming**: Boot loaders, drivers, kernels
3. **Reverse Engineering**: Understanding compiled code
4. **Security**: Exploit analysis, vulnerability research
5. **Embedded Systems**: Resource-constrained environments
6. **Debugging**: Understanding crash dumps and low-level bugs

## 4.2 Assembly Language Basics

### Anatomy of an Assembly Program

```assembly
section .data           ; Initialized data
    msg db 'Hello', 0   ; String with null terminator
    len equ $ - msg     ; Length constant

section .bss            ; Uninitialized data
    buffer resb 256     ; Reserve 256 bytes

section .text           ; Code section
global _start          ; Entry point

_start:                ; Label
    mov eax, 4         ; System call number (write)
    mov ebx, 1         ; File descriptor (stdout)
    mov ecx, msg       ; Message address
    mov edx, len       ; Message length
    int 0x80          ; Linux system call
    
    mov eax, 1         ; System call (exit)
    xor ebx, ebx       ; Exit status 0
    int 0x80          ; Linux system call
```

### Components

**Instructions**: Operations the CPU can perform
```assembly
add eax, ebx       ; EAX = EAX + EBX
```

**Directives**: Instructions to the assembler
```assembly
section .text      ; Define code section
global _start      ; Make symbol visible
```

**Labels**: Symbolic names for addresses
```assembly
loop_start:        ; Can be jumped to
    ; code here
```

**Comments**: Documentation (ignored by assembler)
```assembly
mov eax, 5        ; Load 5 into EAX register
```

## 4.3 x86 Assembly Language

We'll focus on x86-64, the most common desktop/server architecture.

### Registers

**General Purpose (64-bit mode)**
```
64-bit: RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8-R15
32-bit: EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP, R8D-R15D
16-bit: AX,  BX,  CX,  DX,  SI,  DI,  BP,  SP,  R8W-R15W
8-bit:  AL,  BL,  CL,  DL,  SIL, DIL, BPL, SPL, R8B-R15B
        AH,  BH,  CH,  DH  (high bytes of first 4 registers)
```

**Special Purpose**
- RIP: Instruction pointer
- RFLAGS: Status flags
- Segment registers: CS, DS, ES, FS, GS, SS

### Data Movement Instructions

```assembly
; Move data
mov rax, 42           ; Immediate to register
mov rax, rbx          ; Register to register
mov rax, [rbx]        ; Memory to register (dereference)
mov [rax], rbx        ; Register to memory
mov byte [rax], 5     ; Size specifier

; Load effective address
lea rax, [rbx + rcx*4 + 10]  ; Calculate address

; Stack operations
push rax              ; Push onto stack
pop rbx               ; Pop from stack

; Exchange
xchg rax, rbx         ; Swap values
```

### Arithmetic Instructions

```assembly
; Addition/Subtraction
add rax, rbx          ; rax = rax + rbx
sub rax, 10           ; rax = rax - 10
inc rax               ; rax++
dec rbx               ; rbx--
neg rax               ; rax = -rax

; Multiplication
mul rbx               ; Unsigned: RDX:RAX = RAX * RBX
imul rbx              ; Signed: RDX:RAX = RAX * RBX
imul rax, rbx, 5      ; rax = rbx * 5

; Division
div rbx               ; Unsigned: RAX = RDX:RAX / RBX, RDX = remainder
idiv rbx              ; Signed division

; Comparison
cmp rax, rbx          ; Set flags based on rax - rbx
test rax, rbx         ; Set flags based on rax & rbx
```

### Logical and Bit Operations

```assembly
; Boolean operations
and rax, rbx          ; Bitwise AND
or rax, rbx           ; Bitwise OR
xor rax, rbx          ; Bitwise XOR
not rax               ; Bitwise NOT

; Shifts
shl rax, 3            ; Shift left by 3
shr rax, cl           ; Shift right by CL
sal rax, 1            ; Arithmetic shift left
sar rax, 1            ; Arithmetic shift right (sign-extend)

; Rotations
rol rax, 5            ; Rotate left
ror rax, cl           ; Rotate right
```

### Control Flow

```assembly
; Unconditional jump
jmp label             ; Jump to label

; Conditional jumps (after cmp or test)
je  label             ; Jump if equal (ZF=1)
jne label             ; Jump if not equal (ZF=0)
jl  label             ; Jump if less (signed)
jle label             ; Jump if less or equal
jg  label             ; Jump if greater
jge label             ; Jump if greater or equal
jb  label             ; Jump if below (unsigned)
ja  label             ; Jump if above (unsigned)

; Loops
loop label            ; Decrement RCX, jump if not zero

; Procedures
call function         ; Push return address, jump
ret                   ; Pop return address, jump back
```

## 4.4 Memory Addressing Modes

x86 supports complex addressing calculations:

```assembly
; Direct addressing
mov rax, [0x1000]     ; Load from address 0x1000

; Register indirect
mov rax, [rbx]        ; Load from address in RBX

; Register + displacement
mov rax, [rbx + 8]    ; Load from RBX + 8

; Indexed
mov rax, [rbx + rcx]  ; Load from RBX + RCX

; Scaled indexed
mov rax, [rbx + rcx*4] ; Load from RBX + (RCX * 4)

; Full format: [base + index*scale + displacement]
mov rax, [rbx + rcx*8 + 0x10]

; RIP-relative (x86-64)
mov rax, [rip + label] ; PC-relative addressing
```

## 4.5 The Stack

The stack grows downward (toward lower addresses):

```assembly
; Function prologue
push rbp              ; Save old base pointer
mov rbp, rsp          ; Set up stack frame
sub rsp, 32           ; Allocate 32 bytes local space

; Access local variables
mov [rbp - 8], rax    ; First local variable
mov [rbp - 16], rbx   ; Second local variable

; Access parameters (in 64-bit calling convention)
mov rax, rdi          ; First parameter
mov rbx, rsi          ; Second parameter

; Function epilogue
mov rsp, rbp          ; Restore stack pointer
pop rbp               ; Restore base pointer
ret                   ; Return
```

### Stack Frame Layout

```
Higher addresses
+------------------+
| Return address   | <- Pushed by CALL
+------------------+
| Old RBP          | <- RBP points here after prologue
+------------------+
| Local var 1      |
+------------------+
| Local var 2      | <- RSP points to last item
+------------------+
Lower addresses
```

## 4.6 Procedures and Calling Conventions

### System V AMD64 ABI (Linux/Mac)

**Register usage:**
- Parameters: RDI, RSI, RDX, RCX, R8, R9 (then stack)
- Return value: RAX (RDX:RAX for 128-bit)
- Callee-saved: RBX, RBP, R12-R15
- Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11

**Example function call:**
```assembly
; Caller
mov rdi, 10           ; First argument
mov rsi, 20           ; Second argument
call add_numbers
; Result now in RAX

; Callee
add_numbers:
    push rbp          ; Save frame pointer
    mov rbp, rsp      ; Set up frame
    
    mov rax, rdi      ; Get first parameter
    add rax, rsi      ; Add second parameter
    
    pop rbp           ; Restore frame pointer
    ret               ; Return (result in RAX)
```

### Windows x64 Calling Convention

Different from System V:
- Parameters: RCX, RDX, R8, R9 (then stack)
- 32 bytes shadow space always allocated
- Different register preservation rules

## 4.7 System Calls

Direct kernel interface without library functions:

### Linux System Calls (int 0x80 - 32-bit)

```assembly
mov eax, 4            ; System call number (write)
mov ebx, 1            ; First argument (stdout)
mov ecx, buffer       ; Second argument (buffer address)
mov edx, length       ; Third argument (length)
int 0x80             ; Invoke system call
```

### Linux System Calls (syscall - 64-bit)

```assembly
mov rax, 1            ; System call number (write)
mov rdi, 1            ; First argument (stdout)
mov rsi, buffer       ; Second argument
mov rdx, length       ; Third argument
syscall              ; Fast system call
```

Common system calls:
```
| Number | Name   | Description              |
|--------|--------|--------------------------|
| 0      | read   | Read from file           |
| 1      | write  | Write to file            |
| 2      | open   | Open file                |
| 3      | close  | Close file               |
| 9      | mmap   | Map memory               |
| 60     | exit   | Terminate process        |
```

## 4.8 Assembly Programming Examples

### Example 1: String Length

```assembly
; Calculate string length (null-terminated)
strlen:
    push rbp
    mov rbp, rsp
    
    xor rax, rax      ; Counter = 0
    mov rcx, rdi      ; String pointer
    
.loop:
    cmp byte [rcx], 0 ; Check for null
    je .done          ; If null, we're done
    inc rax           ; Increment counter
    inc rcx           ; Next character
    jmp .loop
    
.done:
    pop rbp
    ret               ; Length in RAX
```

### Example 2: Array Sum

```assembly
; Sum array of integers
; RDI = array pointer, RSI = count
array_sum:
    push rbp
    mov rbp, rsp
    
    xor rax, rax      ; Sum = 0
    xor rcx, rcx      ; Index = 0
    
.loop:
    cmp rcx, rsi      ; Check if done
    jge .done
    
    add rax, [rdi + rcx*8]  ; Add element (8 bytes each)
    inc rcx           ; Next element
    jmp .loop
    
.done:
    pop rbp
    ret
```

### Example 3: Bubble Sort

```assembly
; Bubble sort array of integers
; RDI = array pointer, RSI = count
bubble_sort:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r12, rsi      ; Save count
    dec r12           ; n-1 for outer loop
    
.outer_loop:
    xor rbx, rbx      ; Swapped flag = 0
    xor rcx, rcx      ; Inner index = 0
    
.inner_loop:
    mov rdx, r12
    sub rdx, rcx      ; Limit for inner loop
    cmp rcx, rdx
    jge .check_swapped
    
    ; Compare adjacent elements
    mov rax, [rdi + rcx*8]
    mov rdx, [rdi + rcx*8 + 8]
    cmp rax, rdx
    jle .no_swap
    
    ; Swap elements
    mov [rdi + rcx*8], rdx
    mov [rdi + rcx*8 + 8], rax
    mov rbx, 1        ; Set swapped flag
    
.no_swap:
    inc rcx
    jmp .inner_loop
    
.check_swapped:
    test rbx, rbx     ; Check if any swaps
    jnz .outer_loop   ; If swapped, continue
    
    pop r12
    pop rbx
    pop rbp
    ret
```

## 4.9 Machine Code Encoding

Assembly instructions are encoded into binary machine code:

### x86 Instruction Format

```
[Prefixes] [Opcode] [ModR/M] [SIB] [Displacement] [Immediate]
```

**Example: `add rax, rbx`**
```
48 01 D8
│  │  │
│  │  └─ ModR/M byte: 11 011 000 (reg-reg, rbx, rax)
│  └──── Opcode: ADD r/m64, r64
└─────── REX prefix: 64-bit operands
```

### ModR/M Byte

Specifies operands:
```
  7 6   5 4 3   2 1 0
┌─────┬───────┬───────┐
│ Mod │  Reg  │  R/M  │
└─────┴───────┴───────┘

Mod: Addressing mode
00 = [register]
01 = [register + byte displacement]
10 = [register + dword displacement]
11 = register

Reg: Register operand
R/M: Register or memory operand
```

### Variable-Length Encoding

x86 instructions vary from 1 to 15 bytes:

```assembly
ret                   ; C3 (1 byte)
nop                   ; 90 (1 byte)
push rax              ; 50 (1 byte with implicit register)
mov eax, 0x12345678   ; B8 78 56 34 12 (5 bytes)
mov [rbx+rcx*4+0x1000], rax ; Complex, up to 15 bytes
```

## 4.10 Assembler Directives and Macros

### Common Directives

```assembly
; Sections
section .text         ; Code section
section .data         ; Initialized data
section .bss          ; Uninitialized data

; Data definition
db 0x41, 'A', 65     ; Define bytes
dw 0x1234            ; Define word (2 bytes)
dd 0x12345678        ; Define double word (4 bytes)
dq 0x123456789ABCDEF ; Define quad word (8 bytes)

; Space reservation
resb 100             ; Reserve 100 bytes
resw 50              ; Reserve 50 words

; Constants
equ BUFFER_SIZE, 1024 ; Define constant

; Alignment
align 16             ; Align to 16-byte boundary

; Include files
%include "macros.inc" ; Include another file
```

### Macros

Reusable code templates:

```assembly
; Define macro
%macro print_string 2  ; Name and parameter count
    mov rax, 1         ; System call: write
    mov rdi, 1         ; File descriptor: stdout
    mov rsi, %1        ; First macro parameter
    mov rdx, %2        ; Second macro parameter
    syscall
%endmacro

; Use macro
section .data
    msg db "Hello, World!", 10
    len equ $ - msg

section .text
    print_string msg, len  ; Macro expansion
```

### Conditional Assembly

```assembly
%ifdef DEBUG
    ; Debug code
    mov rax, [debug_counter]
    inc rax
    mov [debug_counter], rax
%endif

%if BUFFER_SIZE > 1024
    ; Large buffer handling
%else
    ; Small buffer handling
%endif
```

## 4.11 Optimization Techniques

### Instruction Selection

```assembly
; Clear register - multiple ways
mov eax, 0           ; 5 bytes
xor eax, eax         ; 2 bytes (preferred)

; Multiply by constant
mov rax, rbx
imul rax, 5          ; Multiply instruction
; vs
lea rax, [rbx + rbx*4] ; LEA trick (faster)

; Test for zero
cmp rax, 0           ; Compare with 0
test rax, rax        ; Test (preferred, smaller)
```

### Loop Optimization

```assembly
; Unoptimized loop
loop_start:
    mov rax, [rsi]
    add rax, 1
    mov [rsi], rax
    add rsi, 8
    dec rcx
    jnz loop_start

; Optimized with loop unrolling
loop_start:
    mov rax, [rsi]
    mov rbx, [rsi + 8]
    mov rdx, [rsi + 16]
    mov r8, [rsi + 24]
    
    add rax, 1
    add rbx, 1
    add rdx, 1
    add r8, 1
    
    mov [rsi], rax
    mov [rsi + 8], rbx
    mov [rsi + 16], rdx
    mov [rsi + 24], r8
    
    add rsi, 32
    sub rcx, 4
    jnz loop_start
```

### Alignment

```assembly
; Align critical loops
align 16              ; Align to cache line
critical_loop:
    ; Loop body
```

## 4.12 SIMD Programming

Using SSE/AVX for parallel operations:

### SSE Example

```assembly
; Add two vectors of 4 floats
; xmm0 = [a0, a1, a2, a3]
; xmm1 = [b0, b1, b2, b3]
movaps xmm0, [vector_a]  ; Load aligned
movaps xmm1, [vector_b]
addps xmm0, xmm1         ; Parallel add
movaps [result], xmm0    ; Store result
```

### AVX Example

```assembly
; Add vectors of 8 floats
vmovaps ymm0, [vector_a]  ; 256-bit load
vmovaps ymm1, [vector_b]
vaddps ymm2, ymm0, ymm1   ; Non-destructive add
vmovaps [result], ymm2
```

## 4.13 Debugging Assembly Code

### Using GDB

```bash
# Compile with debug info
nasm -f elf64 -g -F dwarf program.asm
ld -o program program.o

# Debug with GDB
gdb ./program

# GDB commands
(gdb) break _start       # Set breakpoint
(gdb) run               # Start program
(gdb) info registers    # Show registers
(gdb) x/10x $rsp       # Examine stack
(gdb) stepi            # Step one instruction
(gdb) disas            # Disassemble
```

### Common Issues

**Segmentation Fault**
- Dereferencing invalid pointer
- Stack overflow
- Writing to read-only memory

**Wrong Results**
- Register size mismatch
- Signed/unsigned confusion
- Incorrect addressing mode

**Performance Issues**
- Cache misses
- Branch misprediction
- Unaligned memory access

## 4.14 Inline Assembly

Embedding assembly in C/C++:

### GCC Inline Assembly

```c
int add(int a, int b) {
    int result;
    __asm__ (
        "addl %2, %0"
        : "=r" (result)     // Output
        : "0" (a), "r" (b)  // Inputs
    );
    return result;
}

// Extended example
void atomic_increment(int *ptr) {
    __asm__ __volatile__ (
        "lock incl %0"
        : "+m" (*ptr)       // Input/output
        :
        : "memory"          // Clobber
    );
}
```

### MSVC Inline Assembly (32-bit only)

```c
int add(int a, int b) {
    __asm {
        mov eax, a
        add eax, b
        // Result in EAX (return value)
    }
}
```

## Exercises

1. Write an assembly function to:
   - Reverse a string in place
   - Find maximum element in array
   - Implement binary search

2. Convert the following C code to assembly:
   ```c
   int factorial(int n) {
       if (n <= 1) return 1;
       return n * factorial(n - 1);
   }
   ```

3. Implement string comparison (strcmp) in assembly.

4. Write a program that:
   - Reads user input
   - Converts string to integer
   - Performs calculation
   - Outputs result

5. Optimize this loop using SIMD instructions:
   ```c
   for (int i = 0; i < 1000; i++) {
       c[i] = a[i] * b[i];
   }
   ```

6. Debug this assembly code (find the bug):
   ```assembly
   mov rcx, 10
   loop_start:
       add rax, [rsi + rcx*8]
       loop loop_start
   ```

7. Calculate machine code for:
   - `mov rax, 0x123456789ABCDEF`
   - `add qword [rbx + 8], 5`

8. Implement memory copy (memcpy) optimized for:
   - Small copies (< 16 bytes)
   - Large aligned copies
   - Unaligned copies

9. Write position-independent code (PIC) for shared library.

10. Create a simple assembler that can assemble basic instructions.

## Summary

This chapter explored assembly language programming and its relationship to machine code:

- Assembly provides human-readable representation of machine instructions
- Each processor architecture has its own assembly language
- Understanding assembly is crucial for system programming and optimization
- Modern processors provide complex addressing modes and SIMD instructions
- Assembly knowledge helps in debugging and reverse engineering
- Inline assembly allows optimization of critical code sections

Assembly language bridges the gap between high-level programming and hardware. While most programming today uses high-level languages, understanding assembly provides invaluable insights into program execution, performance optimization, and system behavior. The next chapter will explore how operating systems use these low-level capabilities to manage hardware resources and provide services to applications.