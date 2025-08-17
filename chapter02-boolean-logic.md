# Chapter 2: Boolean Logic and Digital Circuits

## Introduction

Boolean logic forms the mathematical foundation of digital computing. Named after mathematician George Boole, this algebra deals with binary values and logical operations. Every computation a computer performs ultimately reduces to billions of simple Boolean operations executed by transistors. This chapter explores Boolean algebra, logic gates, and how these fundamental concepts combine to create the complex digital systems that power modern computers.

### Chapter Objectives
By the end of this chapter, you will understand:
- How Boolean algebra provides the mathematical foundation for digital circuits
- How logic gates implement Boolean operations in hardware
- How to simplify Boolean expressions for efficient circuit design
- How combinational and sequential circuits form the building blocks of processors
- The relationship between Boolean logic and the binary system from Chapter 1

## 2.1 Boolean Algebra Fundamentals

Boolean algebra operates on binary variables that can only have two values: true (1) or false (0). Unlike regular algebra with infinite numbers, Boolean algebra's simplicity makes it perfect for digital circuits.

### Basic Boolean Operations

The three fundamental operations are:

**NOT (Negation)**
- Symbol: ¬ or ' or overbar
- Function: Inverts the input value
- Real-world analogy: A light switch that turns on when off and vice versa
- Truth table:
```
A | ¬A
--|---
0 | 1
1 | 0
```
- Circuit behavior: Output is HIGH when input is LOW

**AND (Conjunction)**
- Symbol: ∧ or · or &
- Function: Output is true only when ALL inputs are true
- Real-world analogy: A security system requiring both a keycard AND a PIN
- Truth table:
```
A | B | A∧B
--|---|----
0 | 0 | 0
0 | 1 | 0
1 | 0 | 0
1 | 1 | 1
```
- Circuit behavior: Output is HIGH only when both inputs are HIGH

**OR (Disjunction)**
- Symbol: ∨ or +
- Function: Output is true when AT LEAST ONE input is true
- Real-world analogy: A doorbell that rings when either the front OR back button is pressed
- Truth table:
```
A | B | A∨B
--|---|----
0 | 0 | 0
0 | 1 | 1
1 | 0 | 1
1 | 1 | 1
```
- Circuit behavior: Output is HIGH when any input is HIGH

### Derived Operations

**XOR (Exclusive OR)**
- Symbol: ⊕
- True when inputs differ
- Truth table:
```
A | B | A⊕B
--|---|----
0 | 0 | 0
0 | 1 | 1
1 | 0 | 1
1 | 1 | 0
```

**NAND (NOT AND)**
- Symbol: ↑
- Truth table:
```
A | B | A↑B
--|---|----
0 | 0 | 1
0 | 1 | 1
1 | 0 | 1
1 | 1 | 0
```

**NOR (NOT OR)**
- Symbol: ↓
- Truth table:
```
A | B | A↓B
--|---|----
0 | 0 | 1
0 | 1 | 0
1 | 0 | 0
1 | 1 | 0
```

## 2.2 Boolean Laws and Theorems

### Identity Laws
- A ∧ 1 = A
- A ∨ 0 = A

### Null Laws
- A ∧ 0 = 0
- A ∨ 1 = 1

### Idempotent Laws
- A ∧ A = A
- A ∨ A = A

### Complement Laws
- A ∧ ¬A = 0
- A ∨ ¬A = 1
- ¬(¬A) = A

### Commutative Laws
- A ∧ B = B ∧ A
- A ∨ B = B ∨ A

### Associative Laws
- (A ∧ B) ∧ C = A ∧ (B ∧ C)
- (A ∨ B) ∨ C = A ∨ (B ∨ C)

### Distributive Laws
- A ∧ (B ∨ C) = (A ∧ B) ∨ (A ∧ C)
- A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)

### De Morgan's Laws
- ¬(A ∧ B) = ¬A ∨ ¬B
- ¬(A ∨ B) = ¬A ∧ ¬B

### Absorption Laws
- A ∨ (A ∧ B) = A
- A ∧ (A ∨ B) = A

## 2.3 Boolean Expression Simplification

Simplifying Boolean expressions reduces circuit complexity, cost, and power consumption.

### Example: Simplify F = A·B + A·¬B

Using distributive law:
```
F = A·B + A·¬B
F = A·(B + ¬B)    [Distributive]
F = A·1           [Complement]
F = A             [Identity]
```

**Practical interpretation**: This expression says "A AND B, OR A AND NOT B", which simplifies to just "A" regardless of B's value. This insight can reduce a circuit from 5 gates to just a wire!

### Karnaugh Maps (K-Maps)

K-Maps provide a visual method for simplifying Boolean expressions for up to 4-6 variables. They work by arranging truth table values in a grid where adjacent cells differ by only one variable (Gray code ordering), making patterns easy to spot.

Example for F = ∑(0,1,2,6):

```
    AB
    00  01  11  10
CD +----+----+----+----+
00 | 1  | 1  | 0  | 1  |
   +----+----+----+----+
01 | 0  | 0  | 0  | 0  |
   +----+----+----+----+
11 | 0  | 0  | 0  | 0  |
   +----+----+----+----+
10 | 0  | 1  | 0  | 0  |
   +----+----+----+----+
```

Groups of 1s give simplified terms:
- Group (0,1,2): ¬B
- Group (2,6): ¬A·C·¬D
- Result: F = ¬B + ¬A·C·¬D

## 2.4 Logic Gates

Logic gates are physical implementations of Boolean operations using transistors.

### Basic Gates

**NOT Gate (Inverter)**
```
    A ---[>o]--- ¬A
```

**AND Gate**
```
    A ---\
          [D]--- A∧B
    B ---/
```

**OR Gate**
```
    A ---\
          [≥1]--- A∨B
    B ---/
```

### Universal Gates

NAND and NOR are called universal gates because any Boolean function can be implemented using only NAND gates or only NOR gates.

**Implementing NOT with NAND:**
```
A ---\
      [NAND]--- ¬A
A ---/
```

**Implementing AND with NAND:**
```
A ---\
      [NAND]---[NAND]--- A∧B
B ---/
```

## 2.5 Combinational Circuits

Combinational circuits produce outputs that depend only on current inputs, with no memory of previous states.

### Half Adder

Adds two single bits:
- Sum = A ⊕ B (gives the least significant bit)
- Carry = A ∧ B (gives the overflow to next position)

```
Inputs: A, B
Outputs: Sum, Carry

A ----+----[XOR]---- Sum
      |
      +----[AND]---- Carry
      |
B ----+
```

**Truth Table:**
```
A | B | Sum | Carry
--|---|-----|------
0 | 0 |  0  |  0
0 | 1 |  1  |  0
1 | 0 |  1  |  0
1 | 1 |  0  |  1    (1+1=10 in binary)
```

### Full Adder

Adds three bits (two inputs plus carry-in from previous position):
- Sum = A ⊕ B ⊕ Cin
- Cout = (A ∧ B) ∨ (Cin ∧ (A ⊕ B))

**Why we need it**: When adding multi-bit numbers, we need to handle the carry from each bit position. The full adder is the building block for multi-bit addition.

### Ripple Carry Adder

Chains full adders to add multi-bit numbers:

```
A3 B3 C2    A2 B2 C1    A1 B1 C0    A0 B0 0
  ↓ ↓ ↓       ↓ ↓ ↓       ↓ ↓ ↓       ↓ ↓ ↓
[Full Adder][Full Adder][Full Adder][Full Adder]
     ↓           ↓           ↓           ↓
    S3          S2          S1          S0
```

### Multiplexer (MUX)

Selects one of multiple inputs based on control signals.

4-to-1 MUX:
```
Inputs: I0, I1, I2, I3
Select: S1, S0
Output: Y = I0·¬S1·¬S0 + I1·¬S1·S0 + I2·S1·¬S0 + I3·S1·S0
```

### Decoder

Activates one output line based on binary input.

2-to-4 Decoder:
```
Inputs: A1, A0
Outputs: D0 = ¬A1·¬A0
         D1 = ¬A1·A0
         D2 = A1·¬A0
         D3 = A1·A0
```

### Encoder

Opposite of decoder - converts active input line to binary code.

### Arithmetic Logic Unit (ALU)

The ALU is the computational heart of a CPU, performing arithmetic and logical operations based on control signals:

```
Inputs: A[n], B[n], Operation[k]
Output: Result[n], Flags (Zero, Negative, Carry, Overflow)

Operations:
- ADD: Result = A + B         (arithmetic addition)
- SUB: Result = A - B         (arithmetic subtraction)
- AND: Result = A ∧ B         (bitwise AND)
- OR:  Result = A ∨ B         (bitwise OR)
- XOR: Result = A ⊕ B         (bitwise XOR)
- NOT: Result = ¬A            (bitwise NOT)
- SHL: Result = A << 1        (shift left = multiply by 2)
- SHR: Result = A >> 1        (shift right = divide by 2)
```

**Flags explained:**
- Zero: Set when result equals zero
- Negative: Set when result is negative (MSB = 1 in two's complement)
- Carry: Set when unsigned operation overflows
- Overflow: Set when signed operation overflows

## 2.6 Sequential Circuits

Sequential circuits have memory - outputs depend on both current inputs and previous states.

### SR Latch (Set-Reset)

Basic memory element using cross-coupled NOR gates:

```
S ---[NOR]----+---- Q
        ↑     |
        +-----+
        |     ↓
R ---[NOR]----+---- ¬Q
```

Truth table:
```
S | R | Q(next)
--|---|--------
0 | 0 | Q (hold)
0 | 1 | 0 (reset)
1 | 0 | 1 (set)
1 | 1 | Invalid
```

### D Latch (Data Latch)

Eliminates invalid state of SR latch:

```
D ----+----[AND]---- S
      |              |
      +--[NOT]--[AND]---- R
      |         |
Enable --------+
```

### D Flip-Flop

Edge-triggered version of D latch:

```
      Master Latch    Slave Latch
D ----[D Latch]------[D Latch]---- Q
         |              |
Clock ---+---[NOT]------+
```

Captures input on clock edge (rising or falling).

### JK Flip-Flop

Enhanced SR flip-flop that toggles when J=K=1:

```
J | K | Q(next)
--|---|--------
0 | 0 | Q (hold)
0 | 1 | 0 (reset)
1 | 0 | 1 (set)
1 | 1 | ¬Q (toggle)
```

### T Flip-Flop (Toggle)

Simplified JK flip-flop with J=K=T:

```
T | Q(next)
--|--------
0 | Q (hold)
1 | ¬Q (toggle)
```

## 2.7 Registers and Counters

### Parallel Register

Stores n-bit values:

```
D0 ---[D-FF]--- Q0
D1 ---[D-FF]--- Q1
D2 ---[D-FF]--- Q2
D3 ---[D-FF]--- Q3
      |
Clock-+---- (common to all)
```

### Shift Register

Shifts bits left or right:

```
Serial In ---[D-FF]---[D-FF]---[D-FF]---[D-FF]--- Serial Out
                |       |       |       |
              Q0      Q1      Q2      Q3
```

Applications:
- Serial-to-parallel conversion
- Multiplication/division by 2
- Delay lines
- Pseudo-random number generation

### Binary Counter

4-bit ripple counter using T flip-flops:

```
     +1----[T-FF]----[T-FF]----[T-FF]----[T-FF]
Clock       Q0        Q1        Q2        Q3
            LSB                           MSB
```

Counts: 0000, 0001, 0010, ..., 1111, 0000, ...

### Ring Counter

Shift register with output fed back to input:

```
Initial: 1000
Clock 1: 0100
Clock 2: 0010
Clock 3: 0001
Clock 4: 1000 (repeats)
```

## 2.8 Memory Elements

### Static RAM (SRAM) Cell

6-transistor cell holds one bit:

```
     +--[Inverter]--+
     |              |
     +--[Inverter]--+
     |              |
WordLine--[Pass]----BitLine
     |              |
WordLine--[Pass]----¬BitLine
```

Features:
- Fast access
- No refresh needed
- More transistors per bit
- Higher power consumption

### Dynamic RAM (DRAM) Cell

1-transistor, 1-capacitor cell:

```
BitLine
   |
[Transistor]
   |
WordLine---+
   |
[Capacitor]
   |
Ground
```

Features:
- Slower than SRAM
- Requires periodic refresh
- Higher density
- Lower cost per bit

### ROM (Read-Only Memory)

Permanent storage programmed during manufacture:

```
Address Decoder
      |
   [Decoder]
      |
+-----+-----+-----+
|     |     |     |
[Bit] [Bit] [Bit] [Bit]
|     |     |     |
Data Lines
```

Types:
- ROM: Factory programmed
- PROM: One-time programmable
- EPROM: UV-erasable
- EEPROM: Electrically erasable
- Flash: Block-erasable EEPROM

## 2.9 Programmable Logic Devices

### Programmable Logic Array (PLA)

Implements sum-of-products expressions:

```
Inputs ---[Programmable AND array]---[Programmable OR array]--- Outputs
```

### Programmable Array Logic (PAL)

Fixed OR array, programmable AND array:

```
Inputs ---[Programmable AND array]---[Fixed OR array]--- Outputs
```

### Field-Programmable Gate Array (FPGA)

Configurable logic blocks connected by programmable interconnects:

Components:
- Configurable Logic Blocks (CLBs)
- Input/Output Blocks (IOBs)
- Programmable interconnects
- Block RAM
- DSP slices
- Clock management

## 2.10 Digital Design Principles

### Design Flow

1. **Specification**: Define requirements
2. **Truth Table**: List all input/output combinations
3. **Boolean Expression**: Derive from truth table
4. **Simplification**: Use Boolean algebra or K-maps
5. **Implementation**: Choose gates/components
6. **Verification**: Test all cases

### Timing Considerations

**Propagation Delay**: Time for signal to pass through gate
```
tpd = time(output change) - time(input change)
```

**Setup Time**: Minimum time input must be stable before clock edge

**Hold Time**: Minimum time input must remain stable after clock edge

**Clock Skew**: Variation in clock arrival times

**Critical Path**: Longest delay path determining maximum clock frequency
```
fmax = 1 / (tcritical + tsetup + tclock_skew)
```

### Hazards

**Static Hazard**: Momentary incorrect output during transition
- Solution: Add redundant terms

**Dynamic Hazard**: Multiple transitions for single input change
- Solution: Careful timing analysis

**Race Condition**: Output depends on signal arrival order
- Solution: Synchronous design

## 2.11 Modern Digital Design

### Hardware Description Languages (HDL)

**Verilog Example:**
```verilog
module full_adder(
    input a, b, cin,
    output sum, cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (cin & (a ^ b));
endmodule
```

**VHDL Example:**
```vhdl
entity full_adder is
    port(a, b, cin : in std_logic;
         sum, cout : out std_logic);
end full_adder;

architecture behavioral of full_adder is
begin
    sum <= a xor b xor cin;
    cout <= (a and b) or (cin and (a xor b));
end behavioral;
```

### Design Optimization

**Area Optimization**: Minimize gate count
- Share common sub-expressions
- Use multiplexers for complex functions

**Speed Optimization**: Minimize delay
- Parallel processing
- Pipelining
- Carry lookahead for adders

**Power Optimization**: Minimize switching
- Clock gating
- Power gating
- Dynamic voltage scaling

## Exercises

1. Prove De Morgan's Laws using truth tables.

2. Simplify the following Boolean expressions:
   - F = A·B·C + A·B·¬C + A·¬B·C
   - G = (A+B)·(A+C)·(B+C+D)

3. Design a 2-bit comparator that outputs:
   - A > B
   - A = B
   - A < B

4. Implement a full adder using only NAND gates.

5. Design a 3-to-8 decoder using basic gates.

6. Create a 4-bit synchronous counter with parallel load capability.

7. Design a circuit that detects if a 4-bit number is prime.

8. Implement a 4-bit arithmetic right shift register.

9. Design a simple 4-word × 4-bit RAM using D flip-flops.

10. Calculate the maximum clock frequency for a circuit with:
    - Critical path delay: 15 ns
    - Setup time: 2 ns
    - Clock skew: 1 ns

11. Design a Moore state machine that detects the sequence "1101" in a serial bit stream.

12. Write Verilog code for a 4-bit ripple carry adder.

## Summary

This chapter explored the mathematical and physical foundations of digital computing:

- Boolean algebra provides the mathematical framework for digital logic
- Logic gates implement Boolean operations in hardware
- Combinational circuits perform calculations without memory
- Sequential circuits add memory and state to digital systems
- Memory elements store data in various forms
- Programmable logic enables flexible hardware design
- Modern design uses HDLs and sophisticated optimization techniques

These concepts form the building blocks for all digital systems, from simple calculators to supercomputers. In the next chapter, we'll see how these components combine to create complete computer architectures, exploring how processors execute instructions and manage data.