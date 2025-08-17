# Chapter 1: Number Systems and Counting

## Introduction

Before we can understand computers, we must first understand how they represent and manipulate information. At the most fundamental level, all information in a computer is represented as numbers, and all computation is manipulation of these numbers. This chapter explores how we count, how different number systems work, and why computers use binary.

## 1.1 The Decimal System

Humans typically count in base 10, or decimal. This system uses ten digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9. The position of each digit determines its value through powers of 10.

Consider the number 2,743:
- 2 × 10³ = 2,000
- 7 × 10² = 700
- 4 × 10¹ = 40
- 3 × 10⁰ = 3
- Total: 2,743

This positional notation system, where the position of a digit determines its contribution to the overall value, is fundamental to all number systems we'll explore.

## 1.2 The Binary System

Computers use binary (base 2) because electronic circuits have two stable states: on and off, high voltage and low voltage, or simply 1 and 0. Binary uses only two digits: 0 and 1.

### Why Binary?

Binary is ideal for computers because:
- **Reliability**: Two states are easier to distinguish than ten, reducing errors
- **Simplicity**: Binary logic gates are simpler to design and manufacture
- **Noise immunity**: Digital signals can tolerate more electrical noise
- **Boolean algebra**: Maps directly to logical operations (AND, OR, NOT)

### Understanding Binary

Each position in a binary number represents a power of 2:

The binary number 1101₂ equals:
- 1 × 2³ = 8
- 1 × 2² = 4
- 0 × 2¹ = 0
- 1 × 2⁰ = 1
- Total: 13₁₀

**Quick conversion tip**: To convert decimal to binary, repeatedly divide by 2 and track remainders:
```
13 ÷ 2 = 6 remainder 1
6  ÷ 2 = 3 remainder 0
3  ÷ 2 = 1 remainder 1
1  ÷ 2 = 0 remainder 1
Reading upward: 1101₂
```

### Binary Arithmetic

**Addition:**
```
  1011  (11 in decimal)
+ 0110  (6 in decimal)
------
 10001  (17 in decimal)
```

Rules:
- 0 + 0 = 0
- 0 + 1 = 1
- 1 + 0 = 1
- 1 + 1 = 10 (0 with carry 1)
- 1 + 1 + 1 = 11 (1 with carry 1)

**Subtraction:**
```
  1011  (11 in decimal)
- 0110  (6 in decimal)
------
  0101  (5 in decimal)
```

Rules:
- 0 - 0 = 0
- 1 - 0 = 1
- 1 - 1 = 0
- 0 - 1 = 1 (with borrow from left)

**Multiplication:**
```
  1011
×  101
------
  1011
 0000
1011
------
110111
```

## 1.3 The Hexadecimal System

Hexadecimal (base 16) provides a more compact way to represent binary numbers. It uses sixteen symbols: 0-9 and A-F (where A=10, B=11, C=12, D=13, E=14, F=15).

### Hex to Binary Conversion

Each hexadecimal digit represents exactly 4 binary digits:

| Hex | Binary | Decimal |
|-----|--------|---------|
| 0   | 0000   | 0       |
| 1   | 0001   | 1       |
| 2   | 0010   | 2       |
| 3   | 0011   | 3       |
| 4   | 0100   | 4       |
| 5   | 0101   | 5       |
| 6   | 0110   | 6       |
| 7   | 0111   | 7       |
| 8   | 1000   | 8       |
| 9   | 1001   | 9       |
| A   | 1010   | 10      |
| B   | 1011   | 11      |
| C   | 1100   | 12      |
| D   | 1101   | 13      |
| E   | 1110   | 14      |
| F   | 1111   | 15      |

Example: 2A3F₁₆ = 0010 1010 0011 1111₂

## 1.4 The Octal System

Octal (base 8) uses eight digits: 0-7. Each octal digit represents exactly 3 binary digits. While less common today, octal was historically important in computing.

Example: 725₈ = 111 010 101₂ = 469₁₀

## 1.5 Representing Negative Numbers

Computers need to represent both positive and negative numbers. Three common methods exist:

### Sign and Magnitude

The leftmost bit represents the sign (0 for positive, 1 for negative):
- +5 = 0101
- -5 = 1101

Problem: Two representations for zero (0000 and 1000).

### One's Complement

Negative numbers are formed by inverting all bits:
- +5 = 0101
- -5 = 1010

Problem: Still two representations for zero.

### Two's Complement (Most Common)

Negative numbers are formed by inverting all bits and adding 1:
- +5 = 0101
- -5 = 1010 + 1 = 1011

Advantages:
- Only one representation for zero
- Addition works the same for positive and negative numbers
- The range for n bits is -2^(n-1) to 2^(n-1) - 1

**Quick trick**: To find the two's complement, scan from right to left, copy all bits up to and including the first 1, then invert all remaining bits.

Example with 8 bits:
- +12 = 00001100
- -12: Keep rightmost 100, invert rest → 11110100

**Overflow detection**: In two's complement, overflow occurs when:
- Adding two positive numbers yields a negative result
- Adding two negative numbers yields a positive result

## 1.6 Floating-Point Numbers

Real numbers with fractional parts require special representation. The IEEE 754 standard defines floating-point formats.

### Structure (32-bit single precision)

| Sign | Exponent | Mantissa |
|------|----------|----------|
| 1 bit | 8 bits | 23 bits |

The number is calculated as: (-1)^sign × 1.mantissa × 2^(exponent-127)

Example: 12.375₁₀
1. Convert to binary: 1100.011₂
   - Integer part: 12 = 1100₂
   - Fractional part: 0.375 = 0.011₂
     (0.375 × 2 = 0.75 → 0)
     (0.75 × 2 = 1.5 → 1)
     (0.5 × 2 = 1.0 → 1)
2. Normalize: 1.100011 × 2³
3. Sign: 0 (positive)
4. Exponent: 3 + 127 = 130 = 10000010₂
5. Mantissa: 10001100000000000000000 (drop leading 1, pad with zeros)
6. Result: 0 10000010 10001100000000000000000

### Special Values

- **Zero**: All bits are 0 (note: -0 exists with sign bit = 1)
- **Infinity**: Exponent all 1s, mantissa all 0s
  - +∞: 0 11111111 00000000000000000000000
  - -∞: 1 11111111 00000000000000000000000
- **NaN (Not a Number)**: Exponent all 1s, mantissa non-zero
  - Results from operations like 0/0, ∞-∞, sqrt(-1)
- **Denormalized numbers**: Exponent all 0s, mantissa non-zero
  - Allows representation of very small numbers near zero
  - Value = (-1)^sign × 0.mantissa × 2^(-126)

## 1.7 Binary Coded Decimal (BCD)

BCD represents each decimal digit with its 4-bit binary equivalent:

```
Decimal: 9 3 7
BCD:     1001 0011 0111
```

While less space-efficient than pure binary, BCD simplifies decimal arithmetic and display operations, making it useful in calculators and financial systems.

## 1.8 Character Encoding

Computers must also represent text. Several encoding standards exist:

### ASCII (American Standard Code for Information Interchange)

Uses 7 bits to represent 128 characters:
- 0-31: Control characters
- 32-127: Printable characters
- Example: 'A' = 65₁₀ = 01000001₂

### Extended ASCII

Uses 8 bits for 256 characters, adding accented letters and symbols.

### Unicode

Modern standard supporting all world languages:
- UTF-8: Variable length (1-4 bytes), backward compatible with ASCII
- UTF-16: Variable length (2 or 4 bytes)
- UTF-32: Fixed length (4 bytes)

## 1.9 Data Units

Understanding data measurement is crucial:

- **Bit**: Single binary digit (0 or 1)
- **Nibble**: 4 bits (one hexadecimal digit)
- **Byte**: 8 bits (standard addressable unit)
- **Word**: Processor-dependent (16, 32, or 64 bits typically)

### Storage Units

Traditional (powers of 2):
- Kilobyte (KB): 2¹⁰ = 1,024 bytes
- Megabyte (MB): 2²⁰ = 1,048,576 bytes
- Gigabyte (GB): 2³⁰ bytes
- Terabyte (TB): 2⁴⁰ bytes

Decimal (SI units):
- Kilobyte: 1,000 bytes
- Megabyte: 1,000,000 bytes
- Gigabyte: 1,000,000,000 bytes

To avoid confusion, IEC introduced binary prefixes:
- KiB (kibibyte): 1,024 bytes
- MiB (mebibyte): 1,048,576 bytes
- GiB (gibibyte): 2³⁰ bytes

## 1.10 Practical Applications

### Memory Addressing

A 32-bit system can address 2³² = 4,294,967,296 unique memory locations (4 GB).
A 64-bit system can address 2⁶⁴ locations (16 exabytes theoretically).

### Color Representation

RGB colors often use 24 bits:
- Red: 8 bits (0-255)
- Green: 8 bits (0-255)
- Blue: 8 bits (0-255)
- Total: 16,777,216 possible colors

### Network Addresses

IPv4 addresses use 32 bits, written as four decimal octets:
192.168.1.1 = 11000000.10101000.00000001.00000001

IPv6 addresses use 128 bits, written in hexadecimal:
2001:0db8:85a3:0000:0000:8a2e:0370:7334

## Exercises

1. Convert the following decimal numbers to binary:
   - 42
   - 255
   - 1000
   
   **Solutions:**
   - 42 = 101010₂
   - 255 = 11111111₂
   - 1000 = 1111101000₂

2. Convert the following binary numbers to hexadecimal:
   - 11011110101011011011111011101111
   - 10101010101010101010101010101010

3. Perform the following binary arithmetic:
   - 1011 + 1101
   - 1100 × 101
   - 10110 - 1011

4. Represent -42 in:
   - Sign and magnitude (8 bits)
   - One's complement (8 bits)
   - Two's complement (8 bits)

5. Convert 3.14159 to IEEE 754 single-precision format.

6. How many bits are needed to represent:
   - All uppercase English letters?
   - All printable ASCII characters?
   - One million distinct values?

7. A computer has 16 GB of RAM. How many addressable bytes is this? Express your answer in:
   - Decimal notation
   - Binary notation (powers of 2)
   - Hexadecimal notation

8. Write a simple algorithm to convert any decimal number to binary.

   **Solution approach:**
   ```python
   def decimal_to_binary(n):
       if n == 0:
           return '0'
       binary = ''
       while n > 0:
           binary = str(n % 2) + binary
           n = n // 2
       return binary
   ```

9. Explain why computers use binary instead of decimal for internal representation.

   **Key points to consider:**
   - Electronic components naturally have two stable states
   - Binary arithmetic circuits are simpler and faster
   - Error detection and correction is more reliable
   - Direct correspondence with Boolean logic

10. Research and explain why floating-point arithmetic can lead to rounding errors in computers.

   **Key concepts:**
   - Limited precision (finite bits)
   - Not all decimal fractions have exact binary representations
   - Example: 0.1₁₀ = 0.00011001100110011...₂ (repeating)
   - Accumulation of small errors in calculations

## Summary

This chapter introduced the fundamental concept of number systems and how computers represent data. We explored:

- Positional notation and different number bases
- Binary arithmetic and its importance in computing
- Methods for representing negative numbers and real numbers
- Character encoding standards
- Data units and measurements

Understanding these number systems forms the foundation for comprehending how computers process information at the lowest level. In the next chapter, we'll build on this knowledge to explore Boolean logic and how simple binary operations can be combined to create complex computational systems.