# Chapter 9: Programming Fundamentals

## Introduction

Programming is the art and science of instructing computers to solve problems. While previous chapters explored how computers work at the hardware and system level, this chapter introduces the fundamental concepts of programming that transcend any particular language. We'll explore programming paradigms, core concepts, and the principles that guide the creation of correct, efficient, and maintainable software.

### Key Programming Principles

1. **Abstraction**: Hide complexity behind simple interfaces
2. **Modularity**: Break problems into manageable pieces
3. **DRY (Don't Repeat Yourself)**: Avoid code duplication
4. **KISS (Keep It Simple, Stupid)**: Favor simplicity over cleverness
5. **YAGNI (You Aren't Gonna Need It)**: Don't over-engineer

## 9.1 Programming Paradigms

### Imperative Programming

Programs as sequences of commands that change state:

```c
// Imperative style - step by step instructions
// "How" to compute factorial
int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;  // Mutate state
    }
    return result;
}

// Characteristics:
// - Explicit control flow
// - Mutable state
// - Sequential execution
// - Close to machine architecture

// Procedural programming (imperative subset)
struct Account {
    int balance;
    char owner[100];
};

void deposit(struct Account* acc, int amount) {
    acc->balance += amount;  // Direct state modification
}

void withdraw(struct Account* acc, int amount) {
    if (acc->balance >= amount) {
        acc->balance -= amount;
    }
}
```

### Object-Oriented Programming (OOP)

Encapsulation of data and behavior:

```cpp
class BankAccount {
private:
    double balance;
    std::string owner;
    std::vector<Transaction> history;

public:
    BankAccount(std::string owner, double initial = 0) 
        : owner(owner), balance(initial) {}
    
    // Encapsulation - controlled access
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            history.push_back(Transaction(DEPOSIT, amount));
        }
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && balance >= amount) {
            balance -= amount;
            history.push_back(Transaction(WITHDRAWAL, amount));
            return true;
        }
        return false;
    }
    
    double getBalance() const { return balance; }
};

// Inheritance
class SavingsAccount : public BankAccount {
private:
    double interestRate;
    int withdrawalsThisMonth;
    
public:
    SavingsAccount(std::string owner, double rate) 
        : BankAccount(owner), interestRate(rate), withdrawalsThisMonth(0) {}
    
    // Polymorphism - override behavior
    bool withdraw(double amount) override {
        if (withdrawalsThisMonth >= 6) {
            return false;  // Monthly limit exceeded
        }
        if (BankAccount::withdraw(amount)) {
            withdrawalsThisMonth++;
            return true;
        }
        return false;
    }
    
    void applyInterest() {
        deposit(getBalance() * interestRate);
    }
};
```

### Functional Programming

Computation as evaluation of mathematical functions:

```haskell
-- Pure functional style (Haskell)
-- "What" factorial is, not "how" to compute it
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Characteristics:
-- - No side effects (pure functions)
-- - Immutable data
-- - Function composition
-- - Declarative style

-- Higher-order functions
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs

filter :: (a -> Bool) -> [a] -> [a]
filter _ [] = []
filter p (x:xs)
    | p x       = x : filter p xs
    | otherwise = filter p xs

-- Function composition
compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g = \x -> f (g x)
```

Functional concepts in modern languages:

```python
# Python functional features
from functools import reduce

# Lambda functions (anonymous functions)
square = lambda x: x ** 2

# Map, filter, reduce - functional trinity
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))  # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]
sum_all = reduce(lambda x, y: x + y, numbers)  # 15

# List comprehensions (functional-inspired)
squared_evens = [x ** 2 for x in numbers if x % 2 == 0]

# Immutability focus
def add_element(lst, elem):
    return lst + [elem]  # Return new list, don't mutate

# Closure
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

times_two = make_multiplier(2)
print(times_two(5))  # 10
```

### Logic Programming

Declaring facts and rules:

```prolog
% Prolog example
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

% Query: ?- grandparent(tom, X).
% Results: X = ann; X = pat; X = jim
```

## 9.2 Variables and Data Types

### Type Systems

**Static Typing** (compile-time checking):

Advantages:
- Catch errors early (at compile time)
- Better performance (no runtime type checks)
- Better IDE support (autocomplete, refactoring)
- Self-documenting code

```java
// Java - statically typed
int count = 42;
String name = "Alice";
// count = "Bob";  // Compile error - type mismatch!

List<Integer> numbers = new ArrayList<>();
numbers.add(10);
// numbers.add("string");  // Compile error!
```

**Dynamic Typing** (runtime checking):

Advantages:
- More flexible
- Faster development
- Less verbose
- Duck typing ("If it quacks like a duck...")

```python
# Python - dynamically typed
count = 42
count = "Bob"  # Legal, type changes at runtime

numbers = [1, 2, 3]
numbers.append("four")  # Legal, mixed types
```

**Type Inference**:

```rust
// Rust - static with inference
let x = 42;  // Compiler infers i32
let y = 3.14;  // Compiler infers f64

let mut vec = Vec::new();
vec.push(1);  // Now compiler knows it's Vec<i32>
```

### Primitive vs Reference Types

```java
// Primitive types - stored directly
int a = 5;
int b = a;  // Copy value
b = 10;     // a is still 5

// Reference types - store references
int[] arr1 = {1, 2, 3};
int[] arr2 = arr1;  // Copy reference
arr2[0] = 99;        // arr1[0] is now 99 too!

// Boxing and unboxing
Integer boxed = 42;  // Auto-boxing
int primitive = boxed;  // Auto-unboxing
```

### Memory Management

**Manual Memory Management**:

```c
// C - explicit allocation/deallocation
int* array = malloc(10 * sizeof(int));
if (array == NULL) {
    // Handle allocation failure
    return -1;
}

// Use array
for (int i = 0; i < 10; i++) {
    array[i] = i * i;
}

free(array);  // Must remember to free!
// array[0] = 5;  // Use-after-free bug!
```

**Automatic Memory Management (Garbage Collection)**:

```java
// Java - automatic garbage collection
public List<String> processData() {
    List<String> temp = new ArrayList<>();
    // ... use temp
    return temp;
    // temp eligible for garbage collection after method returns
}
```

**Ownership-Based Memory Management**:

```rust
// Rust - ownership system
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;  // s1 moved to s2
    // println!("{}", s1);  // Compile error! s1 no longer valid
    
    let s3 = String::from("world");
    let s4 = s3.clone();  // Deep copy
    println!("{} {}", s3, s4);  // Both valid
}

// Borrowing
fn calculate_length(s: &String) -> usize {
    s.len()  // Can read, but not modify
}

fn main() {
    let s = String::from("hello");
    let len = calculate_length(&s);  // Borrow s
    println!("Length of '{}' is {}", s, len);  // s still valid
}
```

## 9.3 Control Structures

### Conditional Execution

```c
// If-else chains
int categorize(int score) {
    if (score >= 90) {
        return GRADE_A;
    } else if (score >= 80) {
        return GRADE_B;
    } else if (score >= 70) {
        return GRADE_C;
    } else if (score >= 60) {
        return GRADE_D;
    } else {
        return GRADE_F;
    }
}

// Switch/case
char* day_name(int day) {
    switch (day) {
        case 1: return "Monday";
        case 2: return "Tuesday";
        case 3: return "Wednesday";
        case 4: return "Thursday";
        case 5: return "Friday";
        case 6: return "Saturday";
        case 7: return "Sunday";
        default: return "Invalid";
    }
}

// Pattern matching (more powerful)
// Rust example
match value {
    0 => println!("zero"),
    1..=9 => println!("single digit"),
    10 | 20 | 30 => println!("round number"),
    _ => println!("something else"),
}
```

### Iteration

```python
# Different iteration patterns

# Count-controlled
for i in range(10):
    print(i)

# Collection iteration
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# Enumeration
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# While loops
count = 0
while count < 10:
    print(count)
    count += 1

# Do-while equivalent
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break

# List comprehensions
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Generator expressions (lazy evaluation)
sum_squares = sum(x**2 for x in range(1000000))
```

### Exception Handling

```java
public class FileProcessor {
    public String readFile(String filename) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(filename));
            StringBuilder content = new StringBuilder();
            String line;
            
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
            
            return content.toString();
            
        } catch (FileNotFoundException e) {
            System.err.println("File not found: " + filename);
            return null;
            
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return null;
            
        } finally {
            // Always executed
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    // Log error
                }
            }
        }
    }
    
    // Try-with-resources (automatic cleanup)
    public String readFileModern(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            return reader.lines()
                        .collect(Collectors.joining("\n"));
        }
    }
}
```

## 9.4 Functions and Procedures

### Function Design Principles

```python
# Pure functions - no side effects
def pure_add(a, b):
    return a + b  # Only depends on inputs, no external state

# Impure function - has side effects
global_counter = 0

def impure_add(a, b):
    global global_counter
    global_counter += 1  # Side effect!
    print(f"Called {global_counter} times")  # Side effect!
    return a + b

# Single Responsibility Principle
def calculate_total_bad(items):
    # Does too many things!
    total = 0
    for item in items:
        if item.is_taxable:
            total += item.price * 1.08
        else:
            total += item.price
    
    # Also sends email? Bad!
    send_email("admin@example.com", f"Total: {total}")
    
    # Also logs? Bad!
    with open("log.txt", "a") as f:
        f.write(f"Calculated total: {total}\n")
    
    return total

# Better: separate concerns
def calculate_subtotal(items):
    return sum(item.price for item in items)

def calculate_tax(items, tax_rate=0.08):
    taxable_total = sum(item.price for item in items if item.is_taxable)
    return taxable_total * tax_rate

def calculate_total(items, tax_rate=0.08):
    return calculate_subtotal(items) + calculate_tax(items, tax_rate)
```

### Parameter Passing

```cpp
// Pass by value
void increment_value(int x) {
    x++;  // Only changes local copy
}

// Pass by reference
void increment_ref(int& x) {
    x++;  // Changes original
}

// Pass by pointer
void increment_ptr(int* x) {
    (*x)++;  // Changes value at address
}

// Const correctness
void process_data(const std::vector<int>& data) {
    // Can read but not modify data
    // Efficient: no copy, safe: no modification
}

// Default parameters
double calculate_interest(double principal, 
                        double rate = 0.05, 
                        int years = 1) {
    return principal * pow(1 + rate, years);
}

// Variable arguments
#include <stdarg.h>

int sum(int count, ...) {
    va_list args;
    va_start(args, count);
    
    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }
    
    va_end(args);
    return total;
}
```

### Recursion

```python
# Direct recursion
def factorial(n):
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# Tail recursion (optimizable)
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)

# Mutual recursion
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)

# Tree recursion
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Memoization for efficiency
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
```

## 9.5 Data Structures Basics

### Arrays and Lists

```c
// Fixed-size arrays
int numbers[10];  // Stack allocation
int* dynamic = malloc(10 * sizeof(int));  // Heap allocation

// Multidimensional arrays
int matrix[3][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

// Dynamic arrays (vector in C++)
#include <vector>
std::vector<int> vec;
vec.push_back(10);  // Grows automatically
vec.push_back(20);
```

### Strings

```python
# String operations
text = "Hello, World!"

# Indexing and slicing
first_char = text[0]  # 'H'
last_char = text[-1]  # '!'
substring = text[7:12]  # 'World'

# String methods
upper = text.upper()  # "HELLO, WORLD!"
lower = text.lower()  # "hello, world!"
parts = text.split(", ")  # ["Hello", "World!"]
joined = "-".join(parts)  # "Hello-World!"

# String formatting
name = "Alice"
age = 30
formatted = f"{name} is {age} years old"  # f-strings
template = "{} is {} years old".format(name, age)
old_style = "%s is %d years old" % (name, age)

# String immutability
s = "hello"
# s[0] = 'H'  # Error! Strings are immutable
s = 'H' + s[1:]  # Create new string instead
```

### Records and Structures

```c
// C structures
struct Point {
    double x;
    double y;
};

struct Rectangle {
    struct Point top_left;
    struct Point bottom_right;
};

double area(struct Rectangle r) {
    double width = r.bottom_right.x - r.top_left.x;
    double height = r.bottom_right.y - r.top_left.y;
    return width * height;
}

// Unions (overlapping storage)
union Value {
    int as_int;
    float as_float;
    char as_char[4];
};

// Bit fields
struct Flags {
    unsigned int is_ready : 1;
    unsigned int is_error : 1;
    unsigned int priority : 3;
    unsigned int reserved : 27;
};
```

## 9.6 Modular Programming

### Modules and Namespaces

```python
# Python modules
# math_utils.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def calculate(self, expression):
        return eval(expression)

# main.py
import math_utils
result = math_utils.add(5, 3)

from math_utils import multiply
result = multiply(4, 5)

from math_utils import Calculator as Calc
calc = Calc()
```

C++ namespaces:

```cpp
namespace Graphics {
    class Point {
        double x, y;
    public:
        Point(double x, double y) : x(x), y(y) {}
    };
    
    class Line {
        Point start, end;
    public:
        Line(Point s, Point e) : start(s), end(e) {}
    };
}

namespace Math {
    class Point {
        double coords[3];  // 3D point
    public:
        Point(double x, double y, double z) {
            coords[0] = x;
            coords[1] = y;
            coords[2] = z;
        }
    };
}

// Usage
Graphics::Point p2d(1.0, 2.0);
Math::Point p3d(1.0, 2.0, 3.0);

using namespace Graphics;  // Import all
Point p(5.0, 10.0);  // Graphics::Point
```

### Information Hiding and Encapsulation

```java
public class BankAccount {
    // Private implementation details
    private double balance;
    private List<Transaction> transactions;
    private AccountType type;
    
    // Public interface
    public void deposit(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Amount must be positive");
        }
        balance += amount;
        recordTransaction(TransactionType.DEPOSIT, amount);
    }
    
    public double getBalance() {
        return balance;  // Read-only access
    }
    
    // Private helper methods
    private void recordTransaction(TransactionType type, double amount) {
        transactions.add(new Transaction(type, amount, new Date()));
    }
    
    // Package-private for testing
    void resetForTesting() {
        balance = 0;
        transactions.clear();
    }
}
```

## 9.7 Generic Programming

### Templates and Generics

```cpp
// C++ templates
template<typename T>
class Stack {
private:
    std::vector<T> elements;
    
public:
    void push(const T& element) {
        elements.push_back(element);
    }
    
    T pop() {
        if (elements.empty()) {
            throw std::runtime_error("Stack is empty");
        }
        T top = elements.back();
        elements.pop_back();
        return top;
    }
    
    bool empty() const {
        return elements.empty();
    }
};

// Template functions
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Template specialization
template<>
const char* max<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

// Usage
Stack<int> intStack;
intStack.push(42);

Stack<std::string> stringStack;
stringStack.push("hello");
```

Java generics:

```java
// Generic class
public class Pair<T, U> {
    private T first;
    private U second;
    
    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }
    
    public T getFirst() { return first; }
    public U getSecond() { return second; }
}

// Bounded type parameters
public class NumberBox<T extends Number> {
    private T value;
    
    public void setValue(T value) {
        this.value = value;
    }
    
    public double getDoubleValue() {
        return value.doubleValue();  // Available because T extends Number
    }
}

// Wildcards
public void printList(List<?> list) {
    for (Object item : list) {
        System.out.println(item);
    }
}

public void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}
```

## 9.8 Programming Best Practices

### Code Style and Readability

```python
# Good naming conventions
def calculate_compound_interest(principal, annual_rate, years):
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount
        annual_rate: Interest rate (as decimal, e.g., 0.05 for 5%)
        years: Number of years
    
    Returns:
        Final amount after compound interest
    """
    return principal * (1 + annual_rate) ** years

# Bad naming
def calc(p, r, t):  # Unclear what this does
    return p * (1 + r) ** t

# Clear control flow
def process_user_input(input_string):
    # Early return for invalid input
    if not input_string:
        return None
    
    # Clean, linear flow
    cleaned = input_string.strip().lower()
    validated = validate_format(cleaned)
    
    if not validated:
        return None
    
    return transform_data(validated)

# Avoid deep nesting
# Bad
def process_bad(data):
    if data:
        if data.is_valid():
            if data.type == "A":
                if data.value > 0:
                    return data.value * 2
    return None

# Good
def process_good(data):
    if not data or not data.is_valid():
        return None
    
    if data.type != "A" or data.value <= 0:
        return None
    
    return data.value * 2
```

### Testing

```python
import unittest

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def test_add_negative_numbers(self):
        self.assertEqual(self.calc.add(-2, -3), -5)
    
    def test_divide_normal(self):
        self.assertAlmostEqual(self.calc.divide(10, 3), 3.333333, places=5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

# Test-Driven Development (TDD) cycle:
# 1. Write failing test
# 2. Write minimal code to pass
# 3. Refactor
```

### Debugging Techniques

```c
#include <stdio.h>
#include <assert.h>

// Assertions for invariants
void process_array(int* arr, int size) {
    assert(arr != NULL);  // Precondition
    assert(size > 0);     // Precondition
    
    // Process array
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    
    assert(sum >= 0);  // Postcondition (if we expect positive sum)
}

// Debug output
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) \
        fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt "\n", \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) // No-op in release
#endif

void complex_algorithm(int n) {
    DEBUG_PRINT("Starting with n=%d", n);
    
    for (int i = 0; i < n; i++) {
        DEBUG_PRINT("Iteration %d", i);
        // Algorithm steps
    }
    
    DEBUG_PRINT("Completed");
}
```

## 9.9 Common Programming Patterns

### Iterator Pattern

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self, root):
        self.root = root
    
    def __iter__(self):
        return self.inorder_iterator(self.root)
    
    def inorder_iterator(self, node):
        if node:
            yield from self.inorder_iterator(node.left)
            yield node.value
            yield from self.inorder_iterator(node.right)

# Usage
tree = BinaryTree(
    TreeNode(2,
        TreeNode(1),
        TreeNode(3))
)

for value in tree:
    print(value)  # Prints 1, 2, 3
```

### Factory Pattern

```java
// Product interface
interface Animal {
    void makeSound();
}

// Concrete products
class Dog implements Animal {
    public void makeSound() {
        System.out.println("Woof!");
    }
}

class Cat implements Animal {
    public void makeSound() {
        System.out.println("Meow!");
    }
}

// Factory
class AnimalFactory {
    public static Animal createAnimal(String type) {
        switch (type.toLowerCase()) {
            case "dog":
                return new Dog();
            case "cat":
                return new Cat();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + type);
        }
    }
}

// Usage
Animal pet = AnimalFactory.createAnimal("dog");
pet.makeSound();  // Woof!
```

### Observer Pattern

```python
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()

class Observer:
    def update(self, state):
        pass

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} received update: {state}")

# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer1")
observer2 = ConcreteObserver("Observer2")

subject.attach(observer1)
subject.attach(observer2)

subject.set_state("New State")
# Output:
# Observer1 received update: New State
# Observer2 received update: New State
```

## 9.10 Performance Considerations

### Time and Space Complexity

```python
# O(1) - Constant time
def get_first(lst):
    return lst[0] if lst else None

# O(log n) - Logarithmic time
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# O(n) - Linear time
def find_max(arr):
    if not arr:
        return None
    
    max_val = arr[0]
    for val in arr[1:]:
        if val > max_val:
            max_val = val
    return max_val

# O(n log n) - Linearithmic time
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

# O(n²) - Quadratic time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# O(2ⁿ) - Exponential time
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```

### Optimization Techniques

```c
// Loop optimization
// Unoptimized
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        result += matrix[i][j] * expensive_function();
    }
}

// Optimized - move invariant computation out
double factor = expensive_function();
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        result += matrix[i][j] * factor;
    }
}

// Cache-friendly access patterns
// Bad - column-major access in row-major layout
for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
        sum += matrix[i][j];  // Poor cache locality
    }
}

// Good - row-major access
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        sum += matrix[i][j];  // Good cache locality
    }
}

// Lazy evaluation
struct LazyValue {
    mutable bool computed;
    mutable int value;
    int (*computation)();
    
    int get() const {
        if (!computed) {
            value = computation();
            computed = true;
        }
        return value;
    }
};
```

## 9.11 Concurrent Programming Basics

### Thread Safety

```java
// Not thread-safe
class Counter {
    private int count = 0;
    
    public void increment() {
        count++;  // Read-modify-write race condition
    }
    
    public int getCount() {
        return count;
    }
}

// Thread-safe with synchronization
class SafeCounter {
    private int count = 0;
    
    public synchronized void increment() {
        count++;
    }
    
    public synchronized int getCount() {
        return count;
    }
}

// Thread-safe with atomic operations
import java.util.concurrent.atomic.AtomicInteger;

class AtomicCounter {
    private AtomicInteger count = new AtomicInteger(0);
    
    public void increment() {
        count.incrementAndGet();
    }
    
    public int getCount() {
        return count.get();
    }
}

// Thread-safe immutable class
final class ImmutablePoint {
    private final int x;
    private final int y;
    
    public ImmutablePoint(int x, int y) {
        this.x = x;
        this.y = y;
    }
    
    public int getX() { return x; }
    public int getY() { return y; }
    
    public ImmutablePoint move(int dx, int dy) {
        return new ImmutablePoint(x + dx, y + dy);  // Return new instance
    }
}
```

### Async Programming

```python
import asyncio

# Async/await pattern
async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network delay
    return f"Data from {url}"

async def process_urls(urls):
    # Concurrent execution
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Callback pattern (JavaScript-style)
def fetch_with_callback(url, callback):
    def simulate_fetch():
        time.sleep(1)
        callback(f"Data from {url}")
    
    thread = threading.Thread(target=simulate_fetch)
    thread.start()

# Promise/Future pattern
from concurrent.futures import ThreadPoolExecutor

def fetch_with_future(url):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fetch_data_sync, url)
    return future

# Usage
future = fetch_with_future("http://example.com")
result = future.result()  # Blocks until complete
```

## 9.12 Language Evolution

### Modern Language Features

```rust
// Rust - Modern systems language
// Pattern matching
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

fn process_message(msg: Message) {
    match msg {
        Message::Quit => println!("Quit"),
        Message::Move { x, y } => println!("Move to ({}, {})", x, y),
        Message::Write(text) => println!("Text: {}", text),
        Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
    }
}

// Traits (interfaces with default implementations)
trait Drawable {
    fn draw(&self);
    
    fn draw_twice(&self) {
        self.draw();
        self.draw();
    }
}

// Option type for null safety
fn divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None
    } else {
        Some(a / b)
    }
}

// Result type for error handling
fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
```

## Exercises

1. Implement a stack data structure in three different paradigms:
   - Procedural (C)
   - Object-oriented (Java/C++)
   - Functional (immutable)

2. Write a program that demonstrates:
   - Pass by value vs pass by reference
   - Deep copy vs shallow copy
   - Stack vs heap allocation

3. Create a generic sorting function that can sort any comparable type.

4. Implement the same algorithm (e.g., finding prime numbers) in:
   - Imperative style
   - Functional style
   - Object-oriented style

5. Write a recursive function to:
   - Traverse a binary tree
   - Solve Tower of Hanoi
   - Generate permutations

6. Design a simple class hierarchy for:
   - Geometric shapes
   - Bank accounts
   - Game characters

7. Implement common design patterns:
   - Singleton
   - Observer
   - Strategy

8. Write unit tests for a calculator class covering:
   - Normal cases
   - Edge cases
   - Error conditions

9. Create a thread-safe data structure:
   - Concurrent queue
   - Read-write lock
   - Thread pool

10. Optimize a program by:
    - Profiling to find bottlenecks
    - Improving algorithm complexity
    - Using appropriate data structures

## Summary

This chapter covered fundamental programming concepts:

- Programming paradigms provide different approaches to problem-solving
- Type systems and memory management affect program safety and performance
- Control structures direct program flow
- Functions and modules organize code for reusability
- Generic programming enables code reuse across types
- Best practices improve code quality and maintainability
- Common patterns solve recurring design problems
- Performance considerations guide optimization efforts
- Concurrent programming handles multiple tasks simultaneously
- Modern languages continue to evolve with new features

These fundamentals apply across all programming languages and form the foundation for writing effective software. The next chapter will explore data structures in detail, building on these programming concepts to examine how data can be organized and manipulated efficiently.