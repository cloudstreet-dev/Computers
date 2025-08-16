# Chapter 10: Data Structures

## Introduction

Data structures are fundamental building blocks of computer programs, providing organized ways to store and access data efficiently. The choice of data structure profoundly impacts program performance, memory usage, and code complexity. This chapter explores essential data structures, their implementations, trade-offs, and applications, providing the foundation for understanding how to organize data for optimal algorithmic performance.

## 10.1 Arrays and Dynamic Arrays

### Static Arrays

Arrays provide contiguous memory storage with constant-time access:

```c
// Fixed-size array
int numbers[100];

// Array access is simple pointer arithmetic
// numbers[i] = *(numbers + i * sizeof(int))

// 2D array in row-major order
int matrix[10][20];
// matrix[i][j] = *(matrix + i * 20 + j)

// Advantages:
// - O(1) random access
// - Cache-friendly for sequential access
// - No memory overhead per element

// Disadvantages:
// - Fixed size
// - Insertion/deletion expensive O(n)
// - Wasted space if not fully utilized
```

### Dynamic Arrays (Vectors)

Resizable arrays that grow as needed:

```cpp
template<typename T>
class DynamicArray {
private:
    T* data;
    size_t size;
    size_t capacity;
    
    void resize() {
        capacity *= 2;  // Double capacity
        T* new_data = new T[capacity];
        for (size_t i = 0; i < size; i++) {
            new_data[i] = std::move(data[i]);
        }
        delete[] data;
        data = new_data;
    }
    
public:
    DynamicArray() : data(new T[1]), size(0), capacity(1) {}
    
    ~DynamicArray() {
        delete[] data;
    }
    
    void push_back(const T& value) {
        if (size == capacity) {
            resize();
        }
        data[size++] = value;
    }
    
    T& operator[](size_t index) {
        if (index >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index];
    }
    
    void insert(size_t index, const T& value) {
        if (size == capacity) {
            resize();
        }
        // Shift elements right
        for (size_t i = size; i > index; i--) {
            data[i] = data[i - 1];
        }
        data[index] = value;
        size++;
    }
    
    void remove(size_t index) {
        // Shift elements left
        for (size_t i = index; i < size - 1; i++) {
            data[i] = data[i + 1];
        }
        size--;
    }
    
    size_t get_size() const { return size; }
    size_t get_capacity() const { return capacity; }
};

// Amortized O(1) append analysis:
// n insertions: 1 + 2 + 4 + 8 + ... + n = 2n - 1
// Average per insertion: (2n - 1) / n ≈ 2 = O(1)
```

## 10.2 Linked Lists

### Singly Linked List

Nodes connected by pointers:

```c
typedef struct Node {
    int data;
    struct Node* next;
} Node;

typedef struct {
    Node* head;
    size_t size;
} LinkedList;

// O(1) insertion at head
void push_front(LinkedList* list, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
}

// O(n) search
Node* find(LinkedList* list, int value) {
    Node* current = list->head;
    while (current != NULL) {
        if (current->data == value) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// O(n) insertion at position
void insert_at(LinkedList* list, size_t index, int value) {
    if (index == 0) {
        push_front(list, value);
        return;
    }
    
    Node* current = list->head;
    for (size_t i = 0; i < index - 1 && current != NULL; i++) {
        current = current->next;
    }
    
    if (current == NULL) return;
    
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    new_node->next = current->next;
    current->next = new_node;
    list->size++;
}

// O(1) deletion with pointer to previous node
void delete_after(Node* prev) {
    if (prev == NULL || prev->next == NULL) return;
    
    Node* to_delete = prev->next;
    prev->next = to_delete->next;
    free(to_delete);
}
```

### Doubly Linked List

Bidirectional traversal:

```cpp
template<typename T>
class DoublyLinkedList {
private:
    struct Node {
        T data;
        Node* prev;
        Node* next;
        
        Node(const T& value) : data(value), prev(nullptr), next(nullptr) {}
    };
    
    Node* head;
    Node* tail;
    size_t size;
    
public:
    DoublyLinkedList() : head(nullptr), tail(nullptr), size(0) {}
    
    void push_front(const T& value) {
        Node* new_node = new Node(value);
        new_node->next = head;
        
        if (head != nullptr) {
            head->prev = new_node;
        } else {
            tail = new_node;  // First element
        }
        
        head = new_node;
        size++;
    }
    
    void push_back(const T& value) {
        Node* new_node = new Node(value);
        new_node->prev = tail;
        
        if (tail != nullptr) {
            tail->next = new_node;
        } else {
            head = new_node;  // First element
        }
        
        tail = new_node;
        size++;
    }
    
    void remove(Node* node) {
        if (node == nullptr) return;
        
        if (node->prev != nullptr) {
            node->prev->next = node->next;
        } else {
            head = node->next;  // Removing head
        }
        
        if (node->next != nullptr) {
            node->next->prev = node->prev;
        } else {
            tail = node->prev;  // Removing tail
        }
        
        delete node;
        size--;
    }
    
    class Iterator {
        Node* current;
    public:
        Iterator(Node* node) : current(node) {}
        
        T& operator*() { return current->data; }
        Iterator& operator++() { 
            current = current->next; 
            return *this; 
        }
        Iterator& operator--() { 
            current = current->prev; 
            return *this; 
        }
        bool operator!=(const Iterator& other) { 
            return current != other.current; 
        }
    };
    
    Iterator begin() { return Iterator(head); }
    Iterator end() { return Iterator(nullptr); }
};
```

### Circular Linked List

Last node points to first:

```c
typedef struct {
    Node* tail;  // Keep tail for O(1) insertion at both ends
    size_t size;
} CircularList;

void insert_after_tail(CircularList* list, int value) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = value;
    
    if (list->tail == NULL) {
        new_node->next = new_node;  // Points to itself
        list->tail = new_node;
    } else {
        new_node->next = list->tail->next;  // Point to head
        list->tail->next = new_node;
        list->tail = new_node;  // Update tail
    }
    list->size++;
}

// Useful for round-robin scheduling, circular buffers
void rotate(CircularList* list) {
    if (list->tail != NULL) {
        list->tail = list->tail->next;
    }
}
```

## 10.3 Stacks and Queues

### Stack (LIFO)

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)  # O(1) amortized
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()  # O(1)
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]  # O(1)
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# Stack applications
def is_balanced(expression):
    """Check if parentheses are balanced"""
    stack = Stack()
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in expression:
        if char in pairs:
            stack.push(char)
        elif char in pairs.values():
            if stack.is_empty():
                return False
            if pairs[stack.pop()] != char:
                return False
    
    return stack.is_empty()

def evaluate_postfix(expression):
    """Evaluate postfix expression"""
    stack = Stack()
    
    for token in expression.split():
        if token.isdigit():
            stack.push(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.push(a + b)
            elif token == '-':
                stack.push(a - b)
            elif token == '*':
                stack.push(a * b)
            elif token == '/':
                stack.push(a // b)
    
    return stack.pop()
```

### Queue (FIFO)

```cpp
template<typename T>
class Queue {
private:
    struct Node {
        T data;
        Node* next;
        Node(const T& value) : data(value), next(nullptr) {}
    };
    
    Node* front;
    Node* rear;
    size_t size;
    
public:
    Queue() : front(nullptr), rear(nullptr), size(0) {}
    
    void enqueue(const T& value) {  // O(1)
        Node* new_node = new Node(value);
        
        if (rear == nullptr) {
            front = rear = new_node;
        } else {
            rear->next = new_node;
            rear = new_node;
        }
        size++;
    }
    
    T dequeue() {  // O(1)
        if (front == nullptr) {
            throw std::runtime_error("Queue is empty");
        }
        
        Node* temp = front;
        T value = front->data;
        front = front->next;
        
        if (front == nullptr) {
            rear = nullptr;  // Queue is now empty
        }
        
        delete temp;
        size--;
        return value;
    }
    
    bool is_empty() const { return front == nullptr; }
    size_t get_size() const { return size; }
};

// Circular queue with array
template<typename T>
class CircularQueue {
private:
    T* data;
    size_t capacity;
    size_t front;
    size_t rear;
    size_t size;
    
public:
    CircularQueue(size_t cap) 
        : capacity(cap), front(0), rear(0), size(0) {
        data = new T[capacity];
    }
    
    void enqueue(const T& value) {
        if (size == capacity) {
            throw std::overflow_error("Queue is full");
        }
        
        data[rear] = value;
        rear = (rear + 1) % capacity;
        size++;
    }
    
    T dequeue() {
        if (size == 0) {
            throw std::underflow_error("Queue is empty");
        }
        
        T value = data[front];
        front = (front + 1) % capacity;
        size--;
        return value;
    }
};
```

### Deque (Double-Ended Queue)

```python
from collections import deque

class Deque:
    def __init__(self):
        self.items = deque()
    
    def add_front(self, item):
        self.items.appendleft(item)  # O(1)
    
    def add_rear(self, item):
        self.items.append(item)  # O(1)
    
    def remove_front(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.popleft()  # O(1)
    
    def remove_rear(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop()  # O(1)
    
    def is_empty(self):
        return len(self.items) == 0

# Application: Palindrome checker
def is_palindrome(string):
    char_deque = Deque()
    
    for char in string.lower():
        if char.isalnum():
            char_deque.add_rear(char)
    
    while char_deque.size() > 1:
        if char_deque.remove_front() != char_deque.remove_rear():
            return False
    
    return True
```

## 10.4 Trees

### Binary Tree

```c
typedef struct TreeNode {
    int data;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

// Tree traversals
void preorder(TreeNode* root) {
    if (root == NULL) return;
    printf("%d ", root->data);
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (root == NULL) return;
    inorder(root->left);
    printf("%d ", root->data);
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->data);
}

// Level-order traversal (BFS)
void level_order(TreeNode* root) {
    if (root == NULL) return;
    
    Queue* queue = create_queue();
    enqueue(queue, root);
    
    while (!is_empty(queue)) {
        TreeNode* current = dequeue(queue);
        printf("%d ", current->data);
        
        if (current->left != NULL) {
            enqueue(queue, current->left);
        }
        if (current->right != NULL) {
            enqueue(queue, current->right);
        }
    }
}

// Tree properties
int height(TreeNode* root) {
    if (root == NULL) return -1;
    
    int left_height = height(root->left);
    int right_height = height(root->right);
    
    return 1 + (left_height > right_height ? left_height : right_height);
}

int size(TreeNode* root) {
    if (root == NULL) return 0;
    return 1 + size(root->left) + size(root->right);
}

bool is_balanced(TreeNode* root) {
    if (root == NULL) return true;
    
    int left_height = height(root->left);
    int right_height = height(root->right);
    
    return abs(left_height - right_height) <= 1 &&
           is_balanced(root->left) &&
           is_balanced(root->right);
}
```

### Binary Search Tree (BST)

```cpp
template<typename T>
class BST {
private:
    struct Node {
        T data;
        Node* left;
        Node* right;
        
        Node(const T& value) 
            : data(value), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    
    Node* insert_helper(Node* node, const T& value) {
        if (node == nullptr) {
            return new Node(value);
        }
        
        if (value < node->data) {
            node->left = insert_helper(node->left, value);
        } else if (value > node->data) {
            node->right = insert_helper(node->right, value);
        }
        
        return node;
    }
    
    Node* find_min(Node* node) {
        while (node->left != nullptr) {
            node = node->left;
        }
        return node;
    }
    
    Node* delete_helper(Node* node, const T& value) {
        if (node == nullptr) return nullptr;
        
        if (value < node->data) {
            node->left = delete_helper(node->left, value);
        } else if (value > node->data) {
            node->right = delete_helper(node->right, value);
        } else {
            // Node to delete found
            if (node->left == nullptr) {
                Node* temp = node->right;
                delete node;
                return temp;
            } else if (node->right == nullptr) {
                Node* temp = node->left;
                delete node;
                return temp;
            }
            
            // Node with two children
            Node* temp = find_min(node->right);
            node->data = temp->data;
            node->right = delete_helper(node->right, temp->data);
        }
        
        return node;
    }
    
    bool search_helper(Node* node, const T& value) {
        if (node == nullptr) return false;
        
        if (value == node->data) return true;
        if (value < node->data) return search_helper(node->left, value);
        return search_helper(node->right, value);
    }
    
public:
    BST() : root(nullptr) {}
    
    void insert(const T& value) {  // O(log n) average, O(n) worst
        root = insert_helper(root, value);
    }
    
    void remove(const T& value) {  // O(log n) average, O(n) worst
        root = delete_helper(root, value);
    }
    
    bool search(const T& value) {  // O(log n) average, O(n) worst
        return search_helper(root, value);
    }
    
    T find_min() {
        if (root == nullptr) {
            throw std::runtime_error("Tree is empty");
        }
        Node* min_node = find_min(root);
        return min_node->data;
    }
    
    T find_max() {
        if (root == nullptr) {
            throw std::runtime_error("Tree is empty");
        }
        Node* current = root;
        while (current->right != nullptr) {
            current = current->right;
        }
        return current->data;
    }
};
```

### AVL Tree (Self-Balancing BST)

```cpp
template<typename T>
class AVLTree {
private:
    struct Node {
        T data;
        Node* left;
        Node* right;
        int height;
        
        Node(const T& value) 
            : data(value), left(nullptr), right(nullptr), height(1) {}
    };
    
    Node* root;
    
    int height(Node* node) {
        return node ? node->height : 0;
    }
    
    int balance_factor(Node* node) {
        return node ? height(node->left) - height(node->right) : 0;
    }
    
    void update_height(Node* node) {
        if (node) {
            node->height = 1 + std::max(height(node->left), 
                                       height(node->right));
        }
    }
    
    Node* rotate_right(Node* y) {
        Node* x = y->left;
        Node* T2 = x->right;
        
        x->right = y;
        y->left = T2;
        
        update_height(y);
        update_height(x);
        
        return x;
    }
    
    Node* rotate_left(Node* x) {
        Node* y = x->right;
        Node* T2 = y->left;
        
        y->left = x;
        x->right = T2;
        
        update_height(x);
        update_height(y);
        
        return y;
    }
    
    Node* insert_helper(Node* node, const T& value) {
        // Standard BST insertion
        if (node == nullptr) {
            return new Node(value);
        }
        
        if (value < node->data) {
            node->left = insert_helper(node->left, value);
        } else if (value > node->data) {
            node->right = insert_helper(node->right, value);
        } else {
            return node;  // Duplicate values not allowed
        }
        
        // Update height
        update_height(node);
        
        // Get balance factor
        int balance = balance_factor(node);
        
        // Left-Left case
        if (balance > 1 && value < node->left->data) {
            return rotate_right(node);
        }
        
        // Right-Right case
        if (balance < -1 && value > node->right->data) {
            return rotate_left(node);
        }
        
        // Left-Right case
        if (balance > 1 && value > node->left->data) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }
        
        // Right-Left case
        if (balance < -1 && value < node->right->data) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        
        return node;
    }
    
public:
    AVLTree() : root(nullptr) {}
    
    void insert(const T& value) {  // O(log n) guaranteed
        root = insert_helper(root, value);
    }
    
    bool is_balanced() {
        return check_balance(root);
    }
    
private:
    bool check_balance(Node* node) {
        if (node == nullptr) return true;
        
        int balance = balance_factor(node);
        
        return abs(balance) <= 1 && 
               check_balance(node->left) && 
               check_balance(node->right);
    }
};
```

### B-Tree

```cpp
template<typename T, int ORDER>
class BTree {
private:
    struct Node {
        T keys[2 * ORDER - 1];
        Node* children[2 * ORDER];
        int num_keys;
        bool is_leaf;
        
        Node(bool leaf = true) : num_keys(0), is_leaf(leaf) {
            for (int i = 0; i < 2 * ORDER; i++) {
                children[i] = nullptr;
            }
        }
    };
    
    Node* root;
    
    void split_child(Node* parent, int index) {
        Node* full_child = parent->children[index];
        Node* new_child = new Node(full_child->is_leaf);
        
        new_child->num_keys = ORDER - 1;
        
        // Copy second half of keys to new child
        for (int i = 0; i < ORDER - 1; i++) {
            new_child->keys[i] = full_child->keys[i + ORDER];
        }
        
        // Copy second half of children if not leaf
        if (!full_child->is_leaf) {
            for (int i = 0; i < ORDER; i++) {
                new_child->children[i] = full_child->children[i + ORDER];
            }
        }
        
        full_child->num_keys = ORDER - 1;
        
        // Insert middle key into parent
        for (int i = parent->num_keys; i > index; i--) {
            parent->children[i + 1] = parent->children[i];
        }
        parent->children[index + 1] = new_child;
        
        for (int i = parent->num_keys - 1; i >= index; i--) {
            parent->keys[i + 1] = parent->keys[i];
        }
        parent->keys[index] = full_child->keys[ORDER - 1];
        parent->num_keys++;
    }
    
    void insert_non_full(Node* node, const T& value) {
        int i = node->num_keys - 1;
        
        if (node->is_leaf) {
            // Insert key in sorted order
            while (i >= 0 && node->keys[i] > value) {
                node->keys[i + 1] = node->keys[i];
                i--;
            }
            node->keys[i + 1] = value;
            node->num_keys++;
        } else {
            // Find child to insert
            while (i >= 0 && node->keys[i] > value) {
                i--;
            }
            i++;
            
            if (node->children[i]->num_keys == 2 * ORDER - 1) {
                split_child(node, i);
                
                if (node->keys[i] < value) {
                    i++;
                }
            }
            
            insert_non_full(node->children[i], value);
        }
    }
    
public:
    BTree() : root(nullptr) {}
    
    void insert(const T& value) {
        if (root == nullptr) {
            root = new Node();
            root->keys[0] = value;
            root->num_keys = 1;
        } else {
            if (root->num_keys == 2 * ORDER - 1) {
                Node* new_root = new Node(false);
                new_root->children[0] = root;
                split_child(new_root, 0);
                root = new_root;
            }
            insert_non_full(root, value);
        }
    }
    
    bool search(const T& value) {
        return search_helper(root, value);
    }
    
private:
    bool search_helper(Node* node, const T& value) {
        if (node == nullptr) return false;
        
        int i = 0;
        while (i < node->num_keys && value > node->keys[i]) {
            i++;
        }
        
        if (i < node->num_keys && value == node->keys[i]) {
            return true;
        }
        
        if (node->is_leaf) return false;
        
        return search_helper(node->children[i], value);
    }
};
```

## 10.5 Heaps

### Binary Heap

```python
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, value):  # O(log n)
        self.heap.append(value)
        self.heapify_up(len(self.heap) - 1)
    
    def heapify_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def extract_min(self):  # O(log n)
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        
        return min_val
    
    def heapify_down(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != i:
            self.swap(i, smallest)
            self.heapify_down(smallest)
    
    def peek(self):  # O(1)
        if not self.heap:
            raise IndexError("Heap is empty")
        return self.heap[0]
    
    def build_heap(self, arr):  # O(n)
        self.heap = arr[:]
        # Start from last non-leaf node
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self.heapify_down(i)

# Priority Queue implementation
class PriorityQueue:
    def __init__(self):
        self.heap = MinHeap()
    
    def enqueue(self, item, priority):
        self.heap.insert((priority, item))
    
    def dequeue(self):
        priority, item = self.heap.extract_min()
        return item
    
    def is_empty(self):
        return len(self.heap.heap) == 0
```

### Fibonacci Heap

```cpp
// Advanced heap with better amortized complexity
template<typename T>
class FibonacciHeap {
private:
    struct Node {
        T key;
        int degree;
        bool marked;
        Node* parent;
        Node* child;
        Node* left;
        Node* right;
        
        Node(const T& k) : key(k), degree(0), marked(false),
                          parent(nullptr), child(nullptr) {
            left = right = this;
        }
    };
    
    Node* min_node;
    int num_nodes;
    
public:
    FibonacciHeap() : min_node(nullptr), num_nodes(0) {}
    
    void insert(const T& key) {  // O(1)
        Node* node = new Node(key);
        
        if (min_node == nullptr) {
            min_node = node;
        } else {
            // Add to root list
            node->left = min_node;
            node->right = min_node->right;
            min_node->right->left = node;
            min_node->right = node;
            
            if (key < min_node->key) {
                min_node = node;
            }
        }
        num_nodes++;
    }
    
    T extract_min() {  // O(log n) amortized
        Node* z = min_node;
        
        if (z != nullptr) {
            // Add children to root list
            if (z->child != nullptr) {
                Node* child = z->child;
                do {
                    Node* next = child->right;
                    child->left = min_node;
                    child->right = min_node->right;
                    min_node->right->left = child;
                    min_node->right = child;
                    child->parent = nullptr;
                    child = next;
                } while (child != z->child);
            }
            
            // Remove z from root list
            z->left->right = z->right;
            z->right->left = z->left;
            
            T min_key = z->key;
            
            if (z == z->right) {
                min_node = nullptr;
            } else {
                min_node = z->right;
                consolidate();
            }
            
            delete z;
            num_nodes--;
            return min_key;
        }
        
        throw std::runtime_error("Heap is empty");
    }
    
    void decrease_key(Node* x, const T& new_key) {  // O(1) amortized
        if (new_key > x->key) {
            throw std::invalid_argument("New key is greater than current key");
        }
        
        x->key = new_key;
        Node* y = x->parent;
        
        if (y != nullptr && x->key < y->key) {
            cut(x, y);
            cascading_cut(y);
        }
        
        if (x->key < min_node->key) {
            min_node = x;
        }
    }
    
private:
    void consolidate() {
        // Implementation of consolidation after extract_min
        // Combines trees of same degree
    }
    
    void cut(Node* x, Node* y) {
        // Remove x from child list of y
    }
    
    void cascading_cut(Node* y) {
        // Cascading cut operation
    }
};
```

## 10.6 Hash Tables

### Hash Table with Chaining

```c
#define TABLE_SIZE 1000

typedef struct HashNode {
    char* key;
    int value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode* buckets[TABLE_SIZE];
    size_t size;
} HashTable;

unsigned int hash(const char* key) {
    unsigned int hash = 5381;
    int c;
    
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    
    return hash % TABLE_SIZE;
}

void insert(HashTable* table, const char* key, int value) {
    unsigned int index = hash(key);
    HashNode* node = table->buckets[index];
    
    // Check if key exists
    while (node != NULL) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;  // Update value
            return;
        }
        node = node->next;
    }
    
    // Add new node
    HashNode* new_node = malloc(sizeof(HashNode));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = table->buckets[index];
    table->buckets[index] = new_node;
    table->size++;
}

int* get(HashTable* table, const char* key) {
    unsigned int index = hash(key);
    HashNode* node = table->buckets[index];
    
    while (node != NULL) {
        if (strcmp(node->key, key) == 0) {
            return &node->value;
        }
        node = node->next;
    }
    
    return NULL;  // Key not found
}

void delete(HashTable* table, const char* key) {
    unsigned int index = hash(key);
    HashNode* node = table->buckets[index];
    HashNode* prev = NULL;
    
    while (node != NULL) {
        if (strcmp(node->key, key) == 0) {
            if (prev == NULL) {
                table->buckets[index] = node->next;
            } else {
                prev->next = node->next;
            }
            free(node->key);
            free(node);
            table->size--;
            return;
        }
        prev = node;
        node = node->next;
    }
}
```

### Hash Table with Open Addressing

```cpp
template<typename K, typename V>
class OpenAddressHashTable {
private:
    struct Entry {
        K key;
        V value;
        bool occupied;
        bool deleted;
        
        Entry() : occupied(false), deleted(false) {}
    };
    
    Entry* table;
    size_t capacity;
    size_t size;
    
    size_t hash1(const K& key) {
        return std::hash<K>{}(key) % capacity;
    }
    
    size_t hash2(const K& key) {
        return 1 + (std::hash<K>{}(key) % (capacity - 1));
    }
    
    size_t probe(const K& key, int i) {
        // Double hashing
        return (hash1(key) + i * hash2(key)) % capacity;
        
        // Linear probing
        // return (hash1(key) + i) % capacity;
        
        // Quadratic probing
        // return (hash1(key) + i * i) % capacity;
    }
    
    void resize() {
        size_t old_capacity = capacity;
        Entry* old_table = table;
        
        capacity *= 2;
        table = new Entry[capacity];
        size = 0;
        
        for (size_t i = 0; i < old_capacity; i++) {
            if (old_table[i].occupied && !old_table[i].deleted) {
                insert(old_table[i].key, old_table[i].value);
            }
        }
        
        delete[] old_table;
    }
    
public:
    OpenAddressHashTable(size_t initial_capacity = 16) 
        : capacity(initial_capacity), size(0) {
        table = new Entry[capacity];
    }
    
    ~OpenAddressHashTable() {
        delete[] table;
    }
    
    void insert(const K& key, const V& value) {
        if (size >= capacity * 0.75) {  // Load factor threshold
            resize();
        }
        
        for (int i = 0; i < capacity; i++) {
            size_t index = probe(key, i);
            
            if (!table[index].occupied || table[index].deleted ||
                table[index].key == key) {
                
                if (!table[index].occupied || table[index].deleted) {
                    size++;
                }
                
                table[index].key = key;
                table[index].value = value;
                table[index].occupied = true;
                table[index].deleted = false;
                return;
            }
        }
    }
    
    V* find(const K& key) {
        for (int i = 0; i < capacity; i++) {
            size_t index = probe(key, i);
            
            if (!table[index].occupied) {
                return nullptr;  // Key not found
            }
            
            if (!table[index].deleted && table[index].key == key) {
                return &table[index].value;
            }
        }
        return nullptr;
    }
    
    void remove(const K& key) {
        for (int i = 0; i < capacity; i++) {
            size_t index = probe(key, i);
            
            if (!table[index].occupied) {
                return;  // Key not found
            }
            
            if (!table[index].deleted && table[index].key == key) {
                table[index].deleted = true;
                size--;
                return;
            }
        }
    }
};
```

## 10.7 Graphs

### Graph Representations

```python
# Adjacency Matrix
class GraphMatrix:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    def add_edge(self, u, v, weight=1):
        self.matrix[u][v] = weight
        # self.matrix[v][u] = weight  # For undirected graph
    
    def has_edge(self, u, v):
        return self.matrix[u][v] != 0
    
    def get_weight(self, u, v):
        return self.matrix[u][v]
    
    def get_neighbors(self, v):
        neighbors = []
        for i in range(self.num_vertices):
            if self.matrix[v][i] != 0:
                neighbors.append((i, self.matrix[v][i]))
        return neighbors

# Adjacency List
class GraphList:
    def __init__(self):
        self.graph = {}
    
    def add_vertex(self, v):
        if v not in self.graph:
            self.graph[v] = []
    
    def add_edge(self, u, v, weight=1):
        if u not in self.graph:
            self.add_vertex(u)
        if v not in self.graph:
            self.add_vertex(v)
        
        self.graph[u].append((v, weight))
        # self.graph[v].append((u, weight))  # For undirected
    
    def get_neighbors(self, v):
        return self.graph.get(v, [])
    
    def has_edge(self, u, v):
        for neighbor, _ in self.graph.get(u, []):
            if neighbor == v:
                return True
        return False

# Edge List
class GraphEdgeList:
    def __init__(self):
        self.edges = []
        self.vertices = set()
    
    def add_edge(self, u, v, weight=1):
        self.edges.append((u, v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
    
    def get_edges(self):
        return self.edges
    
    def get_vertices(self):
        return list(self.vertices)
```

### Graph Traversal

```cpp
#include <vector>
#include <queue>
#include <stack>
#include <unordered_set>

class Graph {
private:
    std::vector<std::vector<int>> adj_list;
    int num_vertices;
    
public:
    Graph(int n) : num_vertices(n), adj_list(n) {}
    
    void add_edge(int u, int v) {
        adj_list[u].push_back(v);
        // adj_list[v].push_back(u);  // For undirected
    }
    
    void dfs(int start) {
        std::vector<bool> visited(num_vertices, false);
        dfs_helper(start, visited);
    }
    
    void dfs_helper(int v, std::vector<bool>& visited) {
        visited[v] = true;
        std::cout << v << " ";
        
        for (int neighbor : adj_list[v]) {
            if (!visited[neighbor]) {
                dfs_helper(neighbor, visited);
            }
        }
    }
    
    void dfs_iterative(int start) {
        std::vector<bool> visited(num_vertices, false);
        std::stack<int> stack;
        
        stack.push(start);
        
        while (!stack.empty()) {
            int v = stack.top();
            stack.pop();
            
            if (!visited[v]) {
                visited[v] = true;
                std::cout << v << " ";
                
                for (int neighbor : adj_list[v]) {
                    if (!visited[neighbor]) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }
    
    void bfs(int start) {
        std::vector<bool> visited(num_vertices, false);
        std::queue<int> queue;
        
        visited[start] = true;
        queue.push(start);
        
        while (!queue.empty()) {
            int v = queue.front();
            queue.pop();
            std::cout << v << " ";
            
            for (int neighbor : adj_list[v]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }
    
    bool has_cycle() {
        std::vector<int> color(num_vertices, 0);  // 0: white, 1: gray, 2: black
        
        for (int i = 0; i < num_vertices; i++) {
            if (color[i] == 0) {
                if (has_cycle_dfs(i, color)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    bool has_cycle_dfs(int v, std::vector<int>& color) {
        color[v] = 1;  // Gray
        
        for (int neighbor : adj_list[v]) {
            if (color[neighbor] == 1) {
                return true;  // Back edge found
            }
            if (color[neighbor] == 0 && has_cycle_dfs(neighbor, color)) {
                return true;
            }
        }
        
        color[v] = 2;  // Black
        return false;
    }
    
    std::vector<int> topological_sort() {
        std::vector<int> in_degree(num_vertices, 0);
        
        // Calculate in-degrees
        for (int v = 0; v < num_vertices; v++) {
            for (int neighbor : adj_list[v]) {
                in_degree[neighbor]++;
            }
        }
        
        std::queue<int> queue;
        for (int i = 0; i < num_vertices; i++) {
            if (in_degree[i] == 0) {
                queue.push(i);
            }
        }
        
        std::vector<int> result;
        
        while (!queue.empty()) {
            int v = queue.front();
            queue.pop();
            result.push_back(v);
            
            for (int neighbor : adj_list[v]) {
                in_degree[neighbor]--;
                if (in_degree[neighbor] == 0) {
                    queue.push(neighbor);
                }
            }
        }
        
        if (result.size() != num_vertices) {
            // Graph has cycle
            return {};
        }
        
        return result;
    }
};
```

## 10.8 Tries

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):  # O(m) where m is word length
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.frequency += 1
    
    def search(self, word):  # O(m)
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def starts_with(self, prefix):  # O(m)
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return True
    
    def delete(self, word):
        def delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete = delete_helper(node.children[char], word, index + 1)
            
            if should_delete:
                del node.children[char]
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        delete_helper(self.root, word, 0)
    
    def autocomplete(self, prefix):
        node = self.root
        
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words with this prefix
        results = []
        self._collect_words(node, prefix, results)
        return results
    
    def _collect_words(self, node, prefix, results):
        if node.is_end_of_word:
            results.append(prefix)
        
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)
```

## 10.9 Disjoint Set (Union-Find)

```cpp
class DisjointSet {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    int num_sets;
    
public:
    DisjointSet(int n) : parent(n), rank(n, 0), num_sets(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {  // With path compression
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    void union_sets(int x, int y) {  // Union by rank
        int root_x = find(x);
        int root_y = find(y);
        
        if (root_x == root_y) return;
        
        if (rank[root_x] < rank[root_y]) {
            parent[root_x] = root_y;
        } else if (rank[root_x] > rank[root_y]) {
            parent[root_y] = root_x;
        } else {
            parent[root_y] = root_x;
            rank[root_x]++;
        }
        
        num_sets--;
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
    
    int count_sets() {
        return num_sets;
    }
    
    // Application: Kruskal's MST algorithm
    struct Edge {
        int u, v, weight;
        bool operator<(const Edge& other) const {
            return weight < other.weight;
        }
    };
    
    static int kruskal_mst(std::vector<Edge>& edges, int num_vertices) {
        std::sort(edges.begin(), edges.end());
        
        DisjointSet ds(num_vertices);
        int mst_weight = 0;
        int edges_added = 0;
        
        for (const Edge& edge : edges) {
            if (!ds.connected(edge.u, edge.v)) {
                ds.union_sets(edge.u, edge.v);
                mst_weight += edge.weight;
                edges_added++;
                
                if (edges_added == num_vertices - 1) {
                    break;  // MST complete
                }
            }
        }
        
        return mst_weight;
    }
};
```

## 10.10 Advanced Data Structures

### Skip List

```python
import random

class SkipListNode:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16):
        self.max_level = max_level
        self.header = SkipListNode(float('-inf'), max_level)
        self.level = 0
    
    def random_level(self):
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level
    
    def insert(self, value):
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find position to insert
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].value < value):
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # Insert new node
        if current is None or current.value != value:
            new_level = self.random_level()
            
            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level
            
            new_node = SkipListNode(value, new_level)
            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node
    
    def search(self, value):
        current = self.header
        
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].value < value):
                current = current.forward[i]
        
        current = current.forward[0]
        
        return current and current.value == value
    
    def delete(self, value):
        update = [None] * (self.max_level + 1)
        current = self.header
        
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].value < value):
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        if current and current.value == value:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
```

### Bloom Filter

```python
import hashlib

class BloomFilter:
    def __init__(self, size, num_hash_functions):
        self.size = size
        self.bit_array = [False] * size
        self.num_hash_functions = num_hash_functions
    
    def _hash(self, item, seed):
        hash_obj = hashlib.md5()
        hash_obj.update(str(item).encode())
        hash_obj.update(str(seed).encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item):
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            self.bit_array[index] = True
    
    def might_contain(self, item):
        for i in range(self.num_hash_functions):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True  # Might be false positive
    
    def definitely_not_contains(self, item):
        return not self.might_contain(item)
```

## Exercises

1. Implement a self-balancing binary search tree (Red-Black tree or AVL tree).

2. Design a LRU (Least Recently Used) cache using:
   - Hash table + Doubly linked list
   - Capacity limit and O(1) operations

3. Implement a graph data structure that supports:
   - Both directed and undirected graphs
   - Weighted edges
   - Common algorithms (BFS, DFS, shortest path)

4. Create a persistent data structure (functional):
   - Immutable list with efficient prepend
   - Persistent stack with versioning

5. Implement a B+ tree for database indexing:
   - Leaf nodes contain data
   - Internal nodes for navigation
   - Support range queries

6. Design a concurrent data structure:
   - Thread-safe queue
   - Lock-free stack
   - Concurrent hash map

7. Implement a space-efficient data structure:
   - Bit vector with rank/select operations
   - Compressed trie (Patricia tree)
   - Succinct tree representation

8. Create a specialized tree:
   - Segment tree for range queries
   - Fenwick tree (Binary Indexed Tree)
   - Interval tree

9. Implement string-specific structures:
   - Suffix array
   - Suffix tree
   - KMP failure function

10. Design a hybrid data structure:
    - Combination of array and linked list
    - Tree with hash table for O(1) lookup
    - Graph with indexed edges

## Summary

This chapter explored essential data structures:

- Arrays provide fast random access but fixed size
- Linked lists offer dynamic size but sequential access
- Stacks and queues implement LIFO and FIFO semantics
- Trees organize data hierarchically for efficient searching
- Heaps maintain priority queue properties
- Hash tables provide near-constant time lookup
- Graphs represent relationships between entities
- Tries optimize string operations
- Advanced structures solve specialized problems

Understanding data structures is crucial for efficient algorithm design and implementation. The choice of data structure significantly impacts program performance, memory usage, and code complexity. The next chapter will explore algorithms that operate on these data structures, examining how to solve computational problems efficiently.