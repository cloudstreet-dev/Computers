# Chapter 11: Algorithms and Complexity

## Introduction

Algorithms are precise sequences of steps for solving computational problems. They form the foundation of computer science, determining how efficiently we can process data, make decisions, and solve complex problems. This chapter explores fundamental algorithms, analyzes their complexity, and examines problem-solving strategies that apply across different domains. Understanding algorithms and their analysis is crucial for writing efficient software and recognizing computational limits.

## 11.1 Algorithm Analysis

### Big O Notation

Describes upper bound of growth rate:

```python
# O(1) - Constant time
def get_first(arr):
    return arr[0] if arr else None

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
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# O(n³) - Cubic time
def matrix_multiply(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# O(2ⁿ) - Exponential time
def power_set(s):
    if not s:
        return [[]]
    
    first = s[0]
    rest = s[1:]
    rest_subsets = power_set(rest)
    
    return rest_subsets + [[first] + subset for subset in rest_subsets]

# O(n!) - Factorial time
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for p in permutations(rest):
            result.append([arr[i]] + p)
    
    return result
```

### Other Complexity Notations

```python
# Big Omega (Ω) - Lower bound
# Ω(n log n) for comparison-based sorting

# Big Theta (Θ) - Tight bound
# Θ(n) for linear search in average case

# Little o - Strict upper bound
# o(n²) means grows slower than n²

# Little omega (ω) - Strict lower bound
# ω(n) means grows faster than n

# Amortized Analysis
class DynamicArray:
    def __init__(self):
        self.data = [None]
        self.size = 0
        self.capacity = 1
    
    def append(self, value):  # O(1) amortized
        if self.size == self.capacity:
            # Double capacity - O(n) but rare
            self.resize(2 * self.capacity)
        
        self.data[self.size] = value
        self.size += 1
    
    def resize(self, new_capacity):
        new_data = [None] * new_capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_capacity

# n appends: Total cost = n + (1 + 2 + 4 + ... + n) ≈ 2n
# Amortized cost per operation = 2n/n = O(1)
```

### Space Complexity

```python
# O(1) space - constant extra space
def reverse_in_place(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# O(n) space - linear extra space
def merge_sort_space(arr):
    if len(arr) <= 1:
        return arr
    
    # Creates new arrays - O(n) space total
    mid = len(arr) // 2
    left = merge_sort_space(arr[:mid])
    right = merge_sort_space(arr[mid:])
    
    return merge(left, right)

# O(log n) space - recursion stack
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

## 11.2 Sorting Algorithms

### Comparison-Based Sorting

```cpp
// Bubble Sort - O(n²)
void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;  // Optimization for sorted arrays
    }
}

// Insertion Sort - O(n²) worst, O(n) best
void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Merge Sort - O(n log n) always
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int* L = new int[n1];
    int* R = new int[n2];
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    
    delete[] L;
    delete[] R;
}

void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Quick Sort - O(n log n) average, O(n²) worst
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// Heap Sort - O(n log n) always
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heap_sort(int arr[], int n) {
    // Build heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    
    // Extract elements
    for (int i = n - 1; i > 0; i--) {
        std::swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

### Non-Comparison Sorting

```python
# Counting Sort - O(n + k) where k is range
def counting_sort(arr, max_val):
    count = [0] * (max_val + 1)
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Cumulative count
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
    
    return output

# Radix Sort - O(d * (n + k)) where d is digits
def radix_sort(arr):
    if not arr:
        return arr
    
    max_num = max(arr)
    exp = 1
    
    while max_num // exp > 0:
        counting_sort_digit(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    for i in range(n - 1, -1, -1):
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
    
    for i in range(n):
        arr[i] = output[i]

# Bucket Sort - O(n) average for uniform distribution
def bucket_sort(arr):
    if not arr:
        return arr
    
    num_buckets = len(arr)
    max_val = max(arr)
    min_val = min(arr)
    
    # Create buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Distribute elements
    for num in arr:
        index = int((num - min_val) / (max_val - min_val + 1) * num_buckets)
        if index == num_buckets:
            index -= 1
        buckets[index].append(num)
    
    # Sort individual buckets
    for bucket in buckets:
        bucket.sort()  # Can use insertion sort for small buckets
    
    # Concatenate buckets
    result = []
    for bucket in buckets:
        result.extend(bucket)
    
    return result
```

## 11.3 Searching Algorithms

### Linear and Binary Search

```c
// Linear Search - O(n)
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

// Binary Search Variants
// Standard binary search
int binary_search(int arr[], int n, int target) {
    int left = 0, right = n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

// Find first occurrence
int find_first(int arr[], int n, int target) {
    int left = 0, right = n - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// Find insertion position
int find_insert_position(int arr[], int n, int target) {
    int left = 0, right = n;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

// Binary search on answer
double sqrt_binary_search(double n, double epsilon) {
    double left = 0, right = n;
    
    while (right - left > epsilon) {
        double mid = left + (right - left) / 2;
        
        if (mid * mid < n) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    return left;
}
```

### Advanced Search Algorithms

```python
# Interpolation Search - O(log log n) for uniform distribution
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1
    
    while low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            return low if arr[low] == target else -1
        
        # Interpolation formula
        pos = low + int((target - arr[low]) * (high - low) / 
                       (arr[high] - arr[low]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return -1

# Exponential Search - O(log n)
def exponential_search(arr, target):
    n = len(arr)
    
    if arr[0] == target:
        return 0
    
    # Find range for binary search
    i = 1
    while i < n and arr[i] <= target:
        i *= 2
    
    # Binary search in range
    return binary_search(arr, target, i // 2, min(i, n - 1))

# Ternary Search - O(log₃ n)
def ternary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3
        
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        
        if target < arr[mid1]:
            right = mid1 - 1
        elif target > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1
    
    return -1
```

## 11.4 Graph Algorithms

### Shortest Path Algorithms

```cpp
#include <vector>
#include <queue>
#include <limits>

// Dijkstra's Algorithm - O((V + E) log V) with min heap
class Dijkstra {
public:
    typedef std::pair<int, int> pii;  // (distance, vertex)
    
    std::vector<int> dijkstra(std::vector<std::vector<pii>>& graph, int src) {
        int n = graph.size();
        std::vector<int> dist(n, INT_MAX);
        std::priority_queue<pii, std::vector<pii>, std::greater<pii>> pq;
        
        dist[src] = 0;
        pq.push({0, src});
        
        while (!pq.empty()) {
            int d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            
            if (d > dist[u]) continue;
            
            for (auto& edge : graph[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        
        return dist;
    }
};

// Bellman-Ford - O(VE), handles negative weights
class BellmanFord {
public:
    struct Edge {
        int src, dest, weight;
    };
    
    std::vector<int> bellman_ford(int V, std::vector<Edge>& edges, int src) {
        std::vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        
        // Relax edges V-1 times
        for (int i = 0; i < V - 1; i++) {
            for (const Edge& e : edges) {
                if (dist[e.src] != INT_MAX && 
                    dist[e.src] + e.weight < dist[e.dest]) {
                    dist[e.dest] = dist[e.src] + e.weight;
                }
            }
        }
        
        // Check for negative cycles
        for (const Edge& e : edges) {
            if (dist[e.src] != INT_MAX && 
                dist[e.src] + e.weight < dist[e.dest]) {
                // Negative cycle exists
                return {};
            }
        }
        
        return dist;
    }
};

// Floyd-Warshall - O(V³), all pairs shortest path
class FloydWarshall {
public:
    std::vector<std::vector<int>> floyd_warshall(std::vector<std::vector<int>>& graph) {
        int n = graph.size();
        std::vector<std::vector<int>> dist = graph;
        
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                        dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        
        return dist;
    }
};

// A* Search - Heuristic-based pathfinding
class AStar {
public:
    struct Node {
        int x, y;
        int g, h, f;  // g: cost from start, h: heuristic, f: g + h
        Node* parent;
        
        Node(int x, int y) : x(x), y(y), g(INT_MAX), h(0), f(INT_MAX), parent(nullptr) {}
    };
    
    int heuristic(int x1, int y1, int x2, int y2) {
        // Manhattan distance
        return abs(x1 - x2) + abs(y1 - y2);
    }
    
    std::vector<std::pair<int, int>> find_path(
        std::vector<std::vector<int>>& grid,
        int start_x, int start_y,
        int goal_x, int goal_y
    ) {
        int rows = grid.size();
        int cols = grid[0].size();
        
        auto cmp = [](Node* a, Node* b) { return a->f > b->f; };
        std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> open_set(cmp);
        
        std::vector<std::vector<bool>> closed(rows, std::vector<bool>(cols, false));
        std::vector<std::vector<Node*>> nodes(rows, std::vector<Node*>(cols));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                nodes[i][j] = new Node(i, j);
            }
        }
        
        Node* start = nodes[start_x][start_y];
        start->g = 0;
        start->h = heuristic(start_x, start_y, goal_x, goal_y);
        start->f = start->h;
        
        open_set.push(start);
        
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        
        while (!open_set.empty()) {
            Node* current = open_set.top();
            open_set.pop();
            
            if (current->x == goal_x && current->y == goal_y) {
                // Reconstruct path
                std::vector<std::pair<int, int>> path;
                while (current != nullptr) {
                    path.push_back({current->x, current->y});
                    current = current->parent;
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
            
            closed[current->x][current->y] = true;
            
            for (int i = 0; i < 4; i++) {
                int nx = current->x + dx[i];
                int ny = current->y + dy[i];
                
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols &&
                    grid[nx][ny] == 0 && !closed[nx][ny]) {
                    
                    Node* neighbor = nodes[nx][ny];
                    int tentative_g = current->g + 1;
                    
                    if (tentative_g < neighbor->g) {
                        neighbor->parent = current;
                        neighbor->g = tentative_g;
                        neighbor->h = heuristic(nx, ny, goal_x, goal_y);
                        neighbor->f = neighbor->g + neighbor->h;
                        
                        open_set.push(neighbor);
                    }
                }
            }
        }
        
        return {};  // No path found
    }
};
```

### Minimum Spanning Tree

```python
# Kruskal's Algorithm - O(E log E)
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        return True

def kruskal(n, edges):
    # edges: list of (weight, u, v)
    edges.sort()
    uf = UnionFind(n)
    mst = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == n - 1:
                break
    
    return mst, total_weight

# Prim's Algorithm - O(E log V)
import heapq

def prim(graph, start=0):
    n = len(graph)
    visited = [False] * n
    min_heap = [(0, start, -1)]  # (weight, vertex, parent)
    mst = []
    total_weight = 0
    
    while min_heap:
        weight, u, parent = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        
        visited[u] = True
        
        if parent != -1:
            mst.append((parent, u, weight))
            total_weight += weight
        
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v, u))
    
    return mst, total_weight
```

## 11.5 Dynamic Programming

### Classic DP Problems

```python
# Fibonacci with memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Fibonacci with tabulation
def fibonacci_tab(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    values[i - 1] + dp[i - 1][w - weights[i - 1]]
                )
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

# Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct LCS
    i, j = m, n
    lcs_str = []
    
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs_str.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], ''.join(reversed(lcs_str))

# Edit Distance (Levenshtein)
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]

# Longest Increasing Subsequence
def lis(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Coin Change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Matrix Chain Multiplication
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k + 1][j] +
                       dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n - 1]
```

## 11.6 Greedy Algorithms

```python
# Activity Selection
def activity_selection(activities):
    # activities: list of (start, end) tuples
    activities.sort(key=lambda x: x[1])  # Sort by end time
    
    selected = [activities[0]]
    last_end = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected

# Huffman Coding
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    if not text:
        return None, {}
    
    # Count frequencies
    freq = Counter(text)
    
    # Create leaf nodes
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        
        heapq.heappush(heap, parent)
    
    root = heap[0]
    
    # Generate codes
    codes = {}
    def generate_codes(node, code):
        if node.char is not None:
            codes[node.char] = code
        else:
            generate_codes(node.left, code + '0')
            generate_codes(node.right, code + '1')
    
    generate_codes(root, '')
    
    # Encode text
    encoded = ''.join(codes[char] for char in text)
    
    return encoded, codes

# Fractional Knapsack
def fractional_knapsack(weights, values, capacity):
    n = len(weights)
    items = [(values[i] / weights[i], weights[i], values[i]) 
             for i in range(n)]
    items.sort(reverse=True)  # Sort by value/weight ratio
    
    total_value = 0
    remaining_capacity = capacity
    
    for ratio, weight, value in items:
        if weight <= remaining_capacity:
            total_value += value
            remaining_capacity -= weight
        else:
            total_value += ratio * remaining_capacity
            break
    
    return total_value

# Job Scheduling with Deadlines
def job_scheduling(jobs):
    # jobs: list of (profit, deadline) tuples
    jobs.sort(reverse=True)  # Sort by profit
    
    max_deadline = max(job[1] for job in jobs)
    slots = [-1] * max_deadline
    total_profit = 0
    
    for profit, deadline in jobs:
        # Find latest available slot
        for i in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[i] == -1:
                slots[i] = profit
                total_profit += profit
                break
    
    return total_profit, slots
```

## 11.7 Divide and Conquer

```cpp
// Closest Pair of Points - O(n log n)
#include <algorithm>
#include <cmath>

struct Point {
    double x, y;
};

double distance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + 
                (p1.y - p2.y) * (p1.y - p2.y));
}

double brute_force(Point points[], int n) {
    double min_dist = DBL_MAX;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            min_dist = std::min(min_dist, distance(points[i], points[j]));
        }
    }
    
    return min_dist;
}

double strip_closest(Point strip[], int n, double d) {
    double min_dist = d;
    
    // Sort by y-coordinate
    std::sort(strip, strip + n, [](const Point& a, const Point& b) {
        return a.y < b.y;
    });
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n && (strip[j].y - strip[i].y) < min_dist; j++) {
            min_dist = std::min(min_dist, distance(strip[i], strip[j]));
        }
    }
    
    return min_dist;
}

double closest_pair_recursive(Point px[], Point py[], int n) {
    if (n <= 3) {
        return brute_force(px, n);
    }
    
    int mid = n / 2;
    Point midpoint = px[mid];
    
    Point pyl[mid];
    Point pyr[n - mid];
    int li = 0, ri = 0;
    
    for (int i = 0; i < n; i++) {
        if (py[i].x <= midpoint.x) {
            pyl[li++] = py[i];
        } else {
            pyr[ri++] = py[i];
        }
    }
    
    double dl = closest_pair_recursive(px, pyl, mid);
    double dr = closest_pair_recursive(px + mid, pyr, n - mid);
    
    double d = std::min(dl, dr);
    
    Point strip[n];
    int j = 0;
    
    for (int i = 0; i < n; i++) {
        if (abs(py[i].x - midpoint.x) < d) {
            strip[j++] = py[i];
        }
    }
    
    return std::min(d, strip_closest(strip, j, d));
}

// Karatsuba Multiplication - O(n^1.585)
std::string karatsuba(std::string num1, std::string num2) {
    int n = std::max(num1.size(), num2.size());
    
    if (n == 1) {
        return std::to_string((num1[0] - '0') * (num2[0] - '0'));
    }
    
    // Pad with zeros
    while (num1.size() < n) num1 = "0" + num1;
    while (num2.size() < n) num2 = "0" + num2;
    
    int mid = n / 2;
    
    std::string a = num1.substr(0, mid);
    std::string b = num1.substr(mid);
    std::string c = num2.substr(0, mid);
    std::string d = num2.substr(mid);
    
    std::string ac = karatsuba(a, c);
    std::string bd = karatsuba(b, d);
    std::string ad_bc = karatsuba(add(a, b), add(c, d));
    ad_bc = subtract(ad_bc, ac);
    ad_bc = subtract(ad_bc, bd);
    
    // Combine results
    std::string result = add(shift(ac, 2 * (n - mid)), 
                            add(shift(ad_bc, n - mid), bd));
    
    return result;
}
```

## 11.8 Backtracking

```python
# N-Queens Problem
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check upper right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(board, row):
        if row == n:
            solutions.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                backtrack(board, row + 1)
                board[row][col] = '.'
    
    solutions = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(board, 0)
    
    return solutions

# Sudoku Solver
def solve_sudoku(board):
    def is_valid(num, row, col):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(num, i, j):
                            board[i][j] = num
                            
                            if solve():
                                return True
                            
                            board[i][j] = '.'
                    
                    return False
        
        return True
    
    solve()
    return board

# Generate Permutations
def permutations(nums):
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for num in nums:
            if num not in path:
                path.append(num)
                backtrack(path)
                path.pop()
    
    result = []
    backtrack([])
    return result

# Subset Sum
def subset_sum(nums, target):
    def backtrack(start, current_sum, path):
        if current_sum == target:
            solutions.append(path[:])
            return
        
        if current_sum > target:
            return
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, current_sum + nums[i], path)
            path.pop()
    
    solutions = []
    backtrack(0, 0, [])
    return solutions
```

## 11.9 String Algorithms

```cpp
// KMP (Knuth-Morris-Pratt) Pattern Matching - O(n + m)
std::vector<int> compute_lps(const std::string& pattern) {
    int m = pattern.length();
    std::vector<int> lps(m, 0);
    int len = 0;
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}

std::vector<int> kmp_search(const std::string& text, const std::string& pattern) {
    int n = text.length();
    int m = pattern.length();
    std::vector<int> lps = compute_lps(pattern);
    std::vector<int> matches;
    
    int i = 0;  // Index for text
    int j = 0;  // Index for pattern
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    return matches;
}

// Rabin-Karp Rolling Hash - O(n + m) average
class RabinKarp {
private:
    const int PRIME = 101;
    
    int hash_value(const std::string& str, int end) {
        int hash = 0;
        for (int i = 0; i <= end; i++) {
            hash = (hash * 256 + str[i]) % PRIME;
        }
        return hash;
    }
    
public:
    std::vector<int> search(const std::string& text, const std::string& pattern) {
        int n = text.length();
        int m = pattern.length();
        std::vector<int> matches;
        
        int pattern_hash = hash_value(pattern, m - 1);
        int text_hash = hash_value(text, m - 1);
        
        int h = 1;
        for (int i = 0; i < m - 1; i++) {
            h = (h * 256) % PRIME;
        }
        
        for (int i = 0; i <= n - m; i++) {
            if (pattern_hash == text_hash) {
                // Check character by character
                bool match = true;
                for (int j = 0; j < m; j++) {
                    if (text[i + j] != pattern[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    matches.push_back(i);
                }
            }
            
            if (i < n - m) {
                text_hash = (256 * (text_hash - text[i] * h) + text[i + m]) % PRIME;
                if (text_hash < 0) {
                    text_hash += PRIME;
                }
            }
        }
        
        return matches;
    }
};

// Z-Algorithm - O(n)
std::vector<int> z_algorithm(const std::string& s) {
    int n = s.length();
    std::vector<int> z(n, 0);
    
    int l = 0, r = 0;
    for (int i = 1; i < n; i++) {
        if (i <= r) {
            z[i] = std::min(r - i + 1, z[i - l]);
        }
        
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        
        if (i + z[i] - 1 > r) {
            l = i;
            r = i + z[i] - 1;
        }
    }
    
    return z;
}
```

## 11.10 Computational Geometry

```python
# Convex Hull - Graham Scan O(n log n)
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def convex_hull(points):
    n = len(points)
    if n < 3:
        return points
    
    # Find the bottom-most point (and left-most if tie)
    start = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort points by polar angle with respect to start
    def polar_angle(p):
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return (math.atan2(dy, dx), dx*dx + dy*dy)
    
    sorted_points = sorted(points, key=polar_angle)
    
    # Graham scan
    hull = []
    for p in sorted_points:
        while len(hull) > 1 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)
    
    return hull

# Line Segment Intersection
def segments_intersect(p1, q1, p2, q2):
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases (collinear)
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False

# Point in Polygon - Ray Casting Algorithm O(n)
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside
```

## 11.11 NP-Complete Problems

```python
# Traveling Salesman Problem - Dynamic Programming O(n²2ⁿ)
def tsp_dp(dist):
    n = len(dist)
    all_visited = (1 << n) - 1
    
    # dp[mask][i] = minimum cost to visit cities in mask ending at i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start from city 0
    
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            
            for v in range(n):
                if mask & (1 << v) or u == v:
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], 
                                      dp[mask][u] + dist[u][v])
    
    # Find minimum cost to visit all cities and return to start
    ans = float('inf')
    for i in range(1, n):
        ans = min(ans, dp[all_visited][i] + dist[i][0])
    
    return ans

# SAT Solver (Simple DPLL)
def dpll(clauses, assignment):
    # Unit propagation
    while True:
        unit_clause = None
        for clause in clauses:
            if len(clause) == 1:
                unit_clause = clause[0]
                break
        
        if unit_clause is None:
            break
        
        # Assign the unit clause
        var = abs(unit_clause)
        value = unit_clause > 0
        assignment[var] = value
        
        # Simplify clauses
        new_clauses = []
        for clause in clauses:
            if unit_clause in clause:
                continue  # Clause satisfied
            
            new_clause = [lit for lit in clause if lit != -unit_clause]
            if not new_clause:
                return False  # Empty clause, unsatisfiable
            
            new_clauses.append(new_clause)
        
        clauses = new_clauses
    
    if not clauses:
        return True  # All clauses satisfied
    
    # Choose a variable
    var = abs(clauses[0][0])
    
    # Try true
    new_assignment = assignment.copy()
    new_assignment[var] = True
    new_clauses = [[lit for lit in clause if lit != -var] 
                   for clause in clauses if var not in clause]
    
    if dpll(new_clauses, new_assignment):
        assignment.update(new_assignment)
        return True
    
    # Try false
    new_assignment = assignment.copy()
    new_assignment[var] = False
    new_clauses = [[lit for lit in clause if lit != var] 
                   for clause in clauses if -var not in clause]
    
    if dpll(new_clauses, new_assignment):
        assignment.update(new_assignment)
        return True
    
    return False

# Vertex Cover - 2-Approximation
def vertex_cover_approx(edges):
    cover = set()
    edges_copy = edges.copy()
    
    while edges_copy:
        # Pick an arbitrary edge
        u, v = edges_copy[0]
        
        # Add both vertices to cover
        cover.add(u)
        cover.add(v)
        
        # Remove all edges incident to u or v
        edges_copy = [(a, b) for a, b in edges_copy 
                     if a != u and a != v and b != u and b != v]
    
    return cover
```

## Exercises

1. Implement and analyze different sorting algorithms:
   - Compare performance on different input types
   - Identify best/worst cases
   - Measure actual runtime vs theoretical complexity

2. Solve the following dynamic programming problems:
   - Maximum subarray sum (Kadane's algorithm)
   - Longest palindromic subsequence
   - Rod cutting problem
   - Word break problem

3. Implement graph algorithms:
   - Strongly connected components (Tarjan's or Kosaraju's)
   - Articulation points and bridges
   - Maximum flow (Ford-Fulkerson)
   - Bipartite matching

4. Design efficient algorithms for:
   - Finding kth largest element
   - Median of two sorted arrays
   - Rotate array by k positions
   - Maximum sum path in binary tree

5. Implement string matching algorithms:
   - Boyer-Moore
   - Aho-Corasick for multiple patterns
   - Longest common substring

6. Solve geometric problems:
   - Closest pair of points
   - Line sweep for rectangle intersection
   - Voronoi diagram construction

7. Analyze algorithm complexity:
   - Prove correctness using loop invariants
   - Derive recurrence relations
   - Solve using Master theorem

8. Implement approximation algorithms:
   - Set cover
   - Bin packing
   - Graph coloring

9. Design algorithms for:
   - LRU cache with O(1) operations
   - Median from data stream
   - Skyline problem

10. Optimize existing algorithms:
    - Cache-efficient matrix multiplication
    - Parallel merge sort
    - Space-optimized dynamic programming

## Summary

This chapter covered fundamental algorithms and complexity analysis:

- Algorithm analysis provides tools to evaluate efficiency
- Sorting algorithms trade off simplicity, speed, and stability
- Search algorithms efficiently locate data in various structures
- Graph algorithms solve connectivity and optimization problems
- Dynamic programming solves problems with overlapping subproblems
- Greedy algorithms make locally optimal choices
- Divide and conquer breaks problems into smaller pieces
- Backtracking systematically explores solution spaces
- String algorithms efficiently process text
- Computational geometry handles spatial problems
- NP-complete problems lack known polynomial solutions

Understanding algorithms and complexity is essential for solving computational problems efficiently and recognizing inherent limitations. The next chapter will explore database systems, examining how these algorithmic principles apply to managing and querying large-scale persistent data.