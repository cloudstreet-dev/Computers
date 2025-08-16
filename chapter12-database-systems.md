# Chapter 12: Database Systems

## Introduction

Database systems are specialized software designed to efficiently store, retrieve, and manage large amounts of structured data. They provide the foundation for most modern applications, from simple websites to complex enterprise systems. This chapter explores database concepts, relational model theory, SQL, transaction processing, and modern database architectures including NoSQL and distributed databases.

## 12.1 Database Fundamentals

### Database Management System (DBMS) Architecture

```
┌─────────────────────────────┐
│     User Applications       │
├─────────────────────────────┤
│      Query Processor        │
│  - Parser                   │
│  - Optimizer                │
│  - Executor                 │
├─────────────────────────────┤
│   Transaction Manager       │
│  - Concurrency Control      │
│  - Recovery Manager         │
├─────────────────────────────┤
│     Storage Manager         │
│  - Buffer Manager           │
│  - File Manager             │
│  - Index Manager            │
├─────────────────────────────┤
│      Physical Storage       │
│  - Data Files               │
│  - Index Files              │
│  - Log Files                │
└─────────────────────────────┘
```

### Data Models

**Hierarchical Model**: Tree structure
**Network Model**: Graph structure
**Relational Model**: Tables with rows and columns
**Object-Oriented Model**: Objects with attributes and methods
**Document Model**: Semi-structured documents (JSON/XML)
**Key-Value Model**: Simple key-value pairs
**Graph Model**: Nodes and edges with properties

## 12.2 Relational Model

### Relations and Schemas

```sql
-- Schema definition
CREATE TABLE Student (
    student_id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE,
    gpa DECIMAL(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0),
    enrollment_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE Course (
    course_id INTEGER PRIMARY KEY,
    course_name VARCHAR(200) NOT NULL,
    credits INTEGER CHECK (credits > 0),
    department VARCHAR(50)
);

CREATE TABLE Enrollment (
    student_id INTEGER,
    course_id INTEGER,
    semester VARCHAR(20),
    grade CHAR(2),
    PRIMARY KEY (student_id, course_id, semester),
    FOREIGN KEY (student_id) REFERENCES Student(student_id),
    FOREIGN KEY (course_id) REFERENCES Course(course_id)
);
```

### Relational Algebra

Fundamental operations:

```python
# Selection (σ): Filter rows
def select(relation, condition):
    return [row for row in relation if condition(row)]

# Projection (π): Select columns
def project(relation, columns):
    return [{col: row[col] for col in columns} for row in relation]

# Union (∪): Combine relations
def union(R, S):
    return list(set(tuple(r.items()) for r in R) | 
                set(tuple(s.items()) for s in S))

# Difference (-): Rows in R but not in S
def difference(R, S):
    return list(set(tuple(r.items()) for r in R) - 
                set(tuple(s.items()) for s in S))

# Cartesian Product (×): All combinations
def cartesian_product(R, S):
    result = []
    for r in R:
        for s in S:
            combined = r.copy()
            combined.update(s)
            result.append(combined)
    return result

# Join (⋈): Combine related rows
def natural_join(R, S):
    common_attrs = set(R[0].keys()) & set(S[0].keys())
    result = []
    
    for r in R:
        for s in S:
            if all(r[attr] == s[attr] for attr in common_attrs):
                combined = r.copy()
                combined.update(s)
                result.append(combined)
    
    return result
```

### Normalization

**First Normal Form (1NF)**: Atomic values only
```sql
-- Violates 1NF (multiple phone numbers in one field)
CREATE TABLE Bad_Contact (
    id INTEGER,
    name VARCHAR(100),
    phones VARCHAR(200)  -- "555-1234, 555-5678"
);

-- Satisfies 1NF
CREATE TABLE Good_Contact (
    id INTEGER,
    name VARCHAR(100)
);

CREATE TABLE Phone (
    contact_id INTEGER,
    phone_number VARCHAR(20),
    FOREIGN KEY (contact_id) REFERENCES Good_Contact(id)
);
```

**Second Normal Form (2NF)**: No partial dependencies
```sql
-- Violates 2NF (course_name depends only on course_id)
CREATE TABLE Bad_Enrollment (
    student_id INTEGER,
    course_id INTEGER,
    course_name VARCHAR(200),  -- Partial dependency
    grade CHAR(2),
    PRIMARY KEY (student_id, course_id)
);

-- Satisfies 2NF
CREATE TABLE Course (
    course_id INTEGER PRIMARY KEY,
    course_name VARCHAR(200)
);

CREATE TABLE Enrollment (
    student_id INTEGER,
    course_id INTEGER,
    grade CHAR(2),
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (course_id) REFERENCES Course(course_id)
);
```

**Third Normal Form (3NF)**: No transitive dependencies
```sql
-- Violates 3NF (dept_head depends on department, not student_id)
CREATE TABLE Bad_Student (
    student_id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    dept_head VARCHAR(100)  -- Transitive dependency
);

-- Satisfies 3NF
CREATE TABLE Department (
    dept_name VARCHAR(50) PRIMARY KEY,
    dept_head VARCHAR(100)
);

CREATE TABLE Student (
    student_id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    FOREIGN KEY (department) REFERENCES Department(dept_name)
);
```

## 12.3 SQL (Structured Query Language)

### Data Manipulation Language (DML)

```sql
-- SELECT with various clauses
SELECT s.name, c.course_name, e.grade
FROM Student s
JOIN Enrollment e ON s.student_id = e.student_id
JOIN Course c ON e.course_id = c.course_id
WHERE c.department = 'Computer Science'
  AND e.semester = 'Fall 2024'
  AND e.grade IS NOT NULL
ORDER BY s.name, c.course_name;

-- Aggregation and grouping
SELECT c.department, 
       COUNT(DISTINCT e.student_id) as num_students,
       AVG(CASE 
           WHEN e.grade = 'A' THEN 4.0
           WHEN e.grade = 'B' THEN 3.0
           WHEN e.grade = 'C' THEN 2.0
           WHEN e.grade = 'D' THEN 1.0
           WHEN e.grade = 'F' THEN 0.0
       END) as avg_gpa
FROM Course c
JOIN Enrollment e ON c.course_id = e.course_id
GROUP BY c.department
HAVING COUNT(DISTINCT e.student_id) > 10
ORDER BY avg_gpa DESC;

-- Subqueries
SELECT name
FROM Student
WHERE student_id IN (
    SELECT student_id
    FROM Enrollment
    GROUP BY student_id
    HAVING COUNT(DISTINCT course_id) > 5
);

-- Common Table Expressions (CTEs)
WITH HighPerformers AS (
    SELECT student_id, AVG(
        CASE 
            WHEN grade = 'A' THEN 4.0
            WHEN grade = 'B' THEN 3.0
            WHEN grade = 'C' THEN 2.0
            WHEN grade = 'D' THEN 1.0
            ELSE 0.0
        END
    ) as gpa
    FROM Enrollment
    GROUP BY student_id
    HAVING gpa > 3.5
)
SELECT s.name, hp.gpa
FROM Student s
JOIN HighPerformers hp ON s.student_id = hp.student_id;

-- Window functions
SELECT student_id,
       course_id,
       grade,
       ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY grade) as rank,
       AVG(credits) OVER (PARTITION BY student_id) as avg_credits
FROM Enrollment e
JOIN Course c ON e.course_id = c.course_id;

-- INSERT, UPDATE, DELETE
INSERT INTO Student (student_id, name, email)
VALUES (1001, 'Alice Smith', 'alice@example.com');

UPDATE Student
SET gpa = 3.75
WHERE student_id = 1001;

DELETE FROM Enrollment
WHERE student_id = 1001 AND semester = 'Spring 2023';
```

### Data Definition Language (DDL)

```sql
-- Create table with constraints
CREATE TABLE Professor (
    professor_id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10, 2),
    INDEX idx_department (department),
    CONSTRAINT chk_salary CHECK (salary > 0)
);

-- Alter table structure
ALTER TABLE Professor
ADD COLUMN office_number VARCHAR(20),
ADD CONSTRAINT fk_dept 
    FOREIGN KEY (department) 
    REFERENCES Department(dept_name)
    ON DELETE SET NULL
    ON UPDATE CASCADE;

-- Create views
CREATE VIEW StudentTranscript AS
SELECT s.name, c.course_name, e.semester, e.grade, c.credits
FROM Student s
JOIN Enrollment e ON s.student_id = e.student_id
JOIN Course c ON e.course_id = c.course_id;

-- Create stored procedures
DELIMITER //
CREATE PROCEDURE CalculateGPA(IN student_id INT, OUT gpa DECIMAL(3,2))
BEGIN
    SELECT AVG(
        CASE 
            WHEN grade = 'A' THEN 4.0 * credits
            WHEN grade = 'B' THEN 3.0 * credits
            WHEN grade = 'C' THEN 2.0 * credits
            WHEN grade = 'D' THEN 1.0 * credits
            ELSE 0.0
        END
    ) / SUM(credits) INTO gpa
    FROM Enrollment e
    JOIN Course c ON e.course_id = c.course_id
    WHERE e.student_id = student_id;
END //
DELIMITER ;

-- Create triggers
CREATE TRIGGER update_gpa
AFTER INSERT ON Enrollment
FOR EACH ROW
BEGIN
    DECLARE new_gpa DECIMAL(3,2);
    CALL CalculateGPA(NEW.student_id, new_gpa);
    UPDATE Student SET gpa = new_gpa WHERE student_id = NEW.student_id;
END;
```

## 12.4 Indexing and Query Optimization

### Index Structures

```python
# B+ Tree Index Implementation
class BPlusNode:
    def __init__(self, order, is_leaf=False):
        self.order = order
        self.keys = []
        self.values = []  # Child pointers or data pointers
        self.is_leaf = is_leaf
        self.next = None  # For leaf nodes

class BPlusTree:
    def __init__(self, order):
        self.root = BPlusNode(order, is_leaf=True)
        self.order = order
    
    def search(self, key):
        node = self.root
        
        while not node.is_leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.values[i]
        
        # Search in leaf
        for i, k in enumerate(node.keys):
            if k == key:
                return node.values[i]
        
        return None
    
    def insert(self, key, value):
        if len(self.root.keys) >= self.order - 1:
            # Split root
            new_root = BPlusNode(self.order)
            new_root.is_leaf = False
            new_root.values.append(self.root)
            self.split_child(new_root, 0)
            self.root = new_root
        
        self.insert_non_full(self.root, key, value)

# Hash Index
class HashIndex:
    def __init__(self, num_buckets):
        self.buckets = [[] for _ in range(num_buckets)]
        self.num_buckets = num_buckets
    
    def hash_function(self, key):
        return hash(key) % self.num_buckets
    
    def insert(self, key, value):
        bucket = self.hash_function(key)
        self.buckets[bucket].append((key, value))
    
    def search(self, key):
        bucket = self.hash_function(key)
        for k, v in self.buckets[bucket]:
            if k == key:
                return v
        return None

# Bitmap Index
class BitmapIndex:
    def __init__(self, column_values):
        self.bitmaps = {}
        
        for i, value in enumerate(column_values):
            if value not in self.bitmaps:
                self.bitmaps[value] = set()
            self.bitmaps[value].add(i)
    
    def query(self, value):
        return self.bitmaps.get(value, set())
    
    def range_query(self, min_val, max_val):
        result = set()
        for value, positions in self.bitmaps.items():
            if min_val <= value <= max_val:
                result |= positions
        return result
```

### Query Optimization

```sql
-- Query optimizer chooses execution plan
EXPLAIN SELECT s.name, COUNT(e.course_id) as course_count
FROM Student s
LEFT JOIN Enrollment e ON s.student_id = e.student_id
WHERE s.gpa > 3.0
GROUP BY s.student_id, s.name
HAVING course_count > 3;

-- Cost-based optimization considers:
-- 1. Table statistics (row count, cardinality)
-- 2. Available indexes
-- 3. Join algorithms (nested loop, hash, merge)
-- 4. Access methods (sequential scan, index scan)

-- Index hints
SELECT /*+ INDEX(Student idx_gpa) */ *
FROM Student
WHERE gpa > 3.5;

-- Query rewriting
-- Original (inefficient)
SELECT * FROM Student
WHERE student_id IN (
    SELECT student_id FROM Enrollment WHERE grade = 'A'
);

-- Rewritten (efficient)
SELECT DISTINCT s.*
FROM Student s
JOIN Enrollment e ON s.student_id = e.student_id
WHERE e.grade = 'A';
```

## 12.5 Transaction Processing

### ACID Properties

```python
class Transaction:
    def __init__(self, db_connection):
        self.conn = db_connection
        self.conn.begin()
        self.savepoint_stack = []
    
    # Atomicity: All or nothing
    def execute(self, operations):
        try:
            for op in operations:
                op.execute(self.conn)
            self.commit()
        except Exception as e:
            self.rollback()
            raise e
    
    # Consistency: Maintain invariants
    def check_constraints(self):
        # Check foreign keys
        # Check unique constraints
        # Check custom business rules
        pass
    
    # Isolation: Concurrent transactions don't interfere
    def set_isolation_level(self, level):
        levels = {
            'READ_UNCOMMITTED': 'SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED',
            'READ_COMMITTED': 'SET TRANSACTION ISOLATION LEVEL READ COMMITTED',
            'REPEATABLE_READ': 'SET TRANSACTION ISOLATION LEVEL REPEATABLE READ',
            'SERIALIZABLE': 'SET TRANSACTION ISOLATION LEVEL SERIALIZABLE'
        }
        self.conn.execute(levels[level])
    
    # Durability: Committed changes persist
    def commit(self):
        self.conn.commit()
        # Write to transaction log
        # Flush to disk
    
    def rollback(self):
        self.conn.rollback()
    
    def savepoint(self, name):
        self.conn.execute(f"SAVEPOINT {name}")
        self.savepoint_stack.append(name)
    
    def rollback_to_savepoint(self, name):
        self.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
```

### Concurrency Control

```python
# Two-Phase Locking (2PL)
class LockManager:
    def __init__(self):
        self.locks = {}  # resource -> (lock_type, transaction_id)
        self.wait_graph = {}  # For deadlock detection
    
    def acquire_lock(self, transaction_id, resource, lock_type):
        if resource not in self.locks:
            self.locks[resource] = []
        
        current_locks = self.locks[resource]
        
        # Check compatibility
        if lock_type == 'SHARED':
            # Can coexist with other shared locks
            for lock in current_locks:
                if lock[0] == 'EXCLUSIVE':
                    self.wait_for(transaction_id, lock[1])
                    return False
            
            self.locks[resource].append(('SHARED', transaction_id))
            return True
        
        else:  # EXCLUSIVE
            if current_locks:
                for lock in current_locks:
                    self.wait_for(transaction_id, lock[1])
                return False
            
            self.locks[resource].append(('EXCLUSIVE', transaction_id))
            return True
    
    def release_lock(self, transaction_id, resource):
        if resource in self.locks:
            self.locks[resource] = [
                lock for lock in self.locks[resource]
                if lock[1] != transaction_id
            ]
    
    def wait_for(self, waiter, holder):
        if waiter not in self.wait_graph:
            self.wait_graph[waiter] = []
        self.wait_graph[waiter].append(holder)
        
        # Check for deadlock
        if self.has_cycle():
            raise DeadlockException()
    
    def has_cycle(self):
        # Implement cycle detection in wait graph
        pass

# Multiversion Concurrency Control (MVCC)
class MVCCDatabase:
    def __init__(self):
        self.data = {}  # key -> [(value, version, deleted)]
        self.current_version = 0
    
    def begin_transaction(self):
        self.current_version += 1
        return self.current_version
    
    def read(self, transaction_version, key):
        if key not in self.data:
            return None
        
        # Find latest version visible to transaction
        for value, version, deleted in reversed(self.data[key]):
            if version <= transaction_version:
                return None if deleted else value
        
        return None
    
    def write(self, transaction_version, key, value):
        if key not in self.data:
            self.data[key] = []
        
        self.data[key].append((value, transaction_version, False))
    
    def delete(self, transaction_version, key):
        if key not in self.data:
            return
        
        self.data[key].append((None, transaction_version, True))
```

## 12.6 Recovery and Logging

```python
# Write-Ahead Logging (WAL)
class WAL:
    def __init__(self, log_file):
        self.log_file = log_file
        self.lsn = 0  # Log Sequence Number
    
    def log_record(self, record):
        self.lsn += 1
        record['lsn'] = self.lsn
        self.log_file.write(json.dumps(record) + '\n')
        self.log_file.flush()
        return self.lsn
    
    def log_begin(self, transaction_id):
        return self.log_record({
            'type': 'BEGIN',
            'transaction_id': transaction_id
        })
    
    def log_update(self, transaction_id, page_id, offset, old_value, new_value):
        return self.log_record({
            'type': 'UPDATE',
            'transaction_id': transaction_id,
            'page_id': page_id,
            'offset': offset,
            'old_value': old_value,
            'new_value': new_value
        })
    
    def log_commit(self, transaction_id):
        return self.log_record({
            'type': 'COMMIT',
            'transaction_id': transaction_id
        })
    
    def log_checkpoint(self, active_transactions):
        return self.log_record({
            'type': 'CHECKPOINT',
            'active_transactions': active_transactions
        })

# ARIES Recovery Algorithm
class ARIESRecovery:
    def recover(self, log_file):
        # Phase 1: Analysis
        checkpoint = self.find_last_checkpoint(log_file)
        redo_lsn = checkpoint['lsn'] if checkpoint else 0
        active_transactions = self.analyze_phase(log_file, redo_lsn)
        
        # Phase 2: Redo
        self.redo_phase(log_file, redo_lsn)
        
        # Phase 3: Undo
        self.undo_phase(log_file, active_transactions)
    
    def analyze_phase(self, log_file, start_lsn):
        active_transactions = set()
        
        for record in self.read_log_from(log_file, start_lsn):
            if record['type'] == 'BEGIN':
                active_transactions.add(record['transaction_id'])
            elif record['type'] == 'COMMIT' or record['type'] == 'ABORT':
                active_transactions.discard(record['transaction_id'])
        
        return active_transactions
    
    def redo_phase(self, log_file, start_lsn):
        for record in self.read_log_from(log_file, start_lsn):
            if record['type'] == 'UPDATE':
                # Check if page needs redo
                page = self.read_page(record['page_id'])
                if page.lsn < record['lsn']:
                    self.apply_update(record)
    
    def undo_phase(self, log_file, active_transactions):
        # Undo uncommitted transactions in reverse order
        for transaction_id in active_transactions:
            self.undo_transaction(log_file, transaction_id)
```

## 12.7 NoSQL Databases

### Document Stores

```python
# MongoDB-style document store
class DocumentStore:
    def __init__(self):
        self.collections = {}
    
    def create_collection(self, name):
        self.collections[name] = []
    
    def insert(self, collection, document):
        if '_id' not in document:
            document['_id'] = generate_object_id()
        
        self.collections[collection].append(document)
        return document['_id']
    
    def find(self, collection, query):
        results = []
        for doc in self.collections[collection]:
            if self.matches_query(doc, query):
                results.append(doc)
        return results
    
    def matches_query(self, document, query):
        for key, value in query.items():
            if key.startswith('$'):
                # Handle operators
                if key == '$gt':
                    return document.get(key) > value
                elif key == '$in':
                    return document.get(key) in value
                # ... more operators
            else:
                # Nested path support
                if '.' in key:
                    keys = key.split('.')
                    current = document
                    for k in keys:
                        if k not in current:
                            return False
                        current = current[k]
                    if current != value:
                        return False
                else:
                    if document.get(key) != value:
                        return False
        return True
    
    def update(self, collection, query, update):
        for doc in self.collections[collection]:
            if self.matches_query(doc, query):
                self.apply_update(doc, update)
    
    def apply_update(self, document, update):
        for key, value in update.items():
            if key == '$set':
                document.update(value)
            elif key == '$inc':
                for field, increment in value.items():
                    document[field] = document.get(field, 0) + increment
            # ... more update operators

# Example usage
db = DocumentStore()
db.create_collection('users')

# Insert document
user_id = db.insert('users', {
    'name': 'Alice',
    'age': 30,
    'email': 'alice@example.com',
    'address': {
        'street': '123 Main St',
        'city': 'Boston'
    }
})

# Query documents
results = db.find('users', {
    'age': {'$gt': 25},
    'address.city': 'Boston'
})
```

### Key-Value Stores

```python
# Redis-style key-value store
class KeyValueStore:
    def __init__(self):
        self.data = {}
        self.expiry = {}
    
    def get(self, key):
        if key in self.expiry:
            if time.time() > self.expiry[key]:
                del self.data[key]
                del self.expiry[key]
                return None
        
        return self.data.get(key)
    
    def set(self, key, value, ttl=None):
        self.data[key] = value
        if ttl:
            self.expiry[key] = time.time() + ttl
    
    def delete(self, key):
        self.data.pop(key, None)
        self.expiry.pop(key, None)
    
    # List operations
    def lpush(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].insert(0, value)
    
    def rpop(self, key):
        if key in self.data and self.data[key]:
            return self.data[key].pop()
        return None
    
    # Set operations
    def sadd(self, key, member):
        if key not in self.data:
            self.data[key] = set()
        self.data[key].add(member)
    
    # Hash operations
    def hset(self, key, field, value):
        if key not in self.data:
            self.data[key] = {}
        self.data[key][field] = value
```

### Graph Databases

```python
# Neo4j-style graph database
class GraphDatabase:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_id_counter = 0
    
    def create_node(self, labels, properties):
        node_id = self.node_id_counter
        self.node_id_counter += 1
        
        self.nodes[node_id] = {
            'id': node_id,
            'labels': labels,
            'properties': properties
        }
        return node_id
    
    def create_edge(self, from_node, to_node, relationship, properties=None):
        edge = {
            'from': from_node,
            'to': to_node,
            'type': relationship,
            'properties': properties or {}
        }
        self.edges.append(edge)
        return edge
    
    def cypher_query(self, query):
        # Simplified Cypher query execution
        # MATCH (n:Person {name: 'Alice'})-[:KNOWS]->(m:Person)
        # RETURN m.name
        pass
    
    def traverse(self, start_node, relationship_type, direction='OUT', max_depth=None):
        visited = set()
        queue = [(start_node, 0)]
        results = []
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if node_id in visited:
                continue
            
            if max_depth and depth > max_depth:
                continue
            
            visited.add(node_id)
            results.append(self.nodes[node_id])
            
            # Find connected nodes
            for edge in self.edges:
                if edge['type'] != relationship_type:
                    continue
                
                if direction in ['OUT', 'BOTH'] and edge['from'] == node_id:
                    queue.append((edge['to'], depth + 1))
                
                if direction in ['IN', 'BOTH'] and edge['to'] == node_id:
                    queue.append((edge['from'], depth + 1))
        
        return results
```

## 12.8 Distributed Databases

### Sharding and Partitioning

```python
# Consistent Hashing for Sharding
class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=150):
        self.nodes = nodes
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self._build_ring()
    
    def _hash(self, key):
        return hashlib.md5(key.encode()).hexdigest()
    
    def _build_ring(self):
        for node in self.nodes:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = self._hash(virtual_key)
                self.ring[hash_value] = node
    
    def get_node(self, key):
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node clockwise from the hash
        sorted_keys = sorted(self.ring.keys())
        for ring_hash in sorted_keys:
            if ring_hash >= hash_value:
                return self.ring[ring_hash]
        
        # Wrap around
        return self.ring[sorted_keys[0]]
    
    def add_node(self, node):
        self.nodes.append(node)
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
    
    def remove_node(self, node):
        self.nodes.remove(node)
        keys_to_remove = []
        for key, n in self.ring.items():
            if n == node:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.ring[key]

# Range-based Partitioning
class RangePartitioner:
    def __init__(self, partitions):
        # partitions: [(min_value, max_value, node), ...]
        self.partitions = sorted(partitions, key=lambda x: x[0])
    
    def get_partition(self, key):
        for min_val, max_val, node in self.partitions:
            if min_val <= key <= max_val:
                return node
        
        raise ValueError(f"Key {key} not in any partition")
```

### CAP Theorem and Consistency Models

```python
# Eventually Consistent System
class EventuallyConsistentStore:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers
        self.data = {}
        self.vector_clock = {node_id: 0}
    
    def write(self, key, value):
        # Update local version
        self.vector_clock[self.node_id] += 1
        
        self.data[key] = {
            'value': value,
            'version': self.vector_clock.copy()
        }
        
        # Async replicate to peers
        self.replicate_async(key, value, self.vector_clock.copy())
    
    def read(self, key, consistency_level='ONE'):
        if consistency_level == 'ONE':
            # Read from local
            return self.data.get(key, {}).get('value')
        
        elif consistency_level == 'QUORUM':
            # Read from majority
            responses = [self.data.get(key)]
            for peer in self.peers[:len(self.peers)//2]:
                responses.append(peer.get(key))
            
            # Return most recent version
            return self.resolve_conflicts(responses)
        
        elif consistency_level == 'ALL':
            # Read from all nodes
            responses = [self.data.get(key)]
            for peer in self.peers:
                responses.append(peer.get(key))
            
            return self.resolve_conflicts(responses)
    
    def resolve_conflicts(self, responses):
        # Use vector clocks to determine most recent
        latest = None
        latest_version = {}
        
        for response in responses:
            if response and self.is_later(response['version'], latest_version):
                latest = response['value']
                latest_version = response['version']
        
        return latest
    
    def is_later(self, v1, v2):
        # Vector clock comparison
        for node in v1:
            if node not in v2 or v1[node] > v2[node]:
                return True
        return False
```

## Exercises

1. Design a relational schema for a library system with books, authors, members, and loans. Normalize to 3NF.

2. Write SQL queries to:
   - Find the top 5 most borrowed books
   - List members with overdue books
   - Calculate the average loan duration by genre

3. Implement a B+ tree that supports:
   - Range queries
   - Bulk loading
   - Node splitting and merging

4. Create a transaction manager that handles:
   - Nested transactions
   - Savepoints
   - Deadlock detection

5. Design a query optimizer that:
   - Estimates query costs
   - Chooses between different join algorithms
   - Reorders joins for efficiency

6. Implement a simple document database with:
   - JSON document storage
   - Query language with operators
   - Indexing support

7. Build a distributed key-value store with:
   - Consistent hashing
   - Replication
   - Eventual consistency

8. Create a graph database that supports:
   - Node and edge creation
   - Path queries
   - Pattern matching

9. Implement MVCC (Multiversion Concurrency Control) for a simple database.

10. Design a system that can:
    - Shard data across multiple nodes
    - Handle node failures
    - Rebalance data when nodes are added/removed

## Summary

This chapter covered database systems comprehensively:

- Database fundamentals provide structured data management
- The relational model organizes data in tables with relationships
- SQL enables declarative data manipulation and definition
- Indexing and optimization improve query performance
- Transaction processing ensures ACID properties
- Recovery mechanisms protect against failures
- NoSQL databases offer alternative data models
- Distributed databases scale across multiple machines

Database systems are essential for managing persistent data in applications. Understanding their principles, from relational theory to distributed systems, enables effective data management at any scale. The combination of traditional RDBMS and modern NoSQL approaches provides tools for diverse data management needs.