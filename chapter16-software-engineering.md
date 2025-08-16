# Chapter 16: Software Engineering

## Introduction

Software engineering is the systematic application of engineering principles to the development, operation, and maintenance of software systems. It encompasses methodologies, practices, and tools that enable teams to build reliable, maintainable, and scalable software. This chapter explores software development lifecycles, design patterns, testing strategies, quality assurance, and modern software engineering practices that have evolved to meet the challenges of increasingly complex systems.

## 16.1 Software Development Methodologies

### Waterfall Model

```
┌─────────────────┐
│  Requirements   │
└────────┬────────┘
         │
┌────────▼────────┐
│     Design      │
└────────┬────────┘
         │
┌────────▼────────┐
│ Implementation  │
└────────┬────────┘
         │
┌────────▼────────┐
│   Verification  │
└────────┬────────┘
         │
┌────────▼────────┐
│  Maintenance    │
└─────────────────┘
```

### Agile Methodologies

```python
class ScrumFramework:
    def __init__(self, team_size, sprint_length=14):
        self.team_size = team_size
        self.sprint_length = sprint_length  # days
        self.product_backlog = []
        self.sprint_backlog = []
        self.completed_items = []
        
    def add_user_story(self, story):
        """Add user story to product backlog"""
        story['status'] = 'backlog'
        story['created_at'] = datetime.now()
        self.product_backlog.append(story)
    
    def sprint_planning(self, velocity):
        """Plan sprint based on team velocity"""
        self.sprint_backlog = []
        remaining_capacity = velocity
        
        # Sort backlog by priority
        sorted_backlog = sorted(self.product_backlog, 
                               key=lambda x: x['priority'])
        
        for story in sorted_backlog:
            if story['story_points'] <= remaining_capacity:
                self.sprint_backlog.append(story)
                self.product_backlog.remove(story)
                remaining_capacity -= story['story_points']
                story['status'] = 'in_sprint'
        
        return self.sprint_backlog
    
    def daily_standup(self):
        """Track daily progress"""
        standup = {
            'date': datetime.now(),
            'updates': []
        }
        
        for member in self.team:
            update = {
                'member': member,
                'yesterday': member.get_completed_tasks(),
                'today': member.get_planned_tasks(),
                'blockers': member.get_blockers()
            }
            standup['updates'].append(update)
        
        return standup
    
    def sprint_review(self):
        """Review completed work"""
        completed = [story for story in self.sprint_backlog 
                    if story['status'] == 'done']
        
        acceptance_rate = len(completed) / len(self.sprint_backlog)
        velocity = sum(story['story_points'] for story in completed)
        
        return {
            'completed_stories': completed,
            'acceptance_rate': acceptance_rate,
            'velocity': velocity
        }
    
    def sprint_retrospective(self):
        """Identify improvements"""
        return {
            'what_went_well': [],
            'what_to_improve': [],
            'action_items': []
        }

class KanbanBoard:
    def __init__(self, wip_limits):
        self.columns = {
            'backlog': [],
            'analysis': [],
            'development': [],
            'testing': [],
            'done': []
        }
        self.wip_limits = wip_limits  # Work in progress limits
        
    def add_item(self, item, column='backlog'):
        """Add item to board"""
        if column != 'backlog' and len(self.columns[column]) >= self.wip_limits.get(column, float('inf')):
            raise Exception(f"WIP limit exceeded for {column}")
        
        item['entered_column'] = datetime.now()
        self.columns[column].append(item)
    
    def move_item(self, item_id, from_column, to_column):
        """Move item between columns"""
        # Check WIP limit
        if to_column != 'done' and len(self.columns[to_column]) >= self.wip_limits.get(to_column, float('inf')):
            return False, "WIP limit exceeded"
        
        # Find and move item
        item = None
        for i, task in enumerate(self.columns[from_column]):
            if task['id'] == item_id:
                item = self.columns[from_column].pop(i)
                break
        
        if item:
            # Calculate cycle time
            item['cycle_time'] = datetime.now() - item['entered_column']
            item['entered_column'] = datetime.now()
            self.columns[to_column].append(item)
            return True, "Item moved"
        
        return False, "Item not found"
    
    def calculate_metrics(self):
        """Calculate flow metrics"""
        metrics = {
            'wip': sum(len(col) for col in self.columns.values() 
                      if col != 'backlog' and col != 'done'),
            'cycle_time': self.calculate_average_cycle_time(),
            'throughput': len(self.columns['done'])
        }
        return metrics
```

### DevOps Practices

```python
class CI_CD_Pipeline:
    def __init__(self, repository):
        self.repository = repository
        self.stages = []
        self.artifacts = {}
        
    def add_stage(self, stage):
        """Add stage to pipeline"""
        self.stages.append(stage)
    
    def execute(self, commit):
        """Execute pipeline for commit"""
        context = {
            'commit': commit,
            'status': 'running',
            'artifacts': {},
            'test_results': {},
            'metrics': {}
        }
        
        for stage in self.stages:
            try:
                result = stage.execute(context)
                context.update(result)
                
                if not result.get('success', True):
                    context['status'] = 'failed'
                    break
                    
            except Exception as e:
                context['status'] = 'error'
                context['error'] = str(e)
                break
        
        if context['status'] == 'running':
            context['status'] = 'success'
        
        return context

class BuildStage:
    def execute(self, context):
        """Compile and build application"""
        return {
            'success': True,
            'artifacts': {
                'binary': 'app.exe',
                'libraries': ['lib1.dll', 'lib2.dll']
            }
        }

class TestStage:
    def execute(self, context):
        """Run automated tests"""
        test_results = {
            'unit_tests': self.run_unit_tests(),
            'integration_tests': self.run_integration_tests(),
            'coverage': self.calculate_coverage()
        }
        
        success = all(result['passed'] for result in test_results.values())
        
        return {
            'success': success,
            'test_results': test_results
        }
    
    def run_unit_tests(self):
        # Simulate test execution
        return {
            'passed': True,
            'total': 150,
            'failed': 0,
            'duration': 12.5
        }
    
    def run_integration_tests(self):
        return {
            'passed': True,
            'total': 25,
            'failed': 0,
            'duration': 45.3
        }
    
    def calculate_coverage(self):
        return {
            'line_coverage': 85.2,
            'branch_coverage': 78.9
        }

class DeploymentStage:
    def __init__(self, environment):
        self.environment = environment
        
    def execute(self, context):
        """Deploy to environment"""
        if self.environment == 'production':
            # Blue-green deployment
            return self.blue_green_deploy(context)
        else:
            return self.standard_deploy(context)
    
    def blue_green_deploy(self, context):
        """Blue-green deployment strategy"""
        steps = [
            self.deploy_to_green(),
            self.run_smoke_tests(),
            self.switch_traffic(),
            self.monitor_metrics()
        ]
        
        for step in steps:
            if not step['success']:
                self.rollback()
                return {'success': False, 'rollback': True}
        
        return {'success': True, 'deployment': 'blue-green'}
    
    def canary_deploy(self, context, percentage=10):
        """Canary deployment strategy"""
        steps = [
            self.deploy_canary(percentage),
            self.monitor_canary(),
            self.gradual_rollout(),
            self.full_deployment()
        ]
        
        for step in steps:
            if not step['success']:
                self.rollback()
                return {'success': False, 'rollback': True}
        
        return {'success': True, 'deployment': 'canary'}
```

## 16.2 Software Design Principles

### SOLID Principles

```python
# Single Responsibility Principle
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user):
        # Save user to database
        pass
    
    def find(self, user_id):
        # Find user in database
        pass

class UserValidator:
    def validate(self, user):
        # Validate user data
        return self.validate_email(user.email)
    
    def validate_email(self, email):
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

# Open/Closed Principle
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

class AreaCalculator:
    def calculate_total_area(self, shapes):
        return sum(shape.area() for shape in shapes)

# Liskov Substitution Principle
class Bird(ABC):
    @abstractmethod
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        return self.fly()
    
    def fly(self):
        return "Flying through the air"

class SwimmingBird(Bird):
    def move(self):
        return self.swim()
    
    def swim(self):
        return "Swimming in water"

# Interface Segregation Principle
class Printable(ABC):
    @abstractmethod
    def print(self):
        pass

class Scannable(ABC):
    @abstractmethod
    def scan(self):
        pass

class Faxable(ABC):
    @abstractmethod
    def fax(self):
        pass

class MultiFunctionPrinter(Printable, Scannable, Faxable):
    def print(self):
        return "Printing document"
    
    def scan(self):
        return "Scanning document"
    
    def fax(self):
        return "Faxing document"

class SimplePrinter(Printable):
    def print(self):
        return "Printing document"

# Dependency Inversion Principle
class MessageSender(ABC):
    @abstractmethod
    def send(self, message):
        pass

class EmailSender(MessageSender):
    def send(self, message):
        return f"Sending email: {message}"

class SMSSender(MessageSender):
    def send(self, message):
        return f"Sending SMS: {message}"

class NotificationService:
    def __init__(self, sender: MessageSender):
        self.sender = sender  # Depends on abstraction
    
    def notify(self, message):
        return self.sender.send(message)
```

### Design Patterns

```python
# Creational Patterns

# Singleton
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.config = {}

# Factory Method
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == 'dog':
            return Dog()
        elif animal_type == 'cat':
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Builder
class QueryBuilder:
    def __init__(self):
        self.query_parts = {
            'select': [],
            'from': None,
            'where': [],
            'order_by': [],
            'limit': None
        }
    
    def select(self, *columns):
        self.query_parts['select'].extend(columns)
        return self
    
    def from_table(self, table):
        self.query_parts['from'] = table
        return self
    
    def where(self, condition):
        self.query_parts['where'].append(condition)
        return self
    
    def order_by(self, column, direction='ASC'):
        self.query_parts['order_by'].append(f"{column} {direction}")
        return self
    
    def limit(self, n):
        self.query_parts['limit'] = n
        return self
    
    def build(self):
        query = f"SELECT {', '.join(self.query_parts['select'])}"
        query += f" FROM {self.query_parts['from']}"
        
        if self.query_parts['where']:
            query += f" WHERE {' AND '.join(self.query_parts['where'])}"
        
        if self.query_parts['order_by']:
            query += f" ORDER BY {', '.join(self.query_parts['order_by'])}"
        
        if self.query_parts['limit']:
            query += f" LIMIT {self.query_parts['limit']}"
        
        return query

# Structural Patterns

# Adapter
class LegacyPrinter:
    def print_document(self, text):
        return f"Legacy printing: {text}"

class ModernPrinter:
    def print(self, document):
        return f"Modern printing: {document}"

class PrinterAdapter:
    def __init__(self, legacy_printer):
        self.legacy_printer = legacy_printer
    
    def print(self, document):
        return self.legacy_printer.print_document(document)

# Decorator
class Coffee:
    def cost(self):
        return 2.0
    
    def description(self):
        return "Coffee"

class CoffeeDecorator:
    def __init__(self, coffee):
        self.coffee = coffee
    
    def cost(self):
        return self.coffee.cost()
    
    def description(self):
        return self.coffee.description()

class MilkDecorator(CoffeeDecorator):
    def cost(self):
        return self.coffee.cost() + 0.5
    
    def description(self):
        return self.coffee.description() + ", Milk"

class SugarDecorator(CoffeeDecorator):
    def cost(self):
        return self.coffee.cost() + 0.2
    
    def description(self):
        return self.coffee.description() + ", Sugar"

# Proxy
class DatabaseProxy:
    def __init__(self, database):
        self.database = database
        self.cache = {}
    
    def query(self, sql):
        if sql in self.cache:
            return self.cache[sql]
        
        result = self.database.query(sql)
        self.cache[sql] = result
        return result

# Behavioral Patterns

# Observer
class Subject:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def detach(self, observer):
        self.observers.remove(observer)
    
    def notify(self, event):
        for observer in self.observers:
            observer.update(event)

class Observer:
    def update(self, event):
        pass

# Strategy
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def sort(self, data):
        return self.strategy.sort(data)

# Command
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class InsertCommand(Command):
    def __init__(self, document, position, text):
        self.document = document
        self.position = position
        self.text = text
    
    def execute(self):
        self.document.insert(self.position, self.text)
    
    def undo(self):
        self.document.delete(self.position, len(self.text))

class CommandHistory:
    def __init__(self):
        self.history = []
        self.current = -1
    
    def execute_command(self, command):
        command.execute()
        self.history = self.history[:self.current + 1]
        self.history.append(command)
        self.current += 1
    
    def undo(self):
        if self.current >= 0:
            self.history[self.current].undo()
            self.current -= 1
    
    def redo(self):
        if self.current < len(self.history) - 1:
            self.current += 1
            self.history[self.current].execute()
```

## 16.3 Software Testing

### Unit Testing

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    
    def calculate_average(self, numbers):
        if not numbers:
            return 0
        return sum(numbers) / len(numbers)

class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = Calculator()
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers"""
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addition of negative numbers"""
        result = self.calculator.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_divide_normal(self):
        """Test normal division"""
        result = self.calculator.divide(10, 2)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises exception"""
        with self.assertRaises(ValueError) as context:
            self.calculator.divide(10, 0)
        
        self.assertEqual(str(context.exception), "Division by zero")
    
    def test_calculate_average_empty_list(self):
        """Test average of empty list"""
        result = self.calculator.calculate_average([])
        self.assertEqual(result, 0)
    
    def test_calculate_average_normal(self):
        """Test average calculation"""
        result = self.calculator.calculate_average([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 3.0)

class TestDatabaseService(unittest.TestCase):
    @patch('database.connection')
    def test_save_user(self, mock_connection):
        """Test saving user with mocked database"""
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        
        service = UserService(mock_connection)
        service.save_user({'name': 'Alice', 'email': 'alice@example.com'})
        
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()
    
    @patch('requests.get')
    def test_fetch_data(self, mock_get):
        """Test API call with mocked response"""
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        service = APIService()
        result = service.fetch_data('http://api.example.com/data')
        
        self.assertEqual(result['status'], 'success')
        mock_get.assert_called_with('http://api.example.com/data')

# Test-Driven Development (TDD)
class StringCalculator:
    """Example of TDD approach"""
    
    def add(self, numbers):
        if not numbers:
            return 0
        
        if ',' in numbers:
            parts = numbers.split(',')
            return sum(int(part.strip()) for part in parts)
        
        return int(numbers)

class TestStringCalculator(unittest.TestCase):
    def test_empty_string_returns_zero(self):
        """Written first, before implementation"""
        calc = StringCalculator()
        self.assertEqual(calc.add(""), 0)
    
    def test_single_number(self):
        """Test drives implementation"""
        calc = StringCalculator()
        self.assertEqual(calc.add("5"), 5)
    
    def test_two_numbers(self):
        """Test drives comma handling"""
        calc = StringCalculator()
        self.assertEqual(calc.add("1,2"), 3)
    
    def test_multiple_numbers(self):
        """Test drives multiple number handling"""
        calc = StringCalculator()
        self.assertEqual(calc.add("1,2,3,4"), 10)
```

### Integration Testing

```python
class IntegrationTest:
    def setup_test_environment(self):
        """Set up test database and services"""
        self.test_db = self.create_test_database()
        self.api_server = self.start_test_server()
        self.test_data = self.load_test_fixtures()
    
    def test_user_registration_flow(self):
        """Test complete user registration"""
        # Create user through API
        response = self.api_client.post('/users', {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'secure123'
        })
        
        assert response.status_code == 201
        user_id = response.json()['id']
        
        # Verify user in database
        user = self.db.query("SELECT * FROM users WHERE id = ?", user_id)
        assert user is not None
        assert user['email'] == 'test@example.com'
        
        # Verify email was sent
        emails = self.email_service.get_sent_emails()
        assert len(emails) == 1
        assert emails[0]['to'] == 'test@example.com'
        assert 'Welcome' in emails[0]['subject']
        
        # Test login with new user
        login_response = self.api_client.post('/login', {
            'email': 'test@example.com',
            'password': 'secure123'
        })
        
        assert login_response.status_code == 200
        assert 'token' in login_response.json()
    
    def test_order_processing_pipeline(self):
        """Test order processing from creation to fulfillment"""
        # Create order
        order = self.create_test_order()
        
        # Process payment
        payment_result = self.payment_service.process(order)
        assert payment_result['status'] == 'success'
        
        # Update inventory
        inventory_result = self.inventory_service.reserve_items(order)
        assert inventory_result['reserved'] == True
        
        # Generate shipping label
        shipping = self.shipping_service.create_label(order)
        assert shipping['tracking_number'] is not None
        
        # Verify order status
        final_order = self.order_service.get_order(order['id'])
        assert final_order['status'] == 'shipped'
```

### End-to-End Testing

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class E2ETest:
    def setup(self):
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
    
    def teardown(self):
        self.driver.quit()
    
    def test_user_journey(self):
        """Test complete user journey through application"""
        # Navigate to homepage
        self.driver.get("https://example.com")
        
        # Click sign up
        signup_button = self.wait.until(
            EC.element_to_be_clickable((By.ID, "signup-button"))
        )
        signup_button.click()
        
        # Fill registration form
        self.driver.find_element(By.ID, "username").send_keys("testuser")
        self.driver.find_element(By.ID, "email").send_keys("test@example.com")
        self.driver.find_element(By.ID, "password").send_keys("secure123")
        self.driver.find_element(By.ID, "submit").click()
        
        # Wait for redirect to dashboard
        self.wait.until(EC.url_contains("/dashboard"))
        
        # Verify welcome message
        welcome = self.driver.find_element(By.CLASS_NAME, "welcome-message")
        assert "Welcome, testuser" in welcome.text
        
        # Create a new item
        self.driver.find_element(By.ID, "new-item").click()
        self.driver.find_element(By.ID, "item-name").send_keys("Test Item")
        self.driver.find_element(By.ID, "save-item").click()
        
        # Verify item appears in list
        items = self.driver.find_elements(By.CLASS_NAME, "item")
        assert len(items) == 1
        assert "Test Item" in items[0].text

class PerformanceTest:
    def test_api_response_time(self):
        """Test API performance under load"""
        import concurrent.futures
        import time
        
        def make_request():
            start = time.time()
            response = requests.get("https://api.example.com/users")
            duration = time.time() - start
            return duration, response.status_code
        
        # Simulate concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(make_request) for _ in range(1000)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        durations = [r[0] for r in results]
        status_codes = [r[1] for r in results]
        
        # Performance assertions
        assert sum(durations) / len(durations) < 0.5  # Average < 500ms
        assert max(durations) < 2.0  # Max < 2 seconds
        assert status_codes.count(200) / len(status_codes) > 0.99  # 99% success
```

## 16.4 Code Quality and Metrics

### Static Analysis

```python
class CodeAnalyzer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.ast = ast.parse(source_code)
        
    def calculate_cyclomatic_complexity(self, node):
        """Calculate McCabe cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def calculate_lines_of_code(self):
        """Calculate LOC metrics"""
        lines = self.source_code.split('\n')
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0
        }
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics['blank_lines'] += 1
            elif stripped.startswith('#'):
                metrics['comment_lines'] += 1
            else:
                metrics['code_lines'] += 1
        
        return metrics
    
    def find_code_smells(self):
        """Identify potential code smells"""
        smells = []
        
        for node in ast.walk(self.ast):
            # Long method
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 20:
                    smells.append({
                        'type': 'long_method',
                        'name': node.name,
                        'lines': len(node.body)
                    })
                
                # Too many parameters
                if len(node.args.args) > 5:
                    smells.append({
                        'type': 'too_many_parameters',
                        'name': node.name,
                        'count': len(node.args.args)
                    })
            
            # Large class
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    smells.append({
                        'type': 'large_class',
                        'name': node.name,
                        'methods': len(methods)
                    })
        
        return smells
    
    def calculate_maintainability_index(self):
        """Calculate maintainability index"""
        loc = self.calculate_lines_of_code()['code_lines']
        complexity = self.calculate_average_complexity()
        
        # Simplified maintainability index formula
        mi = 171 - 5.2 * math.log(loc) - 0.23 * complexity
        mi = max(0, min(100, mi))  # Normalize to 0-100
        
        return mi

class CodeFormatter:
    def format_python(self, code):
        """Format Python code using Black-style rules"""
        import black
        
        try:
            formatted = black.format_str(code, mode=black.Mode())
            return formatted
        except Exception as e:
            return code  # Return original if formatting fails
    
    def check_style_violations(self, code):
        """Check PEP 8 style violations"""
        import pycodestyle
        
        style_guide = pycodestyle.StyleGuide()
        result = style_guide.check_files([code])
        
        violations = []
        for error in result.messages:
            violations.append({
                'line': error.line,
                'column': error.column,
                'code': error.code,
                'message': error.message
            })
        
        return violations
```

## 16.5 Documentation

### Code Documentation

```python
class DocumentationGenerator:
    def generate_function_docs(self, func):
        """
        Generate documentation for a function.
        
        Args:
            func: Function object to document
        
        Returns:
            dict: Documentation structure with signature, docstring, and parameters
        
        Example:
            >>> def add(a: int, b: int) -> int:
            ...     '''Add two numbers'''
            ...     return a + b
            >>> gen = DocumentationGenerator()
            >>> docs = gen.generate_function_docs(add)
            >>> print(docs['signature'])
            'add(a: int, b: int) -> int'
        """
        import inspect
        
        docs = {
            'name': func.__name__,
            'signature': str(inspect.signature(func)),
            'docstring': inspect.getdoc(func),
            'parameters': {},
            'returns': None
        }
        
        # Parse parameters
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            docs['parameters'][param_name] = {
                'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
        
        # Parse return type
        if sig.return_annotation != inspect.Signature.empty:
            docs['returns'] = sig.return_annotation
        
        return docs
    
    def generate_api_docs(self, module):
        """Generate API documentation for a module"""
        import pydoc
        
        api_docs = {
            'module': module.__name__,
            'description': module.__doc__,
            'classes': {},
            'functions': {}
        }
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                api_docs['classes'][name] = self.document_class(obj)
            elif inspect.isfunction(obj):
                api_docs['functions'][name] = self.generate_function_docs(obj)
        
        return api_docs
    
    def generate_markdown(self, docs):
        """Convert documentation to Markdown format"""
        md = []
        
        md.append(f"# {docs['module']}\n")
        md.append(f"{docs['description']}\n")
        
        if docs['classes']:
            md.append("## Classes\n")
            for class_name, class_docs in docs['classes'].items():
                md.append(f"### {class_name}\n")
                md.append(f"{class_docs['docstring']}\n")
                
                for method_name, method_docs in class_docs['methods'].items():
                    md.append(f"#### {method_name}\n")
                    md.append(f"```python\n{method_docs['signature']}\n```\n")
                    md.append(f"{method_docs['docstring']}\n")
        
        if docs['functions']:
            md.append("## Functions\n")
            for func_name, func_docs in docs['functions'].items():
                md.append(f"### {func_name}\n")
                md.append(f"```python\n{func_docs['signature']}\n```\n")
                md.append(f"{func_docs['docstring']}\n")
        
        return '\n'.join(md)
```

## 16.6 Version Control

### Git Workflow

```python
class GitRepository:
    def __init__(self, path):
        self.path = path
        self.repo = git.Repo(path)
    
    def create_feature_branch(self, feature_name):
        """Create and checkout feature branch"""
        branch_name = f"feature/{feature_name}"
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()
        return branch_name
    
    def commit_changes(self, message, files=None):
        """Stage and commit changes"""
        if files:
            self.repo.index.add(files)
        else:
            self.repo.index.add('*')
        
        self.repo.index.commit(message)
        return self.repo.head.commit.hexsha
    
    def merge_branch(self, branch_name, strategy='recursive'):
        """Merge branch into current branch"""
        try:
            self.repo.git.merge(branch_name, strategy=strategy)
            return True, "Merge successful"
        except git.GitCommandError as e:
            # Handle merge conflicts
            conflicts = self.get_merge_conflicts()
            return False, conflicts
    
    def get_merge_conflicts(self):
        """Get list of conflicted files"""
        conflicts = []
        for item in self.repo.index.entries:
            if item.stage != 0:
                conflicts.append(item.path)
        return conflicts
    
    def create_pull_request(self, title, description, base_branch='main'):
        """Create pull request (GitHub example)"""
        current_branch = self.repo.active_branch.name
        
        pr_data = {
            'title': title,
            'body': description,
            'head': current_branch,
            'base': base_branch,
            'draft': False
        }
        
        # This would use GitHub API in practice
        return pr_data
    
    def code_review_checklist(self):
        """Generate code review checklist"""
        return {
            'functionality': [
                'Code accomplishes intended purpose',
                'Edge cases are handled',
                'Error handling is appropriate'
            ],
            'design': [
                'Code follows SOLID principles',
                'Appropriate design patterns used',
                'No unnecessary complexity'
            ],
            'testing': [
                'Unit tests cover new functionality',
                'Integration tests updated if needed',
                'All tests pass'
            ],
            'style': [
                'Code follows style guidelines',
                'Naming is clear and consistent',
                'Comments explain complex logic'
            ],
            'performance': [
                'No obvious performance issues',
                'Efficient algorithms used',
                'Resource usage is reasonable'
            ],
            'security': [
                'Input validation present',
                'No sensitive data exposed',
                'Authentication/authorization correct'
            ]
        }
```

## 16.7 Software Architecture

### Microservices Architecture

```python
class MicroserviceTemplate:
    def __init__(self, service_name, port):
        self.service_name = service_name
        self.port = port
        self.app = Flask(service_name)
        self.setup_routes()
        self.setup_middleware()
    
    def setup_routes(self):
        """Define service endpoints"""
        @self.app.route('/health')
        def health():
            return {'status': 'healthy', 'service': self.service_name}
        
        @self.app.route('/metrics')
        def metrics():
            return self.get_metrics()
    
    def setup_middleware(self):
        """Setup common middleware"""
        # Authentication
        @self.app.before_request
        def authenticate():
            token = request.headers.get('Authorization')
            if not self.validate_token(token):
                abort(401)
        
        # Logging
        @self.app.before_request
        def log_request():
            logger.info(f"{request.method} {request.path}")
        
        # Error handling
        @self.app.errorhandler(Exception)
        def handle_error(error):
            logger.error(f"Error: {str(error)}")
            return {'error': str(error)}, 500
    
    def register_with_discovery(self, discovery_url):
        """Register service with discovery service"""
        registration = {
            'name': self.service_name,
            'host': socket.gethostname(),
            'port': self.port,
            'health_check': f"http://{socket.gethostname()}:{self.port}/health"
        }
        
        response = requests.post(f"{discovery_url}/register", json=registration)
        return response.status_code == 200

class APIGateway:
    def __init__(self):
        self.services = {}
        self.circuit_breakers = {}
        
    def register_service(self, name, url):
        """Register backend service"""
        self.services[name] = url
        self.circuit_breakers[name] = CircuitBreaker(name)
    
    def route_request(self, service_name, path, method='GET', **kwargs):
        """Route request to appropriate service"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        circuit_breaker = self.circuit_breakers[service_name]
        
        if circuit_breaker.is_open():
            return {'error': 'Service unavailable'}, 503
        
        try:
            url = f"{self.services[service_name]}{path}"
            response = requests.request(method, url, **kwargs)
            
            if response.status_code >= 500:
                circuit_breaker.record_failure()
            else:
                circuit_breaker.record_success()
            
            return response.json(), response.status_code
            
        except Exception as e:
            circuit_breaker.record_failure()
            raise

class CircuitBreaker:
    def __init__(self, name, threshold=5, timeout=60):
        self.name = name
        self.failure_threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def is_open(self):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
                return False
            return True
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'closed'
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
```

## 16.8 Monitoring and Observability

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def record_metric(self, name, value, tags=None):
        """Record a metric value"""
        metric = {
            'timestamp': time.time(),
            'value': value,
            'tags': tags or {}
        }
        self.metrics[name].append(metric)
    
    def get_metric_stats(self, name, window=3600):
        """Get statistics for a metric"""
        now = time.time()
        recent = [m for m in self.metrics[name] 
                 if now - m['timestamp'] < window]
        
        if not recent:
            return None
        
        values = [m['value'] for m in recent]
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': self.percentile(values, 50),
            'p95': self.percentile(values, 95),
            'p99': self.percentile(values, 99)
        }
    
    def percentile(self, values, p):
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class Logger:
    def __init__(self, name, level='INFO'):
        self.name = name
        self.level = level
        self.handlers = []
    
    def log(self, level, message, **context):
        """Log message with context"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'context': context
        }
        
        for handler in self.handlers:
            handler.emit(log_entry)
    
    def info(self, message, **context):
        self.log('INFO', message, **context)
    
    def error(self, message, **context):
        self.log('ERROR', message, **context)
    
    def debug(self, message, **context):
        self.log('DEBUG', message, **context)

class DistributedTracing:
    def __init__(self):
        self.traces = {}
    
    def start_trace(self, operation):
        """Start a new trace"""
        trace_id = self.generate_trace_id()
        span_id = self.generate_span_id()
        
        trace = {
            'trace_id': trace_id,
            'spans': [{
                'span_id': span_id,
                'parent_id': None,
                'operation': operation,
                'start_time': time.time(),
                'tags': {},
                'logs': []
            }]
        }
        
        self.traces[trace_id] = trace
        return trace_id, span_id
    
    def start_span(self, trace_id, parent_id, operation):
        """Start a new span within a trace"""
        span_id = self.generate_span_id()
        
        span = {
            'span_id': span_id,
            'parent_id': parent_id,
            'operation': operation,
            'start_time': time.time(),
            'tags': {},
            'logs': []
        }
        
        self.traces[trace_id]['spans'].append(span)
        return span_id
    
    def end_span(self, trace_id, span_id):
        """End a span"""
        trace = self.traces.get(trace_id)
        if trace:
            for span in trace['spans']:
                if span['span_id'] == span_id:
                    span['end_time'] = time.time()
                    span['duration'] = span['end_time'] - span['start_time']
                    break
```

## Exercises

1. Implement a complete Scrum management system with:
   - Sprint planning and tracking
   - Burndown chart generation
   - Velocity calculation

2. Create a design pattern library that:
   - Provides implementations of all GoF patterns
   - Includes usage examples
   - Suggests appropriate patterns for problems

3. Build a comprehensive testing framework with:
   - Test discovery and execution
   - Mocking and stubbing support
   - Coverage reporting

4. Design a code quality analyzer that:
   - Calculates various metrics
   - Identifies code smells
   - Suggests refactoring opportunities

5. Implement a CI/CD pipeline that:
   - Builds projects
   - Runs tests
   - Deploys to multiple environments

6. Create a documentation generator that:
   - Extracts docs from code
   - Generates API documentation
   - Creates UML diagrams

7. Build a microservices framework with:
   - Service discovery
   - Load balancing
   - Circuit breakers

8. Implement a monitoring system that:
   - Collects metrics
   - Provides alerting
   - Visualizes system health

9. Design a version control system that:
   - Handles branching and merging
   - Supports distributed development
   - Includes code review features

10. Create a software architecture analyzer that:
    - Identifies architectural patterns
    - Detects violations
    - Suggests improvements

## Summary

This chapter covered software engineering principles and practices:

- Development methodologies guide project organization and execution
- Design principles and patterns promote maintainable code
- Testing ensures software quality and reliability
- Code quality metrics identify improvement areas
- Documentation aids understanding and maintenance
- Version control enables collaboration and history tracking
- Architecture patterns address system-level concerns
- Monitoring provides visibility into system behavior

Software engineering combines technical skills with processes and practices to deliver quality software efficiently. These principles apply across all development contexts, from individual projects to large-scale enterprise systems.