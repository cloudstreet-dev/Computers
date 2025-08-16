# Chapter 15: Compilers and Interpreters

## Introduction

Compilers and interpreters are fundamental software tools that bridge the gap between human-readable source code and machine-executable instructions. A compiler translates an entire program from a high-level language into machine code before execution, while an interpreter executes source code directly, translating and executing it line by line. This chapter explores the theory and implementation of both approaches, covering lexical analysis, parsing, semantic analysis, code generation, and optimization techniques.

## 15.1 Language Processing Overview

### Compilation vs Interpretation

```
Compilation Pipeline:
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Source  │────>│ Compiler │────>│ Machine  │
│   Code   │     │          │     │   Code   │
└──────────┘     └──────────┘     └──────────┘
                                        │
                                        v
                                   ┌──────────┐
                                   │ Execution│
                                   └──────────┘

Interpretation Pipeline:
┌──────────┐     ┌──────────────┐
│  Source  │────>│ Interpreter  │────> Output
│   Code   │     │              │
└──────────┘     └──────────────┘
```

### Compiler Phases

```python
class CompilerPipeline:
    def compile(self, source_code):
        # Frontend
        tokens = self.lexical_analysis(source_code)
        ast = self.syntax_analysis(tokens)
        symbol_table = self.semantic_analysis(ast)
        
        # Middle-end
        ir = self.intermediate_generation(ast, symbol_table)
        optimized_ir = self.optimization(ir)
        
        # Backend
        assembly = self.code_generation(optimized_ir)
        machine_code = self.assembly(assembly)
        
        return machine_code
```

## 15.2 Lexical Analysis

### Token Recognition

```python
import re
from enum import Enum, auto

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    FUNCTION = auto()
    RETURN = auto()
    VAR = auto()
    CONST = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    
    # Special
    EOF = auto()
    NEWLINE = auto()

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.line}, {self.column})"

class Lexer:
    def __init__(self, source_code):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Define token patterns
        self.patterns = [
            # Whitespace and comments
            (r'[ \t]+', None),  # Skip spaces and tabs
            (r'//[^\n]*', None),  # Skip single-line comments
            (r'/\*.*?\*/', None),  # Skip multi-line comments
            
            # Keywords
            (r'\bif\b', TokenType.IF),
            (r'\belse\b', TokenType.ELSE),
            (r'\bwhile\b', TokenType.WHILE),
            (r'\bfor\b', TokenType.FOR),
            (r'\bfunction\b', TokenType.FUNCTION),
            (r'\breturn\b', TokenType.RETURN),
            (r'\bvar\b', TokenType.VAR),
            (r'\bconst\b', TokenType.CONST),
            
            # Identifiers and literals
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
            (r'\d+\.?\d*', TokenType.NUMBER),
            (r'"([^"\\\\]|\\\\.)*"', TokenType.STRING),
            
            # Operators
            (r'==', TokenType.EQUAL),
            (r'!=', TokenType.NOT_EQUAL),
            (r'<=', TokenType.LESS_THAN),
            (r'>=', TokenType.GREATER_THAN),
            (r'<', TokenType.LESS_THAN),
            (r'>', TokenType.GREATER_THAN),
            (r'\+', TokenType.PLUS),
            (r'-', TokenType.MINUS),
            (r'\*', TokenType.MULTIPLY),
            (r'/', TokenType.DIVIDE),
            (r'=', TokenType.ASSIGN),
            
            # Delimiters
            (r'\(', TokenType.LPAREN),
            (r'\)', TokenType.RPAREN),
            (r'\{', TokenType.LBRACE),
            (r'\}', TokenType.RBRACE),
            (r';', TokenType.SEMICOLON),
            (r',', TokenType.COMMA),
            (r'\n', TokenType.NEWLINE),
        ]
        
        # Compile patterns
        self.regex = [(re.compile(pattern), token_type) 
                     for pattern, token_type in self.patterns]
    
    def tokenize(self):
        while self.position < len(self.source):
            match_found = False
            
            for pattern, token_type in self.regex:
                match = pattern.match(self.source, self.position)
                if match:
                    value = match.group(0)
                    
                    if token_type:  # Not whitespace or comment
                        if token_type == TokenType.NEWLINE:
                            self.line += 1
                            self.column = 1
                        else:
                            token = Token(token_type, value, self.line, self.column)
                            self.tokens.append(token)
                    
                    self.position = match.end()
                    self.column += len(value)
                    match_found = True
                    break
            
            if not match_found:
                raise SyntaxError(f"Unexpected character '{self.source[self.position]}' "
                                f"at line {self.line}, column {self.column}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

# Finite Automaton for Token Recognition
class FiniteAutomaton:
    def __init__(self):
        self.states = {}
        self.start_state = 'START'
        self.accept_states = set()
        self.current_state = self.start_state
    
    def add_transition(self, from_state, char, to_state):
        if from_state not in self.states:
            self.states[from_state] = {}
        self.states[from_state][char] = to_state
    
    def add_accept_state(self, state, token_type):
        self.accept_states.add((state, token_type))
    
    def process(self, string):
        self.current_state = self.start_state
        
        for char in string:
            if self.current_state in self.states:
                if char in self.states[self.current_state]:
                    self.current_state = self.states[self.current_state][char]
                else:
                    return None  # No transition
            else:
                return None
        
        for state, token_type in self.accept_states:
            if self.current_state == state:
                return token_type
        
        return None
```

## 15.3 Syntax Analysis (Parsing)

### Context-Free Grammars

```python
class Grammar:
    def __init__(self):
        # Example grammar for simple expressions
        self.productions = {
            'E': [['E', '+', 'T'], ['E', '-', 'T'], ['T']],
            'T': [['T', '*', 'F'], ['T', '/', 'F'], ['F']],
            'F': [['(', 'E', ')'], ['number'], ['identifier']]
        }
        self.start_symbol = 'E'
        self.terminals = {'+', '-', '*', '/', '(', ')', 'number', 'identifier'}
        self.non_terminals = {'E', 'T', 'F'}
    
    def is_terminal(self, symbol):
        return symbol in self.terminals
    
    def is_non_terminal(self, symbol):
        return symbol in self.non_terminals
    
    def get_productions(self, non_terminal):
        return self.productions.get(non_terminal, [])
```

### Recursive Descent Parser

```python
class RecursiveDescentParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
    
    def advance(self):
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def expect(self, token_type):
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token}")
    
    def parse(self):
        return self.parse_program()
    
    def parse_program(self):
        statements = []
        while self.current_token and self.current_token.type != TokenType.EOF:
            statements.append(self.parse_statement())
        return {'type': 'Program', 'statements': statements}
    
    def parse_statement(self):
        if self.current_token.type == TokenType.VAR:
            return self.parse_variable_declaration()
        elif self.current_token.type == TokenType.IF:
            return self.parse_if_statement()
        elif self.current_token.type == TokenType.WHILE:
            return self.parse_while_statement()
        elif self.current_token.type == TokenType.RETURN:
            return self.parse_return_statement()
        else:
            return self.parse_expression_statement()
    
    def parse_variable_declaration(self):
        self.expect(TokenType.VAR)
        identifier = self.expect(TokenType.IDENTIFIER)
        
        value = None
        if self.current_token and self.current_token.type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expression()
        
        self.expect(TokenType.SEMICOLON)
        
        return {
            'type': 'VariableDeclaration',
            'identifier': identifier.value,
            'value': value
        }
    
    def parse_if_statement(self):
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)
        
        then_branch = self.parse_block()
        
        else_branch = None
        if self.current_token and self.current_token.type == TokenType.ELSE:
            self.advance()
            else_branch = self.parse_block()
        
        return {
            'type': 'IfStatement',
            'condition': condition,
            'then': then_branch,
            'else': else_branch
        }
    
    def parse_expression(self):
        return self.parse_assignment()
    
    def parse_assignment(self):
        expr = self.parse_logical_or()
        
        if self.current_token and self.current_token.type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_assignment()
            return {
                'type': 'Assignment',
                'left': expr,
                'right': value
            }
        
        return expr
    
    def parse_logical_or(self):
        left = self.parse_logical_and()
        
        while self.current_token and self.current_token.value == '||':
            op = self.current_token.value
            self.advance()
            right = self.parse_logical_and()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_logical_and(self):
        left = self.parse_equality()
        
        while self.current_token and self.current_token.value == '&&':
            op = self.current_token.value
            self.advance()
            right = self.parse_equality()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_equality(self):
        left = self.parse_comparison()
        
        while self.current_token and self.current_token.type in [TokenType.EQUAL, TokenType.NOT_EQUAL]:
            op = self.current_token.value
            self.advance()
            right = self.parse_comparison()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_comparison(self):
        left = self.parse_additive()
        
        while self.current_token and self.current_token.type in [TokenType.LESS_THAN, TokenType.GREATER_THAN]:
            op = self.current_token.value
            self.advance()
            right = self.parse_additive()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_additive(self):
        left = self.parse_multiplicative()
        
        while self.current_token and self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplicative()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_multiplicative(self):
        left = self.parse_unary()
        
        while self.current_token and self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            op = self.current_token.value
            self.advance()
            right = self.parse_unary()
            left = {
                'type': 'BinaryExpression',
                'operator': op,
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_unary(self):
        if self.current_token and self.current_token.type in [TokenType.MINUS]:
            op = self.current_token.value
            self.advance()
            expr = self.parse_unary()
            return {
                'type': 'UnaryExpression',
                'operator': op,
                'operand': expr
            }
        
        return self.parse_primary()
    
    def parse_primary(self):
        if self.current_token.type == TokenType.NUMBER:
            value = float(self.current_token.value)
            self.advance()
            return {'type': 'Number', 'value': value}
        
        elif self.current_token.type == TokenType.STRING:
            value = self.current_token.value[1:-1]  # Remove quotes
            self.advance()
            return {'type': 'String', 'value': value}
        
        elif self.current_token.type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            
            # Check for function call
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                return self.parse_function_call(name)
            
            return {'type': 'Identifier', 'name': name}
        
        elif self.current_token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")
```

### LR Parser

```python
class LRParser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.action_table = {}
        self.goto_table = {}
        self.build_parsing_tables()
    
    def build_parsing_tables(self):
        # Build LR(0) automaton
        self.states = self.build_lr0_automaton()
        
        # Build action and goto tables
        for state_id, state in enumerate(self.states):
            for item in state:
                production, dot_pos = item
                
                if dot_pos < len(production):
                    next_symbol = production[dot_pos]
                    
                    if self.grammar.is_terminal(next_symbol):
                        # Shift action
                        next_state = self.goto(state, next_symbol)
                        self.action_table[(state_id, next_symbol)] = ('shift', next_state)
                    else:
                        # Goto entry
                        next_state = self.goto(state, next_symbol)
                        self.goto_table[(state_id, next_symbol)] = next_state
                else:
                    # Reduce action
                    if production[0] == self.grammar.start_symbol:
                        self.action_table[(state_id, '$')] = ('accept', None)
                    else:
                        for terminal in self.grammar.terminals:
                            self.action_table[(state_id, terminal)] = ('reduce', production)
    
    def parse(self, input_tokens):
        stack = [0]  # Initial state
        input_buffer = input_tokens + ['$']  # Add end marker
        position = 0
        
        while True:
            state = stack[-1]
            symbol = input_buffer[position]
            
            action = self.action_table.get((state, symbol))
            
            if not action:
                raise SyntaxError(f"No action for state {state} and symbol {symbol}")
            
            action_type, action_data = action
            
            if action_type == 'shift':
                stack.append(symbol)
                stack.append(action_data)
                position += 1
            
            elif action_type == 'reduce':
                production = action_data
                lhs, rhs = production[0], production[1:]
                
                # Pop 2 * len(rhs) items from stack
                for _ in range(2 * len(rhs)):
                    stack.pop()
                
                state = stack[-1]
                stack.append(lhs)
                stack.append(self.goto_table[(state, lhs)])
            
            elif action_type == 'accept':
                return True  # Successful parse
```

## 15.4 Semantic Analysis

### Symbol Table Management

```python
class Symbol:
    def __init__(self, name, type, scope, attributes=None):
        self.name = name
        self.type = type
        self.scope = scope
        self.attributes = attributes or {}

class SymbolTable:
    def __init__(self):
        self.scopes = [{}]  # Stack of scopes
        self.current_scope = 0
    
    def enter_scope(self):
        self.scopes.append({})
        self.current_scope += 1
    
    def exit_scope(self):
        if self.current_scope > 0:
            self.scopes.pop()
            self.current_scope -= 1
    
    def declare(self, name, symbol_type, attributes=None):
        if name in self.scopes[self.current_scope]:
            raise NameError(f"Symbol '{name}' already declared in current scope")
        
        symbol = Symbol(name, symbol_type, self.current_scope, attributes)
        self.scopes[self.current_scope][name] = symbol
        return symbol
    
    def lookup(self, name):
        # Search from current scope to global scope
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def lookup_in_current_scope(self, name):
        return self.scopes[self.current_scope].get(name)

class TypeChecker:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.type_rules = {
            '+': self.check_arithmetic,
            '-': self.check_arithmetic,
            '*': self.check_arithmetic,
            '/': self.check_arithmetic,
            '==': self.check_equality,
            '!=': self.check_equality,
            '<': self.check_comparison,
            '>': self.check_comparison,
            '&&': self.check_logical,
            '||': self.check_logical
        }
    
    def check(self, ast):
        return self.visit(ast)
    
    def visit(self, node):
        method_name = f'visit_{node["type"]}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def visit_Program(self, node):
        for statement in node['statements']:
            self.visit(statement)
    
    def visit_VariableDeclaration(self, node):
        var_type = 'any'  # Infer type from value
        
        if node['value']:
            var_type = self.visit(node['value'])
        
        self.symbol_table.declare(node['identifier'], var_type)
        return var_type
    
    def visit_BinaryExpression(self, node):
        left_type = self.visit(node['left'])
        right_type = self.visit(node['right'])
        
        checker = self.type_rules.get(node['operator'])
        if checker:
            return checker(left_type, right_type, node['operator'])
        
        return 'any'
    
    def visit_Number(self, node):
        return 'number'
    
    def visit_String(self, node):
        return 'string'
    
    def visit_Identifier(self, node):
        symbol = self.symbol_table.lookup(node['name'])
        if not symbol:
            raise NameError(f"Undefined variable '{node['name']}'")
        return symbol.type
    
    def check_arithmetic(self, left_type, right_type, op):
        if left_type == 'number' and right_type == 'number':
            return 'number'
        elif left_type == 'string' and right_type == 'string' and op == '+':
            return 'string'  # String concatenation
        else:
            raise TypeError(f"Invalid types for {op}: {left_type} and {right_type}")
    
    def check_equality(self, left_type, right_type, op):
        # Allow equality comparison between same types
        if left_type == right_type:
            return 'boolean'
        else:
            raise TypeError(f"Cannot compare {left_type} with {right_type}")
    
    def check_comparison(self, left_type, right_type, op):
        if left_type == 'number' and right_type == 'number':
            return 'boolean'
        else:
            raise TypeError(f"Invalid types for {op}: {left_type} and {right_type}")
    
    def check_logical(self, left_type, right_type, op):
        if left_type == 'boolean' and right_type == 'boolean':
            return 'boolean'
        else:
            raise TypeError(f"Logical operator {op} requires boolean operands")
```

## 15.5 Intermediate Representation

### Three-Address Code

```python
class ThreeAddressCode:
    def __init__(self):
        self.instructions = []
        self.temp_counter = 0
        self.label_counter = 0
    
    def new_temp(self):
        self.temp_counter += 1
        return f"t{self.temp_counter}"
    
    def new_label(self):
        self.label_counter += 1
        return f"L{self.label_counter}"
    
    def emit(self, op, arg1=None, arg2=None, result=None):
        instruction = {
            'op': op,
            'arg1': arg1,
            'arg2': arg2,
            'result': result
        }
        self.instructions.append(instruction)
        return instruction
    
    def generate(self, ast):
        return self.visit(ast)
    
    def visit(self, node):
        method_name = f'visit_{node["type"]}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def visit_Program(self, node):
        for statement in node['statements']:
            self.visit(statement)
    
    def visit_BinaryExpression(self, node):
        left = self.visit(node['left'])
        right = self.visit(node['right'])
        result = self.new_temp()
        
        self.emit(node['operator'], left, right, result)
        return result
    
    def visit_Number(self, node):
        temp = self.new_temp()
        self.emit('=', node['value'], None, temp)
        return temp
    
    def visit_Identifier(self, node):
        return node['name']
    
    def visit_Assignment(self, node):
        value = self.visit(node['right'])
        target = node['left']['name']  # Assuming left is identifier
        self.emit('=', value, None, target)
        return target
    
    def visit_IfStatement(self, node):
        condition = self.visit(node['condition'])
        else_label = self.new_label()
        end_label = self.new_label()
        
        # Jump to else if condition is false
        self.emit('if_false', condition, else_label)
        
        # Then branch
        self.visit(node['then'])
        self.emit('goto', end_label)
        
        # Else branch
        self.emit('label', else_label)
        if node['else']:
            self.visit(node['else'])
        
        self.emit('label', end_label)
    
    def visit_WhileStatement(self, node):
        start_label = self.new_label()
        end_label = self.new_label()
        
        self.emit('label', start_label)
        
        condition = self.visit(node['condition'])
        self.emit('if_false', condition, end_label)
        
        self.visit(node['body'])
        self.emit('goto', start_label)
        
        self.emit('label', end_label)

# Static Single Assignment (SSA) Form
class SSAConverter:
    def __init__(self, three_address_code):
        self.tac = three_address_code
        self.phi_functions = {}
        self.version_counters = {}
    
    def convert_to_ssa(self):
        # Step 1: Insert phi functions
        self.insert_phi_functions()
        
        # Step 2: Rename variables
        self.rename_variables()
        
        return self.tac
    
    def insert_phi_functions(self):
        # Find dominance frontiers
        dominance_frontiers = self.compute_dominance_frontiers()
        
        # For each variable, insert phi functions at dominance frontiers
        for var in self.get_all_variables():
            work_list = self.get_definition_sites(var)
            
            while work_list:
                block = work_list.pop()
                for frontier_block in dominance_frontiers[block]:
                    if frontier_block not in self.phi_functions:
                        self.phi_functions[frontier_block] = {}
                    
                    if var not in self.phi_functions[frontier_block]:
                        self.phi_functions[frontier_block][var] = True
                        work_list.append(frontier_block)
    
    def rename_variables(self):
        for var in self.get_all_variables():
            self.version_counters[var] = 0
        
        self.rename_block(self.get_entry_block())
    
    def rename_block(self, block):
        # Save current versions
        old_versions = dict(self.version_counters)
        
        # Process phi functions
        if block in self.phi_functions:
            for var in self.phi_functions[block]:
                self.version_counters[var] += 1
                new_name = f"{var}_{self.version_counters[var]}"
                # Update phi function with new name
        
        # Process instructions
        for instruction in block.instructions:
            # Rename uses
            if instruction['arg1'] and self.is_variable(instruction['arg1']):
                var = instruction['arg1']
                instruction['arg1'] = f"{var}_{self.version_counters[var]}"
            
            if instruction['arg2'] and self.is_variable(instruction['arg2']):
                var = instruction['arg2']
                instruction['arg2'] = f"{var}_{self.version_counters[var]}"
            
            # Rename definitions
            if instruction['result'] and self.is_variable(instruction['result']):
                var = instruction['result']
                self.version_counters[var] += 1
                instruction['result'] = f"{var}_{self.version_counters[var]}"
        
        # Process successors
        for successor in self.get_successors(block):
            self.rename_block(successor)
        
        # Restore versions
        self.version_counters = old_versions
```

## 15.6 Code Optimization

### Local Optimization

```python
class LocalOptimizer:
    def optimize(self, basic_block):
        self.constant_folding(basic_block)
        self.constant_propagation(basic_block)
        self.dead_code_elimination(basic_block)
        self.common_subexpression_elimination(basic_block)
        return basic_block
    
    def constant_folding(self, block):
        for instruction in block.instructions:
            if instruction['op'] in ['+', '-', '*', '/']:
                if self.is_constant(instruction['arg1']) and self.is_constant(instruction['arg2']):
                    # Evaluate at compile time
                    result = self.evaluate(
                        instruction['op'],
                        instruction['arg1'],
                        instruction['arg2']
                    )
                    
                    # Replace with assignment
                    instruction['op'] = '='
                    instruction['arg1'] = result
                    instruction['arg2'] = None
    
    def constant_propagation(self, block):
        constants = {}
        
        for instruction in block.instructions:
            # Update uses with known constants
            if instruction['arg1'] in constants:
                instruction['arg1'] = constants[instruction['arg1']]
            
            if instruction['arg2'] in constants:
                instruction['arg2'] = constants[instruction['arg2']]
            
            # Track constant assignments
            if instruction['op'] == '=' and self.is_constant(instruction['arg1']):
                constants[instruction['result']] = instruction['arg1']
    
    def dead_code_elimination(self, block):
        # Mark live variables
        live = set()
        
        # Backward pass to find live variables
        for instruction in reversed(block.instructions):
            if instruction['result'] in live or self.has_side_effects(instruction):
                # Mark uses as live
                if instruction['arg1'] and self.is_variable(instruction['arg1']):
                    live.add(instruction['arg1'])
                
                if instruction['arg2'] and self.is_variable(instruction['arg2']):
                    live.add(instruction['arg2'])
            else:
                # Mark for removal
                instruction['dead'] = True
        
        # Remove dead instructions
        block.instructions = [i for i in block.instructions if not i.get('dead')]
    
    def common_subexpression_elimination(self, block):
        expressions = {}
        
        for instruction in block.instructions:
            if instruction['op'] in ['+', '-', '*', '/']:
                # Create expression signature
                expr = (instruction['op'], instruction['arg1'], instruction['arg2'])
                
                if expr in expressions:
                    # Replace with copy
                    instruction['op'] = '='
                    instruction['arg1'] = expressions[expr]
                    instruction['arg2'] = None
                else:
                    expressions[expr] = instruction['result']

# Loop Optimization
class LoopOptimizer:
    def optimize_loops(self, cfg):
        loops = self.find_loops(cfg)
        
        for loop in loops:
            self.loop_invariant_code_motion(loop)
            self.strength_reduction(loop)
            self.loop_unrolling(loop)
    
    def find_loops(self, cfg):
        # Find natural loops using dominators
        dominators = self.compute_dominators(cfg)
        loops = []
        
        for edge in cfg.edges:
            if self.is_back_edge(edge, dominators):
                loop = self.find_natural_loop(edge)
                loops.append(loop)
        
        return loops
    
    def loop_invariant_code_motion(self, loop):
        invariant = []
        
        for block in loop.blocks:
            for instruction in block.instructions:
                if self.is_loop_invariant(instruction, loop):
                    invariant.append(instruction)
        
        # Move invariant code to pre-header
        pre_header = loop.pre_header
        for instruction in invariant:
            pre_header.instructions.append(instruction)
            # Remove from original location
    
    def strength_reduction(self, loop):
        # Replace expensive operations with cheaper ones
        for block in loop.blocks:
            for instruction in block.instructions:
                if instruction['op'] == '*' and self.is_constant(instruction['arg2']):
                    power_of_two = self.is_power_of_two(instruction['arg2'])
                    if power_of_two:
                        # Replace multiplication with shift
                        instruction['op'] = '<<'
                        instruction['arg2'] = power_of_two
    
    def loop_unrolling(self, loop, factor=4):
        if not self.can_unroll(loop):
            return
        
        # Duplicate loop body
        original_body = loop.body.copy()
        
        for i in range(factor - 1):
            duplicated = self.duplicate_body(original_body)
            self.rename_variables_in_body(duplicated, i + 1)
            loop.body.extend(duplicated)
        
        # Adjust loop increment
        loop.increment *= factor
```

## 15.7 Code Generation

### Instruction Selection

```python
class InstructionSelector:
    def __init__(self, target_arch):
        self.target_arch = target_arch
        self.instructions = []
        
        # Define instruction patterns for target architecture
        self.patterns = {
            'x86_64': {
                '+': lambda dst, src1, src2: [
                    f"mov {dst}, {src1}",
                    f"add {dst}, {src2}"
                ],
                '-': lambda dst, src1, src2: [
                    f"mov {dst}, {src1}",
                    f"sub {dst}, {src2}"
                ],
                '*': lambda dst, src1, src2: [
                    f"mov rax, {src1}",
                    f"imul rax, {src2}",
                    f"mov {dst}, rax"
                ],
                '/': lambda dst, src1, src2: [
                    f"mov rax, {src1}",
                    f"xor rdx, rdx",
                    f"div {src2}",
                    f"mov {dst}, rax"
                ],
                '=': lambda dst, src: [f"mov {dst}, {src}"],
            }
        }
    
    def select(self, ir):
        for instruction in ir:
            self.select_instruction(instruction)
        return self.instructions
    
    def select_instruction(self, ir_instruction):
        op = ir_instruction['op']
        pattern = self.patterns[self.target_arch].get(op)
        
        if pattern:
            if op == '=':
                asm = pattern(ir_instruction['result'], ir_instruction['arg1'])
            else:
                asm = pattern(
                    ir_instruction['result'],
                    ir_instruction['arg1'],
                    ir_instruction['arg2']
                )
            
            self.instructions.extend(asm)

# Register Allocation
class RegisterAllocator:
    def __init__(self, num_registers):
        self.num_registers = num_registers
        self.registers = [f"r{i}" for i in range(num_registers)]
        self.allocation = {}
        self.spilled = set()
    
    def allocate(self, live_intervals):
        # Linear scan register allocation
        active = []
        
        for interval in sorted(live_intervals, key=lambda x: x.start):
            # Expire old intervals
            active = [i for i in active if i.end > interval.start]
            
            if len(active) < self.num_registers:
                # Allocate register
                used_registers = {self.allocation[i.var] for i in active}
                for reg in self.registers:
                    if reg not in used_registers:
                        self.allocation[interval.var] = reg
                        active.append(interval)
                        break
            else:
                # Spill variable with furthest end point
                spill = max(active + [interval], key=lambda x: x.end)
                
                if spill == interval:
                    self.spilled.add(interval.var)
                else:
                    self.spilled.add(spill.var)
                    active.remove(spill)
                    active.append(interval)
                    self.allocation[interval.var] = self.allocation[spill.var]
                    del self.allocation[spill.var]
    
    def generate_spill_code(self, var):
        # Generate code to spill variable to memory
        return [
            f"mov [rbp-{self.get_stack_offset(var)}], {self.allocation.get(var, var)}"
        ]
    
    def generate_reload_code(self, var):
        # Generate code to reload variable from memory
        return [
            f"mov {self.allocation.get(var, var)}, [rbp-{self.get_stack_offset(var)}]"
        ]

# Graph Coloring Register Allocation
class GraphColoringAllocator:
    def __init__(self, num_colors):
        self.num_colors = num_colors
        self.graph = {}
        self.colors = {}
    
    def build_interference_graph(self, live_sets):
        for live_set in live_sets:
            for var1 in live_set:
                if var1 not in self.graph:
                    self.graph[var1] = set()
                
                for var2 in live_set:
                    if var1 != var2:
                        self.graph[var1].add(var2)
    
    def color_graph(self):
        # Simplify: Remove nodes with degree < num_colors
        stack = []
        simplified = self.graph.copy()
        
        while simplified:
            # Find node with minimum degree
            node = min(simplified.keys(), key=lambda x: len(simplified[x]))
            
            if len(simplified[node]) < self.num_colors:
                stack.append(node)
                # Remove node from graph
                del simplified[node]
                for neighbors in simplified.values():
                    neighbors.discard(node)
            else:
                # Need to spill
                break
        
        # Select: Assign colors to nodes
        while stack:
            node = stack.pop()
            neighbor_colors = {
                self.colors[neighbor]
                for neighbor in self.graph[node]
                if neighbor in self.colors
            }
            
            for color in range(self.num_colors):
                if color not in neighbor_colors:
                    self.colors[node] = color
                    break
```

## 15.8 Runtime Systems

### Memory Management

```python
class RuntimeHeap:
    def __init__(self, size):
        self.memory = bytearray(size)
        self.free_list = [(0, size)]  # (start, size)
        self.allocated = {}  # ptr -> size
    
    def malloc(self, size):
        # First-fit allocation
        for i, (start, free_size) in enumerate(self.free_list):
            if free_size >= size:
                # Allocate from this block
                self.allocated[start] = size
                
                if free_size > size:
                    # Update free block
                    self.free_list[i] = (start + size, free_size - size)
                else:
                    # Remove exhausted block
                    del self.free_list[i]
                
                return start
        
        return None  # Out of memory
    
    def free(self, ptr):
        if ptr not in self.allocated:
            raise ValueError("Invalid pointer")
        
        size = self.allocated[ptr]
        del self.allocated[ptr]
        
        # Add to free list and coalesce
        self.free_list.append((ptr, size))
        self.coalesce()
    
    def coalesce(self):
        # Merge adjacent free blocks
        self.free_list.sort(key=lambda x: x[0])
        
        merged = []
        for start, size in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == start:
                # Merge with previous block
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((start, size))
        
        self.free_list = merged

# Garbage Collection
class GarbageCollector:
    def __init__(self, heap):
        self.heap = heap
        self.roots = set()
        self.objects = {}  # address -> object metadata
    
    def mark_and_sweep(self):
        # Mark phase
        marked = set()
        work_list = list(self.roots)
        
        while work_list:
            obj = work_list.pop()
            if obj not in marked:
                marked.add(obj)
                
                # Add references from this object
                for ref in self.get_references(obj):
                    if ref not in marked:
                        work_list.append(ref)
        
        # Sweep phase
        to_free = []
        for addr in self.objects:
            if addr not in marked:
                to_free.append(addr)
        
        for addr in to_free:
            self.heap.free(addr)
            del self.objects[addr]
    
    def get_references(self, obj):
        # Extract references from object
        metadata = self.objects[obj]
        references = []
        
        for field in metadata['fields']:
            if field['type'] == 'reference':
                ref_addr = self.read_field(obj, field['offset'])
                if ref_addr:
                    references.append(ref_addr)
        
        return references
```

## 15.9 Interpreter Implementation

### Tree-Walking Interpreter

```python
class Interpreter:
    def __init__(self):
        self.environment = Environment()
        self.call_stack = []
    
    def interpret(self, ast):
        return self.evaluate(ast)
    
    def evaluate(self, node):
        if node['type'] == 'Program':
            result = None
            for statement in node['statements']:
                result = self.evaluate(statement)
            return result
        
        elif node['type'] == 'Number':
            return float(node['value'])
        
        elif node['type'] == 'String':
            return node['value']
        
        elif node['type'] == 'Identifier':
            return self.environment.get(node['name'])
        
        elif node['type'] == 'BinaryExpression':
            left = self.evaluate(node['left'])
            right = self.evaluate(node['right'])
            
            if node['operator'] == '+':
                return left + right
            elif node['operator'] == '-':
                return left - right
            elif node['operator'] == '*':
                return left * right
            elif node['operator'] == '/':
                if right == 0:
                    raise RuntimeError("Division by zero")
                return left / right
            elif node['operator'] == '==':
                return left == right
            elif node['operator'] == '!=':
                return left != right
            elif node['operator'] == '<':
                return left < right
            elif node['operator'] == '>':
                return left > right
            elif node['operator'] == '&&':
                return left and right
            elif node['operator'] == '||':
                return left or right
        
        elif node['type'] == 'UnaryExpression':
            operand = self.evaluate(node['operand'])
            
            if node['operator'] == '-':
                return -operand
            elif node['operator'] == '!':
                return not operand
        
        elif node['type'] == 'Assignment':
            value = self.evaluate(node['right'])
            self.environment.set(node['left']['name'], value)
            return value
        
        elif node['type'] == 'VariableDeclaration':
            value = None
            if node['value']:
                value = self.evaluate(node['value'])
            
            self.environment.define(node['identifier'], value)
            return value
        
        elif node['type'] == 'IfStatement':
            condition = self.evaluate(node['condition'])
            
            if condition:
                return self.evaluate(node['then'])
            elif node['else']:
                return self.evaluate(node['else'])
        
        elif node['type'] == 'WhileStatement':
            while self.evaluate(node['condition']):
                self.evaluate(node['body'])
        
        elif node['type'] == 'FunctionDeclaration':
            func = Function(node['name'], node['params'], node['body'], self.environment)
            self.environment.define(node['name'], func)
            return func
        
        elif node['type'] == 'FunctionCall':
            func = self.evaluate(node['function'])
            args = [self.evaluate(arg) for arg in node['arguments']]
            
            if isinstance(func, Function):
                return self.call_function(func, args)
            else:
                raise RuntimeError(f"'{node['function']}' is not a function")
        
        elif node['type'] == 'ReturnStatement':
            value = self.evaluate(node['value']) if node['value'] else None
            raise ReturnValue(value)
        
        elif node['type'] == 'Block':
            self.environment = Environment(self.environment)
            
            try:
                for statement in node['statements']:
                    self.evaluate(statement)
            finally:
                self.environment = self.environment.parent
    
    def call_function(self, func, args):
        if len(args) != len(func.params):
            raise RuntimeError(f"Expected {len(func.params)} arguments, got {len(args)}")
        
        # Create new environment for function
        env = Environment(func.closure)
        
        # Bind parameters
        for param, arg in zip(func.params, args):
            env.define(param, arg)
        
        # Save current environment
        previous = self.environment
        self.environment = env
        
        try:
            self.evaluate(func.body)
            return None  # No explicit return
        except ReturnValue as ret:
            return ret.value
        finally:
            self.environment = previous

class Environment:
    def __init__(self, parent=None):
        self.values = {}
        self.parent = parent
    
    def define(self, name, value):
        self.values[name] = value
    
    def get(self, name):
        if name in self.values:
            return self.values[name]
        elif self.parent:
            return self.parent.get(name)
        else:
            raise NameError(f"Undefined variable '{name}'")
    
    def set(self, name, value):
        if name in self.values:
            self.values[name] = value
        elif self.parent:
            self.parent.set(name, value)
        else:
            raise NameError(f"Undefined variable '{name}'")

class Function:
    def __init__(self, name, params, body, closure):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value
```

### Bytecode Interpreter

```python
from enum import Enum, auto

class OpCode(Enum):
    LOAD_CONST = auto()
    LOAD_VAR = auto()
    STORE_VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    CALL = auto()
    RETURN = auto()
    POP = auto()
    DUP = auto()

class BytecodeCompiler:
    def __init__(self):
        self.bytecode = []
        self.constants = []
        self.variables = {}
    
    def compile(self, ast):
        self.visit(ast)
        return self.bytecode, self.constants
    
    def emit(self, opcode, operand=None):
        instruction = {'opcode': opcode}
        if operand is not None:
            instruction['operand'] = operand
        
        self.bytecode.append(instruction)
        return len(self.bytecode) - 1
    
    def visit_Number(self, node):
        index = len(self.constants)
        self.constants.append(node['value'])
        self.emit(OpCode.LOAD_CONST, index)
    
    def visit_BinaryExpression(self, node):
        self.visit(node['left'])
        self.visit(node['right'])
        
        opcodes = {
            '+': OpCode.ADD,
            '-': OpCode.SUB,
            '*': OpCode.MUL,
            '/': OpCode.DIV,
            '==': OpCode.EQ,
            '!=': OpCode.NEQ,
            '<': OpCode.LT,
            '>': OpCode.GT
        }
        
        self.emit(opcodes[node['operator']])

class BytecodeVM:
    def __init__(self, bytecode, constants):
        self.bytecode = bytecode
        self.constants = constants
        self.stack = []
        self.variables = {}
        self.ip = 0  # Instruction pointer
    
    def run(self):
        while self.ip < len(self.bytecode):
            instruction = self.bytecode[self.ip]
            opcode = instruction['opcode']
            
            if opcode == OpCode.LOAD_CONST:
                value = self.constants[instruction['operand']]
                self.stack.append(value)
            
            elif opcode == OpCode.LOAD_VAR:
                name = instruction['operand']
                value = self.variables[name]
                self.stack.append(value)
            
            elif opcode == OpCode.STORE_VAR:
                name = instruction['operand']
                value = self.stack.pop()
                self.variables[name] = value
            
            elif opcode == OpCode.ADD:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left + right)
            
            elif opcode == OpCode.SUB:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left - right)
            
            elif opcode == OpCode.MUL:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left * right)
            
            elif opcode == OpCode.DIV:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left / right)
            
            elif opcode == OpCode.EQ:
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left == right)
            
            elif opcode == OpCode.JUMP:
                self.ip = instruction['operand']
                continue
            
            elif opcode == OpCode.JUMP_IF_FALSE:
                condition = self.stack.pop()
                if not condition:
                    self.ip = instruction['operand']
                    continue
            
            elif opcode == OpCode.POP:
                self.stack.pop()
            
            elif opcode == OpCode.DUP:
                self.stack.append(self.stack[-1])
            
            self.ip += 1
        
        return self.stack[-1] if self.stack else None
```

## Exercises

1. Implement a lexer for a simple programming language that supports:
   - Variables and functions
   - Arithmetic and logical operators
   - Control flow statements

2. Build a recursive descent parser for expressions with:
   - Proper operator precedence
   - Parentheses support
   - Error recovery

3. Create an LR(1) parser generator that:
   - Builds parsing tables from grammar
   - Handles shift/reduce conflicts
   - Generates parser code

4. Implement a type checker that supports:
   - Type inference
   - Generic types
   - Type constraints

5. Design an intermediate representation with:
   - SSA form conversion
   - Control flow graph construction
   - Dominance analysis

6. Build an optimizer that performs:
   - Loop optimization
   - Inlining
   - Tail call optimization

7. Create a register allocator using:
   - Graph coloring
   - Spill code generation
   - Coalescing

8. Implement a simple JIT compiler that:
   - Generates machine code at runtime
   - Performs basic optimizations
   - Handles function calls

9. Build a garbage collector with:
   - Generational collection
   - Reference counting
   - Cycle detection

10. Create a debugger for your language that supports:
    - Breakpoints
    - Step execution
    - Variable inspection

## Summary

This chapter covered compiler and interpreter construction:

- Language processing involves multiple phases from source to execution
- Lexical analysis converts text into tokens using finite automata
- Parsing builds abstract syntax trees from token streams
- Semantic analysis ensures program correctness and type safety
- Intermediate representations enable optimization and portability
- Code optimization improves performance at various levels
- Code generation produces efficient machine code
- Runtime systems manage memory and execution
- Interpreters provide direct execution alternatives

Understanding compilers and interpreters is essential for language implementation, optimization, and understanding how high-level code becomes machine instructions. These techniques apply beyond traditional compilers to domain-specific languages, query processors, and configuration systems.