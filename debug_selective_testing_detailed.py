#!/usr/bin/env python3
"""Enhanced debug script to understand why symbol extraction is failing."""

from testing.test_selector import TestSelector
from pathlib import Path
import ast

# Test the symbol extraction logic manually
repo_path = '/Users/antonellaschiavoni/antonella-projects/python_programming_challenges_with_unit_tests'
selector = TestSelector(repo_path)

# Mock patch data similar to what agents generated
class MockPatch:
    def __init__(self, content, file_path, line_start, line_end, patch_id):
        self.content = content
        self.file_path = file_path
        self.line_start = line_start
        self.line_end = line_end
        self.id = patch_id

patch = MockPatch(
    content='    if number in members:\n        members[number][2] = True\n    return members',
    file_path='src/dictionaries.py', 
    line_start=22,
    line_end=22,
    patch_id='test'
)

print("=== DETAILED SYMBOL EXTRACTION DEBUG ===")

# Debug the AST parsing of the patch content
print("1. Trying to parse patch content as AST...")
try:
    tree = ast.parse(patch.content)
    print("   AST parse successful!")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"   Found function: {node.name}")
        elif isinstance(node, ast.ClassDef):
            print(f"   Found class: {node.name}")
except SyntaxError as e:
    print(f"   AST parse failed: {e}")
    print("   Trying regex fallback...")
    
    # Test regex patterns
    import re
    module_name = "dictionaries"
    
    # Function definitions
    func_pattern = r'^[ \t]*def\s+(\w+)\s*\('
    for match in re.finditer(func_pattern, patch.content, re.MULTILINE):
        print(f"   Regex found function: {match.group(1)}")

# Check what's in the target file at the specified line
target_file = Path(repo_path) / patch.file_path
print(f"\n2. Examining target file: {target_file}")

if target_file.exists():
    with open(target_file, 'r') as f:
        lines = f.readlines()
    
    print(f"   File has {len(lines)} lines")
    print(f"   Patch targets lines {patch.line_start}-{patch.line_end}")
    
    # Show lines around the patch
    start_line = max(0, patch.line_start - 2)
    end_line = min(len(lines), patch.line_end + 3)
    
    print(f"   Context (lines {start_line}-{end_line}):")
    for i in range(start_line, end_line):
        marker = " >>>" if patch.line_start <= i <= patch.line_end else "    "
        line_content = lines[i].rstrip() if i < len(lines) else ""
        print(f"{marker} {i:2d}: {repr(line_content)}")

# Now test the actual symbol-in-range extraction
print(f"\n3. Testing _extract_symbols_in_line_range...")
module_name = selector._file_path_to_module_name(Path(patch.file_path))

# Parse the entire file to see function definitions
if target_file.exists():
    with open(target_file, 'r') as f:
        file_content = f.read()
    
    try:
        tree = ast.parse(file_content)
        print("   Functions/classes in target file:")
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    # Convert to 0-indexed
                    symbol_start = node.lineno - 1
                    symbol_end = getattr(node, 'end_lineno', node.lineno) - 1
                    
                    node_type = "function" if isinstance(node, ast.FunctionDef) else "class"
                    print(f"   {node_type} '{node.name}': lines {symbol_start}-{symbol_end} (1-indexed: {node.lineno}-{getattr(node, 'end_lineno', node.lineno)})")
                    
                    # Check overlap with patch
                    if selector._ranges_overlap(patch.line_start, patch.line_end, symbol_start, symbol_end):
                        print(f"   --> OVERLAPS with patch range {patch.line_start}-{patch.line_end}!")
                    else:
                        print(f"   --> No overlap with patch range {patch.line_start}-{patch.line_end}")
    except Exception as e:
        print(f"   Error parsing file: {e}")

# Debug the full selection logic manually
print(f"\n4. Manual symbol matching...")
symbol_index = selector._get_test_symbol_index()

# Show all symbols in the index
print(f"   All symbols in index:")
for symbol, tests in sorted(symbol_index.items()):
    print(f"   {symbol} -> {sorted(tests)}")

print(f"\n5. Why isn't register_payment detected?")
print(f"   Extracted symbols from patch: {selector._extract_modified_symbols([patch])}")
print(f"   Expected to find 'dictionaries.register_payment' or 'register_payment' in index")

# Check if the symbol would match with a different approach
expected_symbol = f"{module_name}.register_payment"
print(f"   Looking for exact match: '{expected_symbol}'")
if expected_symbol in symbol_index:
    print(f"   Found! -> {symbol_index[expected_symbol]}")
else:
    print(f"   Not found exactly. Checking partial matches...")
    for symbol in symbol_index.keys():
        if 'register_payment' in symbol:
            print(f"   Partial match: {symbol} -> {symbol_index[symbol]}")