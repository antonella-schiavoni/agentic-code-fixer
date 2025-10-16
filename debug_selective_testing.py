#!/usr/bin/env python3
"""Debug script to test selective testing logic."""

from testing.test_selector import TestSelector
from pathlib import Path

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

print("=== DEBUGGING SELECTIVE TESTING ===")
print(f"Repository path: {repo_path}")
print(f"Patch file: {patch.file_path}")
print(f"Patch line range: {patch.line_start}-{patch.line_end}")
print(f"Patch content: {repr(patch.content)}")

# 1. Extract symbols
print("\n1. Extracting modified symbols...")
symbols = selector._extract_modified_symbols([patch])
print(f'Extracted symbols: {symbols}')

# 2. Check if register_payment would be found
target_file = Path(repo_path) / patch.file_path
print(f"\n2. Checking target file: {target_file}")
print(f"Target file exists: {target_file.exists()}")

if target_file.exists():
    # Check what symbols are in the line range
    module_name = selector._file_path_to_module_name(Path(patch.file_path))
    print(f"Module name: {module_name}")
    
    symbols_in_range = selector._extract_symbols_in_line_range(
        target_file, patch.line_start, patch.line_end, module_name
    )
    print(f"Symbols in line range: {symbols_in_range}")

# 3. Build test symbol index
print("\n3. Building test symbol index...")
symbol_index = selector._get_test_symbol_index()
print(f"Index contains {len(symbol_index)} symbols")

# 4. Look for register_payment related entries
print("\n4. Looking for register_payment references...")
relevant_symbols = []
for symbol, tests in symbol_index.items():
    if 'register_payment' in symbol.lower():
        relevant_symbols.append((symbol, tests))
        print(f"  Found: {symbol} -> {tests}")

print(f"Found {len(relevant_symbols)} register_payment-related symbols")

# 5. Select tests
print("\n5. Selecting tests for patches...")
selected_tests = selector.select_tests_for_patches([patch])
print(f"Selected tests: {selected_tests}")

# 6. Check if the specific test file would be selected
test_file = "tests/tests_dictionaries.py"
print(f"\n6. Should {test_file} be selected? {test_file in selected_tests}")