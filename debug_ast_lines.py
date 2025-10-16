#!/usr/bin/env python3
"""Debug AST line number detection."""

import ast
from pathlib import Path

# Parse the actual file
file_path = "/Users/antonellaschiavoni/antonella-projects/python_programming_challenges_with_unit_tests/src/dictionaries.py"

with open(file_path, 'r') as f:
    content = f.read()

lines = content.split('\n')
print("=== FILE CONTENT WITH LINE NUMBERS ===")
for i, line in enumerate(lines):
    print(f"{i:2d} (1-idx:{i+1:2d}): {repr(line)}")

print("\n=== AST ANALYSIS ===")
tree = ast.parse(content)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = getattr(node, 'end_lineno', node.lineno) - 1  # Convert to 0-indexed
        
        print(f"\nFunction '{node.name}':")
        print(f"  AST reports: lines {node.lineno}-{getattr(node, 'end_lineno', node.lineno)} (1-indexed)")
        print(f"  Converted to 0-indexed: {start_line}-{end_line}")
        
        # Show the actual lines
        print(f"  Actual content at these lines:")
        for i in range(max(0, start_line), min(len(lines), end_line + 1)):
            print(f"    {i:2d}: {repr(lines[i])}")

print("\n=== PATCH TARGET ANALYSIS ===")
patch_line_22 = 22  # 0-indexed
print(f"Patch targets line {patch_line_22} (0-indexed), which is line {patch_line_22 + 1} (1-indexed)")
print(f"Content at patch line: {repr(lines[patch_line_22])}")

# Check if line 22 should be considered part of register_payment
print(f"\nIs line {patch_line_22} part of register_payment function?")
print(f"  Content: {repr(lines[patch_line_22])}")
print(f"  This is clearly the return statement of the function!")

# Check what happens if we extend the range check by 1
print(f"\n=== TESTING EXTENDED RANGE OVERLAP ===")
def ranges_overlap(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)

register_payment_start = 0  # Function starts at line 0
register_payment_end = 21   # AST says it ends at line 21 (0-indexed)
patch_start = 22
patch_end = 22

print(f"Original range check:")
print(f"  register_payment: {register_payment_start}-{register_payment_end}")
print(f"  patch: {patch_start}-{patch_end}")
print(f"  overlap: {ranges_overlap(register_payment_start, register_payment_end, patch_start, patch_end)}")

print(f"\nExtended range check (function end + 1):")
extended_end = register_payment_end + 1
print(f"  register_payment: {register_payment_start}-{extended_end}")
print(f"  patch: {patch_start}-{patch_end}")
print(f"  overlap: {ranges_overlap(register_payment_start, extended_end, patch_start, patch_end)}")