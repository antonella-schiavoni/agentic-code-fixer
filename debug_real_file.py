#!/usr/bin/env python3
"""Test with the exact real file content to reproduce the bug."""

def debug_real_file():
    # Real file content from math_operations.py
    real_file_lines = [
        '"""\n',                                                          # 0
        'Simple mathematical operations module.\n',                       # 1  
        'Provides basic arithmetic functions: addition, subtraction, multiplication, and division.\n', # 2
        '"""\n',                                                         # 3
        '\n',                                                            # 4
        '\n',                                                            # 5
        'def add(a, b):\n',                                              # 6
        '    """Add two numbers."""\n',                                  # 7
        '    return a * b\n',                                            # 8
        '\n',                                                            # 9
        '\n',                                                            # 10
        'def subtract(a, b):\n',                                         # 11
        '    """Subtract b from a."""\n',                                # 12
        '    return a - b\n',                                            # 13
        '\n',                                                            # 14
        '\n',                                                            # 15
        'def multiply(a, b):\n',                                         # 16
        '    """Multiply two numbers."""\n',                             # 17
        '    return a * b\n',                                            # 18
        '\n',                                                            # 19
        '\n',                                                            # 20
        'def divide(a, b):\n',                                           # 21
        '    """Divide a by b."""\n',                                    # 22  
        '    return a / 0\n'                                             # 23 (no newline at end)
    ]
    
    print("=== REAL FILE CONTENT ===")
    for i, line in enumerate(real_file_lines):
        print(f"  {i}: {repr(line)}")
    print(f"Total lines: {len(real_file_lines)}")
    
    # The patch that the agent likely generated (from error log)
    patch_lines = [
        '    """Divide a by b.\n',
        '    \n', 
        '    Args:\n',
        '        a: The dividend (numerator).\n',
        '        b: The divisor (denominator).\n',
        '    \n',
        '    Returns:\n',
        '        The result of a divided by b.\n',
        '    \n',
        '    Raises:\n',
        '        ValueError: If b is zero.\n',
        '    """\n'
    ]
    
    print(f"\n=== PATCH LINES ({len(patch_lines)} lines) ===")
    for i, line in enumerate(patch_lines):
        print(f"  {i}: {repr(line)}")
    
    # Try different possible line ranges that the agent might have specified
    possible_ranges = [
        (22, 23, "Replace docstring and return statement (22-23)"),
        (21, 23, "Replace entire function except def line (21-23)"),
        (22, 22, "Replace only docstring (22-22)"),
        (23, 23, "Replace only return statement (23-23)")
    ]
    
    for line_start, line_end, description in possible_ranges:
        print(f"\n=== TESTING: {description} ===")
        print(f"Line range: {line_start}-{line_end}")
        
        # Apply patch
        if line_end >= len(real_file_lines):
            print("Path: Appending beyond file length")
            if line_start >= len(real_file_lines):
                padding_lines = ["\n"] * (line_start - len(real_file_lines))
                new_lines = real_file_lines + padding_lines + patch_lines
            else:
                new_lines = real_file_lines[: line_start] + patch_lines
        else:
            print("Path: Normal replacement within file bounds")
            new_lines = (
                real_file_lines[: line_start] + patch_lines + real_file_lines[line_end + 1 :]
            )
        
        print(f"After replacement: {len(new_lines)} lines")
        
        # Show the relevant section
        start_show = max(0, line_start - 2)
        end_show = min(len(new_lines), line_start + len(patch_lines) + 2)
        print("Relevant section:")
        for i in range(start_show, end_show):
            marker = " * " if line_start <= i < line_start + len(patch_lines) else "   "
            print(f"{marker}{i}: {repr(new_lines[i])}")
        
        # Verification slice
        actual_slice = new_lines[line_start:line_start + len(patch_lines)]
        
        print(f"\nVerification slice [{line_start}:{line_start + len(patch_lines)}]: {len(actual_slice)} lines")
        for i, line in enumerate(actual_slice):
            print(f"  {i}: {repr(line)}")
        
        if actual_slice == patch_lines:
            print("✅ This would PASS verification")
        else:
            print("❌ This would FAIL verification")
            print("First difference:")
            for i in range(min(len(actual_slice), len(patch_lines))):
                if actual_slice[i] != patch_lines[i]:
                    print(f"  Line {i}: expected={repr(patch_lines[i])}, actual={repr(actual_slice[i])}")
                    break
            
            # Check if it matches the error pattern we saw
            if len(actual_slice) == len(patch_lines) - 1 and actual_slice == patch_lines[1:]:
                print("🎯 This matches the error pattern! First line is missing.")

if __name__ == "__main__":
    debug_real_file()