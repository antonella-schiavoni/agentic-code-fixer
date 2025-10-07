#!/usr/bin/env python3
"""Debug script to test the exact rstrip split issue."""

def test_rstrip_split_bug():
    # Let's test with a problematic string that might cause the first line to disappear
    
    # Test case 1: Normal case
    content1 = '''    """Divide a by b.
    
    Args:
        a: The numerator
        b: The denominator
        
    Returns:
        The result of a divided by b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    print("=== Test Case 1: Normal content ===")
    print(f"Content ends with newline: {repr(content1.endswith(chr(10)))}")
    print(f"Content repr: {repr(content1)}")
    
    if content1.endswith("\n"):
        result1 = [line + "\n" for line in content1.rstrip("\n").split("\n")]
    else:
        result1 = [line + "\n" for line in content1.split("\n")]
    
    print(f"Result has {len(result1)} lines:")
    for i, line in enumerate(result1):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*60 + "\n")
    
    # Test case 2: Content that might have issues - what if it starts with a newline?
    content2 = '''\n    """Divide a by b.
    
    Args:
        a: The numerator
        b: The denominator
        
    Returns:
        The result of a divided by b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    print("=== Test Case 2: Content starting with newline ===")
    print(f"Content ends with newline: {repr(content2.endswith(chr(10)))}")
    print(f"Content repr: {repr(content2)}")
    
    if content2.endswith("\n"):
        result2 = [line + "\n" for line in content2.rstrip("\n").split("\n")]
    else:
        result2 = [line + "\n" for line in content2.split("\n")]
    
    print(f"Result has {len(result2)} lines:")
    for i, line in enumerate(result2):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*60 + "\n")
    
    # Test case 3: What if the issue is in how agents generate the content?
    # Maybe they're including literal \\n characters?
    content3 = '''    \"""Divide a by b.\\n    \\n    Args:\\n        a: The numerator\\n        b: The denominator\\n        \\n    Returns:\\n        The result of a divided by b\\n        \\n    Raises:\\n        ValueError: If b is zero\\n    \"""\\n    if b == 0:\\n        raise ValueError(\\"Cannot divide by zero\\")\\n    return a / b\\n'''
    
    print("=== Test Case 3: Content with literal \\\\n characters ===")
    print(f"Content ends with newline: {repr(content3.endswith(chr(10)))}")
    print(f"Content repr: {repr(content3)}")
    
    if content3.endswith("\n"):
        result3 = [line + "\n" for line in content3.rstrip("\n").split("\n")]
    else:
        result3 = [line + "\n" for line in content3.split("\n")]
    
    print(f"Result has {len(result3)} lines:")
    for i, line in enumerate(result3):
        print(f"  {i}: {repr(line)}")
    
    # Now let's test the replacement process
    print(f"\n=== Testing line replacement logic ===")
    # Simulate what happens in the actual code
    
    # Original file (from math_operations.py)
    original_lines = [
        'def divide(a, b):\n',
        '    """Divide a by b."""\n', 
        '    return a / 0\n'
    ]
    
    # Test replacing lines 1-2 with result1 
    line_start_0idx = 1
    line_end_0idx = 2
    patch_lines = result1
    
    new_lines = (
        original_lines[: line_start_0idx] + patch_lines + original_lines[line_end_0idx + 1 :]
    )
    
    print(f"After replacement with normal content ({len(new_lines)} lines):")
    for i, line in enumerate(new_lines):
        print(f"  {i}: {repr(line)}")
    
    # Now check what verification slice would return
    actual_slice = new_lines[line_start_0idx:line_start_0idx + len(patch_lines)]
    print(f"\nVerification slice [{line_start_0idx}:{line_start_0idx + len(patch_lines)}] ({len(actual_slice)} lines):")
    for i, line in enumerate(actual_slice):
        print(f"  {i}: {repr(line)}")
    
    if actual_slice == patch_lines:
        print("✅ Verification would pass")
    else:
        print("❌ Verification would fail")

if __name__ == "__main__":
    test_rstrip_split_bug()