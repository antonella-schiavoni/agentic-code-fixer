#!/usr/bin/env python3
"""Debug script to test the specific slicing issue in verification."""

def debug_slicing_issue():
    # Simulate the scenario where a file has content and we're patching some lines
    
    # Original file content (from math_operations.py line 22-24)
    original_lines = [
        'def divide(a, b):\n',
        '    """Divide a by b."""\n', 
        '    return a / 0\n'  # This is the bug we're fixing
    ]
    
    # The patch content we want to apply (this replaces lines 1-2, which is line_start=1, line_end=2)
    patch_content = '''    """Divide a by b.
    
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
    
    # Assume we're patching lines 1-2 (0-indexed), so line_start=1, line_end=2
    line_start_0idx = 1
    line_end_0idx = 2
    
    print("Original file lines:")
    for i, line in enumerate(original_lines):
        print(f"  {i}: {repr(line)}")
    
    print(f"\nPatch content:\n{repr(patch_content)}")
    
    # Process patch content using current logic
    if patch_content.endswith("\n"):
        patch_lines = [line + "\n" for line in patch_content.rstrip("\n").split("\n")]
    else:
        patch_lines = [line + "\n" for line in patch_content.split("\n")]
    
    print(f"\nProcessed patch_lines ({len(patch_lines)} lines):")
    for i, line in enumerate(patch_lines):
        print(f"  {i}: {repr(line)}")
    
    # Apply the patch (simulate the file writing)
    new_lines = (
        original_lines[: line_start_0idx] + patch_lines + original_lines[line_end_0idx + 1 :]
    )
    
    print(f"\nNew file content after patch ({len(new_lines)} lines):")
    for i, line in enumerate(new_lines):
        print(f"  {i}: {repr(line)}")
    
    # Now simulate the verification slicing that's causing the problem
    actual_patched_lines = new_lines[line_start_0idx:line_start_0idx + len(patch_lines)]
    expected_lines = patch_lines
    
    print(f"\nVerification:")
    print(f"Expected lines ({len(expected_lines)} lines):")
    for i, line in enumerate(expected_lines):
        print(f"  {i}: {repr(line)}")
    
    print(f"\nActual patched lines from slice [{line_start_0idx}:{line_start_0idx + len(patch_lines)}] ({len(actual_patched_lines)} lines):")
    for i, line in enumerate(actual_patched_lines):
        print(f"  {i}: {repr(line)}")
    
    if actual_patched_lines == expected_lines:
        print("\n✅ Verification would PASS")
    else:
        print("\n❌ Verification would FAIL")
        print("\nDifferences:")
        for i in range(min(len(expected_lines), len(actual_patched_lines))):
            if i < len(actual_patched_lines) and expected_lines[i] != actual_patched_lines[i]:
                print(f"  Line {i}:")
                print(f"    Expected: {repr(expected_lines[i])}")
                print(f"    Actual:   {repr(actual_patched_lines[i])}")
        
        if len(expected_lines) != len(actual_patched_lines):
            print(f"  Length mismatch: expected {len(expected_lines)}, actual {len(actual_patched_lines)}")

if __name__ == "__main__":
    debug_slicing_issue()