#!/usr/bin/env python3
"""Debug script to replicate the exact verification failure."""

def debug_verification_failure():
    # From the error log, this is what we expected to be written
    expected_lines = [
        '    """Divide a by b.\n',
        '    \n',
        '    Args:\n',
        '        a: The numerator\n',
        '        b: The denominator\n',
        '        \n',
        '    Returns:\n',
        '        The result of a divided by b\n',
        '        \n',
        '    Raises:\n',
        '        ValueError: If b is zero\n',
        '    """\n',
        '    if b == 0:\n',
        '        raise ValueError("Cannot divide by zero")\n',
        '    return a / b\n'
    ]
    
    # This is what was actually written according to the error
    actual_lines = [
        '    \n',
        '    Args:\n',
        '        a: The numerator\n',
        '        b: The denominator\n',
        '        \n',
        '    Returns:\n',
        '        The result of a divided by b\n',
        '        \n',
        '    Raises:\n',
        '        ValueError: If b is zero\n',
        '    """\n',
        '    if b == 0:\n',
        '        raise ValueError("Cannot divide by zero")\n',
        '    return a / b\n'
    ]
    
    print("Expected lines:")
    for i, line in enumerate(expected_lines):
        print(f"  {i}: {repr(line)}")
    
    print("\nActual lines:")
    for i, line in enumerate(actual_lines):
        print(f"  {i}: {repr(line)}")
    
    print(f"\nExpected: {len(expected_lines)} lines")
    print(f"Actual: {len(actual_lines)} lines")
    
    # Find differences
    print("\nDifferences:")
    for i in range(min(len(expected_lines), len(actual_lines))):
        if expected_lines[i] != actual_lines[i]:
            print(f"  Line {i}:")
            print(f"    Expected: {repr(expected_lines[i])}")
            print(f"    Actual:   {repr(actual_lines[i])}")
    
    # The issue is clear: the first line '    """Divide a by b.\n' is missing
    # This suggests the patch content processing is dropping the first line
    
    print(f"\n🔍 Analysis:")
    print(f"  - Expected has {len(expected_lines)} lines")
    print(f"  - Actual has {len(actual_lines)} lines") 
    print(f"  - Missing first line: {repr(expected_lines[0])}")
    print(f"  - All other lines shifted up by 1")
    
    # Let's test what happens when we reconstruct the original patch content
    # that would produce the expected lines
    original_content = ''.join(expected_lines)
    print(f"\nOriginal patch content that should produce expected lines:")
    print(repr(original_content))
    
    # Now test the current patch processing logic
    print(f"\nTesting current patch processing logic:")
    if original_content.endswith("\n"):
        patch_lines = [line + "\n" for line in original_content.rstrip("\n").split("\n")]
    else:
        patch_lines = [line + "\n" for line in original_content.split("\n")]
    
    print(f"Processed patch_lines:")
    for i, line in enumerate(patch_lines):
        print(f"  {i}: {repr(line)}")
    
    if patch_lines == expected_lines:
        print("✅ Current logic produces expected lines")
    else:
        print("❌ Current logic does NOT produce expected lines")
        print("This confirms the bug is in the patch processing logic")

if __name__ == "__main__":
    debug_verification_failure()