#!/usr/bin/env python3
"""Debug script to reproduce the newline handling issue in patch applicator."""

def debug_newline_processing():
    # Simulate the problematic patch content from the error
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
    
    print("Original patch content:")
    print(repr(patch_content))
    print("\nOriginal patch content (visible):")
    print(patch_content)
    
    print("\n" + "="*60)
    
    # Current problematic logic
    print("Current problematic logic:")
    if patch_content.endswith("\n"):
        # Content ends with newline, split and add newlines to each line
        patch_lines_old = [line + "\n" for line in patch_content.rstrip("\n").split("\n")]
    else:
        # Content doesn't end with newline, split and add newlines to all
        patch_lines_old = [line + "\n" for line in patch_content.split("\n")]
    
    print("Processed patch_lines (old logic):")
    for i, line in enumerate(patch_lines_old):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*60)
    
    # Fixed logic
    print("Fixed logic:")
    # Simply split by lines while preserving original newline structure
    patch_lines_new = patch_content.splitlines(keepends=True)
    
    # If the content doesn't end with newline, ensure last line gets one
    if patch_lines_new and not patch_lines_new[-1].endswith('\n'):
        patch_lines_new[-1] += '\n'
    
    print("Processed patch_lines (new logic):")
    for i, line in enumerate(patch_lines_new):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*60)
    print("Comparison:")
    print(f"Old logic produces {len(patch_lines_old)} lines")
    print(f"New logic produces {len(patch_lines_new)} lines")
    
    if patch_lines_old != patch_lines_new:
        print("❌ DIFFERENT RESULTS!")
        print("First few differences:")
        for i in range(min(5, len(patch_lines_old), len(patch_lines_new))):
            if patch_lines_old[i] != patch_lines_new[i]:
                print(f"  Line {i}:")
                print(f"    Old: {repr(patch_lines_old[i])}")
                print(f"    New: {repr(patch_lines_new[i])}")
    else:
        print("✅ Same results")

if __name__ == "__main__":
    debug_newline_processing()