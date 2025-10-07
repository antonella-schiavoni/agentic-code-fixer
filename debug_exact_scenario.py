#!/usr/bin/env python3
"""Test the EXACT scenario from the actual failing patch."""

def test_exact_scenario():
    # Real file content - 24 lines (0-23)
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
        '    return a / 0\n'                                             # 23
    ]
    
    print(f"File has {len(real_file_lines)} lines (indices 0-{len(real_file_lines)-1})")
    
    # Patch content from the JSON - NOTE: NO trailing newline after the last triple quotes!
    # This is EXACTLY as it appears in the JSON
    patch_content = '''    """Divide a by b.
    
    Args:
        a: The dividend (numerator).
        b: The divisor (denominator).
    
    Returns:
        The result of a divided by b.
    
    Raises:
        ValueError: If b is zero.
    """'''
    
    # Make absolutely sure there's no trailing newline (simulate JSON deserialization)
    patch_content = patch_content.rstrip('\n')
    
    print(f"\nPatch content:\n{repr(patch_content)}")
    
    # Process patch content using the current logic from lines 147-152
    if patch_content.endswith("\n"):
        patch_lines = [line + "\n" for line in patch_content.rstrip("\n").split("\n")]
    else:
        patch_lines = [line + "\n" for line in patch_content.split("\n")]
    
    print(f"\nPatch has {len(patch_lines)} lines after processing:")
    for i, line in enumerate(patch_lines):
        print(f"  {i}: {repr(line)}")
    
    # Patch parameters from JSON
    line_start_0idx = 26  # This is BEYOND the file length!
    line_end_0idx = 26
    
    print(f"\n=== APPLYING PATCH ===")
    print(f"line_start: {line_start_0idx}")
    print(f"line_end: {line_end_0idx}")
    print(f"File length: {len(real_file_lines)}")
    
    # Apply the replacement logic (lines 154-169 from patch_applicator.py)
    if line_end_0idx >= len(real_file_lines):
        print("Path: Appending beyond file length")
        if line_start_0idx >= len(real_file_lines):
            print(f"  Sub-path: line_start ({line_start_0idx}) >= file length ({len(real_file_lines)})")
            # Adding completely new lines - pad with empty lines if there's a gap
            padding_lines = ["\n"] * (line_start_0idx - len(real_file_lines))
            new_lines = real_file_lines + padding_lines + patch_lines
            print(f"  Adding {len(padding_lines)} padding lines")
        else:
            print(f"  Sub-path: Replacing from existing line and extending beyond file end")
            new_lines = real_file_lines[: line_start_0idx] + patch_lines
    else:
        print("Path: Normal replacement within file bounds")
        new_lines = (
            real_file_lines[: line_start_0idx] + patch_lines + real_file_lines[line_end_0idx + 1 :]
        )
    
    print(f"\n=== AFTER REPLACEMENT ===")
    print(f"New file has {len(new_lines)} lines")
    print("Last 10 lines of new file:")
    for i in range(max(0, len(new_lines) - 10), len(new_lines)):
        print(f"  {i}: {repr(new_lines[i])}")
    
    # Verification slice
    print(f"\n=== VERIFICATION ===")
    print(f"Extracting slice [{line_start_0idx}:{line_start_0idx + len(patch_lines)}]")
    
    actual_patched_lines = new_lines[line_start_0idx:line_start_0idx + len(patch_lines)]
    expected_lines = patch_lines
    
    print(f"\nExpected {len(expected_lines)} lines:")
    for i, line in enumerate(expected_lines):
        print(f"  {i}: {repr(line)}")
    
    print(f"\nActual {len(actual_patched_lines)} lines:")
    for i, line in enumerate(actual_patched_lines):
        print(f"  {i}: {repr(line)}")
    
    if actual_patched_lines == expected_lines:
        print("\n✅ Verification would PASS")
    else:
        print("\n❌ Verification would FAIL")
        
        # Check if it matches the error pattern
        if len(actual_patched_lines) == len(expected_lines) - 1:
            print(f"Length mismatch: expected {len(expected_lines)}, got {len(actual_patched_lines)}")
            
            # Check if actual matches expected[1:]
            if actual_patched_lines == expected_lines[1:]:
                print("🎯 MATCHES ERROR PATTERN: First line is missing!")
                print("This is the bug!")
        
        # Show differences
        print("\nDifferences:")
        max_len = max(len(expected_lines), len(actual_patched_lines))
        for i in range(max_len):
            exp = expected_lines[i] if i < len(expected_lines) else "MISSING"
            act = actual_patched_lines[i] if i < len(actual_patched_lines) else "MISSING"
            if exp != act:
                print(f"  Line {i}: expected={repr(exp)}, actual={repr(act)}")

if __name__ == "__main__":
    test_exact_scenario()