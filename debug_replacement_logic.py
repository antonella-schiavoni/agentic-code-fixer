#!/usr/bin/env python3
"""Debug the file replacement and slicing logic to find where the first line disappears."""

def debug_replacement_logic():
    # Simulate the exact scenario from math_operations.py
    # Original file content (lines around the divide function)
    original_lines = [
        'def divide(a, b):\n',           # line 0 (22 in actual file)
        '    """Divide a by b."""\n',    # line 1 (23 in actual file) - this will be replaced
        '    return a / 0\n'             # line 2 (24 in actual file) - this will be replaced
    ]
    
    print("=== ORIGINAL FILE CONTENT ===")
    for i, line in enumerate(original_lines):
        print(f"  {i}: {repr(line)}")
    
    # The patch content (what agents generated)
    expected_patch_lines = [
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
    
    print(f"\n=== PATCH CONTENT ({len(expected_patch_lines)} lines) ===")
    for i, line in enumerate(expected_patch_lines):
        print(f"  {i}: {repr(line)}")
    
    # Simulate the patch application - replacing lines 1-2 (0-indexed)
    line_start_0idx = 1
    line_end_0idx = 2  # This should be inclusive (replace lines 1 and 2)
    patch_lines = expected_patch_lines
    
    print(f"\n=== PATCH APPLICATION ===")
    print(f"Replacing lines {line_start_0idx}-{line_end_0idx} (inclusive)")
    print(f"Original file has {len(original_lines)} lines")
    print(f"Patch has {len(patch_lines)} lines")
    
    # Apply the replacement logic from lines 154-169 in patch_applicator.py
    if line_end_0idx >= len(original_lines):
        print("Path: Appending beyond file length")
        if line_start_0idx >= len(original_lines):
            padding_lines = ["\n"] * (line_start_0idx - len(original_lines))
            new_lines = original_lines + padding_lines + patch_lines
        else:
            new_lines = original_lines[: line_start_0idx] + patch_lines
    else:
        print("Path: Normal replacement within file bounds")
        new_lines = (
            original_lines[: line_start_0idx] + patch_lines + original_lines[line_end_0idx + 1 :]
        )
    
    print(f"\n=== AFTER REPLACEMENT ({len(new_lines)} lines) ===")
    for i, line in enumerate(new_lines):
        print(f"  {i}: {repr(line)}")
    
    # Now simulate the verification slicing logic
    print(f"\n=== VERIFICATION SLICING ===")
    print(f"Extracting slice [{line_start_0idx}:{line_start_0idx + len(patch_lines)}]")
    
    actual_patched_lines = new_lines[line_start_0idx:line_start_0idx + len(patch_lines)]
    expected_lines = patch_lines
    
    print(f"Expected ({len(expected_lines)} lines):")
    for i, line in enumerate(expected_lines):
        print(f"  {i}: {repr(line)}")
    
    print(f"\nActual ({len(actual_patched_lines)} lines):")
    for i, line in enumerate(actual_patched_lines):
        print(f"  {i}: {repr(line)}")
    
    if actual_patched_lines == expected_lines:
        print("\n✅ Verification would PASS")
    else:
        print("\n❌ Verification would FAIL")
        print("Differences:")
        max_len = max(len(expected_lines), len(actual_patched_lines))
        for i in range(max_len):
            expected = expected_lines[i] if i < len(expected_lines) else "MISSING"
            actual = actual_patched_lines[i] if i < len(actual_patched_lines) else "MISSING"
            if expected != actual:
                print(f"  Line {i}: expected={repr(expected)}, actual={repr(actual)}")
    
    # Test what happens if the line range is wrong
    print(f"\n=== TESTING DIFFERENT LINE RANGES ===")
    
    # What if it's supposed to be replacing line 1 only (not 1-2)?
    test_line_end = 1  # Only replace line 1
    new_lines_test = (
        original_lines[: line_start_0idx] + patch_lines + original_lines[test_line_end + 1 :]
    )
    
    print(f"If replacing lines {line_start_0idx}-{test_line_end}:")
    for i, line in enumerate(new_lines_test):
        print(f"  {i}: {repr(line)}")
    
    actual_test = new_lines_test[line_start_0idx:line_start_0idx + len(patch_lines)]
    print(f"\nVerification slice would be: {len(actual_test)} lines")
    for i, line in enumerate(actual_test):
        print(f"  {i}: {repr(line)}")
    
    if actual_test == expected_lines:
        print("✅ This would pass verification!")
    else:
        print("❌ This would also fail verification")

if __name__ == "__main__":
    debug_replacement_logic()
