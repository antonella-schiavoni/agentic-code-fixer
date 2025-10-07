#!/usr/bin/env python3
"""Test the exact patch content processing that's causing the first line to disappear."""

def debug_exact_issue():
    # This is the exact expected content from the error log
    expected_content_as_lines = [
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
    
    # Reconstruct what the original patch.content would be
    original_patch_content = ''.join(expected_content_as_lines)
    
    print("Original patch content that should produce the expected lines:")
    print(repr(original_patch_content))
    print("\nOriginal patch content (visible):")
    print(original_patch_content)
    
    print("\n" + "="*60)
    
    # Test the current buggy logic
    print("Testing CURRENT logic (lines 147-152):")
    if original_patch_content.endswith("\n"):
        patch_lines_current = [line + "\n" for line in original_patch_content.rstrip("\n").split("\n")]
    else:
        patch_lines_current = [line + "\n" for line in original_patch_content.split("\n")]
    
    print(f"Current logic produces {len(patch_lines_current)} lines:")
    for i, line in enumerate(patch_lines_current):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*40)
    
    # Test a BETTER logic using splitlines()
    print("Testing FIXED logic using splitlines(keepends=True):")
    patch_lines_fixed = original_patch_content.splitlines(keepends=True)
    
    print(f"Fixed logic produces {len(patch_lines_fixed)} lines:")
    for i, line in enumerate(patch_lines_fixed):
        print(f"  {i}: {repr(line)}")
    
    print("\n" + "="*40)
    
    # Compare results
    print("Comparison:")
    if patch_lines_current == expected_content_as_lines:
        print("✅ Current logic matches expected")
    else:
        print("❌ Current logic does NOT match expected")
        print("Differences:")
        max_len = max(len(patch_lines_current), len(expected_content_as_lines))
        for i in range(max_len):
            current = patch_lines_current[i] if i < len(patch_lines_current) else "MISSING"
            expected = expected_content_as_lines[i] if i < len(expected_content_as_lines) else "MISSING"
            if current != expected:
                print(f"  Line {i}: current={repr(current)}, expected={repr(expected)}")
    
    if patch_lines_fixed == expected_content_as_lines:
        print("✅ Fixed logic matches expected")
    else:
        print("❌ Fixed logic does NOT match expected")
        
    # Most importantly, let's test what happens if there's an issue with the agents
    # generating content that has escaped newlines or other issues
    print("\n" + "="*60)
    print("Testing potential problematic agent-generated content:")
    
    # What if the agent is somehow generating content with literal \\n?
    problematic_content1 = '    """Divide a by b.\\n    \\nArgs:\\n    ..."""\\n'
    print(f"Problematic content 1: {repr(problematic_content1)}")
    
    if problematic_content1.endswith("\n"):
        result1 = [line + "\n" for line in problematic_content1.rstrip("\n").split("\n")]
    else:
        result1 = [line + "\n" for line in problematic_content1.split("\n")]
    
    print(f"Result: {len(result1)} lines: {repr(result1)}")
    
    # What if there's some encoding issue?
    # Let's test with the actual string that might be coming from the agents
    print("\n" + "="*40)
    print("Testing by manually creating the 'disappearing line' scenario:")
    
    # Maybe the issue is that we're getting content that when processed loses the first line
    # Let's simulate file write/read cycle
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as f:
        # Write the patch_lines_current to file
        f.writelines(patch_lines_current)
        temp_file = f.name
    
    # Read it back
    with open(temp_file, 'r', encoding='utf-8') as f:
        read_back_lines = f.readlines()
    
    print(f"After write/read cycle: {len(read_back_lines)} lines")
    for i, line in enumerate(read_back_lines):
        print(f"  {i}: {repr(line)}")
    
    if read_back_lines == patch_lines_current:
        print("✅ File write/read preserves content")
    else:
        print("❌ File write/read CORRUPTS content!")
        
    import os
    os.unlink(temp_file)

if __name__ == "__main__":
    debug_exact_issue()