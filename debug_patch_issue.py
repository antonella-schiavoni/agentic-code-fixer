#!/usr/bin/env python3
"""
Debug script to reproduce the patch application issue.
This will help identify the root cause of why patches are applied but tests still fail.
"""

import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from core.types import PatchCandidate
from patching.patch_applicator import PatchApplicator
from core.config import TestingConfig


def create_test_math_module(temp_dir: Path) -> Path:
    """Create a temporary math operations module with the bug"""
    math_file = temp_dir / "math_operations.py"
    
    content = '''"""
Simple mathematical operations module.
Provides basic arithmetic functions: addition, subtraction, multiplication, and division.
"""


def add(a, b):
    """Add two numbers."""
    return a * b  # BUG: Using multiplication instead of addition


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def divide(a, b):
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    math_file.write_text(content)
    return math_file


def create_test_file(temp_dir: Path) -> Path:
    """Create a test file for the math operations"""
    test_file = temp_dir / "test_math_operations.py"
    
    content = '''"""
Test module for mathematical operations using pytest.
Tests all functions in the math_operations module.
"""

import pytest
from math_operations import add, subtract, multiply, divide


def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(-5, -3) == -8
    assert add(2.5, 3.5) == 6.0


def test_subtract():
    """Test subtraction function."""
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0
    assert subtract(-1, -1) == 0
    assert subtract(-5, 3) == -8
    assert subtract(5.5, 2.5) == 3.0


def test_multiply():
    """Test multiplication function."""
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6
    assert multiply(-2, -3) == 6
    assert multiply(2.5, 4) == 10.0


def test_divide():
    """Test division function."""
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3
    assert divide(-6, 2) == -3
    assert divide(-6, -2) == 3
    assert abs(divide(1, 3) - 0.3333333333333333) < 1e-15


def test_divide_by_zero():
    """Test division by zero raises ValueError."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)
'''
    
    test_file.write_text(content)
    return test_file


def main():
    """Main debug function"""
    print("🔍 Starting patch application debug...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Created temp directory: {temp_path}")
        
        # Create test files
        math_file = create_test_math_module(temp_path)
        test_file = create_test_file(temp_path)
        
        print(f"📄 Created math module: {math_file}")
        print(f"🧪 Created test file: {test_file}")
        
        # Verify initial broken state
        print("\n🔧 Testing initial broken state...")
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, '.'); from math_operations import add; print(f'add(2, 3) = {add(2, 3)}')"
        ], cwd=temp_path, capture_output=True, text=True)
        
        print(f"Initial result: {result.stdout.strip()}")
        print(f"Exit code: {result.returncode}")
        
        if "add(2, 3) = 6" not in result.stdout:
            print("❌ ERROR: Initial state is not broken as expected!")
            return False
            
        print("✅ Confirmed: Initial state is broken (returns 6 instead of 5)")
        
        # Create patch
        patch = PatchCandidate(
            id="test-patch-001",
            content="    return a + b",
            description="Fix add function to use addition instead of multiplication",
            agent_id="debug-agent",
            file_path="math_operations.py",
            line_start=9,
            line_end=9,
            confidence_score=1.0
        )
        
        print(f"\n🔧 Created patch: {patch}")
        
        # Create patch applicator
        config = TestingConfig(
            test_command="python -m pytest test_math_operations.py::test_add -v",
            test_timeout_seconds=30,
            pre_test_commands=[],
            post_test_commands=[]
        )
        
        applicator = PatchApplicator(config)
        
        # Apply patch
        print(f"\n🔧 Applying patch to {math_file}...")
        apply_success = applicator.apply_patch(patch, temp_path, create_backup=True)
        
        print(f"Patch application success: {apply_success}")
        
        if not apply_success:
            print("❌ ERROR: Patch failed to apply!")
            return False
            
        # Check file contents after patch
        print("\n📄 File contents after patch:")
        patched_content = math_file.read_text()
        print("---")
        for i, line in enumerate(patched_content.split('\n'), 1):
            print(f"{i:2d}| {line}")
        print("---")
        
        # Test the patched function directly
        print(f"\n🔧 Testing patched function directly...")
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, '.'); from math_operations import add; print(f'add(2, 3) = {add(2, 3)}')"
        ], cwd=temp_path, capture_output=True, text=True)
        
        print(f"Direct test result: {result.stdout.strip()}")
        print(f"Exit code: {result.returncode}")
        
        if "add(2, 3) = 5" in result.stdout:
            print("✅ SUCCESS: Direct function call works correctly")
        else:
            print("❌ ERROR: Direct function call still broken!")
            print(f"Stderr: {result.stderr}")
            return False
            
        # Run the actual test
        print(f"\n🧪 Running pytest...")
        test_result = applicator.run_tests(temp_path, patch.id)
        
        print(f"Test passed: {test_result.passed}")
        print(f"Exit code: {test_result.exit_code}")
        print(f"Duration: {test_result.duration_seconds:.2f}s")
        print(f"Failed tests: {test_result.failed_tests}")
        
        if test_result.stdout:
            print("STDOUT:")
            print(test_result.stdout)
            
        if test_result.stderr:
            print("STDERR:")
            print(test_result.stderr)
        
        # Check for import cache issues
        print(f"\n🔍 Checking for potential import cache issues...")
        
        # Test with fresh Python process
        result_fresh = subprocess.run([
            sys.executable, "-c", 
            """
import sys
sys.path.insert(0, '.')
print("Python path:", sys.path[0])
print("Importing math_operations...")
import math_operations
import inspect
print("Source code line 9:", inspect.getsourcelines(math_operations.add)[0][1].strip())
print(f"add(2, 3) = {math_operations.add(2, 3)}")
"""
        ], cwd=temp_path, capture_output=True, text=True)
        
        print("Fresh process result:")
        print(result_fresh.stdout)
        if result_fresh.stderr:
            print("Stderr:", result_fresh.stderr)
            
        return test_result.passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)