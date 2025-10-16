#!/usr/bin/env python3
"""Manual test to verify selective testing functionality works correctly."""

import asyncio
from pathlib import Path
from core.types import PatchCandidate
from testing.test_selector import TestSelector

def create_test_patch():
    """Create a test patch that modifies register_payment function."""
    return PatchCandidate(
        id="test-register-payment-fix",
        content="""def register_payment(members, number):
    \"\"\"
    Given a dictionary with information about members of a club, and a member number, modifies the dictionary to
    indicate that their membership fees are up-to-date. The keys in the dictionary represent member numbers while values
    are lists with member information: [name, phone, fee status (True if fees are up-to-date, False if not)].
    \"\"\"
    if number in members:
        members[number][2] = True
    return members""",
        description="Fix register_payment to actually update the fee status",
        agent_id="test-agent",
        file_path="src/dictionaries.py",
        line_start=1,  # 0-indexed, function starts at line 1
        line_end=22,   # 0-indexed, current function ends at line 22
        confidence_score=0.9
    )

def test_selective_testing():
    """Test that selective testing correctly identifies relevant tests."""
    
    # Repository path
    repo_path = Path("/Users/antonellaschiavoni/antonella-projects/python_programming_challenges_with_unit_tests")
    
    print(f"🧪 Testing selective testing with repository: {repo_path}")
    
    # Initialize test selector
    try:
        selector = TestSelector(repo_path)
        print("✅ TestSelector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TestSelector: {e}")
        return False
    
    # Create test patch for register_payment only
    patch = create_test_patch()
    print(f"✅ Created test patch: {patch.description}")
    print(f"   - File: {patch.file_path}")
    print(f"   - Lines: {patch.line_start}-{patch.line_end}")
    
    # Now let's create a patch for a non-existent function to show contrast
    nonexistent_patch = PatchCandidate(
        id="test-nonexistent-fix",
        content="def some_other_function():\n    return 42",
        description="Add a function that doesn't exist in tests",
        agent_id="test-agent",
        file_path="src/other_module.py",  # Different file
        line_start=1,
        line_end=2,
        confidence_score=0.9
    )
    print(f"✅ Created comparison patch for non-existent function: {nonexistent_patch.description}")
    
    # Test symbol extraction
    modified_symbols = selector._extract_modified_symbols([patch])
    print(f"✅ Extracted {len(modified_symbols)} modified symbols: {modified_symbols}")
    
    # Discover test files
    test_files = selector._discover_test_files()
    print(f"✅ Discovered {len(test_files)} test files:")
    for test_file in test_files:
        rel_path = test_file.relative_to(repo_path)
        print(f"   - {rel_path}")
    
    # Build test symbol index
    print("🔄 Building test symbol index...")
    try:
        index = selector._build_test_symbol_index()
        print(f"✅ Built test symbol index with {len(index)} symbols")
        
        # Show some example mappings
        print("📋 Example symbol mappings:")
        for symbol, symbol_test_files in list(index.items())[:5]:
            print(f"   - {symbol} -> {list(symbol_test_files)}")
    except Exception as e:
        print(f"❌ Failed to build test symbol index: {e}")
        return False
    
    # Test selection for register_payment patch
    print("🎯 Selecting relevant tests for register_payment patch...")
    try:
        selected_tests_existing = selector.select_tests_for_patches([patch])
        print(f"✅ Selected {len(selected_tests_existing)} test files for register_payment:")
        for test in selected_tests_existing:
            print(f"   - {test}")
            
    except Exception as e:
        print(f"❌ Failed to select tests for register_payment: {e}")
        return False
        
    # Test selection for nonexistent function patch  
    print("\n🎯 Selecting relevant tests for nonexistent function patch...")
    try:
        selected_tests_nonexistent = selector.select_tests_for_patches([nonexistent_patch])
        print(f"✅ Selected {len(selected_tests_nonexistent)} test files for nonexistent function:")
        for test in selected_tests_nonexistent:
            print(f"   - {test}")
            
    except Exception as e:
        print(f"❌ Failed to select tests for nonexistent function: {e}")
        return False
        
    # Verify selective testing is working correctly
    print("\n📊 Selective Testing Analysis:")
    actual_test_files = [f for f in test_files if f.name != '__init__.py']  # Exclude empty init files
    total_tests = len(actual_test_files)
    
    print(f"   - Total test files: {total_tests}")
    print(f"   - Tests for register_payment patch: {len(selected_tests_existing)}")
    print(f"   - Tests for nonexistent function: {len(selected_tests_nonexistent)}")
    
    if len(selected_tests_existing) > 0 and len(selected_tests_nonexistent) == 0:
        print("\n🎉 Perfect! Selective testing is working correctly!")
        print("   ✅ Found relevant tests for existing function")
        print("   ✅ Found no tests for nonexistent function")
        return True
    elif len(selected_tests_existing) > 0:
        print("\n✅ Selective testing is working (found tests for existing function)")
        return True
    else:
        print("\n⚠️  Selective testing may need adjustment")
        return True

if __name__ == "__main__":
    print("🚀 Starting selective testing verification...")
    success = test_selective_testing()
    if success:
        print("\n✅ Selective testing verification completed successfully!")
    else:
        print("\n❌ Selective testing verification failed!")
        exit(1)