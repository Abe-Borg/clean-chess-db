#!/usr/bin/env python3
"""
Test script to identify and fix the position cache issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_current_cache():
    """Test the current cache to see why it's not working."""
    print("Testing Current Cache Implementation")
    print("=" * 50)
    
    from environment.Environ import Environ
    
    # Create environment
    env = Environ()
    
    print(f"Initial cache state:")
    Environ.print_cache_stats()
    
    print(f"\n1. Getting legal moves for starting position:")
    legal_moves1 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves1)} legal moves")
    Environ.print_cache_stats()
    
    print(f"\n2. Getting legal moves again (should hit cache):")
    legal_moves2 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves2)} legal moves")
    Environ.print_cache_stats()
    
    print(f"\n3. Making a move and getting legal moves:")
    env.board.push_san('e4')
    env.update_curr_state()
    legal_moves3 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves3)} legal moves")
    Environ.print_cache_stats()
    
    # Let's also check what's in the cache
    print(f"\nCache contents:")
    with Environ._global_cache_lock:
        cache_keys = list(Environ._global_position_cache.keys())
        print(f"  Cache has {len(cache_keys)} entries")
        for i, key in enumerate(cache_keys[:3]):  # Show first 3
            print(f"  Key {i+1}: {key[:50]}...")

def test_fixed_cache():
    """Test the fixed cache implementation."""
    print("\n\nTesting Fixed Cache Implementation")
    print("=" * 50)
    
    # Import the fixed version
    from optimization_testing.cache_fix import test_cache_functionality
    
    test_cache_functionality()

if __name__ == "__main__":
    test_current_cache()
    test_fixed_cache()