#!/usr/bin/env python3
"""
Test script to verify Priority 1 optimizations are working correctly.
Run this after updating your files to confirm performance improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils import game_settings

# Import your updated modules
from training.game_simulation import play_games, warm_up_workers
from environment.Environ import Environ

def test_small_sample():
    """Test the optimizations with a small sample to verify functionality."""
    print("Testing Priority 1 optimizations...")
    print("=" * 50)
    
    try:
        chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1)
        print(f"Loaded {len(chess_data)} games")
    except Exception as e:
        print(f"Failed to load chess data {e}")

        # Test with a small sample first
        sample_size = 100
        if len(chess_data) > sample_size:
            chess_data_sample = chess_data.head(sample_size)
            print(f"Using sample of {sample_size} games for testing")
        else:
            chess_data_sample = chess_data
            print(f"Using all {len(chess_data_sample)} games")
        
        # Create a persistent pool to test pool reuse
        num_workers = max(1, cpu_count() - 1)
        print(f"Creating pool with {num_workers} workers...")
        
        with Pool(processes=num_workers) as pool:
            # Warm up workers
            warm_up_workers(pool, num_workers)
            
            # Test the optimized processing
            print("\nRunning optimized game processing...")
            start_time = time.time()
            
            corrupted_games = play_games(chess_data_sample, pool)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Print results
            print(f"\nResults:")
            print(f"  Processed {len(chess_data_sample)} games")
            print(f"  Found {len(corrupted_games)} corrupted games")
            print(f"  Total time: {elapsed:.3f} seconds")
            print(f"  Games per second: {len(chess_data_sample) / elapsed:.2f}")
            
            if 'PlyCount' in chess_data_sample.columns:
                total_moves = chess_data_sample['PlyCount'].sum()
                moves_per_second = total_moves / elapsed
                print(f"  Total moves: {total_moves}")
                print(f"  Moves per second: {moves_per_second:.2f}")
            
            # Print cache statistics
            print(f"\nCache Statistics:")
            Environ.print_cache_stats()
            
        print(f"\nTest completed successfully!")
        print(f"If cache hit rate > 0%, the cache fix is working!")
        print(f"Compare these numbers to your previous profiling results.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please check that all file paths are correct and dependencies are installed.")

def benchmark_methods():
    """Benchmark the old vs new methods to show performance difference."""
    print("\nBenchmarking method performance...")
    print("=" * 50)
    
    from environment.Environ import Environ
    
    # Create environment
    environ = Environ()
    
    # Simulate a few moves to get a more realistic position
    test_moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']
    for move in test_moves:
        try:
            environ.board.push_san(move)
            environ.update_curr_state()
        except:
            break
    
    print(f"Testing position after moves: {' '.join(test_moves)}")
    
    # Test old method (SAN strings)
    iterations = 1000
    
    print(f"\nTesting legacy get_legal_moves() - {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        legal_moves_san = environ.get_legal_moves()
    end_time = time.time()
    legacy_time = end_time - start_time
    print(f"  Legacy method: {legacy_time:.4f} seconds ({legacy_time*1000/iterations:.2f} ms per call)")
    print(f"  Found {len(legal_moves_san)} legal moves")
    
    # Test new method (Move objects)
    print(f"\nTesting optimized get_legal_moves_optimized() - {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        legal_moves_obj = environ.get_legal_moves_optimized()
    end_time = time.time()
    optimized_time = end_time - start_time
    print(f"  Optimized method: {optimized_time:.4f} seconds ({optimized_time*1000/iterations:.2f} ms per call)")
    print(f"  Found {len(legal_moves_obj)} legal moves")
    
    # Calculate improvement
    if legacy_time > 0:
        speedup = legacy_time / optimized_time
        print(f"\nPerformance improvement: {speedup:.1f}x faster")
        print(f"Time reduction: {((legacy_time - optimized_time) / legacy_time * 100):.1f}%")

if __name__ == "__main__":
    print("Priority 1 Optimization Test")
    print("============================")
    
    # Run the main test
    test_small_sample()
    
    # Run benchmark
    benchmark_methods()
    
    print("\nNext steps:")
    print("1. Compare these results with your previous profiling data")
    print("2. If performance improved significantly, proceed to Priority 2")
    print("3. If issues occur, check the integration steps carefully")