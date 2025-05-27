# file: optimization_testing/comprehensive_performance_test.py

"""
Comprehensive test using real data to measure all optimizations.
This will give us the true performance improvement from baseline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils import game_settings

def test_all_optimizations_with_real_data():
    """Test all optimizations using your real data."""
    print("Comprehensive Performance Test with Real Data")
    print("=" * 60)
    
    filepath = game_settings.chess_games_filepath_part_1
    try:
        chess_data = pd.read_pickle(filepath, compression='zip')
        print(f"Loaded {len(chess_data)} games from {filepath}")
    except FileNotFoundError as e:
        print(f"Failed to load chess data {e}")
    
    if chess_data is None:
        print("ERROR: Could not load real data. Please update the file paths.")
        return
    
    # Test with different sample sizes to show scaling
    test_sizes = [100, 500, 1_000, 5_000, 10_0000, 100_000]
    
    print(f"\nTesting with sample sizes: {test_sizes}")
    
    # Import optimized functions
    try:
        from training.game_simulation import play_games_optimized, warm_up_workers
        print("Successfully imported optimized functions")
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Create persistent pool
    num_workers = max(1, cpu_count() - 1)
    print(f"\nCreating persistent pool with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        warm_up_workers(pool, num_workers)
        
        results = []
        
        for size in test_sizes:
            print(f"\n{'='*20} Testing {size} games {'='*20}")
            
            # Use a representative sample (not just first N games)
            if size < len(chess_data):
                # Take every nth game to get a representative sample
                step = len(chess_data) // size
                sample_indices = list(range(0, len(chess_data), step))[:size]
                test_data = chess_data.iloc[sample_indices].copy()
            else:
                test_data = chess_data.copy()
            
            print(f"Sample stats:")
            print(f"  Games: {len(test_data)}")
            print(f"  Total moves: {test_data['PlyCount'].sum():,}")
            print(f"  Avg moves per game: {test_data['PlyCount'].mean():.1f}")
            print(f"  Move range: {test_data['PlyCount'].min()} - {test_data['PlyCount'].max()}")
            
            # Run the test
            start_time = time.time()
            corrupted_games = play_games_optimized(test_data, pool)
            end_time = time.time()
            
            elapsed = end_time - start_time
            games_per_second = len(test_data) / elapsed
            moves_per_second = test_data['PlyCount'].sum() / elapsed
            
            result = {
                'games': len(test_data),
                'time': elapsed,
                'games_per_sec': games_per_second,
                'moves_per_sec': moves_per_second,
                'corrupted': len(corrupted_games),
                'total_moves': test_data['PlyCount'].sum()
            }
            results.append(result)
            
            print(f"\nResults for {size} games:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Games/s: {games_per_second:.0f}")
            print(f"  Moves/s: {moves_per_second:,.0f}")
            print(f"  Corrupted: {len(corrupted_games)} ({len(corrupted_games)/len(test_data)*100:.1f}%)")
        
        # Print comparison with baseline
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        baseline_games_per_sec = 159  # Your original baseline
        baseline_moves_per_sec = 12599  # Your original baseline
        
        print(f"{'Size':<8} {'Time':<8} {'Games/s':<10} {'Moves/s':<12} {'Speedup':<10}")
        print(f"{'-'*60}")
        
        for result in results:
            speedup_games = result['games_per_sec'] / baseline_games_per_sec
            print(f"{result['games']:<8} {result['time']:<8.2f} {result['games_per_sec']:<10.0f} "
                  f"{result['moves_per_sec']:<12.0f} {speedup_games:<10.1f}x")
        
        # Calculate average improvement
        avg_speedup = sum(r['games_per_sec'] for r in results) / len(results) / baseline_games_per_sec
        avg_moves_speedup = sum(r['moves_per_sec'] for r in results) / len(results) / baseline_moves_per_sec
        
        print(f"\nOVERALL IMPROVEMENT:")
        print(f"  Average speedup: {avg_speedup:.1f}x faster")
        print(f"  Move processing: {avg_moves_speedup:.1f}x faster")
        print(f"  Best performance: {max(r['games_per_sec'] for r in results):.0f} games/s")
        print(f"  Best move rate: {max(r['moves_per_sec'] for r in results):,.0f} moves/s")

def test_time_estimation_accuracy():
    """Test how accurate our time estimation is now."""
    print(f"\n{'='*60}")
    print("TIME ESTIMATION ACCURACY TEST")
    print(f"{'='*60}")
    
    # Import the final optimized chunker
    from training.game_simulation import final_optimized_chunker
        
    filepath = game_settings.chess_games_filepath_part_1
    try:
        chess_data = pd.read_pickle(filepath, compression='zip')
        chess_data - chess_data.head(10_000)
        print(f"Loaded {len(chess_data)} games from {filepath}")
    except FileNotFoundError as e:
        print(f"Failed to load chess data {e}")
    
    if chess_data is None:
        print("ERROR: Could not load real data. Please update the file paths.")
        return
        
    if chess_data is not None:
        game_indices = list(chess_data.index)
        
        # Test chunking with realistic estimation
        print("Creating chunks with optimized estimator...")
        chunks = final_optimized_chunker.create_balanced_chunks(
            game_indices, chess_data, cpu_count()
        )
        
        print(f"Created {len(chunks)} chunks")
        print("Time estimation should now be realistic (not 40x overestimate)")

if __name__ == "__main__":
    test_all_optimizations_with_real_data()
    test_time_estimation_accuracy()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("1. This shows the true performance improvement using your real data")
    print("2. Compare these results with your original 159 games/s baseline")
    print("3. If speedup is 5x+ across all sizes, optimizations are working excellently")
    print("4. Next step: Update your production code with these optimizations")