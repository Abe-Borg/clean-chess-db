# file: optimization_testing/test_priorty2_improvements.py

"""
Test script for Priority 2 improvements: Pool Management & Chunking
"""

import sys
import os
# Add your project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils import game_settings


# Import both old and new functions for comparison
from training.game_simulation import (
    play_games_optimized, 
    improved_adaptive_chunker,
    warm_up_workers
)

def test_chunking_algorithm(chess_data):
    """Test the improved chunking algorithm."""
    print("\nTesting Improved Chunking Algorithm")
    print("=" * 50)
        
    if chess_data is None: 
        print("no chess data, exiting test.")
        return
    
    test_data = chess_data.head(1000)
    game_indices = list(test_data.index)
    
    # Test different scenarios
    scenarios = [
        (8, 50, "Standard case"),
        (16, 200, "High worker count"),
        (4, 500, "Low worker count"),
        (8, 10, "Small min chunk size"),
    ]
    
    for num_workers, min_games_per_chunk, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"  Workers: {num_workers}, Min games per chunk: {min_games_per_chunk}")
        
        chunks = improved_adaptive_chunker.create_balanced_chunks(
            game_indices, test_data, num_workers, min_games_per_chunk
        )
        
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            print(f"  Created {len(chunks)} chunks")
            print(f"  Chunk sizes: {chunk_sizes}")
            print(f"  Size range: {min(chunk_sizes)} - {max(chunk_sizes)}")
            print(f"  Total games: {sum(chunk_sizes)}")

def test_pool_reuse_performance(chess_data):
    """Test performance difference between pool creation vs reuse."""
    print("\nTesting Pool Reuse Performance")
    print("=" * 50)
    
    test_data = chess_data.head(500)  # Smaller sample for multiple runs
    num_workers = max(1, cpu_count() - 1)
    
    # Test with pool creation each time (old way)
    print(f"\nTest 1: Creating new pool each time (old method)")
    runs = 3
    creation_times = []
    
    for run in range(runs):
        start_time = time.time()
        
        # Simulate creating pool each time
        with Pool(processes=num_workers) as pool:
            warm_up_workers(pool, num_workers)
            corrupted_games = play_games_optimized(test_data, pool)
        
        end_time = time.time()
        elapsed = end_time - start_time
        creation_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s ({len(test_data)/elapsed:.1f} games/s)")
    
    avg_creation_time = sum(creation_times) / len(creation_times)
    print(f"  Average with pool creation: {avg_creation_time:.2f}s")
    
    # Test with persistent pool (new way)
    print(f"\nTest 2: Using persistent pool (new method)")
    persistent_pool = Pool(processes=num_workers)
    warm_up_workers(persistent_pool, num_workers)
    
    reuse_times = []
    
    for run in range(runs):
        start_time = time.time()
        
        # Use existing pool
        corrupted_games = play_games_optimized(test_data, persistent_pool)
        
        end_time = time.time()
        elapsed = end_time - start_time
        reuse_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.2f}s ({len(test_data)/elapsed:.1f} games/s)")
    
    # Clean up
    persistent_pool.close()
    persistent_pool.join()
    
    avg_reuse_time = sum(reuse_times) / len(reuse_times)
    print(f"  Average with pool reuse: {avg_reuse_time:.2f}s")
    
    # Calculate improvement
    if avg_creation_time > 0:
        speedup = avg_creation_time / avg_reuse_time
        time_saved = avg_creation_time - avg_reuse_time
        print(f"\nPool Reuse Performance Improvement:")
        print(f"  Speedup: {speedup:.1f}x faster")
        print(f"  Time saved per run: {time_saved:.2f}s")
        print(f"  Percentage improvement: {(time_saved/avg_creation_time)*100:.1f}%")

def test_chunking_balance():
    """Test that chunking creates balanced workloads."""
    print("\nTesting Chunking Balance")
    print("=" * 50)
    
    # Create data with varied game lengths to test balancing
    varied_games = []
    for i in range(200):
        # Create some very long games and some short games
        if i < 20:
            ply_count = 150  # Long games
        elif i < 40:
            ply_count = 20   # Short games
        else:
            ply_count = 50 + (i % 40)  # Mixed lengths
            
        game_data = {'PlyCount': ply_count}
        for j in range(1, min(10, ply_count // 2 + 1)):
            game_data[f'W{j}'] = 'e4'
            if j * 2 <= ply_count:
                game_data[f'B{j}'] = 'e5'
        
        varied_games.append(game_data)
    
    test_data = pd.DataFrame(varied_games, index=[f'Game_{i}' for i in range(len(varied_games))])
    game_indices = list(test_data.index)
    
    print(f"Created test data with varied game lengths:")
    print(f"  Total games: {len(test_data)}")
    print(f"  Ply count range: {test_data['PlyCount'].min()} - {test_data['PlyCount'].max()}")
    print(f"  Average ply count: {test_data['PlyCount'].mean():.1f}")
    
    # Test chunking
    chunks = improved_adaptive_chunker.create_balanced_chunks(
        game_indices, test_data, 8, 20
    )
    
    # Analyze balance
    if chunks:
        chunk_stats = []
        for i, chunk in enumerate(chunks):
            chunk_ply_counts = [test_data.loc[game, 'PlyCount'] for game in chunk]
            total_moves = sum(chunk_ply_counts)
            avg_moves = total_moves / len(chunk) if chunk else 0
            chunk_stats.append({
                'chunk': i,
                'games': len(chunk),
                'total_moves': total_moves,
                'avg_moves': avg_moves
            })
        
        print(f"\nChunk balance analysis:")
        for stats in chunk_stats:
            print(f"  Chunk {stats['chunk']}: {stats['games']} games, "
                  f"{stats['total_moves']} moves, {stats['avg_moves']:.1f} avg")
        
        total_moves_per_chunk = [stats['total_moves'] for stats in chunk_stats]
        if total_moves_per_chunk:
            avg_moves_per_chunk = sum(total_moves_per_chunk) / len(total_moves_per_chunk)
            max_deviation = max(abs(moves - avg_moves_per_chunk) for moves in total_moves_per_chunk)
            imbalance = (max_deviation / avg_moves_per_chunk * 100) if avg_moves_per_chunk > 0 else 0
            print(f"\nBalance metrics:")
            print(f"  Average moves per chunk: {avg_moves_per_chunk:.1f}")
            print(f"  Max deviation: {max_deviation:.1f}")
            print(f"  Imbalance: {imbalance:.1f}%")

if __name__ == "__main__":
    print("Priority 2 Optimization Test")
    print("============================")
    
    chess_data = None
    filepath = game_settings.chess_games_filepath_part_1
    try:
        chess_data = pd.read_pickle(filepath, compression='zip')
        print(f"Loaded {len(chess_data)} games from {filepath}")
    except FileNotFoundError as e:
        print(f"Failed to load chess data {e}")
    # Run all tests

    test_chunking_algorithm(chess_data)
    test_pool_reuse_performance(chess_data)  
    test_chunking_balance()
    
    print("\nPriority 2 Testing Complete!")
    print("\nNext steps:")
    print("1. If pool reuse shows significant improvement, Priority 2 is working")
    print("2. Check that chunking creates balanced workloads")
    print("3. Compare overall performance with your original profiling results")
    print("4. If satisfied, proceed to Priority 3 (Chunking Strategy) or Priority 4 (Cache Fix)")