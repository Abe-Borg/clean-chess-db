# optimization_testing/benchmark_script.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
import psutil
from multiprocessing import Pool, cpu_count
import gc  # For garbage collection between runs

from training.game_simulation import play_games
from utils import game_settings
from environment.Environ import Environ

# Define a proper worker initialization function
def init_worker(i):
    """Function to initialize worker processes."""
    return f"Worker {i} initialized"

def warm_up_workers(pool, num_workers):
    """Initialize worker processes with warm-up tasks."""
    print("Warming up worker processes...")
    results = pool.map(init_worker, range(num_workers))
    for result in results:
        print(result)

class Timer:
    """Timer class for measuring execution time"""
    def __init__(self, description):
        self.description = description
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start
        print(f"{self.description}: {self.elapsed:.3f} seconds")

def get_system_info():
    """Get system information for benchmarking context."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
    }
    return info

def print_system_info():
    """Print system information."""
    info = get_system_info()
    print("\nSystem Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# Function that simulates unoptimized implementation
def play_games_unoptimized(chess_data):
    """Simple implementation without optimizations for comparison."""
    from agents.Agent import Agent
    from environment.Environ import Environ
    import pandas as pd
    
    corrupted_games = []
    
    # Process each game sequentially
    for game_id, row in chess_data.iterrows():
        w_agent = Agent('W')
        b_agent = Agent('B')
        environ = Environ()
        
        ply_count = row['PlyCount']
        
        # Extract moves from row
        game_moves = {}
        for col in chess_data.columns:
            if col != 'PlyCount':
                game_moves[col] = row[col]
        
        # Process the game
        game_corrupted = False
        turn_index = 0
        
        while turn_index < ply_count and not environ.board.is_game_over():
            # White's turn
            if turn_index < ply_count:
                curr_turn = f"W{turn_index // 2 + 1}"
                if curr_turn in game_moves and not pd.isna(game_moves[curr_turn]):
                    chess_move = game_moves[curr_turn]
                    legal_moves = environ.get_legal_moves()
                    
                    if chess_move in legal_moves:
                        environ.board.push_san(chess_move)
                        environ.update_curr_state()
                    else:
                        corrupted_games.append(game_id)
                        game_corrupted = True
                        break
                
                turn_index += 1
            
            # Black's turn
            if turn_index < ply_count and not environ.board.is_game_over():
                curr_turn = f"B{turn_index // 2 + 1}"
                if curr_turn in game_moves and not pd.isna(game_moves[curr_turn]):
                    chess_move = game_moves[curr_turn]
                    legal_moves = environ.get_legal_moves()
                    
                    if chess_move in legal_moves:
                        environ.board.push_san(chess_move)
                        environ.update_curr_state()
                    else:
                        corrupted_games.append(game_id)
                        game_corrupted = True
                        break
                
                turn_index += 1
    
    return corrupted_games

def run_benchmark(df, sample_sizes=None, runs_per_sample=3):
    """Run benchmarks on various sample sizes for both implementations."""
    if sample_sizes is None:
        # Default sample sizes
        sample_sizes = [100, 500, 1000]
    
    results = {
        'sample_size': [],
        'unoptimized_time': [],
        'optimized_time': [],
        'speedup_factor': [],
        'games_per_second_unoptimized': [],
        'games_per_second_optimized': [],
        'memory_usage_unoptimized': [],
        'memory_usage_optimized': []
    }
    
    # Create a process pool for optimized implementation
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        # Warm up workers
        warm_up_workers(pool, num_workers)
        
        for size in sample_sizes:
            print(f"\n{'='*50}")
            print(f"Benchmarking with sample size: {size} games")
            print(f"{'='*50}")
            
            # Create a sample of the DataFrame
            if size < len(df):
                sample_df = df.sample(size, random_state=42)
            else:
                sample_df = df
                
            unoptimized_times = []
            optimized_times = []
            unoptimized_memory = []
            optimized_memory = []
            
            # Run multiple times for statistical significance
            for i in range(runs_per_sample):
                print(f"\nRun {i+1}/{runs_per_sample}:")
                
                # Clear environment cache between runs
                Environ._global_position_cache = {}
                Environ._global_cache_hits = 0
                Environ._global_cache_misses = 0
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory before unoptimized run
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                
                # Time unoptimized implementation
                with Timer("Unoptimized implementation") as t:
                    play_games_unoptimized(sample_df)
                unoptimized_times.append(t.elapsed)
                
                # Measure memory after unoptimized run
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                unoptimized_memory.append(memory_after - memory_before)
                
                # Force garbage collection
                gc.collect()
                
                # Clear environment cache between runs
                Environ._global_position_cache = {}
                Environ._global_cache_hits = 0
                Environ._global_cache_misses = 0
                
                # Measure memory before optimized run
                memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                
                # Time optimized implementation
                with Timer("Optimized implementation") as t:
                    play_games(sample_df, pool)
                optimized_times.append(t.elapsed)
                
                # Measure memory after optimized run
                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                optimized_memory.append(memory_after - memory_before)
                
                # Force garbage collection
                gc.collect()
            
            # Calculate average times and metrics
            avg_unoptimized = sum(unoptimized_times) / len(unoptimized_times)
            avg_optimized = sum(optimized_times) / len(optimized_times)
            avg_unoptimized_memory = sum(unoptimized_memory) / len(unoptimized_memory)
            avg_optimized_memory = sum(optimized_memory) / len(optimized_memory)
            
            # Calculate speedup and throughput
            speedup = avg_unoptimized / avg_optimized if avg_optimized > 0 else float('inf')
            games_per_second_unoptimized = size / avg_unoptimized if avg_unoptimized > 0 else 0
            games_per_second_optimized = size / avg_optimized if avg_optimized > 0 else 0
            
            # Print cache statistics from Environ
            Environ.print_cache_stats()
            
            print(f"\nResults for sample size {size}:")
            print(f"  Unoptimized:  {avg_unoptimized:.3f}s")
            print(f"  Optimized:    {avg_optimized:.3f}s")
            print(f"  Speedup:      {speedup:.2f}x")
            print(f"  Throughput:   {games_per_second_unoptimized:.2f} games/s (unopt) vs {games_per_second_optimized:.2f} games/s (opt)")
            print(f"  Memory usage: {avg_unoptimized_memory:.2f} MB (unopt) vs {avg_optimized_memory:.2f} MB (opt)")
            
            # Store results
            results['sample_size'].append(size)
            results['unoptimized_time'].append(avg_unoptimized)
            results['optimized_time'].append(avg_optimized)
            results['speedup_factor'].append(speedup)
            results['games_per_second_unoptimized'].append(games_per_second_unoptimized)
            results['games_per_second_optimized'].append(games_per_second_optimized)
            results['memory_usage_unoptimized'].append(avg_unoptimized_memory)
            results['memory_usage_optimized'].append(avg_optimized_memory)
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot execution times
    sample_sizes = results['sample_size']
    
    ax1.plot(sample_sizes, results['unoptimized_time'], 'o-', label='Unoptimized')
    ax1.plot(sample_sizes, results['optimized_time'], 'o-', label='Optimized')
    ax1.set_xlabel('Sample Size (games)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Plot speedup factors
    ax2.bar(np.arange(len(sample_sizes)), results['speedup_factor'])
    ax2.set_xticks(np.arange(len(sample_sizes)))
    ax2.set_xticklabels([str(size) for size in sample_sizes])
    ax2.set_xlabel('Sample Size (games)')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_title('Performance Improvement')
    ax2.grid(True, axis='y')
    
    # Add speedup values as text
    for i, factor in enumerate(results['speedup_factor']):
        ax2.text(i, factor + 0.1, f'{factor:.2f}x', ha='center')
    
    # Plot games per second
    ax3.plot(sample_sizes, results['games_per_second_unoptimized'], 'o-', label='Unoptimized')
    ax3.plot(sample_sizes, results['games_per_second_optimized'], 'o-', label='Optimized')
    ax3.set_xlabel('Sample Size (games)')
    ax3.set_ylabel('Throughput (games/second)')
    ax3.set_title('Processing Throughput')
    ax3.grid(True)
    ax3.legend()
    
    # Plot memory usage
    ax4.plot(sample_sizes, results['memory_usage_unoptimized'], 'o-', label='Unoptimized')
    ax4.plot(sample_sizes, results['memory_usage_optimized'], 'o-', label='Optimized')
    ax4.set_xlabel('Sample Size (games)')
    ax4.set_ylabel('Memory Usage (MB)')
    ax4.set_title('Memory Usage')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    # Get the dataframe filepath from command line or use a default
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default file path - adjust as needed
        filepath = game_settings.chess_games_filepath_part_1
    
    # Print system info
    print_system_info()
    
    # Load data
    print(f"\nLoading data from {filepath}...")
    df = pd.read_pickle(filepath, compression='zip')
    print(f"Loaded {len(df)} games, average {df['PlyCount'].mean():.2f} moves per game")
    
    # Define sample sizes
    max_size = min(10000, len(df))  # Limit maximum sample size
    sample_sizes = [100, 500, 2000, 5000, max_size]
    
    # Run benchmarks
    results = run_benchmark(df, sample_sizes, runs_per_sample=2)
    
    # Plot results
    plot_results(results)
    
    print("\nBenchmark complete! Results saved to benchmark_results.png")