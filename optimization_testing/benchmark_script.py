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

from training.game_simulation import play_games as original_play_games
from training.optimized_game_simulation import play_games as optimized_play_games

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

def run_benchmark(df, sample_sizes=None, runs_per_sample=3):
    """Run benchmarks on various sample sizes for both implementations."""
    if sample_sizes is None:
        # Default sample sizes
        sample_sizes = [100, 500, 1000]
    
    results = {
        'sample_size': [],
        'original_time': [],
        'optimized_time': [],
        'speedup_factor': []
    }
    
    for size in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Benchmarking with sample size: {size} games")
        print(f"{'='*50}")
        
        # Create a sample of the DataFrame
        if size < len(df):
            sample_df = df.sample(size, random_state=42)
        else:
            sample_df = df
            
        original_times = []
        optimized_times = []
        
        # Run multiple times for statistical significance
        for i in range(runs_per_sample):
            print(f"\nRun {i+1}/{runs_per_sample}:")
            
            # Time original implementation
            with Timer("Original implementation") as t:
                original_play_games(sample_df)
            original_times.append(t.elapsed)
            
            # Time optimized implementation
            with Timer("Optimized implementation") as t:
                optimized_play_games(sample_df)
            optimized_times.append(t.elapsed)
        
        # Calculate average times
        avg_original = sum(original_times) / len(original_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        
        # Calculate speedup
        speedup = avg_original / avg_optimized if avg_optimized > 0 else float('inf')
        
        print(f"\nResults for sample size {size}:")
        print(f"  Original:  {avg_original:.3f}s")
        print(f"  Optimized: {avg_optimized:.3f}s")
        print(f"  Speedup:   {speedup:.2f}x")
        
        # Store results
        results['sample_size'].append(size)
        results['original_time'].append(avg_original)
        results['optimized_time'].append(avg_optimized)
        results['speedup_factor'].append(speedup)
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    sample_sizes = results['sample_size']
    
    ax1.plot(sample_sizes, results['original_time'], 'o-', label='Original')
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
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    # Get the dataframe filepath from command line or use a default
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default file path - adjust as needed
        from utils import game_settings
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
    results = run_benchmark(df, sample_sizes)
    
    # Plot results
    plot_results(results)
    
    print("\nBenchmark complete! Results saved to benchmark_results.png")