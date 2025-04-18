# file: optimization_testing/profiling_script.py

import cProfile
import pstats
import pandas as pd
import time
import platform
import psutil
import os
import sys
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games
from environment.Environ import Environ
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np

# Define worker initialization function
def init_worker(i):
    """Function to initialize worker processes."""
    return f"Worker {i} initialized"

def warm_up_workers(pool, num_workers):
    """Initialize worker processes with warm-up tasks."""
    print("Warming up worker processes...")
    results = pool.map(init_worker, range(num_workers))
    for result in results:
        print(result)

def get_system_info():
    """Get system information for benchmarking context"""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
    }
    return info

def profile_dataframe_processing(df, num_runs=1):
    """Profile the performance of processing a dataframe"""
    # Get DataFrame statistics
    print(f"DataFrame size: {len(df)} games")
    print(f"Average moves per game: {df['PlyCount'].mean():.2f}")
    
    # Time the full process
    total_times = []
    memory_usages = []
    corrupted_games_counts = []
    
    # Create a persistent process pool
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        # Warm up workers
        warm_up_workers(pool, num_workers)
        
        for i in range(num_runs):
            # Reset cache statistics
            Environ._global_position_cache = {}
            Environ._global_cache_hits = 0
            Environ._global_cache_misses = 0
            
            # Track memory before
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Process the DataFrame
            start_time = time.time()
            try:
                corrupted_games = play_games(df, pool)
                corrupted_games_counts.append(len(corrupted_games))
            except Exception as e:
                print(f"Error during DataFrame processing: {e}")
                corrupted_games_counts.append(0)
            end_time = time.time()
            
            # Track memory after
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_diff = memory_after - memory_before
            memory_usages.append(memory_diff)
            
            elapsed = end_time - start_time
            total_times.append(elapsed)
            
            games_per_second = len(df) / elapsed
            moves_per_second = df['PlyCount'].sum() / elapsed
            
            print(f"\nRun {i+1}: Processed {len(df)} games in {elapsed:.2f}s")
            print(f"Performance: {games_per_second:.2f} games/s, {moves_per_second:.2f} moves/s")
            print(f"Corrupted games: {corrupted_games_counts[-1]}")
            print(f"Memory usage: {memory_diff:.2f} MB")
            
            # Print cache statistics
            Environ.print_cache_stats()
            
            # Add a small delay between runs
            time.sleep(1)
    
    if num_runs > 1:
        avg_time = sum(total_times) / num_runs
        avg_memory = sum(memory_usages) / num_runs
        print(f"\nAverage over {num_runs} runs: {avg_time:.2f}s")
        print(f"Average performance: {len(df) / avg_time:.2f} games/s")
        print(f"Average memory usage: {avg_memory:.2f} MB")
    
    return total_times[0]  # Return the first run time

def run_profiling(filepath, sample_size=None, profile_run=True):
    """Run profiling on a dataframe with detailed stats"""
    print("\n" + "="*50)
    print(f"PROFILING: {filepath}")
    print("="*50)
    
    # Display system information
    sys_info = get_system_info()
    print("\nSystem Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(filepath, compression='zip')
    
    # Use a sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    if profile_run:
        # Profile with cProfile
        print("\nRunning cProfile analysis...")
        
        # Create a persistent process pool for the profiled run
        num_workers = max(1, cpu_count() - 1)
        
        try:
            pr = cProfile.Profile()
            pr.enable()
            
            with Pool(processes=num_workers) as pool:
                # Warm up workers first
                warm_up_workers(pool, num_workers)
                
                # Process the DataFrame
                corrupted_games = play_games(df, pool)
                
            pr.disable()
            
            # Print top functions by cumulative time
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            print("\nTop 20 functions by cumulative time:")
            print(s.getvalue())
            
            # Print call stats for key functions
            print("\nDetailed stats for key functions:")
            stats = pstats.Stats(pr)
            for func_pattern in ['handle_agent_turn', 'choose_action', 'get_legal_moves', 'worker_play_games', 'create_shared_data']:
                print(f"\n{func_pattern}:")
                stats.sort_stats('cumulative').print_stats(func_pattern, 5)
                
        except Exception as e:
            print(f"Error during profiling: {e}")
        
        # Return without additional runs
        return 0
    else:
        # Run standard benchmark without cProfile 
        elapsed = profile_dataframe_processing(df, num_runs=3)
        return elapsed

def profile_memory_usage(filepath, sample_sizes):
    """Profile memory usage for different sample sizes"""
    memory_results = {
        'sample_size': [],
        'memory_usage': [],
        'processing_time': []
    }
    
    for size in sample_sizes:
        print(f"\nProfiling memory usage for sample size: {size}")
        
        # Load the full DataFrame
        df = pd.read_pickle(filepath, compression='zip')
        
        # Take a sample
        if size < len(df):
            sample_df = df.sample(size, random_state=42)
        else:
            sample_df = df
        
        # Measure memory before
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Create a persistent process pool
        num_workers = max(1, cpu_count() - 1)
        with Pool(processes=num_workers) as pool:
            # Warm up workers
            warm_up_workers(pool, num_workers)
            
            # Process the DataFrame
            try:
                start_time = time.time()
                corrupted_games = play_games(sample_df, pool)
                end_time = time.time()
                elapsed = end_time - start_time
            except Exception as e:
                print(f"Error processing DataFrame of size {size}: {e}")
                elapsed = 0
        
        # Measure memory after
        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_diff = memory_after - memory_before
        
        print(f"Sample size: {size}")
        print(f"Memory usage: {memory_diff:.2f} MB")
        print(f"Processing time: {elapsed:.2f}s")
        
        memory_results['sample_size'].append(size)
        memory_results['memory_usage'].append(memory_diff)
        memory_results['processing_time'].append(elapsed)
    
    # Plot the results
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(memory_results['sample_size'], memory_results['memory_usage'], 'o-')
        ax1.set_xlabel('Sample Size (games)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage by Sample Size')
        ax1.grid(True)
        
        ax2.plot(memory_results['sample_size'], memory_results['processing_time'], 'o-')
        ax2.set_xlabel('Sample Size (games)')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time by Sample Size')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('memory_profile.png')
        print("Memory profile plot saved to memory_profile.png")
    except Exception as e:
        print(f"Error creating memory profile plot: {e}")
    
    return memory_results

def profile_cpu_utilization(filepath, sample_size=None):
    """Profile CPU utilization during processing"""
    print("\nProfiling CPU utilization...")
    
    # Load data
    df = pd.read_pickle(filepath, compression='zip')
    
    # Use a sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    # Store CPU measurements
    cpu_percents = []
    timestamps = []
    
    # Function to measure CPU usage over time
    def measure_cpu():
        start_time = time.time()
        while not stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_percents.append(cpu_percent)
            timestamps.append(time.time() - start_time)
            time.sleep(0.5)
    
    # Create and start the CPU monitoring thread
    stop_event = threading.Event()
    cpu_thread = threading.Thread(target=measure_cpu)
    cpu_thread.daemon = True
    cpu_thread.start()
    
    # Process the DataFrame
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        # Warm up workers
        warm_up_workers(pool, num_workers)
        
        # Process the DataFrame
        print("\nProcessing DataFrame...")
        try:
            start_time = time.time()
            corrupted_games = play_games(df, pool)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Processed {len(df)} games in {elapsed:.2f}s")
        except Exception as e:
            print(f"Error processing DataFrame: {e}")
            elapsed = 0
    
    # Stop the CPU monitoring thread
    stop_event.set()
    cpu_thread.join(timeout=1.0)
    
    # Calculate statistics
    avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
    max_cpu = max(cpu_percents) if cpu_percents else 0
    
    print(f"Average CPU utilization: {avg_cpu:.2f}%")
    print(f"Maximum CPU utilization: {max_cpu:.2f}%")
    
    # Plot CPU utilization
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cpu_percents)
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Utilization (%)')
        plt.title('CPU Utilization During Processing')
        plt.grid(True)
        plt.savefig('cpu_utilization.png')
        print("CPU utilization plot saved to cpu_utilization.png")
    except Exception as e:
        print(f"Error creating CPU utilization plot: {e}")
    
    return {
        'timestamps': timestamps,
        'cpu_percents': cpu_percents,
        'avg_cpu': avg_cpu,
        'max_cpu': max_cpu,
        'processing_time': elapsed
    }

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Profile chess database processing system")
    parser.add_argument("--filepath", help="Path to the DataFrame file")
    parser.add_argument("--sample_size", type=int, help="Sample size to use")
    parser.add_argument("--mode", choices=["profile", "memory", "cpu", "all"], default="profile",
                       help="Profiling mode: profile, memory, cpu, or all")
    args = parser.parse_args()
    
    # Get file path
    if args.filepath:
        filepath = args.filepath
    else:
        # Default file path
        from utils import game_settings
        filepath = game_settings.chess_games_filepath_part_1
    
    # Get sample size - use smaller samples on Windows
    if platform.system() == 'Windows':
        default_sample_size = 500
    else:
        default_sample_size = 5000
    sample_size = args.sample_size or default_sample_size
    
    # Run the requested profiling mode
    if args.mode == "profile" or args.mode == "all":
        run_profiling(filepath, sample_size, profile_run=True)
    
    if args.mode == "memory" or args.mode == "all":
        if platform.system() == 'Windows':
            sample_sizes = [50, 100, 200, 500]
        else:
            sample_sizes = [100, 500, 2000, min(5000, sample_size)]
        profile_memory_usage(filepath, sample_sizes)
    
    if args.mode == "cpu" or args.mode == "all":
        profile_cpu_utilization(filepath, sample_size)
    
    print("\nProfiling complete!")