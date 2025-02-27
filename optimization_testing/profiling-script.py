# profile_chess.py

import cProfile
import pstats
import pandas as pd
import time
import platform
import psutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games
from io import StringIO
import sys

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
    
    for i in range(num_runs):
        start_time = time.time()
        corrupted_games = play_games(df)
        end_time = time.time()
        elapsed = end_time - start_time
        total_times.append(elapsed)
        
        games_per_second = len(df) / elapsed
        moves_per_second = df['PlyCount'].sum() / elapsed
        
        print(f"Run {i+1}: Processed {len(df)} games in {elapsed:.2f}s")
        print(f"Performance: {games_per_second:.2f} games/s, {moves_per_second:.2f} moves/s")
        print(f"Corrupted games: {len(corrupted_games)}")
    
    if num_runs > 1:
        avg_time = sum(total_times) / num_runs
        print(f"Average over {num_runs} runs: {avg_time:.2f}s")
        print(f"Average performance: {len(df) / avg_time:.2f} games/s")
    
    return total_times[0]  # Return the first run time

def run_profiling(filepath, sample_size=None):
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
    
    # Profile with cProfile
    print("\nRunning cProfile analysis...")
    pr = cProfile.Profile()
    pr.enable()
    elapsed = profile_dataframe_processing(df)
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
    for func_pattern in ['handle_agent_turn', 'choose_action', 'get_legal_moves', 'worker_play_games']:
        print(f"\n{func_pattern}:")
        stats.sort_stats('cumulative').print_stats(func_pattern, 5)
    
    return elapsed

if __name__ == "__main__":
    # Get the dataframe filepath from command line or use a default
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        # Default file path - adjust as needed
        from utils import game_settings
        filepath = game_settings.chess_games_filepath_part_1
        sample_size = 1000  # Use a small sample for quick profiling
    
    run_profiling(filepath, sample_size)
