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
import gc
import multiprocessing
from multiprocessing import Pool, cpu_count
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games
from environment.Environ import Environ
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter
from collections import defaultdict, Counter
import random
import argparse
from utils import game_settings

# Import visualization module
from profiling_utils import (setup_visualization_style, create_visualization_dir, add_visualization_options)
from visualization import (
                            generate_html_dashboard, create_performance_history_dashboard,
                            record_performance_history, create_pdf_report)


def visualize_profiling_stats(stats, n=20, output_file='profile_stats.png'):
    """Create visualization for cProfile results"""
    print(f"\nGenerating visualization of profiling statistics...")
    
    # Extract data from pstats
    function_stats = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        module_name = func[0]
        line_number = func[1] if len(func) > 1 else 0
        function_name = func[2] if len(func) > 2 else "unknown"
        
        # Skip built-in functions and focus on our code
        if 'site-packages' in module_name or '<built-in>' in module_name:
            continue
            
        function_stats.append({
            'function': f"{function_name} ({module_name.split('/')[-1]}:{line_number})",
            'calls': cc,
            'cumulative_time': ct,
            'total_time': tt,
            'time_per_call': tt/cc if cc > 0 else 0
        })
    
    # Sort by cumulative time and get top N
    function_stats.sort(key=lambda x: x['cumulative_time'], reverse=True)
    top_functions = function_stats[:n]
    
    if not top_functions:
        print("No relevant function statistics found to visualize")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot 1: Cumulative time
    ax1 = axes[0]
    functions = [func['function'] for func in top_functions]
    cum_times = [func['cumulative_time'] for func in top_functions]
    
    # Create horizontal bar chart
    bars = ax1.barh(functions, cum_times, color='skyblue')
    ax1.set_xlabel('Cumulative Time (seconds)')
    ax1.set_title('Top Functions by Cumulative Time')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.1, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}s', 
                va='center')
    
    # Plot 2: Time per call
    ax2 = axes[1]
    time_per_call = [func['time_per_call'] for func in top_functions]
    calls = [func['calls'] for func in top_functions]
    
    # Create scatter plot with size based on number of calls
    scatter = ax2.scatter(time_per_call, functions, 
                         s=[min(c*10, 1000) for c in calls],  # Size based on call count
                         alpha=0.7, 
                         c=time_per_call,  # Color by time per call
                         cmap='viridis')
    
    ax2.set_xlabel('Time per Call (seconds)')
    ax2.set_title('Time per Call (point size = number of calls)')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Time per Call (seconds)')
    
    # Add call count annotations
    for i, (x, y, c) in enumerate(zip(time_per_call, functions, calls)):
        ax2.text(x + 0.00001, y, f' {c} calls', va='center')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Profiling statistics visualization saved to {output_file}")

# Add this function to visualize cache performance
def visualize_cache_performance(cache_stats, output_file='cache_performance.png'):
    """Create visualization for cache performance metrics"""
    print("\nGenerating visualization of cache performance...")
    
    # Extract cache statistics
    hits = cache_stats.get('hits', 0)
    misses = cache_stats.get('misses', 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Cache hits/misses pie chart
    labels = ['Cache Hits', 'Cache Misses']
    sizes = [hits, misses]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice (Hits)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Cache Hit/Miss Ratio')
    
    # Plot 2: Cache efficiency gauge
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel('Cache Hit Rate (%)')
    ax2.set_title('Cache Efficiency')
    
    # Add colored regions
    ax2.axvspan(0, 30, color='#ff9999', alpha=0.3)  # Poor
    ax2.axvspan(30, 70, color='#ffcc99', alpha=0.3)  # Fair
    ax2.axvspan(70, 100, color='#99cc99', alpha=0.3)  # Good
    
    # Add the gauge needle
    ax2.plot([hit_rate, hit_rate], [0, 0.5], 'k-', linewidth=2)
    ax2.plot(hit_rate, 0, 'ko', markersize=10)
    
    # Add text labels
    ax2.text(15, 0.7, 'Poor', ha='center')
    ax2.text(50, 0.7, 'Fair', ha='center')
    ax2.text(85, 0.7, 'Good', ha='center')
    ax2.text(hit_rate, 0.2, f'{hit_rate:.1f}%', ha='center', weight='bold')
    
    # Add cache size info if available
    if 'size' in cache_stats:
        ax2.text(50, 0.9, f"Cache Size: {cache_stats['size']} entries", 
                ha='center', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Cache performance visualization saved to {output_file}")

# Add this function to visualize memory usage over time
def visualize_memory_over_time(memory_data, output_file='memory_over_time.png'):
    """Create visualization for memory usage tracked over time"""
    print("\nGenerating visualization of memory usage over time...")
    
    timestamps = memory_data.get('timestamps', [])
    memory_usage = memory_data.get('memory_usage', [])
    
    if not timestamps or not memory_usage:
        print("No memory tracking data available to visualize")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot memory usage over time
    plt.plot(timestamps, memory_usage, 'b-', linewidth=2)
    
    # Add a trend line
    if len(timestamps) > 1:
        z = np.polyfit(timestamps, memory_usage, 1)
        p = np.poly1d(z)
        plt.plot(timestamps, p(timestamps), "r--", alpha=0.7)
        
        # Calculate slope for memory growth rate
        memory_growth_rate = z[0]  # MB per second
        
        # Add annotation for growth rate
        if abs(memory_growth_rate) > 0.01:  # Only show if significant growth
            plt.text(0.05, 0.95, f'Memory growth rate: {memory_growth_rate:.3f} MB/s', 
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Add markers for GC events if available
    if 'gc_events' in memory_data:
        gc_times = memory_data['gc_events']
        gc_y_positions = np.interp(gc_times, timestamps, memory_usage)
        plt.scatter(gc_times, gc_y_positions, color='red', marker='v', s=80, 
                   label='GC Events', zorder=3)
        plt.legend()
    
    # Add annotations
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line for peak memory
    if memory_usage:
        peak_memory = max(memory_usage)
        plt.axhline(y=peak_memory, color='r', linestyle='-', alpha=0.5)
        plt.text(timestamps[-1] * 0.5, peak_memory * 1.02, 
                f'Peak: {peak_memory:.2f} MB',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Memory usage visualization saved to {output_file}")

# Add this function to visualize worker performance
def visualize_worker_performance(worker_stats, output_file='worker_performance.png'):
    """Create visualization for worker thread/process performance"""
    print("\nGenerating visualization of worker performance...")
    
    if not worker_stats:
        print("No worker statistics available to visualize")
        return
    
    worker_ids = list(worker_stats.keys())
    tasks_completed = [worker_stats[w].get('tasks_completed', 0) for w in worker_ids]
    processing_times = [worker_stats[w].get('total_time', 0) for w in worker_ids]
    avg_task_times = [worker_stats[w].get('avg_time_per_task', 0) for w in worker_ids]
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Tasks completed by each worker
    bars = ax1.bar(worker_ids, tasks_completed, color='skyblue')
    ax1.set_xlabel('Worker ID')
    ax1.set_ylabel('Tasks Completed')
    ax1.set_title('Tasks Completed by Worker')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Calculate and display load imbalance
    if tasks_completed:
        min_tasks = min(tasks_completed)
        max_tasks = max(tasks_completed)
        imbalance = (max_tasks - min_tasks) / max_tasks * 100 if max_tasks > 0 else 0
        ax1.text(0.02, 0.95, f'Load Imbalance: {imbalance:.1f}%', 
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 2: Average time per task with relative efficiency
    # Calculate the fastest worker's time (baseline for efficiency)
    min_time = min(avg_task_times) if avg_task_times else 1
    efficiency = [min_time / t * 100 if t > 0 else 0 for t in avg_task_times]
    
    ax2.bar(worker_ids, avg_task_times, color='lightgreen', alpha=0.7, 
            label='Avg time per task (s)')
    
    # Add efficiency line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(worker_ids, efficiency, 'ro-', linewidth=2, label='Relative efficiency')
    ax2_twin.set_ylabel('Efficiency %')
    ax2_twin.set_ylim(0, 105)
    
    # Add legends
    ax2.set_xlabel('Worker ID')
    ax2.set_ylabel('Average Time per Task (s)')
    ax2.set_title('Worker Efficiency')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Combine legends from both y-axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Worker performance visualization saved to {output_file}")

# Add this function to create a comprehensive dashboard
def create_performance_dashboard(results, output_file='performance_dashboard.png'):
    """Create a comprehensive performance dashboard from all metrics"""
    print("\nGenerating comprehensive performance dashboard...")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Performance Summary - Key Metrics
    ax_summary = fig.add_subplot(gs[0, 0])
    metrics = results.get('summary', {})
    
    # Create a table for key metrics
    metric_names = ['Total Processing Time (s)', 'Games Processed', 'Games/Second',
                   'Moves/Second', 'Peak Memory (MB)', 'Corrupted Games']
    metric_values = [
        f"{metrics.get('total_time', 0):.2f}",
        f"{metrics.get('games_processed', 0)}",
        f"{metrics.get('games_per_second', 0):.2f}",
        f"{metrics.get('moves_per_second', 0):.2f}",
        f"{metrics.get('peak_memory', 0):.2f}",
        f"{metrics.get('corrupted_games', 0)}"
    ]
    
    # Hide axes
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    # Create the table
    table = ax_summary.table(cellText=[metric_values], 
                           rowLabels=['Value'],
                           colLabels=metric_names,
                           loc='center',
                           cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_summary.set_title('Performance Summary')
    
    # 2. CPU Utilization over time
    ax_cpu = fig.add_subplot(gs[0, 1:])
    cpu_data = results.get('cpu', {})
    
    if cpu_data:
        timestamps = cpu_data.get('timestamps', [])
        cpu_percents = cpu_data.get('cpu_percents', [])
        
        if timestamps and cpu_percents:
            ax_cpu.plot(timestamps, cpu_percents, 'g-')
            ax_cpu.set_xlabel('Time (seconds)')
            ax_cpu.set_ylabel('CPU Utilization (%)')
            ax_cpu.set_title('CPU Utilization Over Time')
            ax_cpu.grid(True, linestyle='--', alpha=0.7)
            
            # Add average line
            avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
            ax_cpu.axhline(y=avg_cpu, color='r', linestyle='--')
            ax_cpu.text(timestamps[-1] * 0.8 if timestamps else 0, 
                       avg_cpu + 2, f'Avg: {avg_cpu:.1f}%', 
                       bbox=dict(facecolor='white', alpha=0.8))
    else:
        ax_cpu.text(0.5, 0.5, 'No CPU data available', 
                  ha='center', va='center',
                  bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_cpu.set_title('CPU Utilization (No Data)')
    
    # 3. Memory Usage by Sample Size
    ax_mem = fig.add_subplot(gs[1, 0])
    memory_results = results.get('memory_by_sample', {})
    
    if memory_results:
        sample_sizes = memory_results.get('sample_size', [])
        memory_usage = memory_results.get('memory_usage', [])
        
        if sample_sizes and memory_usage:
            ax_mem.plot(sample_sizes, memory_usage, 'bo-')
            ax_mem.set_xlabel('Sample Size (games)')
            ax_mem.set_ylabel('Memory Usage (MB)')
            ax_mem.set_title('Memory Usage by Sample Size')
            ax_mem.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_mem.text(0.5, 0.5, 'No memory scaling data available', 
                  ha='center', va='center',
                  bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_mem.set_title('Memory by Sample Size (No Data)')
    
    # 4. Processing Time by Sample Size
    ax_time = fig.add_subplot(gs[1, 1])
    
    if memory_results:
        sample_sizes = memory_results.get('sample_size', [])
        proc_times = memory_results.get('processing_time', [])
        
        if sample_sizes and proc_times:
            ax_time.plot(sample_sizes, proc_times, 'ro-')
            ax_time.set_xlabel('Sample Size (games)')
            ax_time.set_ylabel('Processing Time (seconds)')
            ax_time.set_title('Processing Time by Sample Size')
            ax_time.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_time.text(0.5, 0.5, 'No processing time scaling data available', 
                   ha='center', va='center',
                   bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_time.set_title('Processing Time by Sample Size (No Data)')
    
    # 5. Cache Performance
    ax_cache = fig.add_subplot(gs[1, 2])
    cache_stats = results.get('cache', {})
    
    if cache_stats:
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        total = hits + misses
        
        if total > 0:
            hit_rate = hits / total * 100
            labels = ['Cache Hits', 'Cache Misses']
            sizes = [hits, misses]
            colors = ['#66b3ff', '#ff9999']
            explode = (0.1, 0)
            
            ax_cache.pie(sizes, explode=explode, labels=labels, colors=colors, 
                       autopct='%1.1f%%', shadow=True, startangle=90)
            ax_cache.axis('equal')
            ax_cache.set_title(f'Cache Performance\nHit Rate: {hit_rate:.1f}%')
        else:
            ax_cache.text(0.5, 0.5, 'No cache hits or misses recorded', 
                        ha='center', va='center',
                        bbox=dict(facecolor='lightgray', alpha=0.3))
            ax_cache.set_title('Cache Performance (No Data)')
    else:
        ax_cache.text(0.5, 0.5, 'No cache data available', 
                    ha='center', va='center',
                    bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_cache.set_title('Cache Performance (No Data)')
    
    # 6. Top Functions by Time
    ax_funcs = fig.add_subplot(gs[2, :2])
    top_funcs = results.get('top_functions', [])
    
    if top_funcs:
        # Get top 5 functions
        n_funcs = min(5, len(top_funcs))
        funcs = [f['name'] for f in top_funcs[:n_funcs]]
        times = [f['time'] for f in top_funcs[:n_funcs]]
        
        # Create horizontal bar chart
        bars = ax_funcs.barh(funcs, times, color='skyblue')
        ax_funcs.set_xlabel('Cumulative Time (seconds)')
        ax_funcs.set_title('Top Functions by Time')
        ax_funcs.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add text labels
        for bar in bars:
            width = bar.get_width()
            ax_funcs.text(width + 0.1, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}s', 
                        va='center')
    else:
        ax_funcs.text(0.5, 0.5, 'No function profiling data available', 
                    ha='center', va='center',
                    bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_funcs.set_title('Top Functions (No Data)')
    
    # 7. Worker Efficiency
    ax_workers = fig.add_subplot(gs[2, 2])
    worker_stats = results.get('workers', {})
    
    if worker_stats:
        worker_ids = list(worker_stats.keys())
        tasks_completed = [worker_stats[w].get('tasks_completed', 0) for w in worker_ids]
        
        if worker_ids and tasks_completed:
            # Create a pie chart of task distribution
            ax_workers.pie(tasks_completed, labels=worker_ids, autopct='%1.1f%%',
                         shadow=True, startangle=90)
            ax_workers.axis('equal')
            
            # Calculate load imbalance
            min_tasks = min(tasks_completed)
            max_tasks = max(tasks_completed)
            imbalance = (max_tasks - min_tasks) / max_tasks * 100 if max_tasks > 0 else 0
            
            ax_workers.set_title(f'Worker Task Distribution\nLoad Imbalance: {imbalance:.1f}%')
        else:
            ax_workers.text(0.5, 0.5, 'No worker task data available', 
                          ha='center', va='center',
                          bbox=dict(facecolor='lightgray', alpha=0.3))
            ax_workers.set_title('Worker Distribution (No Data)')
    else:
        ax_workers.text(0.5, 0.5, 'No worker statistics available', 
                      ha='center', va='center',
                      bbox=dict(facecolor='lightgray', alpha=0.3))
        ax_workers.set_title('Worker Distribution (No Data)')
    
    # Add system info as a text box
    system_info = results.get('system', {})
    if system_info:
        info_text = (
            f"System: {system_info.get('platform', 'Unknown')}\n"
            f"CPU: {system_info.get('processor', 'Unknown')}\n"
            f"CPUs: {system_info.get('cpu_count', 0)} physical, "
            f"{system_info.get('logical_cpus', 0)} logical\n"
            f"Memory: {system_info.get('memory_total', 0):.1f} GB\n"
            f"Python: {system_info.get('python_version', 'Unknown')}"
        )
        fig.text(0.01, 0.01, info_text, fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
    
    # Add title and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(f'Performance Profiling Dashboard\n{timestamp}', fontsize=16)
    
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Performance dashboard saved to {output_file}")
    
    return fig

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

# Modify run_profiling to collect and visualize results
def run_profiling(filepath, sample_size=None, profile_run=True):
    """Run profiling on a dataframe with detailed stats and visualizations"""
    print("\n" + "="*50)
    print(f"PROFILING: {filepath}")
    print("="*50)
    
    # Store results for visualization
    results = {
        'summary': {},
        'system': {},
        'top_functions': [],
        'cache': {},
        'workers': {},
        'cpu': {},
        'memory_by_sample': {}
    }
    
    # Display system information
    sys_info = get_system_info()
    print("\nSystem Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    results['system'] = sys_info
    
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
            
            # Track memory before
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Track performance metrics
            start_time = time.time()
            
            with Pool(processes=num_workers) as pool:
                # Warm up workers first
                warm_up_workers(pool, num_workers)
                
                # Process the DataFrame
                corrupted_games = play_games(df, pool)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Track memory after
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_diff = memory_after - memory_before
            
            pr.disable()
            
            # Calculate performance metrics
            games_per_second = len(df) / elapsed
            moves_per_second = df['PlyCount'].sum() / elapsed
            
            # Store in results
            results['summary'] = {
                'total_time': elapsed,
                'games_processed': len(df),
                'games_per_second': games_per_second,
                'moves_per_second': moves_per_second,
                'peak_memory': memory_diff,
                'corrupted_games': len(corrupted_games) if corrupted_games else 0
            }
            
            print(f"\nProcessed {len(df)} games in {elapsed:.2f}s")
            print(f"Performance: {games_per_second:.2f} games/s, {moves_per_second:.2f} moves/s")
            print(f"Memory usage: {memory_diff:.2f} MB")
            
            # Print top functions by cumulative time
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            print("\nTop 20 functions by cumulative time:")
            print(s.getvalue())
            
            # Extract top functions for visualization
            stats = pstats.Stats(pr)
            function_stats = []
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                function_name = func[2] if len(func) > 2 else "unknown"
                module_name = func[0] if len(func) > 0 else "unknown"
                
                # Skip built-in functions and focus on our code
                if not ('site-packages' in module_name or '<built-in>' in module_name):
                    function_stats.append({
                        'name': f"{function_name} ({module_name.split('/')[-1]})",
                        'calls': cc,
                        'time': ct,
                        'time_per_call': tt/cc if cc > 0 else 0
                    })
            
            # Sort by time and store in results
            function_stats.sort(key=lambda x: x['time'], reverse=True)
            results['top_functions'] = function_stats[:10]  # Top 10 functions
            
            # Create visualizations
            visualize_profiling_stats(stats)
            
            # Get cache statistics
            cache_stats = {
                'hits': Environ._global_cache_hits,
                'misses': Environ._global_cache_misses,
                'size': len(Environ._global_position_cache)
            }
            results['cache'] = cache_stats
            
            # Visualize cache performance
            visualize_cache_performance(cache_stats)
            
            # Memory tracking is already done in the existing memory profile function
            # CPU tracking is already done in the existing CPU profile function
            
            # Create comprehensive dashboard
            create_performance_dashboard(results)
            
        except Exception as e:
            print(f"Error during profiling: {e}")
        
        # Return the collected results
        return results
    else:
        # Run standard benchmark without cProfile 
        elapsed = profile_dataframe_processing(df, num_runs=3)
        
        # Store basic results
        results['summary'] = {
            'total_time': elapsed,
            'games_processed': len(df),
            'games_per_second': len(df) / elapsed if elapsed > 0 else 0,
            'moves_per_second': df['PlyCount'].sum() / elapsed if elapsed > 0 else 0,
        }
        
        # Create simplified dashboard with available data
        create_performance_dashboard(results, 'basic_performance_dashboard.png')
        
        return results
    

def track_memory_usage(stop_event, memory_data, interval=0.5):
    """Thread function to track memory usage over time"""
    start_time = time.time()
    memory_data['timestamps'] = []
    memory_data['memory_usage'] = []
    memory_data['gc_events'] = []
    
    # Track initial memory
    memory_data['timestamps'].append(0)
    memory_data['memory_usage'].append(psutil.Process().memory_info().rss / (1024 * 1024))
    
    while not stop_event.is_set():
        # Record memory usage
        current_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        memory_data['timestamps'].append(current_time)
        memory_data['memory_usage'].append(memory_usage)
        
        # Detect if garbage collection happened
        gc_count = gc.get_count()
        if len(memory_data['timestamps']) > 1:
            # Simple heuristic: if memory decreased significantly, might be a GC event
            if memory_data['memory_usage'][-2] - memory_data['memory_usage'][-1] > 10:  # 10MB drop
                memory_data['gc_events'].append(current_time)
        
        time.sleep(interval)


def monitor_worker_performance(games_df, chunk_size=100):
    """
    Process DataFrame in chunks with worker performance monitoring.
    
    Args:
        games_df: DataFrame containing games to process
        chunk_size: Number of games per chunk
    
    Returns:
        Dict with worker performance stats
    """

    # Store worker stats
    worker_stats = {}
    worker_lock = threading.Lock()
    
    # Store memory tracking data
    memory_data = {}
    stop_event = threading.Event()
    
    # Create worker wrapper to track performance
    def worker_wrapper(args):
        worker_id = multiprocessing.current_process().name
        chunk_id, game_chunk = args
        
        # Initialize worker stats if needed
        with worker_lock:
            if worker_id not in worker_stats:
                worker_stats[worker_id] = {
                    'tasks_completed': 0,
                    'total_time': 0,
                    'games_processed': 0
                }
        
        # Process the chunk
        start_time = time.time()
        result = play_games_chunk(game_chunk)
        elapsed = time.time() - start_time
        
        # Update worker stats
        with worker_lock:
            worker_stats[worker_id]['tasks_completed'] += 1
            worker_stats[worker_id]['total_time'] += elapsed
            worker_stats[worker_id]['games_processed'] += len(game_chunk)
            worker_stats[worker_id]['avg_time_per_task'] = (
                worker_stats[worker_id]['total_time'] / worker_stats[worker_id]['tasks_completed']
            )
            worker_stats[worker_id]['avg_time_per_game'] = (
                worker_stats[worker_id]['total_time'] / worker_stats[worker_id]['games_processed']
            )
        
        return result
    
    # Split DataFrame into chunks
    chunks = []
    for i in range(0, len(games_df), chunk_size):
        chunk = games_df.iloc[i:i+chunk_size]
        chunks.append((i // chunk_size, chunk))
    
    # Create a thread to track memory usage
    memory_thread = threading.Thread(
        target=track_memory_usage, 
        args=(stop_event, memory_data)
    )
    memory_thread.daemon = True
    
    # Start processing
    print(f"Processing {len(games_df)} games in {len(chunks)} chunks...")
    corrupted_games = []
    
    try:
        # Start memory tracking
        memory_thread.start()
        
        # Create process pool and process chunks
        num_workers = max(1, psutil.cpu_count(logical=False) - 1)
        with Pool(processes=num_workers) as pool:
            # Warm up workers
            warm_up_workers(pool, num_workers)
            
            # Process chunks and collect results
            results = pool.map(worker_wrapper, chunks)
            
            # Combine results (corrupted games lists)
            for res in results:
                if res:  # If there were corrupted games
                    corrupted_games.extend(res)
    
    finally:
        # Stop memory tracking
        stop_event.set()
        memory_thread.join(timeout=1.0)
    
    # Calculate aggregate statistics
    for worker_id in worker_stats:
        tasks = worker_stats[worker_id]['tasks_completed']
        total_time = worker_stats[worker_id]['total_time']
        games = worker_stats[worker_id]['games_processed']
        
        worker_stats[worker_id]['games_per_second'] = games / total_time if total_time > 0 else 0
    
    # Calculate worker efficiency metrics
    if worker_stats:
        # Find the fastest worker (baseline for efficiency)
        fastest_worker = min(
            worker_stats.items(),
            key=lambda x: x[1]['avg_time_per_game'] if x[1]['games_processed'] > 0 else float('inf')
        )[0]
        
        baseline_speed = worker_stats[fastest_worker]['avg_time_per_game']
        
        # Calculate relative efficiency for each worker
        for worker_id in worker_stats:
            if worker_stats[worker_id]['games_processed'] > 0:
                worker_stats[worker_id]['efficiency'] = (
                    baseline_speed / worker_stats[worker_id]['avg_time_per_game'] * 100
                )
            else:
                worker_stats[worker_id]['efficiency'] = 0
    
    return {
        'worker_stats': worker_stats,
        'memory_data': memory_data,
        'corrupted_games': corrupted_games
    }

def play_games_chunk(game_chunk):
    """
    Process a chunk of games from the DataFrame.
    This is a wrapper around play_games to support chunked processing.
    
    Args:
        game_chunk: DataFrame chunk containing games to process
    
    Returns:
        List of corrupted games
    """    
    # Process this chunk
    corrupted_games = play_games(game_chunk, None)  # None = no pool, local execution
    return corrupted_games


# Add a new function to run profiling with detailed worker tracking
def run_profiling_with_worker_tracking(filepath, sample_size=None, chunk_size=100):
    """Run profiling with detailed worker performance tracking"""
    print("\n" + "="*50)
    print(f"PROFILING WITH WORKER TRACKING: {filepath}")
    print("="*50)
    
    # Store results for visualization
    results = {
        'summary': {},
        'system': {},
        'workers': {},
        'memory_tracking': {}
    }
    
    # Display system information
    sys_info = get_system_info()
    print("\nSystem Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    results['system'] = sys_info
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(filepath, compression='zip')
    
    # Use a sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    # Track overall performance
    start_time = time.time()
    
    # Run worker monitoring
    print("\nRunning worker performance monitoring...")
    monitoring_results = monitor_worker_performance(df, chunk_size)
    
    # Record total time
    elapsed = time.time() - start_time
    
    # Store results
    results['workers'] = monitoring_results['worker_stats']
    results['memory_tracking'] = monitoring_results['memory_data']
    
    # Calculate performance metrics
    games_per_second = len(df) / elapsed
    moves_per_second = df['PlyCount'].sum() / elapsed
    
    # Store in summary
    results['summary'] = {
        'total_time': elapsed,
        'games_processed': len(df),
        'games_per_second': games_per_second,
        'moves_per_second': moves_per_second,
        'corrupted_games': len(monitoring_results['corrupted_games'])
    }
    
    print(f"\nProcessed {len(df)} games in {elapsed:.2f}s")
    print(f"Performance: {games_per_second:.2f} games/s, {moves_per_second:.2f} moves/s")
    print(f"Corrupted games: {len(monitoring_results['corrupted_games'])}")
    
    # Visualize worker performance
    if monitoring_results['worker_stats']:
        visualize_worker_performance(monitoring_results['worker_stats'])
    
    # Visualize memory usage over time
    if monitoring_results['memory_data']:
        visualize_memory_over_time(monitoring_results['memory_data'])
    
    # Create comprehensive dashboard
    create_performance_dashboard(results, 'worker_performance_dashboard.png')
    
    return results



def analyze_game_level_performance(games_df, sample_size=None):
    """
    Analyze performance characteristics at the individual game level
    
    Args:
        games_df: DataFrame containing games to process
        sample_size: Optional sample size to use
    
    Returns:
        Dict with game-level performance metrics
    """
    import time
    import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from training.game_simulation import play_game
    
    # Take a sample if needed
    if sample_size and sample_size < len(games_df):
        games_sample = games_df.sample(sample_size, random_state=42)
    else:
        games_sample = games_df
    
    print(f"\nAnalyzing performance for {len(games_sample)} individual games...")
    
    # Performance metrics to track for each game
    game_metrics = {
        'game_id': [],
        'ply_count': [],
        'processing_time': [],
        'moves_per_second': [],
        'opening_time': [],  # Time for first 10 moves
        'middlegame_time': [],  # Time for middle moves
        'endgame_time': [],  # Time for last 10 moves
        'cache_hits': [],
        'cache_misses': [],
        'position_complexity': []  # Estimated by average branching factor
    }
    
    # Process each game individually
    for idx, game in games_sample.iterrows():
        # Initialize cache stats for this game
        Environ._local_cache_hits = 0
        Environ._local_cache_misses = 0
        
        # Run the game and measure performance
        start_time = time.time()
        
        # Measure opening (first 10 moves)
        opening_start = time.time()
        opening_end = None
        
        # Measure endgame (last 10 moves)
        endgame_start = None
        
        try:
            result = play_game(game, measure_sections=True)
            
            # Get section timing from result if available
            if result and isinstance(result, dict) and 'section_timing' in result:
                opening_time = result['section_timing'].get('opening', 0)
                middlegame_time = result['section_timing'].get('middlegame', 0)
                endgame_time = result['section_timing'].get('endgame', 0)
                branching_factor = result.get('avg_branching_factor', 0)
            else:
                # Estimate if not available
                elapsed = time.time() - start_time
                ply_count = game['PlyCount'] if 'PlyCount' in game else 0
                
                # Rough estimates based on typical game phase distribution
                opening_time = elapsed * 0.2
                middlegame_time = elapsed * 0.6
                endgame_time = elapsed * 0.2
                branching_factor = 0  # Unknown
        
        except Exception as e:
            print(f"Error processing game {idx}: {e}")
            continue
        
        # Record total processing time
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Calculate metrics
        ply_count = game['PlyCount'] if 'PlyCount' in game else 0
        moves_per_second = ply_count / elapsed if elapsed > 0 else 0
        
        # Store metrics
        game_metrics['game_id'].append(idx)
        game_metrics['ply_count'].append(ply_count)
        game_metrics['processing_time'].append(elapsed)
        game_metrics['moves_per_second'].append(moves_per_second)
        game_metrics['opening_time'].append(opening_time)
        game_metrics['middlegame_time'].append(middlegame_time)
        game_metrics['endgame_time'].append(endgame_time)
        game_metrics['cache_hits'].append(Environ._local_cache_hits)
        game_metrics['cache_misses'].append(Environ._local_cache_misses)
        game_metrics['position_complexity'].append(branching_factor)
    
    # Create visualizations for game-level metrics
    create_game_level_visualizations(game_metrics)
    
    return game_metrics


def create_game_level_visualizations(metrics, output_prefix='game_level'):
    """Create visualizations for game-level performance metrics"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Relationship between game length and processing time
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['ply_count'], metrics['processing_time'], 
               alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
    
    # Add trendline
    if len(metrics['ply_count']) > 1:
        z = np.polyfit(metrics['ply_count'], metrics['processing_time'], 1)
        p = np.poly1d(z)
        plt.plot(metrics['ply_count'], p(metrics['ply_count']), "r--", alpha=0.8, 
                linewidth=2, label=f'Trend: y={z[0]:.5f}x+{z[1]:.5f}')
    
    plt.xlabel('Game Length (ply count)')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Relationship Between Game Length and Processing Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_length_vs_time.png')
    
    # 2. Game phase timing distribution
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    game_ids = metrics['game_id']
    opening_times = metrics['opening_time']
    middlegame_times = metrics['middlegame_time']
    endgame_times = metrics['endgame_time']
    
    # Limit to 30 games for readability
    if len(game_ids) > 30:
        indices = list(range(len(game_ids)))
        selected_indices = sorted(random.sample(indices, 30))
        game_ids = [game_ids[i] for i in selected_indices]
        opening_times = [opening_times[i] for i in selected_indices]
        middlegame_times = [middlegame_times[i] for i in selected_indices]
        endgame_times = [endgame_times[i] for i in selected_indices]
    
    # Create stacked bar chart
    width = 0.8
    game_indices = range(len(game_ids))
    
    p1 = plt.bar(game_indices, opening_times, width, color='#66c2a5', label='Opening')
    p2 = plt.bar(game_indices, middlegame_times, width, bottom=opening_times, 
                color='#fc8d62', label='Middlegame')
    
    # Calculate bottom position for endgame
    bottom_endgame = [o + m for o, m in zip(opening_times, middlegame_times)]
    p3 = plt.bar(game_indices, endgame_times, width, bottom=bottom_endgame,
                color='#8da0cb', label='Endgame')
    
    plt.xlabel('Game ID')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Time Distribution Across Game Phases')
    plt.legend()
    plt.xticks(game_indices, [str(g)[-4:] for g in game_ids], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_phase_distribution.png')
    
    # 3. Violin plot of moves per second distribution
    plt.figure(figsize=(10, 6))
    
    # Create a violin plot
    sns.violinplot(y=metrics['moves_per_second'], color='skyblue')
    plt.axhline(y=np.median(metrics['moves_per_second']), color='r', linestyle='--', 
               label=f'Median: {np.median(metrics["moves_per_second"]):.2f}')
    
    plt.ylabel('Moves per Second')
    plt.title('Distribution of Processing Speed (Moves per Second)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_speed_distribution.png')
    
    # 4. Cache efficiency by game length
    plt.figure(figsize=(10, 6))
    
    # Calculate cache hit ratio
    cache_hit_ratio = []
    for hits, misses in zip(metrics['cache_hits'], metrics['cache_misses']):
        total = hits + misses
        ratio = hits / total * 100 if total > 0 else 0
        cache_hit_ratio.append(ratio)
    
    # Scatter plot with color gradient based on hit ratio
    scatter = plt.scatter(metrics['ply_count'], metrics['processing_time'], 
                         c=cache_hit_ratio, cmap='viridis', 
                         s=60, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    plt.colorbar(scatter, label='Cache Hit Ratio (%)')
    plt.xlabel('Game Length (ply count)')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Game Processing Time with Cache Efficiency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_cache_efficiency.png')
    
    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare data for correlation
    import pandas as pd
    corr_data = pd.DataFrame({
        'ply_count': metrics['ply_count'],
        'processing_time': metrics['processing_time'],
        'moves_per_second': metrics['moves_per_second'],
        'opening_time': metrics['opening_time'],
        'middlegame_time': metrics['middlegame_time'],
        'endgame_time': metrics['endgame_time'],
        'cache_hits': metrics['cache_hits'],
        'cache_misses': metrics['cache_misses']
    })
    
    # Calculate correlation
    corr = corr_data.corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', square=True)
    plt.title('Correlation Matrix of Performance Metrics')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation_heatmap.png')
    
    print(f"Game-level visualizations saved with prefix '{output_prefix}'")

# Modified main function to include game-level analysis
def run_comprehensive_profiling(filepath, sample_size=None):
    """Run comprehensive profiling with all visualization features"""
    print("\n" + "="*50)
    print(f"COMPREHENSIVE PROFILING: {filepath}")
    print("="*50)
    
    # Store results for all visualizations
    results = {
        'summary': {},
        'system': get_system_info(),
        'top_functions': [],
        'cache': {},
        'workers': {},
        'cpu': {},
        'memory_tracking': {},
        'game_level': {}
    }
    
    # Display system information
    print("\nSystem Information:")
    for key, value in results['system'].items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(filepath, compression='zip')
    
    # Use a sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    # Run standard profiling with cProfile
    print("\nRunning cProfile analysis...")
    pr = cProfile.Profile()
    pr.enable()
    
    # Track overall performance
    start_time = time.time()
    
    # Create a persistent process pool for processing
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        # Warm up workers
        warm_up_workers(pool, num_workers)
        
        # Process the DataFrame
        corrupted_games = play_games(df, pool)
    
    # Record total time
    elapsed = time.time() - start_time
    pr.disable()
    
    # Store basic performance metrics
    results['summary'] = {
        'total_time': elapsed,
        'games_processed': len(df),
        'games_per_second': len(df) / elapsed,
        'moves_per_second': df['PlyCount'].sum() / elapsed,
        'corrupted_games': len(corrupted_games) if corrupted_games else 0
    }
    
    print(f"\nProcessed {len(df)} games in {elapsed:.2f}s")
    print(f"Performance: {results['summary']['games_per_second']:.2f} games/s, "
          f"{results['summary']['moves_per_second']:.2f} moves/s")
    
    # Extract and visualize cProfile results
    stats = pstats.Stats(pr)
    visualize_profiling_stats(stats)
    
    # Get cache statistics
    results['cache'] = {
        'hits': Environ._global_cache_hits,
        'misses': Environ._global_cache_misses,
        'size': len(Environ._global_position_cache)
    }
    
    # Visualize cache performance
    visualize_cache_performance(results['cache'])
    
    # Run CPU utilization profiling
    print("\nProfiling CPU utilization...")
    cpu_results = profile_cpu_utilization(filepath, min(500, sample_size or len(df)))
    results['cpu'] = cpu_results
    
    # Run memory scaling profiling
    print("\nProfiling memory scaling...")
    if sample_size and sample_size > 1000:
        # Use smaller samples for memory scaling test
        sample_sizes = [100, 250, 500, 1000]
    else:
        # Use even smaller samples
        sample_sizes = [50, 100, 200, min(500, sample_size or len(df))]
    
    memory_results = profile_memory_usage(filepath, sample_sizes)
    results['memory_by_sample'] = memory_results
    
    # Run worker performance monitoring
    print("\nRunning worker performance monitoring...")
    # Use smaller sample for worker monitoring
    worker_sample_size = min(500, sample_size or len(df))
    worker_sample = df.sample(worker_sample_size) if worker_sample_size < len(df) else df
    
    monitoring_results = monitor_worker_performance(worker_sample, chunk_size=50)
    results['workers'] = monitoring_results['worker_stats']
    results['memory_tracking'] = monitoring_results['memory_data']
    
    # Visualize worker performance
    visualize_worker_performance(results['workers'])
    
    # Visualize memory usage over time
    visualize_memory_over_time(results['memory_tracking'])
    
    # Run game-level analysis
    print("\nRunning game-level performance analysis...")
    # Use even smaller sample for game-level analysis
    game_level_sample_size = min(100, sample_size or len(df))
    game_sample = df.sample(game_level_sample_size) if game_level_sample_size < len(df) else df
    
    game_metrics = analyze_game_level_performance(game_sample)
    results['game_level'] = game_metrics
    
    # Create comprehensive dashboard
    create_performance_dashboard(results, 'comprehensive_dashboard.png')
    
    # Analyze results and provide suggestions
    print("\nAnalyzing results and generating suggestions...")
    suggestions = analyze_and_suggest_improvements(results)
    
    print("\nSuggested Improvements:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # Save results for later analysis
    try:
        import pickle
        with open('profiling_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\nProfiling results saved to 'profiling_results.pkl'")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

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
    parser = argparse.ArgumentParser(description="Profile chess database processing system")
    parser.add_argument("--filepath", help="Path to the DataFrame file")
    parser.add_argument("--sample_size", type=int, help="Sample size to use")
    parser.add_argument("--mode", choices=["profile", "memory", "cpu", "all", "game_level", "worker", "comprehensive"],
                       default="profile", help="Profiling mode")
    parser.add_argument("--chunk_size", type=int, default=100, 
                        help="Chunk size for parallel processing (worker mode)")
    
    # Add visualization options
    parser = add_visualization_options(parser)
    
    args = parser.parse_args()
    
    # Set up visualization style
    setup_visualization_style()
    
    # Create visualization directory if specified
    vis_dir = args.vis_dir if args.vis_dir else "."
    if args.vis_dir:
        vis_dir = create_visualization_dir(args.vis_dir)
    
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
    results = None
    
    if args.mode == "profile" or args.mode == "all":
        results = run_profiling(filepath, sample_size, profile_run=True)
    
    elif args.mode == "memory" or args.mode == "all":
        if platform.system() == 'Windows':
            sample_sizes = [50, 100, 200, 500]
        else:
            sample_sizes = [100, 500, 2000, min(5000, sample_size)]
        memory_results = profile_memory_usage(filepath, sample_sizes)
        results = {'memory_by_sample': memory_results}
    
    elif args.mode == "cpu" or args.mode == "all":
        cpu_results = profile_cpu_utilization(filepath, sample_size)
        results = {'cpu': cpu_results}
    
    elif args.mode == "game_level":
        # Run game-level analysis
        print("\nRunning game-level performance analysis...")
        # Use smaller sample for game-level analysis
        game_level_sample_size = min(100, sample_size)
        game_metrics = analyze_game_level_performance(filepath, game_level_sample_size)
        results = {'game_level': game_metrics}
    
    elif args.mode == "worker":
        # Run worker performance monitoring
        chunk_size = args.chunk_size
        results = run_profiling_with_worker_tracking(filepath, sample_size, chunk_size)
    
    elif args.mode == "comprehensive":
        # Run all profiling modes with appropriate sample sizes
        results = run_comprehensive_profiling(filepath, sample_size)
    
    # Generate visualizations if results are available
    if results:
        # Generate HTML dashboard if requested
        if args.html_dashboard:
            html_output = os.path.join(vis_dir, "profiling_dashboard.html")
            generate_html_dashboard(results, html_output)
        
        # Generate PDF report if requested
        if args.pdf_report:
            pdf_output = os.path.join(vis_dir, "profiling_report.pdf")
            create_pdf_report(results, pdf_output)
        
        # Record performance history if requested
        if args.track_history:
            history_file = args.history_file
            record_performance_history(results, history_file)
            
            # Create history dashboard
            history_dashboard = os.path.join(vis_dir, "performance_history.png")
            create_performance_history_dashboard(history_file, history_dashboard)
    
    print("\nProfiling complete!")