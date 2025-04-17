# Example script to demonstrate the various visualization options
# Save this file as: optimization_testing/visualization_examples.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization_testing.profiling_script import get_system_info, analyze_and_suggest_improvements
from optimization_testing.visualization import (
    setup_visualization_style, create_visualization_dir,
    generate_html_dashboard, create_performance_history_dashboard,
    record_performance_history, create_pdf_report
)

def generate_sample_results():
    """Generate sample profiling results for visualization demonstrations"""
    # System info
    system_info = get_system_info()
    
    # Sample summary data
    summary = {
        'total_time': 145.32,
        'games_processed': 5000,
        'games_per_second': 34.41,
        'moves_per_second': 1236.78,
        'peak_memory': 842.56,
        'corrupted_games': 12
    }
    
    # Sample cache stats
    cache_stats = {
        'hits': 34567,
        'misses': 12345,
        'size': 25000
    }
    
    # Sample top functions
    top_functions = [
        {'name': 'handle_agent_turn (game_simulation.py)', 'time': 55.23, 'calls': 78432},
        {'name': 'get_legal_moves (game_rules.py)', 'time': 32.71, 'calls': 164380},
        {'name': 'choose_action (agent.py)', 'time': 24.92, 'calls': 78432},
        {'name': 'apply_move (board.py)', 'time': 18.45, 'calls': 164380},
        {'name': 'evaluate_position (evaluation.py)', 'time': 12.31, 'calls': 45678},
        {'name': 'worker_play_games (pool.py)', 'time': 10.87, 'calls': 8},
        {'name': 'create_shared_data (common.py)', 'time': 5.62, 'calls': 5000},
        {'name': 'parse_move (notation.py)', 'time': 3.82, 'calls': 164380}
    ]
    
    # Sample CPU data (time series)
    timestamps = np.linspace(0, 145, 300)
    base_cpu = 50 + 30 * np.sin(timestamps / 20)  # Oscillating CPU usage
    noise = np.random.normal(0, 5, len(timestamps))
    cpu_percents = np.clip(base_cpu + noise, 0, 100)
    
    cpu_data = {
        'timestamps': timestamps.tolist(),
        'cpu_percents': cpu_percents.tolist(),
        'avg_cpu': np.mean(cpu_percents),
        'max_cpu': np.max(cpu_percents)
    }
    
    # Sample memory by sample size
    memory_by_sample = {
        'sample_size': [100, 500, 1000, 2000, 5000],
        'memory_usage': [95.2, 242.6, 415.8, 618.3, 842.5],
        'processing_time': [3.12, 15.45, 30.87, 62.34, 145.32]
    }
    
    # Sample memory tracking time series
    mem_timestamps = np.linspace(0, 145, 100)
    memory_usage = 200 + 600 * (1 - np.exp(-mem_timestamps / 50))  # Memory grows and plateaus
    mem_noise = np.random.normal(0, 10, len(mem_timestamps))
    memory_usage += mem_noise
    
    # Add some GC events (memory drops)
    gc_events = [30, 60, 90, 120]
    for gc in gc_events:
        idx = np.argmin(np.abs(mem_timestamps - gc))
        memory_usage[idx:idx+3] -= 40
    
    memory_tracking = {
        'timestamps': mem_timestamps.tolist(),
        'memory_usage': memory_usage.tolist(),
        'gc_events': gc_events
    }
    
    # Sample worker stats
    worker_stats = {
        'Worker-1': {'tasks_completed': 64, 'total_time': 142.31, 'avg_time_per_task': 2.22, 'efficiency': 100.0},
        'Worker-2': {'tasks_completed': 58, 'total_time': 140.25, 'avg_time_per_task': 2.42, 'efficiency': 91.8},
        'Worker-3': {'tasks_completed': 62, 'total_time': 141.87, 'avg_time_per_task': 2.29, 'efficiency': 97.0},
        'Worker-4': {'tasks_completed': 52, 'total_time': 138.54, 'avg_time_per_task': 2.66, 'efficiency': 83.4},
        'Worker-5': {'tasks_completed': 60, 'total_time': 139.76, 'avg_time_per_task': 2.33, 'efficiency': 95.3},
        'Worker-6': {'tasks_completed': 54, 'total_time': 138.98, 'avg_time_per_task': 2.57, 'efficiency': 86.3},
        'Worker-7': {'tasks_completed': 56, 'total_time': 140.12, 'avg_time_per_task': 2.50, 'efficiency': 88.8}
    }
    
    # Sample game-level metrics
    n_games = 100
    ply_counts = np.random.randint(30, 120, n_games)
    processing_times = 0.1 + 0.02 * ply_counts + np.random.normal(0, 0.1, n_games)
    moves_per_second = ply_counts / processing_times
    
    # Generate game phase timing
    opening_times = processing_times * 0.2 + np.random.normal(0, 0.02, n_games)
    middlegame_times = processing_times * 0.6 + np.random.normal(0, 0.04, n_games)
    endgame_times = processing_times * 0.2 + np.random.normal(0, 0.02, n_games)
    
    # Generate cache statistics
    total_ops = ply_counts * 10  # Rough estimate of operations per game
    cache_hits = np.random.randint(int(total_ops * 0.5), int(total_ops * 0.8), n_games)
    cache_misses = total_ops - cache_hits
    position_complexity = 15 + 10 * np.random.random(n_games)  # Average branching factor
    
    game_level = {
        'game_id': list(range(1, n_games + 1)),
        'ply_count': ply_counts.tolist(),
        'processing_time': processing_times.tolist(),
        'moves_per_second': moves_per_second.tolist(),
        'opening_time': opening_times.tolist(),
        'middlegame_time': middlegame_times.tolist(),
        'endgame_time': endgame_times.tolist(),
        'cache_hits': cache_hits.tolist(),
        'cache_misses': cache_misses.tolist(),
        'position_complexity': position_complexity.tolist()
    }
    
    # Combine all results
    results = {
        'summary': summary,
        'system': system_info,
        'top_functions': top_functions,
        'cache': cache_stats,
        'cpu': cpu_data,
        'memory_by_sample': memory_by_sample,
        'memory_tracking': memory_tracking,
        'workers': worker_stats,
        'game_level': game_level
    }
    
    return results

def generate_sample_history():
    """Generate sample performance history data for visualizations"""
    # Create a DataFrame with sample history
    current_time = datetime.now()
    timestamps = [(current_time - timedelta(days=i)) for i in range(14, -1, -1)]
    
    # Generate performance metrics with a slight improving trend
    base_games_per_second = 30
    games_per_second = [base_games_per_second + i * 0.5 + np.random.normal(0, 1) for i in range(15)]
    
    base_memory = 800
    peak_memory = [base_memory + i * (-5) + np.random.normal(0, 20) for i in range(15)]
    
    # Cache metrics
    cache_hits = [30000 + i * 500 + np.random.normal(0, 1000) for i in range(15)]
    cache_misses = [15000 - i * 200 + np.random.normal(0, 500) for i in range(15)]
    cache_size = [25000 + i * 100 + np.random.normal(0, 200) for i in range(15)]
    
    # Create DataFrame
    history_df = pd.DataFrame({
        'timestamp': timestamps,
        'games_per_second': games_per_second,
        'moves_per_second': [g * 35 for g in games_per_second],
        'peak_memory': peak_memory,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'cache_size': cache_size
    })
    
    # Save to CSV
    history_file = 'sample_performance_history.csv'
    history_df.to_csv(history_file, index=False)
    print(f"Sample performance history saved to {history_file}")
    
    return history_file

def main():
    """Main function to demonstrate visualizations"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Profiling visualization examples")
    parser.add_argument("--output_dir", default="visualization_examples",
                        help="Output directory for visualizations")
    parser.add_argument("--type", choices=["html", "pdf", "history", "all"],
                       default="all", help="Type of visualization to generate")
    args = parser.parse_args()
    
    # Set up visualization style
    setup_visualization_style()
    
    # Create output directory
    output_dir = create_visualization_dir(args.output_dir)
    print(f"Generating visualizations in {output_dir}")
    
    # Generate sample data
    print("Generating sample profiling results...")
    results = generate_sample_results()
    
    # Generate visualizations based on requested type
    if args.type == "html" or args.type == "all":
        # Generate HTML dashboard
        html_output = os.path.join(output_dir, "sample_dashboard.html")
        generate_html_dashboard(results, html_output)
    
    if args.type == "pdf" or args.type == "all":
        # Generate PDF report
        pdf_output = os.path.join(output_dir, "sample_report.pdf")
        create_pdf_report(results, pdf_output)
    
    if args.type == "history" or args.type == "all":
        # Generate performance history
        history_file = generate_sample_history()
        history_dashboard = os.path.join(output_dir, "sample_history.png")
        create_performance_history_dashboard(history_file, history_dashboard)
    
    print("Visualization examples generated successfully!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()