
# ------- File: optimization_testing/profiling_utils.py -------

import matplotlib.pyplot as plt
import seaborn as sns
import platform
import psutil
import os

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

def analyze_and_suggest_improvements(results):
    """Analyze profiling results and suggest possible improvements"""
    suggestions = []
    
    # Check CPU utilization
    cpu_data = results.get('cpu', {})
    if cpu_data:
        avg_cpu = cpu_data.get('avg_cpu', 0)
        if avg_cpu < 50:
            suggestions.append("CPU utilization is below 50%. Consider increasing parallelism or "
                              "chunk size to better utilize available CPU resources.")
        elif avg_cpu > 90:
            suggestions.append("CPU utilization is very high (>90%). This is good for throughput "
                              "but might cause system responsiveness issues. Consider setting "
                              "process nice values if running in production.")
    
    # Check worker balance
    worker_stats = results.get('workers', {})
    if worker_stats and len(worker_stats) > 1:
        # Calculate task imbalance
        tasks = [w.get('tasks_completed', 0) for w in worker_stats.values()]
        if tasks:
            min_tasks = min(tasks)
            max_tasks = max(tasks)
            imbalance = (max_tasks - min_tasks) / max_tasks * 100 if max_tasks > 0 else 0
            
            if imbalance > 20:
                suggestions.append(f"Worker load is imbalanced ({imbalance:.1f}% difference). "
                                 f"Consider using smaller chunk sizes or a dynamic work scheduler.")
    
    # Check memory usage pattern
    memory_data = results.get('memory_tracking', {})
    if memory_data:
        memory_usage = memory_data.get('memory_usage', [])
        if memory_usage and len(memory_usage) > 2:
            # Check for steady memory growth
            first_half = memory_usage[:len(memory_usage)//2]
            second_half = memory_usage[len(memory_usage)//2:]
            
            first_half_avg = sum(first_half) / len(first_half)
            second_half_avg = sum(second_half) / len(second_half)
            
            growth_pct = (second_half_avg - first_half_avg) / first_half_avg * 100
            
            if growth_pct > 50:
                suggestions.append(f"Memory usage shows significant growth ({growth_pct:.1f}%). "
                                 f"There might be a memory leak or inefficient cache management.")
    
    # Check cache efficiency
    cache_stats = results.get('cache', {})
    if cache_stats:
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        total = hits + misses
        
        if total > 0:
            hit_rate = hits / total * 100
            if hit_rate < 30:
                suggestions.append(f"Cache hit rate is low ({hit_rate:.1f}%). Consider "
                                 f"reviewing cache strategy or preloading common positions.")
    
    # Check top functions for optimization opportunities
    top_funcs = results.get('top_functions', [])
    if top_funcs:
        time_heavy_funcs = [f for f in top_funcs if f.get('time', 0) > 1.0]
        if time_heavy_funcs:
            func_names = ", ".join(f['name'] for f in time_heavy_funcs[:3])
            suggestions.append(f"Functions consuming significant time: {func_names}. "
                             f"These are prime candidates for optimization.")
    
    # Generate overall recommendations
    if not suggestions:
        suggestions.append("No specific performance issues detected. The system appears to be "
                         "functioning efficiently based on the available metrics.")
    
    return suggestions

def setup_visualization_style():
    """Set up consistent styling for all visualizations"""
    sns.set(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Use a color palette that is colorblind-friendly
    sns.set_palette("colorblind")

def create_visualization_dir(base_dir='profiling_visualizations'):
    """Create a timestamped directory for visualization outputs"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{base_dir}_{timestamp}"
    
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def add_visualization_options(parser):
    """Add visualization-specific command line options"""
    parser.add_argument("--vis_dir", help="Output directory for visualizations")
    parser.add_argument("--html_dashboard", action="store_true", 
                        help="Generate interactive HTML dashboard")
    parser.add_argument("--pdf_report", action="store_true",
                        help="Generate comprehensive PDF report")
    parser.add_argument("--track_history", action="store_true",
                        help="Record performance to history file")
    parser.add_argument("--history_file", default="performance_history.csv",
                        help="CSV file for performance history")
    
    return parser
