# file: optimization_testing/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from profiling_utils import analyze_and_suggest_improvements
                             
# Generate an interactive HTML dashboard
def generate_html_dashboard(results, output_file='profiling_dashboard.html'):
    """
    Generate an interactive HTML dashboard with all profiling visualizations
    
    This creates a self-contained HTML file with interactive charts
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.io as pio
        from plotly.offline import plot
    except ImportError:
        print("Plotly is required for interactive dashboards. Install with: pip install plotly")
        return
    
    # Create the dashboard structure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "pie"}],
            [{"colspan": 2, "type": "table"}, None],
        ],
        subplot_titles=[
            "Performance Metrics", "CPU Utilization",
            "Top Functions by Time", "Cache Performance",
            "Performance Recommendations"
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )
    
    # 1. Performance Metrics Gauge
    summary = results.get('summary', {})
    
    if summary:
        games_per_second = summary.get('games_per_second', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=games_per_second,
                title={"text": "Games Processed per Second"},
                delta={"reference": 10},  # Baseline for comparison
                gauge={
                    "axis": {"range": [None, max(20, games_per_second * 1.5)]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 5], "color": "lightgray"},
                        {"range": [5, 10], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 10,
                    },
                },
            ),
            row=1, col=1
        )
    
    # 2. CPU Utilization Line Chart
    cpu_data = results.get('cpu', {})
    
    if cpu_data:
        timestamps = cpu_data.get('timestamps', [])
        cpu_percents = cpu_data.get('cpu_percents', [])
        
        if timestamps and cpu_percents:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu_percents,
                    mode='lines',
                    name='CPU Utilization',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            # Add average line
            avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
            fig.add_trace(
                go.Scatter(
                    x=[min(timestamps), max(timestamps)],
                    y=[avg_cpu, avg_cpu],
                    mode='lines',
                    name=f'Avg: {avg_cpu:.1f}%',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
            fig.update_yaxes(title_text="CPU Utilization (%)", row=1, col=2)
    
    # 3. Top Functions Bar Chart
    top_funcs = results.get('top_functions', [])
    
    if top_funcs:
        # Get top 5 functions
        n_funcs = min(5, len(top_funcs))
        funcs = [f.get('name', 'Unknown') for f in top_funcs[:n_funcs]]
        times = [f.get('time', 0) for f in top_funcs[:n_funcs]]
        
        fig.add_trace(
            go.Bar(
                x=times,
                y=funcs,
                orientation='h',
                marker=dict(color='skyblue'),
                name='Function Time'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Cumulative Time (seconds)", row=2, col=1)
    
    # 4. Cache Performance Pie Chart
    cache_stats = results.get('cache', {})
    
    if cache_stats:
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        total = hits + misses
        
        if total > 0:
            labels = ['Cache Hits', 'Cache Misses']
            values = [hits, misses]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo='label+percent',
                    marker=dict(colors=['#66b3ff', '#ff9999'])
                ),
                row=2, col=2
            )
    
    # 5. Performance Recommendations Table
    suggestions = analyze_and_suggest_improvements(results)
    
    if suggestions:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Optimization Recommendations"],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=14)
                ),
                cells=dict(
                    values=[suggestions],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12),
                    height=30
                )
            ),
            row=3, col=1
        )
    
    # Add system info as annotation
    system_info = results.get('system', {})
    if system_info:
        info_text = (
            f"System: {system_info.get('platform', 'Unknown')}<br>"
            f"CPU: {system_info.get('processor', 'Unknown')}<br>"
            f"CPUs: {system_info.get('cpu_count', 0)} physical, "
            f"{system_info.get('logical_cpus', 0)} logical<br>"
            f"Memory: {system_info.get('memory_total', 0):.1f} GB<br>"
            f"Python: {system_info.get('python_version', 'Unknown')}"
        )
        
        fig.add_annotation(
            x=0,
            y=-0.2,
            xref="paper",
            yref="paper",
            text=info_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    # Update layout and add title
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.update_layout(
        title=f"Performance Profiling Dashboard ({timestamp})",
        showlegend=True,
        height=900,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Write to HTML file
    try:
        plot(fig, filename=output_file, auto_open=False)
        print(f"Interactive dashboard saved to {output_file}")
    except Exception as e:
        print(f"Error saving interactive dashboard: {e}")

# Create a time series dashboard for tracking performance over multiple runs
def create_performance_history_dashboard(history_file='performance_history.csv', 
                                         output_file='performance_history.png'):
    """
    Create a dashboard showing performance trends over time from a history file
    
    Args:
        history_file: CSV file with performance history
        output_file: Output image file
    """
    try:
        # Load performance history
        history_df = pd.read_csv(history_file)
        
        if len(history_df) < 2:
            print("Not enough data for historical trends (need at least 2 data points)")
            return
        
        # Extract timestamps and convert to datetime
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Set up the figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # 1. Games per second over time
        ax1 = axes[0]
        ax1.plot(history_df['timestamp'], history_df['games_per_second'], 
                'o-', color='blue', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Games per Second')
        ax1.set_title('Processing Speed Over Time')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add trendline
        z = np.polyfit(mdates.date2num(history_df['timestamp']), 
                       history_df['games_per_second'], 1)
        p = np.poly1d(z)
        x_dates = mdates.date2num(history_df['timestamp'])
        ax1.plot(history_df['timestamp'], p(x_dates), "r--", 
                 label=f'Trend: {"+" if z[0]>0 else ""}{z[0]:.6f} per day')
        ax1.legend()
        
        # 2. Memory usage over time
        ax2 = axes[1]
        ax2.plot(history_df['timestamp'], history_df['peak_memory'], 
                'o-', color='green', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Usage Over Time')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Cache hit rate over time
        ax3 = axes[2]
        # Calculate cache hit rate if components are available
        if 'cache_hits' in history_df.columns and 'cache_misses' in history_df.columns:
            history_df['cache_hit_rate'] = (
                history_df['cache_hits'] / 
                (history_df['cache_hits'] + history_df['cache_misses']) * 100
            )
            
            ax3.plot(history_df['timestamp'], history_df['cache_hit_rate'], 
                    'o-', color='purple', linewidth=2)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Cache Hit Rate (%)')
            ax3.set_title('Cache Efficiency Over Time')
            ax3.set_ylim(0, 100)
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis dates
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'No cache history data available', 
                     ha='center', va='center', transform=ax3.transAxes,
                     bbox=dict(facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Performance history dashboard saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating performance history dashboard: {e}")

# Record performance results to history file
def record_performance_history(results, history_file='performance_history.csv'):
    """
    Record current performance results to a history file
    
    Args:
        results: Dictionary with performance results
        history_file: CSV file to store history
    """
    import os
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract metrics to record
    summary = results.get('summary', {})
    cache_stats = results.get('cache', {})
    
    metrics = {
        'timestamp': timestamp,
        'games_per_second': summary.get('games_per_second', 0),
        'moves_per_second': summary.get('moves_per_second', 0),
        'peak_memory': summary.get('peak_memory', 0),
        'cache_hits': cache_stats.get('hits', 0),
        'cache_misses': cache_stats.get('misses', 0),
        'cache_size': cache_stats.get('size', 0)
    }
    
    # Check if file exists
    file_exists = os.path.isfile(history_file)
    
    try:
        with open(history_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write current metrics
            writer.writerow(metrics)
            
        print(f"Performance metrics recorded to {history_file}")
    except Exception as e:
        print(f"Error recording performance history: {e}")

# Create a comprehensive multi-page PDF report
def create_pdf_report(results, output_file='profiling_report.pdf'):
    """
    Create a comprehensive PDF report with all profiling results
    
    Args:
        results: Dictionary with all profiling results
        output_file: Output PDF file
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("This feature requires matplotlib with PDF backend")
        return
    
    with PdfPages(output_file) as pdf:
        # 1. Title page with summary
        fig = plt.figure(figsize=(12, 9))
        plt.axis('off')
        
        # Get summary and system info
        summary = results.get('summary', {})
        system_info = results.get('system', {})
        
        # Title
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title_text = f"Performance Profiling Report\n{timestamp}"
        plt.text(0.5, 0.9, title_text, fontsize=24, ha='center')
        
        # Summary table
        summary_text = (
            "Performance Summary:\n\n"
            f"Total Processing Time: {summary.get('total_time', 0):.2f} seconds\n"
            f"Games Processed: {summary.get('games_processed', 0)}\n"
            f"Games per Second: {summary.get('games_per_second', 0):.2f}\n"
            f"Moves per Second: {summary.get('moves_per_second', 0):.2f}\n"
            f"Peak Memory Usage: {summary.get('peak_memory', 0):.2f} MB\n"
            f"Corrupted Games: {summary.get('corrupted_games', 0)}\n"
        )
        plt.text(0.5, 0.7, summary_text, fontsize=14, ha='center', va='top')
        
        # System information
        if system_info:
            system_text = (
                "System Information:\n\n"
                f"Platform: {system_info.get('platform', 'Unknown')}\n"
                f"Processor: {system_info.get('processor', 'Unknown')}\n"
                f"CPU Count: {system_info.get('cpu_count', 0)} physical, "
                f"{system_info.get('logical_cpus', 0)} logical\n"
                f"Memory: {system_info.get('memory_total', 0):.1f} GB\n"
                f"Python Version: {system_info.get('python_version', 'Unknown')}\n"
            )
            plt.text(0.5, 0.4, system_text, fontsize=14, ha='center', va='top')
        
        pdf.savefig()
        plt.close()
        
        # 2. CPU Utilization
        cpu_data = results.get('cpu', {})
        if cpu_data and cpu_data.get('timestamps') and cpu_data.get('cpu_percents'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cpu_data['timestamps'], cpu_data['cpu_percents'], 'g-')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('CPU Utilization (%)')
            ax.set_title('CPU Utilization Over Time')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add average line
            avg_cpu = sum(cpu_data['cpu_percents']) / len(cpu_data['cpu_percents'])
            ax.axhline(y=avg_cpu, color='r', linestyle='--')
            ax.text(cpu_data['timestamps'][-1] * 0.8, 
                   avg_cpu + 2, f'Avg: {avg_cpu:.1f}%', 
                   bbox=dict(facecolor='white', alpha=0.8))
            
            pdf.savefig()
            plt.close()
        
        # 3. Memory Scaling
        memory_results = results.get('memory_by_sample', {})
        if memory_results and memory_results.get('sample_size') and memory_results.get('memory_usage'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(memory_results['sample_size'], memory_results['memory_usage'], 'bo-')
            ax.set_xlabel('Sample Size (games)')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage by Sample Size')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            pdf.savefig()
            plt.close()
            
            # Processing time by sample size
            if memory_results.get('processing_time'):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(memory_results['sample_size'], memory_results['processing_time'], 'ro-')
                ax.set_xlabel('Sample Size (games)')
                ax.set_ylabel('Processing Time (seconds)')
                ax.set_title('Processing Time by Sample Size')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                pdf.savefig()
                plt.close()
        
        # 4. Cache Performance
        cache_stats = results.get('cache', {})
        if cache_stats:
            hits = cache_stats.get('hits', 0)
            misses = cache_stats.get('misses', 0)
            total = hits + misses
            
            if total > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                labels = ['Cache Hits', 'Cache Misses']
                sizes = [hits, misses]
                colors = ['#66b3ff', '#ff9999']
                explode = (0.1, 0)
                
                ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                       autopct='%1.1f%%', shadow=True, startangle=90)
                ax.axis('equal')
                
                hit_rate = hits / total * 100
                ax.set_title(f'Cache Performance\nHit Rate: {hit_rate:.1f}%')
                
                pdf.savefig()
                plt.close()
        
        # 5. Top Functions
        top_funcs = results.get('top_functions', [])
        if top_funcs:
            # Get top 10 functions or fewer
            n_funcs = min(10, len(top_funcs))
            funcs = [f.get('name', 'Unknown') for f in top_funcs[:n_funcs]]
            times = [f.get('time', 0) for f in top_funcs[:n_funcs]]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(funcs, times, color='skyblue')
            ax.set_xlabel('Cumulative Time (seconds)')
            ax.set_title('Top Functions by Time')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add text labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, 
                      bar.get_y() + bar.get_height()/2, 
                      f'{width:.2f}s', 
                      va='center')
            
            pdf.savefig()
            plt.close()
        
        # 6. Worker Performance
        worker_stats = results.get('workers', {})
        if worker_stats:
            worker_ids = list(worker_stats.keys())
            tasks_completed = [worker_stats[w].get('tasks_completed', 0) for w in worker_ids]
            
            if worker_ids and tasks_completed:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.bar(worker_ids, tasks_completed, color='skyblue')
                ax.set_xlabel('Worker ID')
                ax.set_ylabel('Tasks Completed')
                ax.set_title('Tasks Completed by Worker')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add text labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom')
                
                # Calculate and display load imbalance
                min_tasks = min(tasks_completed)
                max_tasks = max(tasks_completed)
                imbalance = (max_tasks - min_tasks) / max_tasks * 100 if max_tasks > 0 else 0
                ax.text(0.02, 0.95, f'Load Imbalance: {imbalance:.1f}%', 
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8))
                
                pdf.savefig()
                plt.close()
                
                # Worker efficiency if available
                if any('efficiency' in worker_stats[w] for w in worker_ids):
                    efficiencies = [worker_stats[w].get('efficiency', 0) for w in worker_ids]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    bars = ax.bar(worker_ids, efficiencies, color='lightgreen')
                    ax.set_xlabel('Worker ID')
                    ax.set_ylabel('Relative Efficiency (%)')
                    ax.set_title('Worker Efficiency')
                    ax.set_ylim(0, 105)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add text labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}%', ha='center', va='bottom')
                    
                    pdf.savefig()
                    plt.close()
        
        # 7. Game Level Metrics (if available)
        game_level = results.get('game_level', {})
        if game_level and game_level.get('ply_count') and game_level.get('processing_time'):
            # Game length vs processing time
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(game_level['ply_count'], game_level['processing_time'], 
                      alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
            
            # Add trendline
            if len(game_level['ply_count']) > 1:
                z = np.polyfit(game_level['ply_count'], game_level['processing_time'], 1)
                p = np.poly1d(z)
                ax.plot(game_level['ply_count'], p(game_level['ply_count']), "r--", alpha=0.7, 
                       linewidth=2, label=f'Trend: y={z[0]:.5f}x+{z[1]:.5f}')
            
            ax.set_xlabel('Game Length (ply count)')
            ax.set_ylabel('Processing Time (seconds)')
            ax.set_title('Relationship Between Game Length and Processing Time')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            pdf.savefig()
            plt.close()
            
            # If cache metrics are available at game level
            if 'cache_hits' in game_level and 'cache_misses' in game_level:
                # Calculate cache hit ratio
                cache_hit_ratio = []
                for hits, misses in zip(game_level['cache_hits'], game_level['cache_misses']):
                    total = hits + misses
                    ratio = hits / total * 100 if total > 0 else 0
                    cache_hit_ratio.append(ratio)
                
                # Plot cache efficiency by game length
                fig, ax = plt.subplots(figsize=(10, 6))
                
                scatter = ax.scatter(game_level['ply_count'], game_level['processing_time'], 
                                   c=cache_hit_ratio, cmap='viridis', 
                                   s=60, alpha=0.8, edgecolor='k', linewidth=0.5)
                
                plt.colorbar(scatter, label='Cache Hit Ratio (%)')
                ax.set_xlabel('Game Length (ply count)')
                ax.set_ylabel('Processing Time (seconds)')
                ax.set_title('Game Processing Time with Cache Efficiency')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                pdf.savefig()
                plt.close()
        
        # 8. Recommendations
        suggestions = analyze_and_suggest_improvements(results)
        if suggestions:
            fig = plt.figure(figsize=(10, 8))
            plt.axis('off')
            
            plt.text(0.5, 0.95, "Performance Optimization Recommendations", 
                    fontsize=18, ha='center')
            
            recommendation_text = ""
            for i, suggestion in enumerate(suggestions, 1):
                recommendation_text += f"{i}. {suggestion}\n\n"
            
            plt.text(0.1, 0.85, recommendation_text, fontsize=12, va='top')
            
            pdf.savefig()
            plt.close()
    
    print(f"Performance report saved to {output_file}")

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