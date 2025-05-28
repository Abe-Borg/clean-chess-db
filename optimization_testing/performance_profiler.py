# file: optimization_testing/performance_profiler.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import pandas as pd
import numpy as np
import time
import psutil
import gc
import tracemalloc
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from io import StringIO

# Import the chess processing modules
from training.game_simulation import play_games, create_shared_data, cleanup_shared_data
from training.game_simulation import worker_process_games, play_one_game
from training.game_simulation import AdaptiveChunker
from environment.Environ import Environ
from agents.Agent import Agent
from main.play_games import create_persistent_pool
from utils import game_settings

# For line profiling
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    print("Warning: line_profiler not installed. Install with: pip install line_profiler")

# For memory profiling
try:
    from memory_profiler import profile as memory_profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("Warning: memory_profiler not installed. Install with: pip install memory-profiler")


class PerformanceProfiler:
    """Comprehensive performance profiling for chess game processing."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        """Initialize the profiler with output directory."""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(output_dir, f"profile_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results = {
            'timestamp': self.timestamp,
            'system_info': self._get_system_info(),
            'configurations': {},
            'profiling_data': {},
            'performance_metrics': {},
            'bottlenecks': [],
            'recommendations': []
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect detailed system information."""
        import platform
        
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
        
        memory_info = psutil.virtual_memory()._asdict()
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu': cpu_info,
            'memory': memory_info,
            'process_info': {
                'pid': os.getpid(),
                'nice': psutil.Process().nice(),
                'cpu_affinity': psutil.Process().cpu_affinity() if hasattr(psutil.Process(), 'cpu_affinity') else None
            }
        }
    
    def profile_configurations(self, chess_data: pd.DataFrame, 
                             worker_counts: List[int] = None,
                             chunk_strategies: List[str] = None) -> Dict[str, Any]:
        """Test different configurations to find optimal settings."""
        print("\n=== Testing Different Configurations ===")
        
        if worker_counts is None:
            # Test from 1 to number of physical cores
            max_workers = psutil.cpu_count(logical=False)
            worker_counts = [1, 2] + list(range(4, max_workers + 1, 2))
            worker_counts = [w for w in worker_counts if w <= max_workers]
        
        if chunk_strategies is None:
            chunk_strategies = ['balanced', 'fixed_size', 'adaptive']
        
        config_results = {}
        
        for num_workers in worker_counts:
            for chunk_strategy in chunk_strategies:
                config_name = f"workers_{num_workers}_{chunk_strategy}"
                print(f"\nTesting configuration: {config_name}")
                
                # Clear any caches
                gc.collect()
                
                # Create pool
                with Pool(processes=num_workers) as pool:
                    # Measure performance
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    try:
                        corrupted_games = play_games(chess_data, pool)
                        success = True
                    except Exception as e:
                        print(f"Error in configuration {config_name}: {e}")
                        corrupted_games = []
                        success = False
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    elapsed = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    # Calculate metrics
                    games_per_second = len(chess_data) / elapsed if elapsed > 0 else 0
                    moves_per_second = chess_data['PlyCount'].sum() / elapsed if elapsed > 0 else 0
                    
                    config_results[config_name] = {
                        'num_workers': num_workers,
                        'chunk_strategy': chunk_strategy,
                        'success': success,
                        'elapsed_time': elapsed,
                        'memory_used_mb': memory_used,
                        'games_per_second': games_per_second,
                        'moves_per_second': moves_per_second,
                        'corrupted_games': len(corrupted_games),
                        'efficiency': games_per_second / num_workers if num_workers > 0 else 0
                    }
                    
                    print(f"  Games/s: {games_per_second:.2f}, Moves/s: {moves_per_second:.2f}")
                    print(f"  Memory: {memory_used:.2f} MB, Efficiency: {config_results[config_name]['efficiency']:.2f}")
        
        # Find best configuration
        best_config = max(config_results.items(), 
                         key=lambda x: x[1]['games_per_second'] if x[1]['success'] else 0)
        
        print(f"\nBest configuration: {best_config[0]}")
        print(f"  Performance: {best_config[1]['games_per_second']:.2f} games/s")
        
        self.results['configurations'] = config_results
        return config_results
    
    def profile_components(self, chess_data: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """Profile individual components with detailed timing."""
        print("\n=== Component-Level Profiling ===")
        
        # Sample data
        sample_data = chess_data.sample(min(sample_size, len(chess_data)), random_state=42)
        
        component_timings = {
            'game_loading': [],
            'move_parsing': [],
            'move_validation': [],
            'board_updates': [],
            'state_queries': [],
            'shared_memory_ops': [],
            'worker_communication': []
        }
        
        # Test 1: Profile single game processing
        game_id = sample_data.index[0]
        row = sample_data.loc[game_id]
        ply_count = int(row['PlyCount'])
        moves = {col: row[col] for col in sample_data.columns if col != 'PlyCount'}
        
        # Create agents and environment
        w_agent = Agent('W')
        b_agent = Agent('B')
        environ = Environ()
        
        # Profile individual operations
        move_timings = []
        
        for turn_idx in range(ply_count):
            # Time state query
            t0 = time.perf_counter()
            curr_state, legal_moves = environ.get_curr_state_and_legal_moves()
            t1 = time.perf_counter()
            component_timings['state_queries'].append(t1 - t0)
            
            # Time move selection
            curr_turn = curr_state['curr_turn']
            agent = w_agent if curr_state['white_to_move'] else b_agent
            
            t0 = time.perf_counter()
            chess_move_san = agent.choose_action(moves, curr_turn)
            t1 = time.perf_counter()
            component_timings['move_parsing'].append(t1 - t0)
            
            if chess_move_san and not pd.isna(chess_move_san):
                # Time move validation
                t0 = time.perf_counter()
                try:
                    move_obj = environ.convert_san_to_move_object(chess_move_san)
                    valid = move_obj in legal_moves
                except:
                    valid = False
                t1 = time.perf_counter()
                component_timings['move_validation'].append(t1 - t0)
                
                if valid:
                    # Time board update
                    t0 = time.perf_counter()
                    environ.push_move_object(move_obj)
                    t1 = time.perf_counter()
                    component_timings['board_updates'].append(t1 - t0)
                else:
                    break
            else:
                environ.update_curr_state()
        
        # Test 2: Profile shared memory operations
        print("\nProfiling shared memory operations...")
        
        t0 = time.perf_counter()
        shared_data = create_shared_data(sample_data)
        t1 = time.perf_counter()
        component_timings['shared_memory_ops'].append(t1 - t0)
        
        # Cleanup
        cleanup_shared_data(shared_data)
        
        # Calculate statistics
        component_stats = {}
        for component, timings in component_timings.items():
            if timings:
                component_stats[component] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'mean_time': np.mean(timings),
                    'median_time': np.median(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'std_time': np.std(timings)
                }
        
        self.results['profiling_data']['components'] = component_stats
        
        # Print summary
        print("\nComponent timing summary:")
        for component, stats in component_stats.items():
            if stats['count'] > 0:
                print(f"  {component}: {stats['mean_time']*1000:.3f}ms avg "
                      f"({stats['total_time']:.3f}s total, {stats['count']} calls)")
        
        return component_stats
    
    def profile_with_line_profiler(self, chess_data: pd.DataFrame, sample_size: int = 10) -> Optional[Dict]:
        """Use line_profiler for detailed line-by-line analysis."""
        if not HAS_LINE_PROFILER:
            print("\nSkipping line profiling (line_profiler not installed)")
            return None
        
        print("\n=== Line-by-Line Profiling ===")
        
        # Sample data
        sample_data = chess_data.sample(min(sample_size, len(chess_data)), random_state=42)
        
        # Create line profiler
        lp = LineProfiler()
        
        # Add functions to profile
        lp.add_function(play_one_game)
        lp.add_function(worker_process_games)
        lp.add_function(Environ.get_curr_state_and_legal_moves)
        lp.add_function(Environ.convert_san_to_move_object)
        lp.add_function(Environ.push_move_object)
        
        # Wrap the processing function
        lp.enable()
        
        # Process games
        with Pool(processes=1) as pool:
            corrupted_games = play_games(sample_data, pool)
        
        lp.disable()
        
        # Get stats
        stats = StringIO()
        lp.print_stats(stats)
        
        # Save to file
        with open(os.path.join(self.results_dir, 'line_profile.txt'), 'w') as f:
            f.write(stats.getvalue())
        
        print(f"\nLine profiling results saved to {self.results_dir}/line_profile.txt")
        
        # Parse key findings
        self._analyze_line_profile(stats.getvalue())
        
        return {'line_profile': stats.getvalue()}
    
    def profile_memory_detailed(self, chess_data: pd.DataFrame, 
                               sample_sizes: List[int] = None) -> Dict[str, Any]:
        """Profile memory usage patterns in detail."""
        print("\n=== Detailed Memory Profiling ===")
        
        if sample_sizes is None:
            sample_sizes = [100, 500, 1000, 5000, 10000]
        
        # Filter to available data
        sample_sizes = [s for s in sample_sizes if s <= len(chess_data)]
        
        memory_results = {
            'by_sample_size': {},
            'allocation_patterns': {},
            'gc_stats': {}
        }
        
        # Enable tracemalloc
        tracemalloc.start()
        
        for size in sample_sizes:
            print(f"\nProfiling memory for {size} games...")
            
            # Sample data
            sample_data = chess_data.sample(size, random_state=42)
            
            # Clear memory
            gc.collect()
            
            # Get baseline
            snapshot1 = tracemalloc.take_snapshot()
            gc_stats_before = gc.get_stats()
            
            # Process games
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            with Pool(processes=cpu_count() - 1) as pool:
                corrupted_games = play_games(sample_data, pool)
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Get memory snapshot
            snapshot2 = tracemalloc.take_snapshot()
            gc_stats_after = gc.get_stats()
            
            # Analyze memory usage
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            
            # Store results
            memory_results['by_sample_size'][size] = {
                'memory_used_mb': end_memory - start_memory,
                'memory_per_game': (end_memory - start_memory) / size,
                'gc_collections': len(gc_stats_after) - len(gc_stats_before),
                'top_allocations': [
                    {
                        'file': stat.traceback[0].filename,
                        'line': stat.traceback[0].lineno,
                        'size_diff': stat.size_diff / 1024 / 1024,  # MB
                        'count_diff': stat.count_diff
                    }
                    for stat in top_stats[:10]
                ]
            }
            
            print(f"  Memory used: {memory_results['by_sample_size'][size]['memory_used_mb']:.2f} MB")
            print(f"  Memory per game: {memory_results['by_sample_size'][size]['memory_per_game']:.3f} MB")
        
        # Stop tracemalloc
        tracemalloc.stop()
        
        self.results['profiling_data']['memory'] = memory_results
        return memory_results
    
    def profile_bottlenecks(self, chess_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify and analyze performance bottlenecks."""
        print("\n=== Bottleneck Analysis ===")
        
        bottlenecks = []
        
        # 1. CPU bottleneck analysis
        print("\nAnalyzing CPU bottlenecks...")
        cpu_bottleneck = self._analyze_cpu_bottleneck(chess_data)
        if cpu_bottleneck:
            bottlenecks.append(cpu_bottleneck)
        
        # 2. Memory bottleneck analysis
        print("\nAnalyzing memory bottlenecks...")
        memory_bottleneck = self._analyze_memory_bottleneck(chess_data)
        if memory_bottleneck:
            bottlenecks.append(memory_bottleneck)
        
        # 3. I/O bottleneck analysis
        print("\nAnalyzing I/O bottlenecks...")
        io_bottleneck = self._analyze_io_bottleneck(chess_data)
        if io_bottleneck:
            bottlenecks.append(io_bottleneck)
        
        # 4. Synchronization bottleneck analysis
        print("\nAnalyzing synchronization bottlenecks...")
        sync_bottleneck = self._analyze_sync_bottleneck(chess_data)
        if sync_bottleneck:
            bottlenecks.append(sync_bottleneck)
        
        self.results['bottlenecks'] = bottlenecks
        
        # Print summary
        print("\nBottleneck Summary:")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck['type']}: {bottleneck['description']}")
            print(f"    Impact: {bottleneck['impact']}")
            print(f"    Recommendation: {bottleneck['recommendation']}")
        
        return bottlenecks
    
    def _analyze_cpu_bottleneck(self, chess_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze CPU utilization patterns."""
        sample_data = chess_data.sample(min(1000, len(chess_data)), random_state=42)
        
        # Monitor CPU during processing
        cpu_samples = []
        stop_event = threading.Event()
        
        def monitor_cpu():
            while not stop_event.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Process games
        with Pool(processes=cpu_count() - 1) as pool:
            play_games(sample_data, pool)
        
        stop_event.set()
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = np.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        cpu_variance = np.var(cpu_samples)
        
        if avg_cpu < 50:
            return {
                'type': 'CPU Underutilization',
                'description': f'Average CPU usage is only {avg_cpu:.1f}%',
                'impact': 'High',
                'recommendation': 'Increase worker count or optimize chunk distribution',
                'metrics': {
                    'avg_cpu': avg_cpu,
                    'max_cpu': max_cpu,
                    'variance': cpu_variance
                }
            }
        
        return None
    
    def _analyze_memory_bottleneck(self, chess_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze memory usage patterns."""
        sample_data = chess_data.sample(min(1000, len(chess_data)), random_state=42)
        
        # Track memory allocations
        memory_samples = []
        stop_event = threading.Event()
        
        def monitor_memory():
            process = psutil.Process()
            while not stop_event.is_set():
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Process games
        with Pool(processes=cpu_count() - 1) as pool:
            play_games(sample_data, pool)
        
        stop_event.set()
        monitor_thread.join()
        
        # Analyze memory pattern
        if len(memory_samples) > 10:
            memory_growth = memory_samples[-1] - memory_samples[0]
            growth_rate = memory_growth / len(memory_samples)
            
            if growth_rate > 0.5:  # More than 0.5 MB per sample
                return {
                    'type': 'Memory Growth',
                    'description': f'Memory growing at {growth_rate:.2f} MB/sample',
                    'impact': 'Medium',
                    'recommendation': 'Check for memory leaks or excessive caching',
                    'metrics': {
                        'initial_memory': memory_samples[0],
                        'final_memory': memory_samples[-1],
                        'growth_rate': growth_rate
                    }
                }
        
        return None
    
    def _analyze_io_bottleneck(self, chess_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze I/O patterns."""
        # For this chess processing system, I/O is mainly about shared memory access
        sample_data = chess_data.sample(min(100, len(chess_data)), random_state=42)
        
        # Time shared memory operations
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            shared_data = create_shared_data(sample_data)
            t1 = time.perf_counter()
            cleanup_shared_data(shared_data)
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        
        if avg_time > 0.1:  # More than 100ms
            return {
                'type': 'Shared Memory I/O',
                'description': f'Shared memory creation taking {avg_time*1000:.1f}ms',
                'impact': 'Low',
                'recommendation': 'Consider optimizing shared memory structure',
                'metrics': {
                    'avg_time': avg_time,
                    'times': times
                }
            }
        
        return None
    
    def _analyze_sync_bottleneck(self, chess_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze synchronization overhead."""
        # Test with different worker counts
        sample_data = chess_data.sample(min(1000, len(chess_data)), random_state=42)
        
        worker_counts = [1, 2, 4, cpu_count() - 1]
        efficiencies = []
        
        for num_workers in worker_counts:
            with Pool(processes=num_workers) as pool:
                t0 = time.time()
                play_games(sample_data, pool)
                elapsed = time.time() - t0
                
                games_per_second = len(sample_data) / elapsed
                efficiency = games_per_second / num_workers
                efficiencies.append(efficiency)
        
        # Check for decreasing efficiency
        if len(efficiencies) > 2 and efficiencies[-1] < efficiencies[0] * 0.5:
            return {
                'type': 'Synchronization Overhead',
                'description': 'Efficiency decreases significantly with more workers',
                'impact': 'Medium',
                'recommendation': 'Optimize chunk size or reduce inter-process communication',
                'metrics': {
                    'worker_counts': worker_counts,
                    'efficiencies': efficiencies
                }
            }
        
        return None
    
    def _analyze_line_profile(self, profile_output: str):
        """Parse line profiling output for insights."""
        # Extract hot spots from line profile
        lines = profile_output.split('\n')
        hot_spots = []
        
        for line in lines:
            if '%' in line and 'Time' in line:
                parts = line.split()
                if len(parts) > 2:
                    try:
                        time_percent = float(parts[0].strip('%'))
                        if time_percent > 5:  # Lines taking more than 5% of time
                            hot_spots.append({
                                'line': line,
                                'percent': time_percent
                            })
                    except:
                        pass
        
        if hot_spots:
            self.results['profiling_data']['hot_spots'] = hot_spots
    
    def generate_report(self):
        """Generate comprehensive profiling report."""
        print("\n=== Generating Profiling Report ===")
        
        # Save raw results
        with open(os.path.join(self.results_dir, 'raw_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate markdown report
        report_path = os.path.join(self.results_dir, 'performance_report.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Chess Processing Performance Report\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            sys_info = self.results['system_info']
            f.write(f"- Platform: {sys_info['platform']}\n")
            f.write(f"- CPU: {sys_info['cpu']['physical_cores']} physical cores, "
                   f"{sys_info['cpu']['logical_cores']} logical cores\n")
            f.write(f"- Memory: {sys_info['memory']['total'] / 1024 / 1024 / 1024:.1f} GB\n\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            if 'configurations' in self.results:
                best_config = max(self.results['configurations'].items(),
                                key=lambda x: x[1].get('games_per_second', 0))
                f.write(f"**Best Configuration:** {best_config[0]}\n")
                f.write(f"- Performance: {best_config[1]['games_per_second']:.2f} games/s\n")
                f.write(f"- Moves/s: {best_config[1]['moves_per_second']:.2f}\n\n")
            
            # Bottlenecks
            f.write("## Identified Bottlenecks\n\n")
            for bottleneck in self.results.get('bottlenecks', []):
                f.write(f"### {bottleneck['type']}\n")
                f.write(f"- **Description:** {bottleneck['description']}\n")
                f.write(f"- **Impact:** {bottleneck['impact']}\n")
                f.write(f"- **Recommendation:** {bottleneck['recommendation']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self._generate_recommendations()
            for i, rec in enumerate(self.results.get('recommendations', []), 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\nReport saved to: {report_path}")
        return report_path
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        # Check if we're at peak performance
        if 'configurations' in self.results:
            best_perf = max(cfg.get('games_per_second', 0) 
                          for cfg in self.results['configurations'].values())
            
            if best_perf >= 750:  # Near your current 784 games/s
                recommendations.append(
                    "System is performing at near-peak levels (>750 games/s). "
                    "Further optimizations would likely require architectural changes."
                )
        
        # Check for CPU utilization
        for bottleneck in self.results.get('bottlenecks', []):
            if bottleneck['type'] == 'CPU Underutilization':
                recommendations.append(
                    "Consider using more aggressive chunking or increasing worker count "
                    "to better utilize available CPU resources."
                )
        
        # Memory recommendations
        if 'memory' in self.results.get('profiling_data', {}):
            memory_data = self.results['profiling_data']['memory']
            if 'by_sample_size' in memory_data:
                # Check memory scaling
                sizes = list(memory_data['by_sample_size'].keys())
                if len(sizes) > 1:
                    mem_per_game = [
                        memory_data['by_sample_size'][s]['memory_per_game'] 
                        for s in sizes
                    ]
                    if max(mem_per_game) > 0.1:  # More than 0.1 MB per game
                        recommendations.append(
                            "Memory usage per game is relatively high. "
                            "Consider optimizing data structures or using memory mapping."
                        )
        
        # Component timing recommendations
        if 'components' in self.results.get('profiling_data', {}):
            components = self.results['profiling_data']['components']
            
            # Find slowest component
            slowest = max(components.items(), 
                         key=lambda x: x[1].get('mean_time', 0) if x[1] else 0)
            
            if slowest[1]['mean_time'] > 0.001:  # More than 1ms
                recommendations.append(
                    f"Component '{slowest[0]}' is taking {slowest[1]['mean_time']*1000:.2f}ms on average. "
                    f"Consider optimizing this component."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "No significant performance issues detected. "
                "Consider profiling under different workloads or with larger datasets."
            )
        
        self.results['recommendations'] = recommendations


def run_comprehensive_profiling(filepath: str, sample_size: int = None):
    """Run comprehensive profiling on a chess database file."""
    # Load data
    print(f"Loading data from {filepath}...")
    chess_data = pd.read_pickle(filepath, compression='zip')
    
    if sample_size and sample_size < len(chess_data):
        chess_data = chess_data.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    print(f"Loaded {len(chess_data)} games")
    print(f"Average moves per game: {chess_data['PlyCount'].mean():.2f}")
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Run profiling suite
    print("\n" + "="*60)
    print("STARTING COMPREHENSIVE PERFORMANCE PROFILING")
    print("="*60)
    
    # 1. Configuration testing
    worker_counts = [1, 2, 4, 8, psutil.cpu_count(logical=False)]
    worker_counts = [w for w in worker_counts if w <= psutil.cpu_count(logical=False)]
    profiler.profile_configurations(chess_data, worker_counts=worker_counts)
    
    # 2. Component profiling
    profiler.profile_components(chess_data, sample_size=100)
    
    # 3. Line profiling (if available)
    profiler.profile_with_line_profiler(chess_data, sample_size=10)
    
    # 4. Memory profiling
    profiler.profile_memory_detailed(chess_data)
    
    # 5. Bottleneck analysis
    profiler.profile_bottlenecks(chess_data)
    
    # 6. Generate report
    report_path = profiler.generate_report()
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {profiler.results_dir}")
    
    return profiler.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive performance profiling for chess processing")
    parser.add_argument("--filepath", type=str, help="Path to chess database file")
    parser.add_argument("--sample-size", type=int, help="Sample size for profiling")
    parser.add_argument("--quick", action="store_true", help="Run quick profiling (smaller samples)")
    
    args = parser.parse_args()
    
    # Get filepath
    if args.filepath:
        filepath = args.filepath
    else:
        filepath = game_settings.chess_games_filepath_part_1
    
    # Determine sample size
    if args.quick:
        sample_size = 1000
    elif args.sample_size:
        sample_size = args.sample_size
    else:
        sample_size = 10000
    
    # Run profiling
    results = run_comprehensive_profiling(filepath, sample_size)