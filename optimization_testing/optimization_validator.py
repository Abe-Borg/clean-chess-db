# file: optimization_testing/optimization_validator.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
import psutil
import multiprocessing
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from training.game_simulation import play_games
from main.play_games import create_persistent_pool
from utils import game_settings


class OptimizationValidator:
    """Validate optimizations and confirm peak performance."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {},
            'performance_tests': {},
            'stress_tests': {},
            'final_metrics': {}
        }
        
    def validate_all_optimizations(self, chess_data: pd.DataFrame):
        """Run all validation tests."""
        print("\n" + "="*60)
        print("OPTIMIZATION VALIDATION SUITE")
        print("="*60)
        
        # 1. Validate no caching (ensure no false corruption)
        self.validate_no_false_corruption(chess_data)
        
        # 2. Validate performance improvements
        self.validate_performance_targets(chess_data)
        
        # 3. Stress test with different workloads
        self.run_stress_tests(chess_data)
        
        # 4. Validate scalability
        self.validate_scalability(chess_data)
        
        # 5. Final performance measurement
        self.measure_final_performance(chess_data)
        
        # Generate report
        self.generate_validation_report()
        
        return self.results
    
    def validate_no_false_corruption(self, chess_data: pd.DataFrame, sample_size: int = 1000):
        """Validate that the caching bug fix eliminated false corruption."""
        print("\n--- Validating No False Corruption ---")
        
        sample_data = chess_data.sample(min(sample_size, len(chess_data)), random_state=42)
        
        # Process games multiple times to ensure consistency
        corruption_rates = []
        
        for run in range(3):
            print(f"Run {run + 1}/3...")
            
            with Pool(processes=cpu_count() - 1) as pool:
                corrupted_games = play_games(sample_data, pool)
                corruption_rate = len(corrupted_games) / len(sample_data) * 100
                corruption_rates.append(corruption_rate)
                
                print(f"  Corruption rate: {corruption_rate:.2f}%")
                print(f"  Corrupted games: {corrupted_games[:5] if corrupted_games else 'None'}")
        
        # Validate results
        avg_corruption = np.mean(corruption_rates)
        consistency = all(rate == corruption_rates[0] for rate in corruption_rates)
        
        self.results['validations']['false_corruption'] = {
            'corruption_rates': corruption_rates,
            'average_corruption': avg_corruption,
            'consistent_results': consistency,
            'passed': avg_corruption < 5.0 and consistency  # Less than 5% corruption
        }
        
        if self.results['validations']['false_corruption']['passed']:
            print("✓ PASSED: No false corruption detected")
        else:
            print("✗ FAILED: Corruption issues detected")
    
    def validate_performance_targets(self, chess_data: pd.DataFrame):
        """Validate that performance meets targets."""
        print("\n--- Validating Performance Targets ---")
        
        # Define performance targets based on your achievements
        targets = {
            'games_per_second': 700,  # Target: > 700 games/s
            'moves_per_second': 50000,  # Target: > 50k moves/s
            'speedup_vs_baseline': 4.0  # Target: > 4x speedup
        }
        
        # Test with different sample sizes
        sample_sizes = [1000, 5000, 10_000]
        performance_results = {}
        
        for size in sample_sizes:
            if size > len(chess_data):
                continue
            
            print(f"\nTesting with {size} games...")
            sample_data = chess_data.sample(size, random_state=42)
            
            # Warm up
            with Pool(processes=cpu_count() - 1) as pool:
                _ = play_games(sample_data.head(100), pool)
            
            # Actual test
            with Pool(processes=cpu_count() - 1) as pool:
                start_time = time.time()
                corrupted_games = play_games(sample_data, pool)
                elapsed_time = time.time() - start_time
            
            games_per_second = len(sample_data) / elapsed_time
            moves_per_second = sample_data['PlyCount'].sum() / elapsed_time
            
            performance_results[size] = {
                'games_per_second': games_per_second,
                'moves_per_second': moves_per_second,
                'elapsed_time': elapsed_time,
                'corrupted_count': len(corrupted_games)
            }
            
            print(f"  Performance: {games_per_second:.2f} games/s, {moves_per_second:.0f} moves/s")
        
        # Check if targets are met
        avg_games_per_sec = np.mean([r['games_per_second'] for r in performance_results.values()])
        avg_moves_per_sec = np.mean([r['moves_per_second'] for r in performance_results.values()])
        
        # Calculate speedup (baseline was 159 games/s)
        baseline_performance = 159
        speedup = avg_games_per_sec / baseline_performance
        
        passed = (avg_games_per_sec >= targets['games_per_second'] and
                 avg_moves_per_sec >= targets['moves_per_second'] and
                 speedup >= targets['speedup_vs_baseline'])
        
        self.results['validations']['performance_targets'] = {
            'targets': targets,
            'results': performance_results,
            'average_games_per_second': avg_games_per_sec,
            'average_moves_per_second': avg_moves_per_sec,
            'speedup': speedup,
            'passed': passed
        }
        
        print(f"\nPerformance Summary:")
        print(f"  Average: {avg_games_per_sec:.2f} games/s (target: {targets['games_per_second']})")
        print(f"  Speedup: {speedup:.2f}x (target: {targets['speedup_vs_baseline']}x)")
        
        if passed:
            print("✓ PASSED: Performance targets met")
        else:
            print("✗ FAILED: Performance below targets")
    
    def run_stress_tests(self, chess_data: pd.DataFrame):
        """Run stress tests with extreme workloads."""
        print("\n--- Running Stress Tests ---")
        
        stress_scenarios = {
            'high_concurrency': {
                'workers': cpu_count(),  # Use all cores
                'sample_size': min(50_000, len(chess_data)),
                'description': 'Maximum worker concurrency'
            },
            'memory_pressure': {
                'workers': cpu_count() - 1,
                'sample_size': min(80_000, len(chess_data)),
                'description': 'Large dataset to test memory handling'
            },
            'long_games': {
                'workers': cpu_count() - 1,
                'sample_size': 1000,
                'filter': lambda df: df.nlargest(1000, 'PlyCount'),
                'description': 'Games with high move counts'
            },
            'rapid_processing': {
                'workers': cpu_count() - 1,
                'sample_size': 10000,
                'iterations': 5,
                'description': 'Repeated rapid processing'
            }
        }
        
        stress_results = {}
        
        for scenario_name, config in stress_scenarios.items():
            print(f"\n{scenario_name}: {config['description']}")
            
            # Prepare data
            if 'filter' in config:
                test_data = config['filter'](chess_data).head(config['sample_size'])
            else:
                test_data = chess_data.sample(
                    min(config['sample_size'], len(chess_data)), 
                    random_state=42
                )
            
            # Monitor resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run stress test
            iterations = config.get('iterations', 1)
            iteration_times = []
            
            try:
                for i in range(iterations):
                    with Pool(processes=config['workers']) as pool:
                        start_time = time.time()
                        corrupted_games = play_games(test_data, pool)
                        elapsed = time.time() - start_time
                        iteration_times.append(elapsed)
                
                # Calculate metrics
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = final_memory - initial_memory
                
                avg_time = np.mean(iteration_times)
                avg_games_per_sec = len(test_data) / avg_time
                
                stress_results[scenario_name] = {
                    'success': True,
                    'iterations': iterations,
                    'avg_time': avg_time,
                    'games_per_second': avg_games_per_sec,
                    'memory_growth_mb': memory_growth,
                    'worker_count': config['workers'],
                    'sample_size': len(test_data)
                }
                
                print(f"  ✓ Completed: {avg_games_per_sec:.2f} games/s, "
                      f"Memory growth: {memory_growth:.2f} MB")
                
            except Exception as e:
                stress_results[scenario_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ Failed: {e}")
        
        self.results['stress_tests'] = stress_results
    
    def validate_scalability(self, chess_data: pd.DataFrame):
        """Validate system scalability with different configurations."""
        print("\n--- Validating Scalability ---")
        
        # Test scaling dimensions
        worker_counts = [1, 2, 4, 8, cpu_count() - 1, cpu_count()]
        worker_counts = [w for w in worker_counts if w > 0 and w <= cpu_count()]
        
        data_sizes = [100, 500, 1000, 5000, 10000, 50000]
        data_sizes = [s for s in data_sizes if s <= len(chess_data)]
        
        scalability_results = {
            'worker_scaling': {},
            'data_scaling': {}
        }
        
        # 1. Worker scaling (fixed data size)
        print("\nTesting worker scalability...")
        fixed_size = min(10000, len(chess_data))
        sample_data = chess_data.sample(fixed_size, random_state=42)
        
        for num_workers in worker_counts:
            with Pool(processes=num_workers) as pool:
                start_time = time.time()
                play_games(sample_data, pool)
                elapsed = time.time() - start_time
            
            games_per_sec = fixed_size / elapsed
            efficiency = games_per_sec / num_workers
            
            scalability_results['worker_scaling'][num_workers] = {
                'games_per_second': games_per_sec,
                'efficiency': efficiency,
                'elapsed_time': elapsed
            }
            
            print(f"  {num_workers} workers: {games_per_sec:.2f} games/s "
                  f"(efficiency: {efficiency:.2f})")
        
        # 2. Data scaling (fixed workers)
        print("\nTesting data scalability...")
        fixed_workers = max(1, cpu_count() - 1)
        
        for data_size in data_sizes:
            sample_data = chess_data.sample(data_size, random_state=42)
            
            with Pool(processes=fixed_workers) as pool:
                start_time = time.time()
                play_games(sample_data, pool)
                elapsed = time.time() - start_time
            
            games_per_sec = data_size / elapsed
            
            scalability_results['data_scaling'][data_size] = {
                'games_per_second': games_per_sec,
                'elapsed_time': elapsed
            }
            
            print(f"  {data_size} games: {games_per_sec:.2f} games/s")
        
        self.results['validations']['scalability'] = scalability_results
        
        # Generate scalability plots
        self._plot_scalability(scalability_results)
    
    def measure_final_performance(self, chess_data: pd.DataFrame):
        """Measure final optimized performance with production settings."""
        print("\n--- Final Performance Measurement ---")
        
        # Use full dataset or large sample
        test_size = min(len(chess_data), 80_000)
        test_data = chess_data.sample(test_size, random_state=42)
        
        print(f"Testing with {test_size} games...")
        print(f"Average game length: {test_data['PlyCount'].mean():.2f} moves")
        
        # Create persistent pool with optimal settings
        optimal_workers = max(1, cpu_count() - 1)
        pool = create_persistent_pool(optimal_workers)
        
        # Warm up
        print("Warming up...")
        _ = play_games(test_data.head(1000), pool)
        
        # Clear any caches
        import gc
        gc.collect()
        
        # Measure performance multiple times
        measurements = []
        
        for run in range(3):
            print(f"\nRun {run + 1}/3...")
            
            # Monitor resources
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process games
            start_time = time.time()
            corrupted_games = play_games(test_data, pool)
            elapsed = time.time() - start_time
            
            # Get resource usage
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            games_per_sec = len(test_data) / elapsed
            moves_per_sec = test_data['PlyCount'].sum() / elapsed
            
            measurement = {
                'run': run + 1,
                'elapsed_time': elapsed,
                'games_per_second': games_per_sec,
                'moves_per_second': moves_per_sec,
                'corrupted_count': len(corrupted_games),
                'corruption_rate': len(corrupted_games) / len(test_data) * 100,
                'cpu_usage': cpu_after - cpu_before,
                'memory_growth_mb': mem_after - mem_before
            }
            
            measurements.append(measurement)
            
            print(f"  Performance: {games_per_sec:.2f} games/s, {moves_per_sec:.0f} moves/s")
            print(f"  Corruption rate: {measurement['corruption_rate']:.2f}%")
            print(f"  Memory growth: {measurement['memory_growth_mb']:.2f} MB")
        
        # Clean up pool
        pool.close()
        pool.join()
        
        # Calculate final metrics
        avg_games_per_sec = np.mean([m['games_per_second'] for m in measurements])
        std_games_per_sec = np.std([m['games_per_second'] for m in measurements])
        avg_moves_per_sec = np.mean([m['moves_per_second'] for m in measurements])
        avg_corruption_rate = np.mean([m['corruption_rate'] for m in measurements])
        
        self.results['final_metrics'] = {
            'test_size': test_size,
            'worker_count': optimal_workers,
            'measurements': measurements,
            'average_games_per_second': avg_games_per_sec,
            'std_games_per_second': std_games_per_sec,
            'average_moves_per_second': avg_moves_per_sec,
            'average_corruption_rate': avg_corruption_rate,
            'speedup_vs_baseline': avg_games_per_sec / 159  # Original baseline
        }
        
        print(f"\nFINAL PERFORMANCE METRICS:")
        print(f"  Games/second: {avg_games_per_sec:.2f} ± {std_games_per_sec:.2f}")
        print(f"  Moves/second: {avg_moves_per_sec:.0f}")
        print(f"  Speedup: {avg_games_per_sec / 159:.2f}x vs baseline")
        print(f"  Corruption rate: {avg_corruption_rate:.2f}%")
    
    def _plot_scalability(self, scalability_results: Dict):
        """Create scalability visualization plots."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('seaborn-darkgrid')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Worker scaling plot
        worker_data = scalability_results['worker_scaling']
        workers = sorted(worker_data.keys())
        games_per_sec = [worker_data[w]['games_per_second'] for w in workers]
        efficiency = [worker_data[w]['efficiency'] for w in workers]
        
        # Plot games/sec
        ax1.plot(workers, games_per_sec, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Games/second')
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Games per Second', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Plot efficiency on secondary axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(workers, efficiency, 's--', color='red', linewidth=2, 
                     markersize=8, label='Efficiency')
        ax1_twin.set_ylabel('Efficiency (games/sec/worker)', color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Worker Scalability')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Data scaling plot
        data_scaling = scalability_results['data_scaling']
        data_sizes = sorted(data_scaling.keys())
        perf = [data_scaling[s]['games_per_second'] for s in data_sizes]
        
        ax2.plot(data_sizes, perf, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Dataset Size (games)')
        ax2.set_ylabel('Games per Second')
        ax2.set_title('Data Scalability')
        ax2.grid(True, alpha=0.3)
        
        # Add ideal scaling line
        if len(data_sizes) > 1:
            # Ideal would be constant performance
            ax2.axhline(y=perf[0], color='gray', linestyle='--', 
                       label='Ideal (constant)')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
        print("\nScalability plots saved to scalability_analysis.png")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        report_path = f"optimization_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n" + "="*60)
        print("VALIDATION REPORT SUMMARY")
        print("="*60)
        
        # Check all validations
        all_passed = True
        
        if 'false_corruption' in self.results['validations']:
            passed = self.results['validations']['false_corruption']['passed']
            all_passed &= passed
            print(f"False Corruption Fix: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        if 'performance_targets' in self.results['validations']:
            passed = self.results['validations']['performance_targets']['passed']
            all_passed &= passed
            print(f"Performance Targets: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        # Stress test summary
        if self.results['stress_tests']:
            stress_passed = all(test.get('success', False) 
                              for test in self.results['stress_tests'].values())
            all_passed &= stress_passed
            print(f"Stress Tests: {'✓ PASSED' if stress_passed else '✗ FAILED'}")
        
        # Final metrics
        if self.results['final_metrics']:
            final_perf = self.results['final_metrics']['average_games_per_second']
            speedup = self.results['final_metrics']['speedup_vs_baseline']
            
            print(f"\nFinal Performance: {final_perf:.2f} games/s ({speedup:.2f}x speedup)")
            
            if final_perf >= 750:
                print("✓ EXCELLENT: Performance exceeds optimization targets!")
            elif final_perf >= 600:
                print("✓ GOOD: Performance meets optimization targets")
            else:
                print("✗ BELOW TARGET: Performance needs improvement")
                all_passed = False
        
        print(f"\nOverall Validation: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print(f"\nDetailed report saved to: {report_path}")
        
        return report_path


def run_optimization_validation(filepath: str, quick: bool = False):
    """Run the optimization validation suite."""
    # Load data
    print(f"Loading data from {filepath}...")
    chess_data = pd.read_pickle(filepath, compression='zip')
    print(f"Loaded {len(chess_data)} games")
    
    if quick:
        # Use smaller sample for quick validation
        chess_data = chess_data.sample(min(10000, len(chess_data)), random_state=42)
        print(f"Quick mode: Using {len(chess_data)} games")
    
    # Create validator and run tests
    validator = OptimizationValidator()
    results = validator.validate_all_optimizations(chess_data)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate chess processing optimizations")
    parser.add_argument("--filepath", type=str, help="Path to chess database file")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    
    args = parser.parse_args()
    
    # Get filepath
    if args.filepath:
        filepath = args.filepath
    else:
        filepath = game_settings.chess_games_filepath_part_1
    
    # Run validation
    results = run_optimization_validation(filepath, args.quick)