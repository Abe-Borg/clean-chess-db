# file: optimization_testing/run_profiling.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime
import json
import pandas as pd

# Import all profiling modules
from performance_profiler import run_comprehensive_profiling
from microbenchmarks import run_microbenchmarks
from optimization_validator import run_optimization_validation
from profiling_dashboard import create_dashboard_from_results, create_summary_report

from utils import game_settings


def run_full_profiling_suite(filepath: str, mode: str = 'comprehensive', 
                           sample_size: int = None, output_dir: str = None):
    """
    Run the complete profiling suite on a chess database.
    
    Args:
        filepath: Path to chess database file
        mode: Profiling mode ('quick', 'standard', 'comprehensive')
        sample_size: Optional sample size override
        output_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("CHESS PROCESSING PERFORMANCE PROFILING SUITE")
    print("="*80)
    print(f"Mode: {mode.upper()}")
    print(f"Database: {filepath}")
    
    # Create output directory
    if output_dir is None:
        output_dir = f"profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load database info
    print("\nLoading database...")
    chess_data = pd.read_pickle(filepath, compression='zip')
    print(f"Total games: {len(chess_data):,}")
    print(f"Average game length: {chess_data['PlyCount'].mean():.2f} moves")
    
    # Determine sample sizes based on mode
    if mode == 'quick':
        profiling_sample = min(1000, len(chess_data))
        benchmark_sample = min(5000, len(chess_data))
        validation_quick = True
    elif mode == 'standard':
        profiling_sample = min(10000, len(chess_data))
        benchmark_sample = min(10000, len(chess_data))
        validation_quick = False
    else:  # comprehensive
        profiling_sample = min(50000, len(chess_data))
        benchmark_sample = min(20000, len(chess_data))
        validation_quick = False
    
    # Override with custom sample size if provided
    if sample_size:
        profiling_sample = min(sample_size, len(chess_data))
        benchmark_sample = min(sample_size, len(chess_data))
    
    all_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'database': filepath,
            'total_games': len(chess_data),
            'profiling_sample': profiling_sample,
            'benchmark_sample': benchmark_sample
        }
    }
    
    # Step 1: Micro-benchmarks
    print(f"\n{'='*60}")
    print("STEP 1: MICRO-BENCHMARKS")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        benchmark_results = run_microbenchmarks(filepath, benchmark_sample)
        benchmark_time = time.time() - start_time
        
        all_results['microbenchmarks'] = benchmark_results
        all_results['meta']['benchmark_time'] = benchmark_time
        
        print(f"\nMicro-benchmarks completed in {benchmark_time:.2f}s")
        
        # Save intermediate results
        with open(os.path.join(output_dir, 'microbenchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
            
    except Exception as e:
        print(f"Error in micro-benchmarks: {e}")
        all_results['microbenchmarks'] = {'error': str(e)}
    
    # Step 2: Comprehensive profiling
    if mode != 'quick':
        print(f"\n{'='*60}")
        print("STEP 2: COMPREHENSIVE PROFILING")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            profiling_results = run_comprehensive_profiling(filepath, profiling_sample)
            profiling_time = time.time() - start_time
            
            all_results['comprehensive_profiling'] = profiling_results
            all_results['meta']['profiling_time'] = profiling_time
            
            print(f"\nComprehensive profiling completed in {profiling_time:.2f}s")
            
            # Save profiling results
            with open(os.path.join(output_dir, 'profiling_results.json'), 'w') as f:
                json.dump(profiling_results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error in comprehensive profiling: {e}")
            all_results['comprehensive_profiling'] = {'error': str(e)}
    
    # Step 3: Optimization validation
    print(f"\n{'='*60}")
    print("STEP 3: OPTIMIZATION VALIDATION")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        validation_results = run_optimization_validation(filepath, validation_quick)
        validation_time = time.time() - start_time
        
        all_results['validation'] = validation_results
        all_results['meta']['validation_time'] = validation_time
        
        print(f"\nValidation completed in {validation_time:.2f}s")
        
        # Save validation results
        with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
            
    except Exception as e:
        print(f"Error in validation: {e}")
        all_results['validation'] = {'error': str(e)}
    
    # Step 4: Generate reports and visualizations
    print(f"\n{'='*60}")
    print("STEP 4: GENERATING REPORTS")
    print(f"{'='*60}")
    
    # Save combined results
    combined_results_path = os.path.join(output_dir, 'all_results.json')
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate dashboard
    try:
        # Prepare data for dashboard
        dashboard_data = {}
        
        if 'comprehensive_profiling' in all_results:
            dashboard_data.update(all_results['comprehensive_profiling'])
        
        if 'validation' in all_results:
            dashboard_data.update(all_results['validation'])
        
        # Create dashboard
        dashboard_path = os.path.join(output_dir, 'profiling_dashboard.html')
        create_dashboard_from_results(combined_results_path, dashboard_path)
        
        # Create summary report
        summary_path = os.path.join(output_dir, 'summary_report.html')
        create_summary_report(dashboard_data, summary_path)
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("PROFILING COMPLETE")
    print(f"{'='*80}")
    
    total_time = sum([
        all_results['meta'].get('benchmark_time', 0),
        all_results['meta'].get('profiling_time', 0),
        all_results['meta'].get('validation_time', 0)
    ])
    
    print(f"\nTotal execution time: {total_time:.2f}s")
    print(f"Results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  - Micro-benchmarks: microbenchmark_results.json")
    print(f"  - Profiling results: profiling_results.json")
    print(f"  - Validation results: validation_results.json")
    print(f"  - Combined results: all_results.json")
    print(f"  - Dashboard: profiling_dashboard.html")
    print(f"  - Summary: summary_report.html")
    
    # Print key performance metrics
    if 'validation' in all_results and 'final_metrics' in all_results['validation']:
        metrics = all_results['validation']['final_metrics']
        print(f"\nFINAL PERFORMANCE:")
        print(f"  Games/second: {metrics.get('average_games_per_second', 0):.2f}")
        print(f"  Moves/second: {metrics.get('average_moves_per_second', 0):.0f}")
        print(f"  Speedup vs baseline: {metrics.get('speedup_vs_baseline', 0):.2f}x")
        
        # Performance assessment
        perf = metrics.get('average_games_per_second', 0)
        if perf >= 750:
            print("\n✓ EXCELLENT: Performance exceeds optimization targets!")
            print("  Your system is running at peak efficiency.")
        elif perf >= 600:
            print("\n✓ GOOD: Performance meets optimization targets")
            print("  Minor optimizations might still be possible.")
        else:
            print("\n✗ BELOW TARGET: Performance needs improvement")
            print("  Review the bottleneck analysis and recommendations.")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive performance profiling for chess processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick profiling (< 5 minutes)
  python run_profiling.py --mode quick
  
  # Standard profiling (10-20 minutes)
  python run_profiling.py --mode standard
  
  # Comprehensive profiling (30+ minutes)
  python run_profiling.py --mode comprehensive
  
  # Custom sample size
  python run_profiling.py --sample-size 5000
  
  # Specific database file
  python run_profiling.py --filepath /path/to/chess_games.pkl
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['quick', 'standard', 'comprehensive'],
                       default='standard',
                       help='Profiling mode (default: standard)')
    
    parser.add_argument('--filepath', 
                       type=str,
                       help='Path to chess database file')
    
    parser.add_argument('--sample-size', 
                       type=int,
                       help='Override sample size for profiling')
    
    parser.add_argument('--output-dir',
                       type=str,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get filepath
    if args.filepath:
        filepath = args.filepath
    else:
        filepath = game_settings.chess_games_filepath_part_1
    
    # Run profiling
    results = run_full_profiling_suite(
        filepath=filepath,
        mode=args.mode,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )