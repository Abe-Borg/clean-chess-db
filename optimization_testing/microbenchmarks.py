# file: optimization_testing/microbenchmarks.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import timeit
import pandas as pd
import numpy as np
import chess
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import gc

from environment.Environ import Environ
from agents.Agent import Agent
from training.game_simulation import (
    play_one_game, create_shared_data, cleanup_shared_data,
    AdaptiveChunker, worker_process_games
)
from utils import game_settings


class MicroBenchmark:
    """Micro-benchmarking suite for chess processing components."""
    
    def __init__(self, chess_data: pd.DataFrame):
        self.chess_data = chess_data
        self.results = defaultdict(dict)
        
    def benchmark_all(self):
        """Run all micro-benchmarks."""
        print("\n" + "="*60)
        print("CHESS PROCESSING MICRO-BENCHMARKS")
        print("="*60)
        
        # Core operations
        self.benchmark_board_operations()
        self.benchmark_move_parsing()
        self.benchmark_state_queries()
        
        # Data structures
        self.benchmark_shared_memory()
        self.benchmark_chunking()
        
        # Game processing
        self.benchmark_game_processing()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def benchmark_board_operations(self, iterations: int = 10000):
        """Benchmark core chess board operations."""
        print("\n--- Benchmarking Board Operations ---")
        
        # Setup
        environ = Environ()
        test_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b8c6"),
            chess.Move.from_uci("f1c4"),
            chess.Move.from_uci("g8f6")
        ]
        
        # 1. Board reset
        def bench_reset():
            environ.reset_environ()
        
        reset_time = timeit.timeit(bench_reset, number=iterations)
        self.results['board_operations']['reset'] = {
            'total_time': reset_time,
            'per_operation': reset_time / iterations,
            'operations_per_second': iterations / reset_time
        }
        
        # 2. Move push (object)
        environ.reset_environ()
        def bench_push_object():
            for move in test_moves:
                if move in environ.board.legal_moves:
                    environ.push_move_object(move)
            environ.reset_environ()
        
        push_time = timeit.timeit(bench_push_object, number=iterations // 10)
        self.results['board_operations']['push_move_object'] = {
            'total_time': push_time,
            'per_operation': push_time / (iterations // 10) / len(test_moves),
            'operations_per_second': (iterations // 10 * len(test_moves)) / push_time
        }
        
        # 3. Legal move generation
        environ.reset_environ()
        def bench_legal_moves():
            list(environ.board.legal_moves)
        
        legal_moves_time = timeit.timeit(bench_legal_moves, number=iterations)
        self.results['board_operations']['legal_moves'] = {
            'total_time': legal_moves_time,
            'per_operation': legal_moves_time / iterations,
            'operations_per_second': iterations / legal_moves_time
        }
        
        # 4. Game over check
        def bench_game_over():
            environ.board.is_game_over()
        
        game_over_time = timeit.timeit(bench_game_over, number=iterations)
        self.results['board_operations']['is_game_over'] = {
            'total_time': game_over_time,
            'per_operation': game_over_time / iterations,
            'operations_per_second': iterations / game_over_time
        }
        
        # Print results
        print(f"Board reset: {self.results['board_operations']['reset']['per_operation']*1e6:.2f} μs")
        print(f"Push move object: {self.results['board_operations']['push_move_object']['per_operation']*1e6:.2f} μs")
        print(f"Legal moves generation: {self.results['board_operations']['legal_moves']['per_operation']*1e6:.2f} μs")
        print(f"Game over check: {self.results['board_operations']['is_game_over']['per_operation']*1e6:.2f} μs")
    
    def benchmark_move_parsing(self, iterations: int = 10000):
        """Benchmark move parsing operations."""
        print("\n--- Benchmarking Move Parsing ---")
        
        environ = Environ()
        test_sans = ["e4", "Nf3", "Bc4", "O-O", "d4", "Bxf7+", "Qh5", "Re1"]
        test_ucis = ["e2e4", "g1f3", "f1c4", "e1g1", "d2d4", "c4f7", "d1h5", "f1e1"]
        
        # 1. SAN to Move object
        def bench_san_parsing():
            environ.reset_environ()
            for san in test_sans[:3]:  # Use first few moves
                try:
                    environ.convert_san_to_move_object(san)
                except:
                    pass
        
        san_time = timeit.timeit(bench_san_parsing, number=iterations // 10)
        self.results['move_parsing']['san_to_move'] = {
            'total_time': san_time,
            'per_operation': san_time / (iterations // 10) / 3,
            'operations_per_second': (iterations // 10 * 3) / san_time
        }
        
        # 2. UCI parsing
        def bench_uci_parsing():
            for uci in test_ucis:
                chess.Move.from_uci(uci)
        
        uci_time = timeit.timeit(bench_uci_parsing, number=iterations)
        self.results['move_parsing']['uci_to_move'] = {
            'total_time': uci_time,
            'per_operation': uci_time / iterations / len(test_ucis),
            'operations_per_second': (iterations * len(test_ucis)) / uci_time
        }
        
        # 3. Move validation
        environ.reset_environ()
        move_e4 = chess.Move.from_uci("e2e4")
        legal_moves = list(environ.board.legal_moves)
        
        def bench_move_validation():
            move_e4 in legal_moves
        
        validation_time = timeit.timeit(bench_move_validation, number=iterations * 10)
        self.results['move_parsing']['move_validation'] = {
            'total_time': validation_time,
            'per_operation': validation_time / (iterations * 10),
            'operations_per_second': (iterations * 10) / validation_time
        }
        
        print(f"SAN to Move: {self.results['move_parsing']['san_to_move']['per_operation']*1e6:.2f} μs")
        print(f"UCI to Move: {self.results['move_parsing']['uci_to_move']['per_operation']*1e6:.2f} μs")
        print(f"Move validation: {self.results['move_parsing']['move_validation']['per_operation']*1e6:.2f} μs")
    
    def benchmark_state_queries(self, iterations: int = 10000):
        """Benchmark state query operations."""
        print("\n--- Benchmarking State Queries ---")
        
        environ = Environ()
        
        # 1. Get current state and legal moves
        def bench_state_and_moves():
            environ.get_curr_state_and_legal_moves()
        
        state_time = timeit.timeit(bench_state_and_moves, number=iterations)
        self.results['state_queries']['get_state_and_moves'] = {
            'total_time': state_time,
            'per_operation': state_time / iterations,
            'operations_per_second': iterations / state_time
        }
        
        # 2. Get current turn
        def bench_get_turn():
            environ.get_curr_turn()
        
        turn_time = timeit.timeit(bench_get_turn, number=iterations * 10)
        self.results['state_queries']['get_turn'] = {
            'total_time': turn_time,
            'per_operation': turn_time / (iterations * 10),
            'operations_per_second': (iterations * 10) / turn_time
        }
        
        # 3. Update state
        def bench_update_state():
            environ.update_curr_state()
            environ.turn_index = 0  # Reset for next iteration
        
        update_time = timeit.timeit(bench_update_state, number=iterations)
        self.results['state_queries']['update_state'] = {
            'total_time': update_time,
            'per_operation': update_time / iterations,
            'operations_per_second': iterations / update_time
        }
        
        print(f"Get state and moves: {self.results['state_queries']['get_state_and_moves']['per_operation']*1e6:.2f} μs")
        print(f"Get turn: {self.results['state_queries']['get_turn']['per_operation']*1e6:.2f} μs")
        print(f"Update state: {self.results['state_queries']['update_state']['per_operation']*1e6:.2f} μs")
    
    def benchmark_shared_memory(self):
        """Benchmark shared memory operations."""
        print("\n--- Benchmarking Shared Memory ---")
        
        # Test with different sample sizes
        sample_sizes = [10, 100, 1000]
        
        for size in sample_sizes:
            if size > len(self.chess_data):
                continue
            
            sample_data = self.chess_data.sample(size, random_state=42)
            
            # Time creation
            times = []
            for _ in range(5):
                gc.collect()
                
                t0 = time.perf_counter()
                shared_data = create_shared_data(sample_data)
                t1 = time.perf_counter()
                
                times.append(t1 - t0)
                
                # Cleanup
                cleanup_shared_data(shared_data)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            self.results['shared_memory'][f'create_{size}_games'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'time_per_game': avg_time / size,
                'games_per_second': size / avg_time
            }
            
            print(f"Create shared memory ({size} games): {avg_time*1000:.2f}ms "
                  f"({avg_time/size*1000:.3f}ms per game)")
    
    def benchmark_chunking(self):
        """Benchmark chunking strategies."""
        print("\n--- Benchmarking Chunking ---")
        
        chunker = AdaptiveChunker()
        
        # Test with different game counts and worker counts
        game_counts = [100, 1000, 10000]
        worker_counts = [4, 8, 16]
        
        for game_count in game_counts:
            if game_count > len(self.chess_data):
                continue
            
            sample_data = self.chess_data.sample(game_count, random_state=42)
            game_indices = list(sample_data.index)
            
            for num_workers in worker_counts:
                # Time chunking
                times = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    chunks = chunker.create_balanced_chunks(
                        game_indices, sample_data, num_workers
                    )
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                
                avg_time = np.mean(times)
                
                # Analyze chunk balance
                chunk_sizes = [len(chunk) for chunk in chunks]
                imbalance = (max(chunk_sizes) - min(chunk_sizes)) / np.mean(chunk_sizes) * 100
                
                key = f'{game_count}_games_{num_workers}_workers'
                self.results['chunking'][key] = {
                    'avg_time': avg_time,
                    'num_chunks': len(chunks),
                    'avg_chunk_size': np.mean(chunk_sizes),
                    'imbalance_percent': imbalance
                }
                
                print(f"Chunking {game_count} games for {num_workers} workers: "
                      f"{avg_time*1000:.2f}ms, {len(chunks)} chunks, "
                      f"{imbalance:.1f}% imbalance")
    
    def benchmark_game_processing(self):
        """Benchmark full game processing."""
        print("\n--- Benchmarking Game Processing ---")
        
        # Select games of different lengths
        game_lengths = [40, 60, 80, 100, 150]
        
        for target_length in game_lengths:
            # Find games close to target length
            length_diff = abs(self.chess_data['PlyCount'] - target_length)
            closest_games = self.chess_data.nsmallest(5, length_diff)
            
            if len(closest_games) == 0:
                continue
            
            # Benchmark each game
            times = []
            for game_id, row in closest_games.iterrows():
                ply_count = int(row['PlyCount'])
                moves = {col: row[col] for col in self.chess_data.columns if col != 'PlyCount'}
                
                # Create agents and environment
                w_agent = Agent('W')
                b_agent = Agent('B')
                environ = Environ()
                
                # Time game processing
                t0 = time.perf_counter()
                result = play_one_game(game_id, ply_count, moves, 
                                     w_agent, b_agent, environ)
                t1 = time.perf_counter()
                
                times.append(t1 - t0)
            
            avg_time = np.mean(times)
            avg_length = closest_games['PlyCount'].mean()
            
            self.results['game_processing'][f'length_{target_length}'] = {
                'avg_time': avg_time,
                'avg_length': avg_length,
                'moves_per_second': avg_length / avg_time,
                'sample_size': len(times)
            }
            
            print(f"Games ~{target_length} moves (avg {avg_length:.1f}): "
                  f"{avg_time*1000:.2f}ms, {avg_length/avg_time:.0f} moves/s")
    
    def generate_report(self):
        """Generate micro-benchmark report."""
        report_path = "microbenchmark_report.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        results_dict = {k: dict(v) for k, v in self.results.items()}
        
        with open(report_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nMicro-benchmark report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("MICRO-BENCHMARK SUMMARY")
        print("="*60)
        
        # Find fastest operations
        print("\nFastest operations:")
        all_ops = []
        for category, ops in self.results.items():
            for op_name, stats in ops.items():
                if 'per_operation' in stats:
                    all_ops.append((f"{category}.{op_name}", stats['per_operation']))
        
        all_ops.sort(key=lambda x: x[1])
        for op, time in all_ops[:5]:
            print(f"  {op}: {time*1e6:.2f} μs")
        
        # Find slowest operations
        print("\nSlowest operations:")
        for op, time in all_ops[-5:]:
            print(f"  {op}: {time*1e6:.2f} μs")
        
        # Key performance indicators
        print("\nKey Performance Indicators:")
        
        if 'board_operations' in self.results:
            moves_per_sec = self.results['board_operations']['push_move_object']['operations_per_second']
            print(f"  Move pushing: {moves_per_sec:.0f} moves/s")
        
        if 'game_processing' in self.results:
            avg_game_speed = np.mean([
                stats['moves_per_second'] 
                for stats in self.results['game_processing'].values()
            ])
            print(f"  Average game processing: {avg_game_speed:.0f} moves/s")
        
        return results_dict


def run_microbenchmarks(filepath: str, sample_size: int = 10000):
    """Run micro-benchmarks on chess database."""
    # Load data
    print(f"Loading data from {filepath}...")
    chess_data = pd.read_pickle(filepath, compression='zip')
    
    if sample_size and sample_size < len(chess_data):
        chess_data = chess_data.sample(sample_size, random_state=42)
    
    print(f"Using {len(chess_data)} games for benchmarking")
    
    # Create and run benchmarks
    benchmark = MicroBenchmark(chess_data)
    results = benchmark.benchmark_all()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Micro-benchmarks for chess processing")
    parser.add_argument("--filepath", type=str, help="Path to chess database file")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for benchmarks")
    
    args = parser.parse_args()
    
    # Get filepath
    if args.filepath:
        filepath = args.filepath
    else:
        filepath = game_settings.chess_games_filepath_part_1
    
    # Run benchmarks
    results = run_microbenchmarks(filepath, args.sample_size)