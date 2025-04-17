# file: optimization_testing/detailed_profiling.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import pandas as pd
import time
import platform
import psutil
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from agents.Agent import Agent
from environment.Environ import Environ
from training.game_simulation import handle_agent_turn, play_one_game

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

def profile_single_game(game_id, chess_data):
    """Profile the processing of a single game to see detailed function performance."""
    print(f"\nProfiling single game: {game_id}")
    
    # Extract the game data
    row = chess_data.loc[game_id]
    ply_count = int(row['PlyCount'])
    
    # Extract moves from row
    game_moves = {}
    for col in chess_data.columns:
        if col != 'PlyCount':
            game_moves[col] = row[col]
    
    # Create the agent and environment objects
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    # Store in game data dictionary
    game_data = {
        'PlyCount': ply_count,
        'moves': game_moves
    }
    
    # Profile the game processing
    pr = cProfile.Profile()
    pr.enable()
    
    result = play_one_game(game_id, game_data, w_agent, b_agent, environ)
    
    pr.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\nTop 20 functions by cumulative time:")
    print(s.getvalue())
    
    # Print stats for key functions
    print("\nDetailed stats for key functions:")
    stats = pstats.Stats(pr)
    for func_pattern in ['handle_agent_turn', 'get_legal_moves', 'get_curr_state', 'push_san', 'choose_action']:
        print(f"\n{func_pattern}:")
        stats.sort_stats('cumulative').print_stats(func_pattern, 3)
    
    return result

def profile_multiple_games(chess_data, num_games=5):
    """Profile multiple individual games to get a representative sample."""
    # Use games with different PlyCount values for better representation
    bins = pd.qcut(chess_data['PlyCount'], num_games, duplicates='drop')
    sample_games = chess_data.groupby(bins).apply(lambda x: x.iloc[0]).index.get_level_values(1)
    
    results = []
    for game_id in sample_games:
        result = profile_single_game(game_id, chess_data)
        results.append((game_id, result))
        
    return results

def profile_legal_moves_cache(chess_data, num_games=10):
    """Profile the effectiveness of the legal moves cache."""
    print("\nProfiling legal moves cache...")
    
    # Create environment
    environ = Environ()
    
    # Reset cache statistics
    Environ._global_position_cache = {}
    Environ._global_cache_hits = 0
    Environ._global_cache_misses = 0
    
    # Process games sequentially to build up cache
    sample_games = chess_data.iloc[:num_games].index
    
    move_counts = []
    cache_hits = []
    cache_misses = []
    
    for i, game_id in enumerate(sample_games):
        # Extract the game data
        row = chess_data.loc[game_id]
        ply_count = int(row['PlyCount'])
        
        # Extract moves from row
        game_moves = {}
        for col in chess_data.columns:
            if col != 'PlyCount':
                game_moves[col] = row[col]
        
        # Create agents
        w_agent = Agent('W')
        b_agent = Agent('B')
        
        # Process the game
        start_time = time.time()
        play_one_game(game_id, {'PlyCount': ply_count, 'moves': game_moves}, w_agent, b_agent, environ)
        end_time = time.time()
        
        # Record statistics
        move_counts.append(ply_count)
        cache_hits.append(Environ._global_cache_hits)
        cache_misses.append(Environ._global_cache_misses)
        
        # Print cache statistics
        hit_rate = (Environ._global_cache_hits / (Environ._global_cache_hits + Environ._global_cache_misses) * 100
                   if (Environ._global_cache_hits + Environ._global_cache_misses) > 0 else 0)
        
        print(f"Game {i+1}: {ply_count} moves, {end_time - start_time:.3f}s")
        print(f"  Cache size: {len(Environ._global_position_cache)} positions")
        print(f"  Cache hits: {Environ._global_cache_hits}")
        print(f"  Cache misses: {Environ._global_cache_misses}")
        print(f"  Hit rate: {hit_rate:.1f}%")
        
        # Reset environment for next game
        environ.reset_environ()
    
    # Plot cache performance
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(sample_games) + 1), move_counts)
    plt.xlabel('Game Number')
    plt.ylabel('Number of Moves')
    plt.title('Moves per Game')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(sample_games) + 1), cache_hits, label='Hits')
    plt.plot(range(1, len(sample_games) + 1), cache_misses, label='Misses')
    plt.xlabel('Game Number')
    plt.ylabel('Count')
    plt.title('Cache Hits and Misses')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    hit_rates = [h/(h+m)*100 if (h+m) > 0 else 0 for h, m in zip(cache_hits, cache_misses)]
    plt.plot(range(1, len(sample_games) + 1), hit_rates)
    plt.xlabel('Game Number')
    plt.ylabel('Hit Rate (%)')
    plt.title('Cache Hit Rate')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.scatter(move_counts, hit_rates)
    plt.xlabel('Number of Moves')
    plt.ylabel('Hit Rate (%)')
    plt.title('Hit Rate vs. Game Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cache_performance.png')
    print("\nCache performance plot saved to cache_performance.png")
    
def profile_chunking_strategy(chess_data, chunk_sizes=[50, 100, 200, 500]):
    """Profile the effectiveness of the chunking strategy."""
    from training.game_simulation import adaptive_chunker, process_games_in_parallel
    
    print("\nProfiling chunking strategy...")
    
    results = {
        'chunk_size': [],
        'avg_chunk_size': [],
        'max_imbalance': [],
        'chunk_count': []
    }
    
    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size}")
        sample_df = chess_data.iloc[:size] if size < len(chess_data) else chess_data
        
        # Create balanced chunks
        game_indices = list(sample_df.index)
        num_processes = max(1, min(psutil.cpu_count() - 1, len(game_indices)))
        
        chunks = adaptive_chunker.create_balanced_chunks(game_indices, sample_df, num_processes)
        
        # Calculate statistics
        chunk_sizes = [len(chunk) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunks) if chunks else 0
        
        # Calculate imbalance (difference between largest and smallest chunk)
        imbalance = (max(chunk_sizes) - min(chunk_sizes)) / avg_chunk_size * 100 if avg_chunk_size > 0 else 0
        
        print(f"  Chunks: {len(chunks)}")
        print(f"  Avg size: {avg_chunk_size:.1f} games/chunk")
        print(f"  Sizes: {chunk_sizes}")
        print(f"  Imbalance: {imbalance:.1f}%")
        
        # Store results
        results['chunk_size'].append(size)
        results['avg_chunk_size'].append(avg_chunk_size)
        results['max_imbalance'].append(imbalance)
        results['chunk_count'].append(len(chunks))
    
    # Plot chunking performance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['chunk_size'], results['max_imbalance'], 'o-')
    plt.xlabel('Input Size (games)')
    plt.ylabel('Maximum Imbalance (%)')
    plt.title('Chunking Imbalance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['chunk_size'], results['avg_chunk_size'], 'o-')
    plt.xlabel('Input Size (games)')
    plt.ylabel('Average Chunk Size (games)')
    plt.title('Average Chunk Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('chunking_performance.png')
    print("\nChunking performance plot saved to chunking_performance.png")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Detailed profiling of chess database processing")
    parser.add_argument("--filepath", help="Path to the DataFrame file")
    parser.add_argument("--mode", choices=["game", "cache", "chunking", "all"], default="all",
                       help="Profiling mode: game, cache, chunking, or all")
    args = parser.parse_args()
    
    # Get file path
    if args.filepath:
        filepath = args.filepath
    else:
        # Default file path
        from utils import game_settings
        filepath = game_settings.chess_games_filepath_part_1
    
    # Display system information
    sys_info = get_system_info()
    print("\nSystem Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Load data
    print(f"\nLoading data from {filepath}...")
    chess_data = pd.read_pickle(filepath, compression='zip')
    print(f"Loaded {len(chess_data)} games, average {chess_data['PlyCount'].mean():.2f} moves per game")
    
    # Take a smaller sample for profiling
    sample_size = 200
    if len(chess_data) > sample_size:
        chess_data = chess_data.sample(sample_size, random_state=42)
        print(f"Using sample of {sample_size} games")
    
    # Run the appropriate profiling mode
    if args.mode == "game" or args.mode == "all":
        # Profile a few individual games
        profile_multiple_games(chess_data, num_games=3)
    
    if args.mode == "cache" or args.mode == "all":
        # Profile the legal moves cache
        profile_legal_moves_cache(chess_data, num_games=20)
    
    if args.mode == "chunking" or args.mode == "all":
        # Profile the chunking strategy
        profile_chunking_strategy(chess_data)
    
    print("\nDetailed profiling complete!")