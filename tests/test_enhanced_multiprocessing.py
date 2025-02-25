# File: tests/test_enhanced_multiprocessing.py
"""
Enhanced stress tests for multiprocessing functionality.
These tests focus on:
1. Very large datasets (100k+ games)
2. Error handling during parallel processing
3. Resource usage during parallel processing
4. Mixed valid/invalid games at scale
"""

import sys
import os
import time
import random
import psutil
import pytest
import pandas as pd
import numpy as np
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import process_games_in_parallel, worker_play_games, chunkify
from agents.Agent import Agent
from environment.Environ import Environ

# Helper function to generate synthetic chess games
def generate_synthetic_games(num_games, valid_ratio=1.0):
    """Generate a DataFrame with synthetic chess games.
    
    Args:
        num_games: The number of games to generate
        valid_ratio: The ratio of valid games (0.0 to 1.0)
    
    Returns:
        A DataFrame with synthetic chess games
    """
    num_valid = int(num_games * valid_ratio)
    num_invalid = num_games - num_valid
    
    # Generate valid games
    valid_data = {
        'W1': ['e4'] * num_valid,
        'B1': ['e5'] * num_valid,
        'W2': ['Nf3'] * num_valid,
        'B2': ['Nc6'] * num_valid,
        'PlyCount': [4] * num_valid
    }
    
    # Generate invalid games
    invalid_data = {
        'W1': ['invalid_move'] * num_invalid,
        'B1': ['e5'] * num_invalid,
        'W2': ['Nf3'] * num_invalid,
        'B2': ['Nc6'] * num_invalid,
        'PlyCount': [4] * num_invalid
    }
    
    # Combine and shuffle
    all_data = {k: valid_data[k] + invalid_data[k] for k in valid_data}
    df = pd.DataFrame(all_data, index=[f'Game {i}' for i in range(num_games)])
    
    # Shuffle the rows to mix valid and invalid games
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Test multiprocessing with a very large number of games
def test_multiprocessing_large_scale():
    """Test multiprocessing with 100,000 valid games."""
    num_games = 100000
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games
    }, index=[f'Game {i}' for i in range(num_games)])
    
    start_time = time.time()
    corrupted_games = process_games_in_parallel(list(df.index), worker_play_games, df)
    end_time = time.time()
    
    print(f"Processed {num_games} games in {end_time - start_time:.2f} seconds")
    
    assert len(corrupted_games) == 0, "Expected no corrupted games for valid dataset"

# Test mixed valid/invalid games at scale
def test_multiprocessing_mixed_games():
    """Test multiprocessing with a mix of valid and invalid games."""
    num_games = 10000
    valid_ratio = 0.9  # 90% valid, 10% invalid
    
    df = generate_synthetic_games(num_games, valid_ratio)
    expected_invalid = int(num_games * (1 - valid_ratio))
    
    corrupted_games = process_games_in_parallel(list(df.index), worker_play_games, df)
    
    assert len(corrupted_games) == expected_invalid, f"Expected {expected_invalid} corrupted games"

# Test resource usage during multiprocessing
def test_multiprocessing_resource_usage():
    """Test resource usage during multiprocessing."""
    num_games = 50000
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games
    }, index=[f'Game {i}' for i in range(num_games)])
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    process_games_in_parallel(list(df.index), worker_play_games, df)
    end_time = time.time()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = final_memory - initial_memory
    
    processing_time = end_time - start_time
    games_per_second = num_games / processing_time
    
    print(f"Memory usage: {memory_usage:.2f} MB")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing rate: {games_per_second:.2f} games/second")
    
    # This is not an assert, but prints useful information for performance tuning

# Test chunking with different configurations
def test_chunkify_configurations():
    """Test different chunk configurations for multiprocessing."""
    num_games = 10000
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games
    }, index=[f'Game {i}' for i in range(num_games)])
    
    # Test with different numbers of chunks
    for num_chunks in [2, 4, 8, 16]:
        chunks = chunkify(list(df.index), num_chunks)
        
        assert len(chunks) == num_chunks
        assert sum(len(chunk) for chunk in chunks) == num_games
        
        # Test processing with these chunks
        start_time = time.time()
        corrupted_games = process_games_in_parallel(list(df.index), worker_play_games, df)
        end_time = time.time()
        
        print(f"{num_chunks} chunks: {end_time - start_time:.2f} seconds")
        assert len(corrupted_games) == 0

# Custom worker function that can simulate random failures
def failing_worker_function(game_indices_chunk, chess_data, failure_rate=0.0):
    """Worker function that randomly fails to process some games."""
    corrupted_games = []
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    for game_number in game_indices_chunk:
        # Randomly fail with probability failure_rate
        if random.random() < failure_rate:
            corrupted_games.append(game_number)
            continue
            
        try:
            # For this test, just check if the move is 'invalid_move'
            if chess_data.at[game_number, 'W1'] == 'invalid_move':
                corrupted_games.append(game_number)
                
            environ.reset_environ()
        except Exception as e:
            corrupted_games.append(game_number)
            
    return corrupted_games

# Test error handling during multiprocessing
def test_multiprocessing_error_handling():
    """Test handling of errors during multiprocessing."""
    num_games = 10000
    failure_rate = 0.05  # 5% random failure rate
    
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games
    }, index=[f'Game {i}' for i in range(num_games)])
    
    # Create a partial function with the failure rate
    worker_func = partial(failing_worker_function, failure_rate=failure_rate)
    
    corrupted_games = process_games_in_parallel(list(df.index), worker_func, df)
    
    # With a 5% failure rate, we expect approximately 5% of games to be marked as corrupted
    expected_failures = int(num_games * failure_rate)
    failure_margin = expected_failures * 0.2  # Allow 20% margin of error
    
    assert abs(len(corrupted_games) - expected_failures) <= failure_margin, \
        f"Expected approximately {expected_failures} failures, got {len(corrupted_games)}"

# Benchmark different chunk sizes
@pytest.mark.parametrize("num_chunks", [2, 4, 8, 16, 32])
def test_chunk_size_benchmark(benchmark, num_chunks):
    """Benchmark performance with different chunk sizes."""
    num_games = 5000  # Smaller number for benchmarking
    
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games
    }, index=[f'Game {i}' for i in range(num_games)])
    
    game_indices = list(df.index)
    chunks = chunkify(game_indices, num_chunks)
    
    # Define a function to benchmark
    def process_with_chunks():
        return process_games_in_parallel(game_indices, worker_play_games, df)
    
    # Benchmark the function
    result = benchmark(process_with_chunks)
    
    print(f"Chunks: {num_chunks}, Time: {result}")
    assert len(result) == 0  # No corrupted games expected