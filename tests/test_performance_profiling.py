# File: tests/test_performance_profiling.py
"""
Performance profiling tests for the chess database processing system.
These tests focus on:
1. Detailed profiling of performance bottlenecks
2. Comparative performance across different configurations
3. Memory usage profiling
4. Scaling tests
"""

import sys
import os
import time
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
import cProfile
import pstats
import io
import multiprocessing
import psutil
from memory_profiler import profile as memory_profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games, process_games_in_parallel, worker_play_games, handle_agent_turn, play_one_game
from agents.Agent import Agent
from environment.Environ import Environ
from utils import constants

# Helper function to create synthetic game data
def create_game_data(num_games, move_count=4, corrupt_percentage=0):
    """Create synthetic game data for performance testing.
    
    Args:
        num_games: Number of games to generate
        move_count: Number of moves per game (ply count)
        corrupt_percentage: Percentage of games to make invalid (0-100)
    
    Returns:
        DataFrame with synthetic game data
    """
    # Standard opening sequence
    valid_moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6']
    
    # Create DataFrame columns
    data = {'PlyCount': [move_count] * num_games}
    
    # Add move columns for each ply
    for i in range(1, (move_count // 2) + 1):
        if i * 2 <= len(valid_moves):
            data[f'W{i}'] = [valid_moves[(i-1)*2]] * num_games
            data[f'B{i}'] = [valid_moves[(i-1)*2 + 1]] * num_games
    
    # Convert to DataFrame
    df = pd.DataFrame(data, index=[f'Game {i}' for i in range(num_games)])
    
    # Corrupt some percentage of games if requested
    if corrupt_percentage > 0:
        num_corrupt = int(num_games * corrupt_percentage / 100)
        corrupt_indices = np.random.choice(num_games, num_corrupt, replace=False)
        for idx in corrupt_indices:
            df.at[f'Game {idx}', 'W1'] = 'invalid_move'
    
    return df

# -----------------------------------------------------------------------------
# PERFORMANCE BENCHMARKS
# -----------------------------------------------------------------------------

# Basic performance benchmark - tests the raw speed of processing valid games
@pytest.mark.benchmark(group="basic")
def test_basic_performance_benchmark(benchmark):
    """Benchmark performance with a moderate number of valid games."""
    num_games = 5000
    df = create_game_data(num_games)
    
    result = benchmark(play_games, df)
    assert len(result) == 0, "Expected no corrupted games"

# Benchmark different game sizes
@pytest.mark.parametrize("num_games", [1000, 5000, 10000])
@pytest.mark.benchmark(group="scaling")
def test_scaling_with_game_count(benchmark, num_games):
    """Benchmark how performance scales with number of games."""
    df = create_game_data(num_games)
    
    result = benchmark(play_games, df)
    assert len(result) == 0, "Expected no corrupted games"

# Benchmark different move counts (game lengths)
@pytest.mark.parametrize("move_count", [2, 4, 8, 16])
@pytest.mark.benchmark(group="game_length")
def test_scaling_with_move_count(benchmark, move_count):
    """Benchmark how performance scales with move count."""
    num_games = 1000
    df = create_game_data(num_games, move_count=move_count)
    
    result = benchmark(play_games, df)
    assert len(result) == 0, "Expected no corrupted games"

# Benchmark different ratios of valid/invalid games
@pytest.mark.parametrize("corrupt_percentage", [0, 5, 10, 50])
@pytest.mark.benchmark(group="corruption")
def test_scaling_with_corruption_rate(benchmark, corrupt_percentage):
    """Benchmark how performance scales with percentage of corrupted games."""
    num_games = 1000
    df = create_game_data(num_games, corrupt_percentage=corrupt_percentage)
    
    result = benchmark(play_games, df)
    expected_corrupted = int(num_games * corrupt_percentage / 100)
    assert len(result) == expected_corrupted, f"Expected {expected_corrupted} corrupted games"

# Benchmark different numbers of processes for multiprocessing
@pytest.mark.parametrize("num_processes", [2, 4, 8, max(2, multiprocessing.cpu_count() - 1)])
@pytest.mark.benchmark(group="processes")
def test_scaling_with_process_count(benchmark, num_processes):
    """Benchmark how performance scales with number of processes."""
    num_games = 5000
    df = create_game_data(num_games)
    game_indices = list(df.index)
    
    # Override the cpu_count method to return our specified number
    with patch('multiprocessing.cpu_count', return_value=num_processes):
        result = benchmark(process_games_in_parallel, game_indices, worker_play_games, df)
    
    assert len(result) == 0, "Expected no corrupted games"

# -----------------------------------------------------------------------------
# DETAILED PROFILING
# -----------------------------------------------------------------------------

def test_detailed_profiling():
    """Run detailed profiling on the processing of games."""
    num_games = 1000
    df = create_game_data(num_games)
    
    # Set up the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the code to profile
    play_games(df)
    
    # Disable the profiler and print stats
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print(s.getvalue())

# Memory usage test for individual game processing
@memory_profile
def test_memory_usage_individual_game():
    """Profile memory usage for processing individual games."""
    df = create_game_data(1, move_count=16)  # One game with 16 moves
    
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    # Process the game
    play_one_game('Game 0', df, w_agent, b_agent, environ)

# Memory usage test for batch processing
def test_memory_usage_batch_processing():
    """Profile memory usage for batch processing of games."""
    num_games = 10000
    df = create_game_data(num_games)
    
    # Get current process for memory monitoring
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process all games
    play_games(df)
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Memory usage increased by {memory_increase:.2f} MB")
    print(f"Per game: {memory_increase / num_games:.6f} MB")

# -----------------------------------------------------------------------------
# COMPONENT-SPECIFIC BENCHMARKS
# -----------------------------------------------------------------------------

@pytest.mark.benchmark(group="components")
def test_agent_performance(benchmark):
    """Benchmark the Agent's choose_action method."""
    df = create_game_data(1)
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4', 'd4', 'Nf3']}
    
    # Benchmark just the agent's choose_action method
    result = benchmark(agent.choose_action, df, environ_state, 'Game 0')
    assert result == 'e4'

@pytest.mark.benchmark(group="components")
def test_environ_performance(benchmark):
    """Benchmark the Environ's get_legal_moves method."""
    environ = Environ()
    
    # Benchmark getting legal moves
    result = benchmark(environ.get_legal_moves)
    assert len(result) > 0

@pytest.mark.benchmark(group="components")
def test_handle_agent_turn_performance(benchmark):
    """Benchmark the handle_agent_turn function."""
    df = create_game_data(1)
    agent = Agent('W')
    environ = Environ()
    environ_state = environ.get_curr_state()
    
    # Benchmark handling a single agent turn
    result = benchmark(handle_agent_turn, agent, df, environ_state, 'Game 0', environ)
    assert result is None