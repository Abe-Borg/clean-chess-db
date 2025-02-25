# File: tests/test_performance_profiling.py
"""
Performance profiling tests for the chess database processing system.
These tests focus on:
1. Detailed profiling of performance bottlenecks
2. Comparative performance across different configurations
3. Memory usage profiling
4. Scaling tests
"""

import logging
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

# Add this at the top of your test_performance_profiling.py file after the imports

# Configure logging to prevent console output during tests
def configure_silent_logging():
    root_logger = logging.getLogger()
    # Remove all handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up null handler to suppress output
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)
    
    # Also configure the game_simulation logger specifically
    game_logger = logging.getLogger("game_simulation.py")
    for handler in game_logger.handlers[:]:
        game_logger.removeHandler(handler)
    game_logger.addHandler(null_handler)

# Run the configuration
configure_silent_logging()

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
    # Get a valid sequence of moves
    valid_moves = get_valid_move_sequence(length=min(32, move_count))
    
    # If we need more moves than available, repeat the sequence
    while len(valid_moves) < move_count:
        valid_moves.extend(valid_moves[:min(8, move_count - len(valid_moves))])
    
    # Create DataFrame columns
    data = {'PlyCount': [move_count] * num_games}
    
    # Add move columns for each ply
    for i in range(1, (move_count // 2) + 2):  # +2 to ensure we cover odd numbers of moves
        white_move_idx = (i-1) * 2
        black_move_idx = white_move_idx + 1
        
        if white_move_idx < len(valid_moves):
            data[f'W{i}'] = [valid_moves[white_move_idx]] * num_games
        
        if black_move_idx < len(valid_moves):
            data[f'B{i}'] = [valid_moves[black_move_idx]] * num_games
    
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
    # Create a game with more moves but ensure all columns exist
    move_count = 16
    df = create_game_data(1, move_count=move_count)
    
    # Verify that all required columns exist
    for i in range(1, (move_count // 2) + 1):
        if f'W{i}' not in df.columns:
            df[f'W{i}'] = 'e4'  # Add default move if missing
        if f'B{i}' not in df.columns:
            df[f'B{i}'] = 'e5'  # Add default move if missing
    
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

# @pytest.mark.benchmark(group="components")
# def test_handle_agent_turn_performance(benchmark):
#     """Benchmark the handle_agent_turn function."""
#     df = create_game_data(1)
#     agent = Agent('W')
#     environ = Environ()
    
#     # We need to ensure that the move in the DataFrame is actually
#     # in the list of legal moves for the starting position
#     legal_moves = environ.get_legal_moves()
    
#     # Use a valid move from legal_moves
#     if legal_moves:
#         df.at['Game 0', 'W1'] = legal_moves[0]
    
#     environ_state = environ.get_curr_state()
    
#     # Benchmark handling a single agent turn
#     result = benchmark(handle_agent_turn, agent, df, environ_state, 'Game 0', environ)
#     assert result is None

def get_valid_move_sequence(length=8):
    """
    Generate a valid sequence of chess moves from the starting position.
    
    Args:
        length: Number of half-moves to generate
        
    Returns:
        List of valid moves in Standard Algebraic Notation
    """
    # Some known valid opening sequences
    openings = [
        # Ruy Lopez
        ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 
         'O-O', 'Be7', 'Re1', 'b5', 'Bb3', 'O-O', 'd4', 'exd4'],
        # Italian Game
        ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'd3', 'Nf6',
         'O-O', 'd6', 'c3', 'a6', 'Re1', 'Ba7', 'Nbd2', 'O-O'],
        # Sicilian Defense
        ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6',
         'Nc3', 'a6', 'Be2', 'e6', 'O-O', 'Be7', 'f4', 'O-O'],
        # Queen's Gambit Accepted
        ['d4', 'd5', 'c4', 'dxc4', 'e3', 'e6', 'Bxc4', 'Nf6',
         'Nf3', 'c5', 'O-O', 'a6', 'e4', 'b5', 'Bd3', 'Bb7'],
    ]
    
    # Pick a random opening
    import random
    opening = random.choice(openings)
    
    # Return the requested number of moves (or all if length > len(opening))
    return opening[:min(length, len(opening))] 

