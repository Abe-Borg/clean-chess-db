# File: tests/test_long_games.py
"""
Test performance with games of varying lengths up to 100 moves.
"""

import sys
import os
import time
import logging
import numpy as np
import pytest
import pandas as pd
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games, play_one_game
from agents.Agent import Agent
from environment.Environ import Environ

# Configure silent logging
def configure_silent_logging():
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)
    
    game_logger = logging.getLogger("game_simulation.py")
    for handler in game_logger.handlers[:]:
        game_logger.removeHandler(handler)
    game_logger.addHandler(null_handler)

configure_silent_logging()

def generate_long_valid_game(num_moves=100):
    """
    Generate a long valid chess game by simulating actual gameplay.
    
    Args:
        num_moves: Target number of half-moves to generate
        
    Returns:
        Dictionary with move columns and PlyCount
    """
    # Start with a basic opening
    opening_moves = [
        'e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'd3', 'Nf6', 
        'O-O', 'd6', 'c3', 'O-O'
    ]
    
    # Continue the game with random valid moves
    board = chess.Board()
    all_moves = []
    
    # Add the opening moves
    for san_move in opening_moves:
        try:
            move = board.parse_san(san_move)
            board.push(move)
            all_moves.append(san_move)
        except ValueError:
            break
    
    # Generate additional random moves up to the target count
    # Use a safety counter to avoid infinite loops
    safety_counter = 0
    max_attempts = 1000
    
    while len(all_moves) < num_moves and safety_counter < max_attempts:
        safety_counter += 1
        
        if board.is_game_over():
            # If game is over, reset and use the moves we have
            break
            
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        # Choose a random move
        move = np.random.choice(legal_moves)
        
        # Convert to SAN before pushing
        san_move = board.san(move)
        board.push(move)
        all_moves.append(san_move)
    
    # Create the game data dictionary
    data = {'PlyCount': min(len(all_moves), num_moves)}
    
    # Add move columns
    for i, move in enumerate(all_moves[:num_moves]):
        # Determine if white or black move
        if i % 2 == 0:
            col = f'W{i//2 + 1}'
        else:
            col = f'B{i//2 + 1}'
        data[col] = move
        
    return data

@pytest.mark.parametrize("move_count", [20, 40, 60, 80, 100])
def test_performance_with_long_games(move_count, benchmark):
    """Test performance with games of varying lengths."""
    # Generate a single long valid game
    game_data = generate_long_valid_game(move_count)
    
    # Create a DataFrame with this single game
    df = pd.DataFrame([game_data], index=['Game 0'])
    
    # Ensure all move columns exist up to the move count
    for i in range(1, (move_count // 2) + 2):
        if f'W{i}' not in df.columns:
            df[f'W{i}'] = None
        if f'B{i}' not in df.columns:
            df[f'B{i}'] = None
    
    # Benchmark the play_one_game function
    # This is more reliable than play_games for single game performance testing
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    benchmark(play_one_game, 'Game 0', df, w_agent, b_agent, environ)
    
    # Print some additional information
    print(f"Moves: {move_count}, Final position:")
    print(environ.board)

@pytest.mark.parametrize("move_count", [20, 40, 60, 80, 100])
def test_bulk_performance_with_long_games(move_count):
    """
    Test performance with multiple games of the same length.
    This doesn't use benchmark to allow for more realistic timing of multiple games.
    """
    num_games = 10
    games_data = []
    
    # Generate multiple long games
    for i in range(num_games):
        game_data = generate_long_valid_game(move_count)
        games_data.append(game_data)
    
    # Create DataFrame with all games
    df = pd.DataFrame(games_data, index=[f'Game {i}' for i in range(num_games)])
    
    # Time the full game processing
    start_time = time.time()
    corrupted_games = play_games(df)
    end_time = time.time()
    
    # Report results
    duration = end_time - start_time
    games_per_second = num_games / duration
    moves_per_second = (num_games * move_count) / duration
    
    print(f"\nMove count: {move_count}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Games per second: {games_per_second:.2f}")
    print(f"Moves per second: {moves_per_second:.2f}")
    print(f"Corrupted games: {len(corrupted_games)}/{num_games}")