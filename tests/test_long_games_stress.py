# File: tests/test_long_games_stress.py
"""
Stress test for processing large numbers of long chess games.
"""

import sys
import os
import time
import logging
import numpy as np
import pytest
import pandas as pd
import chess
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games, process_games_in_parallel, worker_play_games
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

# Using a class-based context manager for proper return values
class MeasureTimeAndMemory:
    def __enter__(self):
        self.process = psutil.Process(os.getpid())
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.end_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        self.duration = self.end_time - self.start_time
        self.memory_used = self.end_mem - self.start_mem
        
        print(f"Time: {self.duration:.2f} seconds")
        print(f"Memory change: {self.memory_used:.2f} MB")
        return False  # Don't suppress exceptions

def generate_long_valid_game(num_moves=100):
    """
    Generate a long valid chess game by simulating actual gameplay.
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
    safety_counter = 0
    max_attempts = 1000
    
    while len(all_moves) < num_moves and safety_counter < max_attempts:
        safety_counter += 1
        
        if board.is_game_over():
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
        if i % 2 == 0:
            col = f'W{i//2 + 1}'
        else:
            col = f'B{i//2 + 1}'
        data[col] = move
        
    return data

def create_long_games_dataframe(num_games, move_count):
    """Create a DataFrame with multiple long chess games."""
    games_data = []
    
    for i in range(num_games):
        if i % 10 == 0:
            print(f"Generating game {i+1}/{num_games}...")
        game_data = generate_long_valid_game(move_count)
        games_data.append(game_data)
    
    # Create DataFrame with all games
    df = pd.DataFrame(games_data, index=[f'Game {i}' for i in range(num_games)])
    
    # Ensure all move columns exist up to the move count
    for i in range(1, (move_count // 2) + 2):
        if f'W{i}' not in df.columns:
            df[f'W{i}'] = None
        if f'B{i}' not in df.columns:
            df[f'B{i}'] = None
    
    return df

def test_stress_with_varying_move_counts():
    """
    Stress test processing different move counts with a significant number of games.
    """
    num_games = 100
    move_counts = [20, 50, 100]
    
    results = []
    
    for move_count in move_counts:
        print(f"\n===== Testing {num_games} games with {move_count} moves each =====")
        
        print("Generating games...")
        df = create_long_games_dataframe(num_games, move_count)
        
        print(f"Processing {num_games} games with {move_count} moves each...")
        with MeasureTimeAndMemory() as metrics:
            corrupted_games = play_games(df)
        
        # Calculate metrics
        games_per_second = num_games / metrics.duration
        moves_per_second = (num_games * move_count) / metrics.duration
        memory_per_game = metrics.memory_used / num_games
        
        print(f"Games per second: {games_per_second:.2f}")
        print(f"Moves per second: {moves_per_second:.2f}")
        print(f"Memory per game: {memory_per_game:.2f} MB")
        print(f"Corrupted games: {len(corrupted_games)}/{num_games}")
        
        result = {
            'move_count': move_count,
            'duration': metrics.duration,
            'memory_used': metrics.memory_used,
            'games_per_second': games_per_second,
            'moves_per_second': moves_per_second,
            'memory_per_game': memory_per_game,
            'corrupted_games': len(corrupted_games)
        }
        results.append(result)
    
    # Display comparative results
    print("\n===== Comparative Results =====")
    print(f"{'Move Count':<10} | {'Time (s)':<10} | {'GPS':<10} | {'MPS':<10} | {'Memory/Game':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['move_count']:<10} | {r['duration']:<10.2f} | {r['games_per_second']:<10.2f} | {r['moves_per_second']:<10.2f} | {r['memory_per_game']:<12.2f}")

def test_detailed_component_performance():
    """
    Measure the performance of individual components with long games.
    """
    # Create one long game for detailed profiling
    move_count = 100
    game_data = generate_long_valid_game(move_count)
    df = pd.DataFrame([game_data], index=['Game 0'])
    
    # Ensure all columns exist
    for i in range(1, (move_count // 2) + 2):
        if f'W{i}' not in df.columns:
            df[f'W{i}'] = None
        if f'B{i}' not in df.columns:
            df[f'B{i}'] = None
    
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    # Measure get_legal_moves performance throughout the game
    print("\n===== Measuring get_legal_moves performance throughout game =====")
    
    # Reset the environment and play through the game
    environ.reset_environ()
    move_times = []
    
    for move_num in range(1, move_count + 1):
        if move_num % 2 == 1:  # White's move
            turn = f'W{(move_num // 2) + 1}'
            agent = w_agent
        else:  # Black's move
            turn = f'B{move_num // 2}'
            agent = b_agent
        
        # Get current state (includes get_legal_moves call)
        start_time = time.time()
        curr_state = environ.get_curr_state()
        legal_moves_time = time.time() - start_time
        
        # Get the move for this turn
        try:
            chess_move = df.at['Game 0', turn]
            if chess_move and chess_move in curr_state['legal_moves']:
                # Apply the move
                environ.board.push_san(chess_move)
                environ.update_curr_state()
                
                # Record the time
                move_times.append((move_num, chess_move, legal_moves_time))
        except (KeyError, ValueError):
            break
    
    # Display performance data
    if move_times:
        print(f"{'Move #':<6} | {'Move':<6} | {'Time (ms)':<10}")
        print("-" * 30)
        
        # Create properly formatted separators
        separator = (None, None, None)
        
        # Get early, middle, and late moves
        early_moves = move_times[:5]
        
        # Handle the case where we don't have enough moves for mid/late sections
        if len(move_times) >= 10:
            mid_point = len(move_times) // 2
            mid_moves = move_times[mid_point-2:mid_point+3]
            late_moves = move_times[-5:]
            
            # Print all sections with separators
            for data in early_moves + [separator] + mid_moves + [separator] + late_moves:
                if data[0] is None:
                    print("..." + " " * 23)
                else:
                    move_num, move, duration = data
                    print(f"{move_num:<6} | {move:<6} | {duration*1000:<10.2f}")
        else:
            # If we have fewer than 10 moves, just print what we have
            for move_num, move, duration in move_times:
                print(f"{move_num:<6} | {move:<6} | {duration*1000:<10.2f}")
        
        # Calculate average times
        avg_time = sum(time for _, _, time in move_times) / len(move_times)
        print(f"\nAverage get_legal_moves time: {avg_time*1000:.2f} ms")
        print(f"Total get_legal_moves time for {len(move_times)} moves: {sum(time for _, _, time in move_times):.4f} seconds")


if __name__ == "__main__":
    # Run the tests directly when the script is executed
    test_stress_with_varying_move_counts()
    test_detailed_component_performance()