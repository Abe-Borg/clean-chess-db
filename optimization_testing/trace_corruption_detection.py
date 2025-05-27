#!/usr/bin/env python3
"""
Trace through the corruption detection logic to find the exact issue.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import game_settings

def trace_game_processing():
    """Trace through the game processing step by step."""
    print("Tracing Game Processing Logic")
    print("=" * 50)
    
    filepath = game_settings.chess_games_filepath_part_1
    try:
        chess_data = pd.read_pickle(filepath, compression='zip')
        chess_data = chess_data.head(10)
        print(f"Loaded {len(chess_data)} games from {filepath}")
    except FileNotFoundError as e:
        print(f"Failed to load chess data {e}")
    
    if chess_data is None:
        print("ERROR: Could not load real data. Please update the file paths.")
        return
    
    # Get the first game
    game_id = chess_data.index[0]
    row = chess_data.loc[game_id]
    ply_count = int(row['PlyCount'])
    
    print(f"Game: {game_id}")
    print(f"PlyCount: {ply_count}")
    
    # Extract moves
    moves = {}
    move_columns = [col for col in chess_data.columns if col != 'PlyCount']
    for col in move_columns:
        moves[col] = row[col]
    
    game_data = {
        'PlyCount': ply_count,
        'moves': moves
    }
    
    # Import and set up
    from agents.Agent import Agent
    from environment.Environ import Environ
    
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    print(f"\nTracing through play_one_game_optimized...")
    
    # Manually trace through the game logic
    move_objects_cache = {}
    turn_count = 0
    
    while True:
        # Get state
        curr_state, legal_moves = environ.get_curr_state_and_legal_moves()
        turn_index = curr_state['turn_index']
        curr_turn = curr_state['curr_turn']
        
        print(f"\nTurn {turn_count + 1}:")
        print(f"  Turn index: {turn_index}")
        print(f"  Current turn: {curr_turn}")
        print(f"  Legal moves: {len(legal_moves)}")
        
        # Check termination conditions
        if turn_index >= ply_count:
            print(f"  → Game ended: turn_index ({turn_index}) >= ply_count ({ply_count})")
            break
        if environ.board.is_game_over():
            print(f"  → Game ended: board.is_game_over() = True")
            break
        
        # Get the move for this turn
        chess_move_san = moves.get(curr_turn, '')
        print(f"  Move from data: '{chess_move_san}'")
        
        if chess_move_san == '' or pd.isna(chess_move_san):
            print(f"  → Move is empty/NaN - continuing...")
            environ.update_curr_state()
            turn_count += 1
            continue
        
        # Convert SAN to Move object
        if chess_move_san not in move_objects_cache:
            try:
                move_obj = environ.convert_san_to_move_object(chess_move_san)
                move_objects_cache[chess_move_san] = move_obj
                print(f"  Converted '{chess_move_san}' to move object: {move_obj}")
            except ValueError as e:
                print(f"  → CORRUPTION: Invalid move format '{chess_move_san}': {e}")
                return f"CORRUPTED: {game_id}"
        else:
            move_obj = move_objects_cache[chess_move_san]
            print(f"  Using cached move object for '{chess_move_san}': {move_obj}")
        
        # Check if move is legal
        if move_obj not in legal_moves:
            print(f"  → CORRUPTION: Move {move_obj} not in legal moves")
            print(f"      Legal moves were: {[str(m) for m in legal_moves[:5]]}...")
            return f"CORRUPTED: {game_id}"
        
        print(f"  Move is legal - applying it")
        
        # Apply move
        environ.push_move_object(move_obj)
        turn_count += 1
        
        # Safety limit
        if turn_count > 200:
            print(f"  → Safety limit reached")
            break
    
    print(f"\nGame completed successfully - NOT corrupted")
    return None

def check_agent_logic():
    """Check if the agent logic is causing issues."""
    print(f"\n" + "=" * 50)
    print("Checking Agent Logic")
    print("=" * 50)
    
    # The original choose_action method expects different parameters
    # Let's see what the agent is actually doing
    
    from agents.Agent import Agent
    
    agent = Agent('W')
    
    # Test the choose_action method
    test_game_moves = {'W1': 'e4', 'B1': 'd5'}
    test_environ_state = {
        'curr_turn': 'W1',
        'legal_moves': ['e4', 'e3', 'd4', 'd3', 'Nf3', 'Nc3']
    }
    
    choice = agent.choose_action(test_game_moves, test_environ_state, 'Game 1')
    print(f"Agent choice for W1: '{choice}'")
    
    # The agent is supposed to return the move from game_moves
    # But our optimized version might not be using this correctly

if __name__ == "__main__":
    result = trace_game_processing()
    if result:
        print(f"\nFinal result: {result}")
    else:
        print(f"\nFinal result: Game processed successfully")
    
    check_agent_logic()