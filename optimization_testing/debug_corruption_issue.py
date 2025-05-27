#!/usr/bin/env python3
"""
Debug the high corruption rate issue.
96-97% corruption rate suggests a systematic problem.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_corruption_issue():
    """Debug why 96-97% of games are being flagged as corrupted."""
    print("Debugging High Corruption Rate")
    print("=" * 50)
    
    # Load a small sample to debug
    possible_paths = [
        "chess_data/chess_games_part_1.pkl",
        "utils/../chess_data/chess_games_part_1.pkl",
        "../chess_data/chess_games_part_1.pkl"
    ]
    
    chess_data = None
    for path in possible_paths:
        if os.path.exists(path):
            chess_data = pd.read_pickle(path).head(10)  # Just 10 games
            break
    
    if chess_data is None:
        print("Could not load data")
        return
    
    print(f"Loaded {len(chess_data)} games for debugging")
    print(f"Columns: {list(chess_data.columns)}")
    print(f"Index: {list(chess_data.index)}")
    
    # Look at the first game in detail
    first_game_id = chess_data.index[0]
    first_game = chess_data.loc[first_game_id]
    
    print(f"\nFirst game details:")
    print(f"  Game ID: {first_game_id}")
    print(f"  PlyCount: {first_game['PlyCount']}")
    
    # Show the moves
    move_columns = [col for col in chess_data.columns if col != 'PlyCount']
    moves = {}
    for col in move_columns[:10]:  # First 10 move columns
        move_value = first_game[col]
        moves[col] = move_value
        print(f"  {col}: '{move_value}' (type: {type(move_value)})")
    
    # Test if moves are valid
    print(f"\nTesting move validity:")
    import chess
    from environment.Environ import Environ
    
    board = chess.Board()
    environ = Environ()
    
    print(f"Starting position legal moves: {len(list(board.legal_moves))}")
    
    # Try to play the first few moves
    for i, (col, move) in enumerate(moves.items()):
        if pd.isna(move) or move == '':
            print(f"  {col}: Empty move - game likely ended")
            break
            
        try:
            # Try to parse the move
            if isinstance(move, str):
                legal_moves = [board.san(m) for m in board.legal_moves]
                if move in legal_moves:
                    board.push_san(move)
                    print(f"  {col}: '{move}' - VALID")
                else:
                    print(f"  {col}: '{move}' - INVALID (not in legal moves)")
                    print(f"    Legal moves were: {legal_moves[:5]}...")
                    break
            else:
                print(f"  {col}: '{move}' - INVALID (not string, type: {type(move)})")
                break
                
        except Exception as e:
            print(f"  {col}: '{move}' - ERROR: {e}")
            break
    
    # Test the actual processing function
    print(f"\nTesting actual game processing:")
    
    try:
        from training.game_simulation import play_one_game_optimized
        from agents.Agent import Agent
        
        # Create test data structure
        game_data = {
            'PlyCount': first_game['PlyCount'],
            'moves': {col: first_game[col] for col in move_columns}
        }
        
        # Create agents and environment
        w_agent = Agent('W')
        b_agent = Agent('B')
        environ = Environ()
        
        print(f"Testing game processing...")
        result = play_one_game_optimized(first_game_id, game_data, w_agent, b_agent, environ)
        
        if result is None:
            print(f"  Game processed successfully (not corrupted)")
        else:
            print(f"  Game flagged as corrupted: {result}")
            
    except Exception as e:
        print(f"  Error during processing: {e}")
        import traceback
        traceback.print_exc()

def check_data_format():
    """Check if the data format matches what the code expects."""
    print(f"\nChecking Data Format Compatibility")
    print("=" * 50)
    
    # Load data
    possible_paths = [
        "chess_data/chess_games_part_1.pkl",
        "utils/../chess_data/chess_games_part_1.pkl",
        "../chess_data/chess_games_part_1.pkl"
    ]
    
    chess_data = None
    for path in possible_paths:
        if os.path.exists(path):
            chess_data = pd.read_pickle(path)
            break
    
    if chess_data is None:
        print("Could not load data")
        return
    
    print(f"Total games: {len(chess_data)}")
    print(f"Columns: {len(chess_data.columns)}")
    
    # Check PlyCount distribution
    print(f"\nPlyCount statistics:")
    print(f"  Min: {chess_data['PlyCount'].min()}")
    print(f"  Max: {chess_data['PlyCount'].max()}")
    print(f"  Mean: {chess_data['PlyCount'].mean():.1f}")
    print(f"  Games with PlyCount=1: {(chess_data['PlyCount'] == 1).sum()}")
    print(f"  Games with PlyCount<10: {(chess_data['PlyCount'] < 10).sum()}")
    
    # Check for missing moves
    move_columns = [col for col in chess_data.columns if col != 'PlyCount']
    print(f"\nMove columns: {len(move_columns)}")
    print(f"First few move columns: {move_columns[:10]}")
    
    # Check for NaN values in first moves
    first_moves = ['W1', 'B1', 'W2', 'B2']
    for col in first_moves:
        if col in chess_data.columns:
            nan_count = chess_data[col].isna().sum()
            empty_count = (chess_data[col] == '').sum()
            print(f"  {col}: {nan_count} NaN, {empty_count} empty")

if __name__ == "__main__":
    debug_corruption_issue()
    check_data_format()