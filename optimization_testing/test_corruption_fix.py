# file: optimization_testing/test_corruption_fix.py

"""
Test the corruption fix.
"""
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import game_settings

def test_corruption_fix():
    """Test that the fix resolves the corruption issue."""
    print("Testing Corruption Fix")
    print("=" * 50)
    
    filepath = game_settings.chess_games_filepath_part_50
    try:
        chess_data = pd.read_pickle(filepath, compression='zip')
        chess_data = chess_data.sample(10_000)
        print(f"Loaded {len(chess_data)} games from {filepath}")
    except FileNotFoundError as e:
        print(f"Failed to load chess data {e}")

    if chess_data is None:
        print("ERROR: Could not load real data. Please update the file paths.")
        return
    
    # Import fixed functions
    from agents.Agent import Agent
    from environment.Environ import Environ
    
    # Copy the fixed functions here (since they're not in your files yet)
    from training.game_simulation import play_one_game_optimized
    
    print(f"Testing {len(chess_data)} games with fixed logic...")
    
    corrupted_count = 0
    successful_count = 0
    
    for game_id in chess_data.index:
        row = chess_data.loc[game_id]
        ply_count = int(row['PlyCount'])
        
        # Extract moves
        moves = {}
        move_columns = [col for col in chess_data.columns if col != 'PlyCount']
        for col in move_columns:
            moves[col] = row[col]
        
        game_data = {
            'PlyCount': ply_count,
            'moves': moves
        }
        
        # Test with fixed function
        w_agent = Agent('W')
        b_agent = Agent('B')
        environ = Environ()
        
        try:
            result = play_one_game_optimized(game_id, game_data, w_agent, b_agent, environ)
            if result is None:
                successful_count += 1
            else:
                corrupted_count += 1
                print(f"  {game_id}: ❌ CORRUPTED")
        except Exception as e:
            corrupted_count += 1
            print(f"  {game_id}: ❌ ERROR: {e}")
    
    print(f"\nResults:")
    print(f"  Successful: {successful_count}/{len(chess_data)} ({successful_count/len(chess_data)*100:.1f}%)")
    print(f"  Corrupted: {corrupted_count}/{len(chess_data)} ({corrupted_count/len(chess_data)*100:.1f}%)")
    
    if successful_count > corrupted_count:
        print(f"✅ FIX SUCCESSFUL! Corruption rate dropped significantly.")
    else:
        print(f"❌ More debugging needed.")

if __name__ == "__main__":
    test_corruption_fix()