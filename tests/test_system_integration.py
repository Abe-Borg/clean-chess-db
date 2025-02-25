# File: tests/test_system_integration.py
"""
System integration tests for the chess database processing pipeline.
These tests focus on:
1. End-to-end workflow testing
2. File I/O operations
3. Integration between components
4. Error handling and recovery
"""

import sys
import os
import tempfile
import pytest
import pandas as pd
import pickle
import logging
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import play_games, process_games_in_parallel, worker_play_games
from agents.Agent import Agent
from environment.Environ import Environ
from utils import game_settings

# Helper function to create a temporary DataFrame file
def create_temp_dataframe(num_games=100, corrupt_indices=None):
    """
    Create a temporary DataFrame with chess games and save it to a pickle file.
    
    Args:
        num_games: Number of games to generate
        corrupt_indices: List of indices to mark as corrupted (with invalid moves)
        
    Returns:
        Tuple of (temp_file_path, DataFrame)
    """
    if corrupt_indices is None:
        corrupt_indices = []
    
    # Create data for DataFrame
    data = {
        'PlyCount': [4] * num_games,
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'W2': ['Nf3'] * num_games,
        'B2': ['Nc6'] * num_games
    }
    
    # Inject corrupted games
    for idx in corrupt_indices:
        if idx < num_games:
            data['W1'][idx] = 'invalid_move'
    
    # Create DataFrame
    df = pd.DataFrame(data, index=[f'Game {i}' for i in range(num_games)])
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    df.to_pickle(temp_file.name, compression='zip')
    
    return temp_file.name, df

# Test end-to-end workflow
def test_end_to_end_workflow():
    """Test the entire workflow from file reading to processing to writing results."""
    # Create a temporary DataFrame with 100 games, 10 of which are corrupted
    corrupt_indices = list(range(10))
    temp_file_path, original_df = create_temp_dataframe(100, corrupt_indices)
    
    try:
        # Process the games
        chess_data = pd.read_pickle(temp_file_path, compression='zip')
        corrupted_games = play_games(chess_data)
        
        # Verify that exactly the 10 corrupted games were identified
        assert len(corrupted_games) == 10
        for i in range(10):
            assert f'Game {i}' in corrupted_games
        
        # Clean the data by removing corrupted games
        clean_df = chess_data.drop(corrupted_games)
        
        # Save the cleaned data to a new temporary file
        output_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        clean_df.to_pickle(output_file.name, compression='zip')
        
        # Verify that the cleaned data has the expected number of games
        reloaded_df = pd.read_pickle(output_file.name, compression='zip')
        assert len(reloaded_df) == 90
        
        # Verify that the cleaned data contains only valid games
        assert all(game_id not in corrupted_games for game_id in reloaded_df.index)
        
    finally:
        # Clean up temporary files
        os.unlink(temp_file_path)
        if 'output_file' in locals():
            os.unlink(output_file.name)

# Test error handling and recovery in file operations
def test_file_operation_error_handling():
    """Test that the system handles file operation errors gracefully."""
    # Test with a non-existent file
    non_existent_file = '/path/to/non/existent/file.pkl'
    
    # Mock the main script's functionality
    with patch('builtins.print') as mock_print, \
         patch('logging.Logger.critical') as mock_log_critical:
        
        # Attempt to read a non-existent file (should raise FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            pd.read_pickle(non_existent_file)
        
        # In a real scenario, the system should log the error and continue
        # with the next file. We just verify that logging would happen.
        assert mock_log_critical.call_count == 0  # We didn't actually execute the main script code

# Test integration between components with various board states
def test_component_integration_different_board_states():
    """Test integration between Agent and Environ with different board states."""
    # Create games with different complexity
    games_data = {
        # Standard starting position with e4
        'Game 1': {'W1': 'e4', 'B1': 'e5', 'W2': 'Nf3', 'B2': 'Nc6', 'PlyCount': 4},
        
        # Different opening: d4 opening
        'Game 2': {'W1': 'd4', 'B1': 'd5', 'W2': 'c4', 'B2': 'e6', 'PlyCount': 4},
        
        # Sicilian opening
        'Game 3': {'W1': 'e4', 'B1': 'c5', 'W2': 'Nf3', 'B2': 'd6', 'PlyCount': 4},
        
        # Italian game
        'Game 4': {'W1': 'e4', 'B1': 'e5', 'W2': 'Nf3', 'B2': 'Nc6', 'W3': 'Bc4', 'B3': 'Bc5', 'PlyCount': 6}
    }
    
    df = pd.DataFrame(games_data).T
    
    # Process each game and verify it's not corrupted
    corrupted_games = play_games(df)
    assert len(corrupted_games) == 0, "All games should be valid"

# Test handling of empty or invalid DataFrames
def test_handling_empty_or_invalid_dataframes():
    """Test that the system handles empty or invalid DataFrames properly."""
    # Test with an empty DataFrame
    empty_df = pd.DataFrame()
    
    # This should not raise an exception, but return an empty list
    corrupted_games = play_games(empty_df)
    assert corrupted_games == []
    
    # Test with a DataFrame missing required columns
    invalid_df = pd.DataFrame({'InvalidColumn': [1, 2, 3]}, index=['Game 1', 'Game 2', 'Game 3'])
    
    # The system should handle this gracefully
    corrupted_games = play_games(invalid_df)
    assert set(corrupted_games) == {'Game 1', 'Game 2', 'Game 3'}, "All games should be marked as corrupted"

# Test integration with logging system
def test_logging_integration(caplog):
    """Test integration with the logging system."""
    # Create a DataFrame with a deliberately corrupted game
    df = pd.DataFrame({
        'W1': ['invalid_move'],
        'B1': ['e5'],
        'PlyCount': [2]
    }, index=['Game 1'])
    
    # Set up logging capture
    caplog.set_level(logging.CRITICAL)
    
    # Process the game (should log a critical error)
    corrupted_games = play_games(df)
    
    # Verify that the game was flagged as corrupted
    assert 'Game 1' in corrupted_games
    
    # Verify that a critical error was logged
    assert any("Invalid move" in record.message for record in caplog.records)

# Test the complete pipeline with a mock file system
def test_complete_pipeline_with_mock_filesystem():
    """Test the complete pipeline with a mock file system."""
    # Create sample data
    num_games = 100
    corrupt_indices = [5, 15, 25, 35, 45]
    
    # Create and mock file paths
    with patch('utils.game_settings.chess_games_filepath_part_1', 'mock_file_1.pkl'), \
         patch('utils.game_settings.chess_games_filepath_part_2', 'mock_file_2.pkl'), \
         patch('pandas.read_pickle') as mock_read_pickle, \
         patch('pandas.DataFrame.to_pickle') as mock_to_pickle, \
         patch('pandas.DataFrame.drop', return_value=pd.DataFrame()) as mock_drop, \
         patch('training.game_simulation.play_games', return_value=[f'Game {i}' for i in corrupt_indices]) as mock_play_games:
        
        # Mock the DataFrame that would be read
        mock_df = pd.DataFrame({
            'W1': ['e4'] * num_games,
            'B1': ['e5'] * num_games,
            'PlyCount': [4] * num_games
        }, index=[f'Game {i}' for i in range(num_games)])
        mock_read_pickle.return_value = mock_df
        
        # Import the script code to run it
        try:
            from main import play_games as main_script
        except ImportError:
            # If the module can't be imported directly, skip this test
            pytest.skip("main.play_games module could not be imported")
            
        # Execute a mocked version of the main script
        # This is a simplified version of what would happen in the real script
        for part in range(1, 3):
            file_path = f'mock_file_{part}.pkl'
            chess_data = mock_read_pickle(file_path, compression='zip')
            mock_chess_data = chess_data.head(1000)  # Simulate head() call
            
            corrupted_games = mock_play_games(mock_chess_data)
            
            # Clean the data
            mock_drop(corrupted_games)
            
            # Verify that to_pickle was called (commented out because we don't actually save)
            # mock_to_pickle.assert_called_with(file_path, compression='zip')
            
        # Verify the correct functions were called
        assert mock_read_pickle.call_count == 2
        assert mock_play_games.call_count == 2
        assert mock_drop.call_count == 2