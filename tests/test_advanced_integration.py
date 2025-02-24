# Advanced Integration Test - Simulate Game with Forced Invalid Moves
# Testing faulty game sequences.

# File: tests/test_advanced_integration.py

import pytest
import pandas as pd
import chess
from training.game_simulation import play_one_game, play_games
from agents.Agent import Agent
from environment.Environ import Environ

def test_simulate_game_with_forced_illegal_move():
    """Simulate a game where an invalid move is forced,
    check if the game is flagged as corrupted.
    """

    df = pd.DataFrame({
        'W1': ['e4'],  # Forced incorrect move
        'B1': ['illegal'],  # Forced illegal move for black
        'PlyCount': [2]
    }, index=['Game 1'])

    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()

    result = play_one_game('Game 1', df, w_agent, b_agent, environ)

    assert result == 'Game 1', "Game with forced illegal move should be flagged as corrupted."

# Integration Test 1: Valid Single Game Simulation
def test_integration_valid_game():
    """
    Simulate a valid game with two moves:
      - White: e4
      - Black: e5
    PlyCount is set to 2, so both moves are processed.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'B1': ['e5'],
        'PlyCount': [2]
    }, index=['Game 1'])
    
    # Run the full simulation on the DataFrame.
    corrupted = play_games(df)
    # Expect no corrupted games.
    assert corrupted == []

# Integration Test 2: Forced Illegal Move in Single Game
def test_integration_forced_illegal_game():
    """
    Simulate a game where the white agent returns an illegal move.
    The simulation should flag the game as corrupted.
    """
    df = pd.DataFrame({
        'W1': ['illegal_move'],  # Forced illegal move
        'B1': ['e5'],            # Won't be reached
        'PlyCount': [2]
    }, index=['Game 1'])
    
    corrupted = play_games(df)
    assert 'Game 1' in corrupted

# Integration Test 3: Mixed Games in One DataFrame
def test_integration_mixed_games():
    """
    Simulate two games in one DataFrame:
      - Game 1: Valid moves
      - Game 2: Forced illegal move on white's turn.
    The simulation should flag only Game 2 as corrupted.
    """
    df = pd.DataFrame({
        'W1': ['e4', 'illegal_move'],
        'B1': ['e5', 'e5'],
        'PlyCount': [2, 2]
    }, index=['Game 1', 'Game 2'])
    
    corrupted = play_games(df)
    assert 'Game 2' in corrupted
    assert 'Game 1' not in corrupted

# Integration Test 4: Simulation Stops at PlyCount
def test_integration_game_stops_at_plycount():
    """
    Simulate a game where PlyCount is set to 1.
    Although both W1 and B1 columns exist, only the first move should be processed.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'B1': ['e5'],  # Should not be processed
        'PlyCount': [1]
    }, index=['Game 1'])
    
    result = play_one_game('Game 1', df, Agent('W'), Agent('B'), Environ())
    # A valid move with PlyCount = 1 should process only one move and complete normally.
    assert result is None

# Integration Test 5: Multi-Move Game Simulation and Board State Verification
def test_integration_multi_move_game():
    """
    Simulate a full game with four moves:
      - W1: e4, B1: e5, W2: Nf3, B2: Nc6
    Verify that the simulation completes and the board state is updated.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'B1': ['e5'],
        'W2': ['Nf3'],
        'B2': ['Nc6'],
        'PlyCount': [4]
    }, index=['Game 1'])
    
    environ = Environ()
    result = play_one_game('Game 1', df, Agent('W'), Agent('B'), environ)
    assert result is None

    # After a valid 4-move game, the board state should differ from the initial position.
    starting_board = chess.Board()
    assert environ.board.fen() != starting_board.fen()

# Integration Test 6: Concurrency/Multiprocessing with a Larger Dataset
def test_integration_concurrency_large_scale():
    """
    Create a DataFrame with 1000 valid games to test the multiprocessing flow.
    Each game is valid with a single move (e4 by white, e5 by black) and PlyCount of 1.
    """
    num_games = 1000
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games,
    }, index=[f"Game {i+1}" for i in range(num_games)])
    
    corrupted = play_games(df)
    assert len(corrupted) == 0