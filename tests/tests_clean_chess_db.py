import pandas as pd
import numpy as np
import chess
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules under test.
from agents.Agent import Agent
from training.training_functions import (
    handle_agent_turn,
    play_one_game,
    play_games,
)
from environment.Environ import Environ


from utils import game_settings


# -----------------------------------------------------------------------------
# Test 1: Agent.choose_action
# -----------------------------------------------------------------------------

def test_agent_choose_action_valid():
    """
    Test that the Agent returns the correct move from the DataFrame
    when there are legal moves.
    """
    # Create a simple DataFrame representing a game.
    df = pd.DataFrame({
        'W1': ['e4'],
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4', 'd4']}
    move = agent.choose_action(df, environ_state, 'Game 1')
    assert move == 'e4'

def test_agent_choose_action_no_legal_moves():
    """
    If there are no legal moves available, the Agent should return an empty string.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': []}
    move = agent.choose_action(df, environ_state, 'Game 1')
    assert move == ''

# -----------------------------------------------------------------------------
# Dummy Environment for testing handle_agent_turn
# -----------------------------------------------------------------------------

class DummyBoard:
    def push_san(self, move):
        # For testing, simply pass (simulate a successful move push)
        pass

class DummyEnviron:
    def __init__(self):
        self.board = DummyBoard()
        self.turn_index = 0

    def get_legal_moves(self):
        # For simplicity, we assume that 'e4' is the only legal move.
        return ['e4']

    def update_curr_state(self):
        self.turn_index += 1

# -----------------------------------------------------------------------------
# Test 2: handle_agent_turn (individual turn tests)
# -----------------------------------------------------------------------------

def test_handle_agent_turn_valid():
    """
    When a valid move is returned by the Agent (and it is in the legal moves),
    the function should apply the move and return an empty string.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}
    dummy_environ = DummyEnviron()
    result = handle_agent_turn(agent, df, environ_state, 'Game 1', dummy_environ)
    # Expected: valid move, so result should be empty.
    assert result == ''

def test_handle_agent_turn_invalid_move():
    """
    If the Agent returns a move that is not in the current legal moves,
    the function should log the error and return the game identifier.
    """
    df = pd.DataFrame({
        'W1': ['e5'],  # e5 is not legal (our dummy legal move is only 'e4')
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}
    dummy_environ = DummyEnviron()
    result = handle_agent_turn(agent, df, environ_state, 'Game 1', dummy_environ)
    assert result == 'Game 1'



# -----------------------------------------------------------------------------
# Test 3: play_one_game
# -----------------------------------------------------------------------------

def test_play_one_game_valid():
    """
    Test play_one_game for a game that should complete normally.
    We simulate a game with a single valid move.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'PlyCount': [1]
    }, index=['Game 1'])
    w_agent = Agent('W')
    b_agent = Agent('B')
    # Use a real Environ instance from the environment module.
    environ = Environ()
    result = play_one_game('Game 1', df, w_agent, b_agent, environ)
    # For a valid game, the result should be an empty string.
    assert result == ''

def test_play_one_game_invalid():
    """
    Test play_one_game for a game that contains an invalid move.
    """
    df = pd.DataFrame({
        'W1': ['e5'],  # e5 is invalid in the starting position.
        'PlyCount': [1]
    }, index=['Game 1'])
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    result = play_one_game('Game 1', df, w_agent, b_agent, environ)
    # The game should be flagged as corrupted.
    assert result == 'Game 1'

# -----------------------------------------------------------------------------
# Test 4: play_games (end-to-end on a small DataFrame)
# -----------------------------------------------------------------------------

def test_play_games_mixed():
    """
    Create a small DataFrame with two games: one valid and one corrupted.
    The valid game has a legal move, the corrupted game has an invalid move.
    The overall play_games function should return a list containing the identifier of the corrupted game.
    """
    df = pd.DataFrame({
        'W1': ['e4', 'e5'],  # Game 1: valid (e4); Game 2: invalid (e5)
        'PlyCount': [1, 1]
    }, index=['Game 1', 'Game 2'])
    corrupted = play_games(df)
    # Expect that Game 2 is flagged as corrupted.
    assert 'Game 2' in corrupted
    assert 'Game 1' not in corrupted

