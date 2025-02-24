# Advanced Integration Test - Simulate Game with Forced Invalid Moves
# Testing faulty game sequences.

# File: tests/test_advanced_integration.py

import pytest
import pandas as pd
from training.game_simulation import play_one_game
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
