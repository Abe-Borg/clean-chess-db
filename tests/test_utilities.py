# test_utilities.py
import pytest
import chess
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import chunkify, apply_move_and_update_state
from environment.Environ import Environ
from agents.Agent import Agent

def test_chunkify_even():
    # Test that chunkify splits a list evenly when possible.
    lst = list(range(10))
    chunks = chunkify(lst, 2)
    assert len(chunks) == 2
    assert sum(len(chunk) for chunk in chunks) == 10

def test_chunkify_odd():
    # Test that chunkify handles lists that don't split evenly.
    lst = list(range(10))
    chunks = chunkify(lst, 3)
    assert len(chunks) == 3
    assert sum(len(chunk) for chunk in chunks) == 10
    lengths = [len(chunk) for chunk in chunks]
    # The difference in length between any two chunks should be at most 1.
    assert max(lengths) - min(lengths) <= 1

def test_environ_initial_state():
    # Ensure that a new Environ has the correct starting state.
    environ = Environ()
    state = environ.get_curr_state()
    legal_moves = environ.get_legal_moves()
    # Standard chess starting position should have 20 legal moves.
    assert len(legal_moves) == 20
    assert state['turn_index'] == 0
    assert state['curr_turn'] == environ.turn_list[0]

def test_environ_reset():
    # Verify that reset_environ correctly resets the board and turn index.
    environ = Environ()
    environ.update_curr_state()
    assert environ.turn_index == 1
    environ.reset_environ()
    assert environ.turn_index == 0
    new_board = chess.Board()
    assert environ.board.fen() == new_board.fen()

def test_agent_choose_action_missing_column():
    # Test Agent.choose_action when the expected column is missing in the DataFrame.
    df = pd.DataFrame({
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}
    # Expect a KeyError because the 'W1' column is missing.
    with pytest.raises(KeyError):
        agent.choose_action(df, environ_state, 'Game 1')

def test_apply_move_and_update_state():
    # Verify that apply_move_and_update_state increments the turn index and updates the board.
    environ = Environ()
    initial_turn_index = environ.turn_index
    starting_fen = environ.board.fen()
    # Use the first legal move.
    move = environ.get_legal_moves()[0]
    apply_move_and_update_state(move, environ)
    # Turn index should increment by 1.
    assert environ.turn_index == initial_turn_index + 1
    # The board FEN should differ from the starting position.
    assert environ.board.fen() != starting_fen
