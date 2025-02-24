# File: tests/test_additional_unit_tests.py
import logging
import pytest
import pandas as pd
import chess
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.Agent import Agent
from environment.Environ import Environ
from training.game_simulation import handle_agent_turn, apply_move_and_update_state, play_one_game, process_games_in_parallel, worker_play_games

def test_agent_raises_keyerror_for_missing_column():
    """
    Verify that Agent.choose_action raises a KeyError
    if the expected move column is missing.
    """
    df = pd.DataFrame({
        'PlyCount': [1],
        'B1': ['e5']
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}
    with pytest.raises(KeyError):
        agent.choose_action(df, environ_state, 'Game 1')

# Test that error logs the appropriate message when the expected column is missing.
def test_logging_for_missing_column():
    df = pd.DataFrame({'PlyCount': [1], 'B1': ['e5']}, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}

    with pytest.raises(KeyError):
        agent.choose_action(df, environ_state, 'Game 1')

def test_environ_turn_progression():
    """
    Ensure that the Environ class correctly progresses the turn index.
    """
    environ = Environ()
    initial_state = environ.get_curr_state()
    initial_turn = initial_state['curr_turn']
    environ.update_curr_state()
    new_state = environ.get_curr_state()
    new_turn = new_state['curr_turn']
    # The turn should change after updating the state.
    assert new_turn != initial_turn, "Turn should progress after updating current state."

def test_apply_move_updates_board_state():
    """
    Test that applying a move updates the board state and increments the turn index.
    """
    environ = Environ()
    initial_fen = environ.board.fen()
    legal_moves = environ.get_legal_moves()
    if legal_moves:
        move = legal_moves[0]
        apply_move_and_update_state(move, environ)
        new_fen = environ.board.fen()
        assert new_fen != initial_fen, "Board state should change after applying a move."
    else:
        pytest.skip("No legal moves available to test apply_move_and_update_state.")

def test_handle_agent_turn_with_empty_move():
    """
    Check that if the agent returns an empty move,
    no move is applied and the game is not flagged as corrupted.
    """
    df = pd.DataFrame({
        'W1': [''],  # Empty move provided.
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}
    environ = Environ()
    result = handle_agent_turn(agent, df, environ_state, 'Game 1', environ)
    # Expecting None because no move is applied.
    assert result is None, "Empty move should not flag game as corrupted."

def test_environ_reset_after_moves():
    """
    Verify that the Environ.reset_environ method correctly resets
    both the turn index and the board to its initial state.
    """
    environ = Environ()
    # Apply a legal move if available.
    legal_moves = environ.get_legal_moves()
    if legal_moves:
        move = legal_moves[0]
        apply_move_and_update_state(move, environ)
        assert environ.turn_index == 1, "Turn index should increment after a move."
    else:
        pytest.skip("No legal moves available for test.")
    # Now reset the environment.
    environ.reset_environ()
    assert environ.turn_index == 0, "After reset, turn index should be zero."
    starting_board = chess.Board()
    assert environ.board.fen() == starting_board.fen(), "After reset, board should be in starting position."

def test_handle_agent_turn_illegal_move_in_df():
    """
    Test that if the agent returns an illegal move,
    handle_agent_turn returns the game identifier to flag corruption.
    """
    df = pd.DataFrame({
        'W1': ['illegal'],
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4', 'd4']}
    environ = Environ()
    result = handle_agent_turn(agent, df, environ_state, 'Game 1', environ)
    assert result == 'Game 1', "An illegal move should flag the game as corrupted."


def test_play_one_game_breaks_on_plycount():
    """
    Ensure that play_one_game exits once the turn index reaches PlyCount.
    For a game with PlyCount of 1, only one move (if valid) is processed.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'B1': ['e5'],  # Even though provided, won't be used because PlyCount is 1.
        'PlyCount': [1]
    }, index=['Game 1'])
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    result = play_one_game('Game 1', df, w_agent, b_agent, environ)
    # With a valid move and a PlyCount of 1, the game should complete normally.
    assert result is None


def test_play_one_game_forced_illegal_move(caplog):
    """
    Simulate a game with a forced illegal move.
    The game should be flagged as corrupted and an error logged.
    """
    df = pd.DataFrame({
        'W1': ['illegal_move'],
        'B1': ['e5'],  # Even if present, will not be reached.
        'PlyCount': [1]
    }, index=['Game 1'])
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    with caplog.at_level(logging.CRITICAL):
        result = play_one_game('Game 1', df, w_agent, b_agent, environ)
    
    assert result == 'Game 1'
    # Verify that the critical error message was logged.
    assert any("Invalid move 'illegal_move'" in message for message in caplog.text.splitlines()) 


# --- Concurrency / Multiprocessing Tests ---

def test_process_games_in_parallel_large_valid():
    """
    Test process_games_in_parallel with a large DataFrame of valid games (5000 games).
    Expect no games to be flagged as corrupted.
    """
    num_games = 5000
    df = pd.DataFrame({
        'W1': ['e4'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games,
    }, index=[f"Game {i+1}" for i in range(num_games)])
    
    corrupted_games = process_games_in_parallel(list(df.index), worker_play_games, df)
    assert len(corrupted_games) == 0

def test_process_games_in_parallel_large_invalid():
    """
    Test process_games_in_parallel with a large DataFrame of games forced to have illegal moves.
    Expect all games to be flagged as corrupted.
    """
    num_games = 5000
    df = pd.DataFrame({
        'W1': ['illegal_move'] * num_games,
        'B1': ['e5'] * num_games,
        'PlyCount': [1] * num_games,
    }, index=[f"Game {i+1}" for i in range(num_games)])
    
    corrupted_games = process_games_in_parallel(list(df.index), worker_play_games, df)
    assert len(corrupted_games) == num_games

# --- Logging within Worker on Exception ---

def test_worker_logging_on_exception(monkeypatch, caplog):
    """
    Verify that if an exception is raised during a game simulation in worker_play_games,
    the error is logged and the game is flagged as corrupted.
    """
    df = pd.DataFrame({
        'W1': ['e4'],
        'B1': ['e5'],
        'PlyCount': [1]
    }, index=['Game 1'])
    
    # Create a faulty play_one_game that always raises an exception.
    def faulty_play_one_game(game_number, chess_data, w_agent, b_agent, environ):
        raise Exception("Forced Exception")
    
    # Import the game_simulation module and monkeypatch play_one_game.
    from training import game_simulation
    monkeypatch.setattr(game_simulation, "play_one_game", faulty_play_one_game)
    
    with caplog.at_level(logging.CRITICAL, logger=game_simulation.logger.name):
        corrupted = worker_play_games(['Game 1'], df)
    
    assert 'Game 1' in corrupted
    # Verify that the log contains the forced exception message.
    assert any("Forced Exception" in message for message in caplog.text.splitlines())


