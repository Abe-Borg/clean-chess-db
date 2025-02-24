# test_game_simulation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.game_simulation import apply_move_and_update_state
from environment.Environ import Environ
import pytest

# Unit test to check if after applying a move, the turn index increments and the board updates
def test_apply_move_and_update_state():
    environ = Environ()
    initial_turn_index = environ.turn_index
    initial_fen = environ.board.fen()

    # Let's just pick the first legal move
    move = environ.get_legal_moves()[0]

    # Apply move and check turn index and board FEN
    apply_move_and_update_state(move, environ)

    # Turn index should now be incremented
    assert environ.turn_index == initial_turn_index + 1

    # The FEN of the board should differ after the move
    assert environ.board.fen() != initial_fen
