# file: environment/Environ.py

import chess
from utils import constants
from typing import Union, Dict, List

class Environ:
    # Class variable for turn list to avoid recreating it for each instance
    _turn_list = None
    
    @classmethod
    def _get_turn_list(cls):
        """Initialize the turn list once and reuse it."""
        if cls._turn_list is None:
            max_turns = constants.max_num_turns_per_player * constants.num_players
            cls._turn_list = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' 
                             for i in range(max_turns)]
        return cls._turn_list

    def __init__(self):
        """Initialize the environment with a chess board."""
        self.board = chess.Board()
        self.turn_list = self._get_turn_list()
        self.turn_index = 0
        # Cache for legal moves to avoid recalculation
        self._legal_moves_cache = None
        self._last_board_fen = None

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        """Get the current state of the environment."""
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    
    def update_curr_state(self) -> None:
        """Update the current state after a move."""
        self.turn_index += 1
        # Invalidate legal moves cache
        self._legal_moves_cache = None
    
    def get_curr_turn(self) -> str:
        """Get the current turn identifier."""
        return self.turn_list[self.turn_index]
    
    def reset_environ(self) -> None:
        """Reset the environment to initial state."""
        self.board.reset()
        self.turn_index = 0
        self._legal_moves_cache = None
        self._last_board_fen = None
    
    def get_legal_moves(self) -> List[str]:
        """Get legal moves for the current board position with caching."""
        current_fen = self.board.fen()
        
        # If we already calculated legal moves for this position, return cached result
        if self._legal_moves_cache is not None and self._last_board_fen == current_fen:
            return self._legal_moves_cache
        
        # Calculate legal moves
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        # Cache result
        self._legal_moves_cache = legal_moves
        self._last_board_fen = current_fen
        
        return legal_moves
