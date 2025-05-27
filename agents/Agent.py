# file: agents/Agent.py

from typing import Union, Dict, List
import chess

class Agent:
    def __init__(self, color: str):
        self.color = color

    def choose_action(self, game_moves: Dict[str, str], environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        """Choose an action based on the current game state.        
        Args:
            game_moves: Dictionary containing moves for the current game
            environ_state: Current state of the environment
            curr_game: Identifier for the current game
        Returns:
            The chosen move in SAN format
        """
        curr_turn = environ_state.get("curr_turn", "")
        legal_moves = environ_state.get('legal_moves', [])
        
        if not legal_moves:
            return ''
        return game_moves.get(curr_turn, '')
    
    def choose_action_optimized(self, game_moves: Dict[str, str], curr_state: Dict[str, Union[int, str]], 
                               legal_moves: List[chess.Move], curr_game: str = 'Game 1') -> str:
        """Optimized action selection that works with Move objects.
        
        Args:
            game_moves: Dictionary containing moves for the current game
            curr_state: Current state of the environment (without expensive legal_moves list)
            legal_moves: Pre-computed legal moves as Move objects
            curr_game: Identifier for the current game
        Returns:
            The chosen move in SAN format
        """
        curr_turn = curr_state.get("curr_turn", "")
        
        if not legal_moves:
            return ''
        return game_moves.get(curr_turn, '')