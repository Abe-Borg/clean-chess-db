# file: agents/Agent.py
from typing import Union, Dict, List, Any

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
