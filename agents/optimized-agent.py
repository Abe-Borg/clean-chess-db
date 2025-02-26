# optimized_agent.py
import logging
from typing import Union, Dict, List, Any

class OptimizedAgent:
    def __init__(self, color: str):
        self.color = color
      
    def choose_action(self, game_moves: Dict[str, str], environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        """Choose an action based on the current game state.
        
        OPTIMIZATION: Uses pre-loaded game moves dictionary instead of DataFrame lookups
        
        Args:
            game_moves: Dictionary containing moves for the current game
            environ_state: Current state of the environment
            curr_game: Identifier for the current game
            
        Returns:
            The chosen move in SAN format
        """
        # Retrieve the current turn from the state
        curr_turn = environ_state.get("curr_turn", "")
        legal_moves = environ_state.get('legal_moves', [])
        
        if not legal_moves:
            return ''
        
        # Get the move directly from the dictionary 
        # This is much faster than DataFrame.at lookup
        return game_moves.get(curr_turn, '')
