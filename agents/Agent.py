# Agent.py
import logging
import pandas as pd
from typing import Union, Dict, List

class Agent:
    def __init__(self, color: str):
        self.color = color
      
    def choose_action(self, chess_data: pd.DataFrame, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        # Retrieve the current turn from the state.
        curr_turn = environ_state.get("curr_turn", "")
        legal_moves = environ_state.get('legal_moves', [])
        if not legal_moves:
            return ''
        
        # Explicitly check for the existence of the expected column.
        if curr_turn not in chess_data.columns:
            raise KeyError(f"Column {curr_turn} missing in chess_data")
        
        return chess_data.at[curr_game, curr_turn]