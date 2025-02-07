# Agent.py
import pandas as pd
from typing import Union, Dict, List

class Agent:
    def __init__(self, color: str):
        self.color = color
      
    def choose_action(self, chess_data: pd.DataFrame, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        legal_moves = environ_state['legal_moves']
        if not legal_moves:
            return ''
          
        return chess_data.at[curr_game, environ_state["curr_turn"]]