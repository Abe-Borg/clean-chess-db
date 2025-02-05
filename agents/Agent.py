# Agent.py
import pandas as pd
import numpy as np
from typing import Union, Dict, List

class Agent:
    def __init__(self, color: str):
        self.color = color
      
    def choose_action(self, chess_data: pd.DataFrame, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        legal_moves = environ_state['legal_moves']
        if not legal_moves:
            return ''
          
        return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    
    def policy_training_mode(self, chess_data: pd.DataFrame, curr_game: str, curr_turn: str) -> str:
        chess_move = chess_data.at[curr_game, curr_turn]
        return chess_move