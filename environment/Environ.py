# Environ.py

import chess
from utils import constants
from typing import Union, Dict, List

class Environ:
    def __init__(self):
        self.board: chess.Board = chess.Board()            
        max_turns = constants.max_num_turns_per_player * constants.num_players
        self.turn_list: List[str] = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' for i in range(max_turns)]
        self.turn_index: int = 0

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    
    def update_curr_state(self) -> None:   
        self.turn_index += 1
    
    def get_curr_turn(self) -> str:                        
        return self.turn_list[self.turn_index]
    
    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    
    def get_legal_moves(self) -> List[str]:
        return [self.board.san(move) for move in self.board.legal_moves]