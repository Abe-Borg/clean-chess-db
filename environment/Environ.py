# file: environment/Environ.py

import chess
from utils import constants
from typing import Union, Dict, List
import functools
import threading

class Environ:
    # Class-level cache for positions across all games and instances
    _global_position_cache = {}
    _global_cache_lock = threading.RLock()  # Thread-safe cache access
    _global_cache_hits = 0
    _global_cache_misses = 0
    _max_cache_size = 100000  # Limit cache size to prevent excessive memory usage
    
    # Class-level turn list to avoid recreating for each instance
    _turn_list = None
    
    @classmethod
    def _get_turn_list(cls):
        """Initialize the turn list once and reuse it."""
        if cls._turn_list is None:
            max_turns = constants.max_num_turns_per_player * constants.num_players
            cls._turn_list = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' 
                             for i in range(max_turns)]
        return cls._turn_list
    
    @classmethod
    def initialize_common_positions(cls):
        """Pre-compute legal moves for common opening positions."""
        # Starting position
        board = chess.Board()
        with cls._global_cache_lock:
            cls._global_position_cache[board.fen()] = [board.san(move) for move in board.legal_moves]
        
        # Common first moves
        common_first_moves = ['e4', 'e5', 'd4', 'd5', 'c4', 'Nf3', 'Nc3']
        for move in common_first_moves:
            board = chess.Board()
            try:
                board.push_san(move)
                with cls._global_cache_lock:
                    cls._global_position_cache[board.fen()] = [board.san(m) for m in board.legal_moves]
                
                # Common responses
                common_responses = ['e5', 'e6', 'd5', 'd6', 'c5', 'c6', 'Nf6', 'Nc6']
                for response in common_responses:
                    try:
                        response_board = board.copy()
                        response_board.push_san(response)
                        with cls._global_cache_lock:
                            cls._global_position_cache[response_board.fen()] = [
                                response_board.san(m) for m in response_board.legal_moves
                            ]
                    except:
                        # Skip invalid responses for this position
                        pass
            except:
                # Skip if move is invalid
                pass
    
    @classmethod
    def print_cache_stats(cls):
        """Print statistics about the global cache usage."""
        with cls._global_cache_lock:
            total_requests = cls._global_cache_hits + cls._global_cache_misses
            hit_rate = cls._global_cache_hits / total_requests * 100 if total_requests > 0 else 0
            print(f"Position cache statistics:")
            print(f"  Cache size: {len(cls._global_position_cache)} positions")
            print(f"  Cache hits: {cls._global_cache_hits}")
            print(f"  Cache misses: {cls._global_cache_misses}")
            print(f"  Hit rate: {hit_rate:.1f}%")
    
    def __init__(self):
        """Initialize the environment with a chess board."""
        self.board = chess.Board()
        self.turn_list = self._get_turn_list()
        self.turn_index = 0
        # Instance-level cache for recently accessed positions
        self._legal_moves_cache = None
        self._last_board_fen = None
        # Track cache hits for this instance
        self._instance_cache_hits = 0

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        """Get the current state of the environment."""
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    
    def update_curr_state(self) -> None:
        """Update the current state after a move."""
        self.turn_index += 1
        # Invalidate instance-level cache
        self._legal_moves_cache = None
        self._last_board_fen = None
    
    def get_curr_turn(self) -> str:
        """Get the current turn identifier."""
        return self.turn_list[self.turn_index]
    
    def reset_environ(self) -> None:
        """Reset the environment to initial state."""
        self.board.reset()
        self.turn_index = 0
        self._legal_moves_cache = None
        self._last_board_fen = None
    
    @functools.lru_cache(maxsize=8)  # Small LRU cache for very recent positions
    def _calculate_legal_moves(self, fen: str) -> List[str]:
        """Calculate legal moves for a position (used for caching)."""
        # Create a temporary board with this position
        board = chess.Board(fen)
        return [board.san(move) for move in board.legal_moves]
    
    def get_legal_moves(self) -> List[str]:
        """Get legal moves for the current board position with enhanced caching."""
        current_fen = self.board.fen()
        
        # Check instance-level cache first (fastest)
        if self._legal_moves_cache is not None and self._last_board_fen == current_fen:
            self._instance_cache_hits += 1
            return self._legal_moves_cache
        
        # Check global position cache next
        with self._global_cache_lock:
            if current_fen in self._global_position_cache:
                legal_moves = self._global_position_cache[current_fen]
                self._global_cache_hits += 1
                
                # Update instance cache
                self._legal_moves_cache = legal_moves
                self._last_board_fen = current_fen
                
                return legal_moves
        
        # Cache miss - calculate legal moves
        with self._global_cache_lock:
            self._global_cache_misses += 1
        
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        # Update instance cache
        self._legal_moves_cache = legal_moves
        self._last_board_fen = current_fen
        
        # Update global cache if not too large
        with self._global_cache_lock:
            if len(self._global_position_cache) < self._max_cache_size:
                self._global_position_cache[current_fen] = legal_moves
            
        return legal_moves

# Initialize common positions at module import time
Environ.initialize_common_positions()