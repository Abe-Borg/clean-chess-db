import chess
from utils import constants
from typing import Union, Dict, List, Tuple
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
        legal_moves = list(board.legal_moves)  # Store Move objects directly
        with cls._global_cache_lock:
            cls._global_position_cache[board.fen()] = legal_moves
        
        # Common first moves
        common_first_moves = ['e4', 'e5', 'd4', 'd5', 'c4', 'Nf3', 'Nc3']
        for move in common_first_moves:
            board = chess.Board()
            try:
                board.push_san(move)
                legal_moves = list(board.legal_moves)  # Store Move objects
                with cls._global_cache_lock:
                    cls._global_position_cache[board.fen()] = legal_moves
                
                # Common responses
                common_responses = ['e5', 'e6', 'd5', 'd6', 'c5', 'c6', 'Nf6', 'Nc6']
                for response in common_responses:
                    try:
                        response_board = board.copy()
                        response_board.push_san(response)
                        legal_moves = list(response_board.legal_moves)
                        with cls._global_cache_lock:
                            cls._global_position_cache[response_board.fen()] = legal_moves
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

    def get_curr_state_and_legal_moves(self) -> Tuple[Dict[str, Union[int, str]], List[chess.Move]]:
        """Get current state and legal moves in one call to avoid duplicate work."""
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = self.get_legal_moves_optimized()
        
        # Return structured state without string conversions
        state = {
            'turn_index': self.turn_index, 
            'curr_turn': curr_turn,
        }
        
        return state, legal_moves

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        """Legacy method - get the current state of the environment with SAN conversion."""
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = self.get_legal_moves()  # This still converts to SAN for backward compatibility
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

    def get_legal_moves_optimized(self) -> List[chess.Move]:
        """Get legal moves as Move objects (fastest - no string conversion)."""
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
        
        legal_moves = list(self.board.legal_moves)  # Store as Move objects
        
        # Update instance cache
        self._legal_moves_cache = legal_moves
        self._last_board_fen = current_fen
        
        # Update global cache if not too large
        with self._global_cache_lock:
            if len(self._global_position_cache) < self._max_cache_size:
                self._global_position_cache[current_fen] = legal_moves
            
        return legal_moves

    def get_legal_moves(self) -> List[str]:
        """Legacy method - get legal moves as SAN strings (slower due to string conversion)."""
        legal_move_objects = self.get_legal_moves_optimized()
        return [self.board.san(move) for move in legal_move_objects]

    def push_move_object(self, move: chess.Move) -> None:
        """Push a move object directly (fastest)."""
        self.board.push(move)
        self.update_curr_state()

    def push_uci_move(self, uci_move: str) -> None:
        """Push a move from UCI format (e.g. 'e2e4')."""
        move = chess.Move.from_uci(uci_move)
        self.push_move_object(move)

    def convert_san_to_move_object(self, san_move: str) -> chess.Move:
        """Convert SAN string to Move object for the current position."""
        return self.board.parse_san(san_move)

# Initialize common positions at module import time
Environ.initialize_common_positions()