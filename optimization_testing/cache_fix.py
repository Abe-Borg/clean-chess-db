"""
Debug script to fix the position cache issue.
The cache shows 0 hits/0 misses, which means it's not being used at all.
"""

import chess
from typing import List, Dict
import threading

class FixedEnviron:
    """Fixed version of Environ with working position cache."""
    
    # Class-level cache for positions across all games and instances
    _global_position_cache = {}
    _global_cache_lock = threading.RLock()
    _global_cache_hits = 0
    _global_cache_misses = 0
    _max_cache_size = 100000

    # Class-level turn list to avoid recreating for each instance
    _turn_list = None

    @classmethod
    def _get_turn_list(cls):
        """Initialize the turn list once and reuse it."""
        if cls._turn_list is None:
            from utils import constants
            max_turns = constants.max_num_turns_per_player * constants.num_players
            cls._turn_list = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' 
                             for i in range(max_turns)]
        return cls._turn_list

    @classmethod
    def initialize_common_positions(cls):
        """Pre-compute legal moves for common opening positions."""
        print("Initializing position cache with common positions...")
        
        # Starting position
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        position_key = board.fen()
        
        with cls._global_cache_lock:
            cls._global_position_cache[position_key] = legal_moves
            print(f"Cached starting position with {len(legal_moves)} legal moves")
        
        # Common first moves
        common_first_moves = ['e4', 'e5', 'd4', 'd5', 'c4', 'Nf3', 'Nc3']
        cached_count = 1  # Starting position
        
        for move in common_first_moves:
            board = chess.Board()
            try:
                board.push_san(move)
                legal_moves = list(board.legal_moves)
                position_key = board.fen()
                
                with cls._global_cache_lock:
                    cls._global_position_cache[position_key] = legal_moves
                    cached_count += 1
                
                # Common responses
                common_responses = ['e5', 'e6', 'd5', 'd6', 'c5', 'c6', 'Nf6', 'Nc6']
                for response in common_responses:
                    try:
                        response_board = board.copy()
                        response_board.push_san(response)
                        legal_moves = list(response_board.legal_moves)
                        position_key = response_board.fen()
                        
                        with cls._global_cache_lock:
                            if position_key not in cls._global_position_cache:
                                cls._global_position_cache[position_key] = legal_moves
                                cached_count += 1
                    except:
                        pass
            except:
                pass
        
        print(f"Position cache initialized with {cached_count} positions")

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
            
            if total_requests == 0:
                print("  WARNING: Cache not being used! (0 hits + 0 misses)")

    def __init__(self):
        """Initialize the environment with a chess board."""
        self.board = chess.Board()
        self.turn_list = self._get_turn_list()
        self.turn_index = 0
        # Instance-level cache for recently accessed positions
        self._legal_moves_cache = None
        self._last_board_fen = None
        self._instance_cache_hits = 0

    def get_curr_state_and_legal_moves(self):
        """Get current state and legal moves in one call to avoid duplicate work."""
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = self.get_legal_moves_optimized()
        
        state = {
            'turn_index': self.turn_index, 
            'curr_turn': curr_turn,
            'white_to_move': self.board.turn == chess.WHITE,
            'castling_rights': self.board.castling_rights,
            'en_passant_square': self.board.ep_square,
            'halfmove_clock': self.board.halfmove_clock,
            'fullmove_number': self.board.fullmove_number
        }
        
        return state, legal_moves

    def get_legal_moves_optimized(self) -> List[chess.Move]:
        """Get legal moves as Move objects with FIXED caching."""
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
                self._legal_moves_cache = legal_moves.copy()  # Important: copy the list
                self._last_board_fen = current_fen
                
                return self._legal_moves_cache
        
        # Cache miss - calculate legal moves
        with self._global_cache_lock:
            self._global_cache_misses += 1
        
        legal_moves = list(self.board.legal_moves)
        
        # Update instance cache
        self._legal_moves_cache = legal_moves.copy()
        self._last_board_fen = current_fen
        
        # Update global cache if not too large
        with self._global_cache_lock:
            if len(self._global_position_cache) < self._max_cache_size:
                self._global_position_cache[current_fen] = legal_moves.copy()
            
        return self._legal_moves_cache

    def update_curr_state(self) -> None:
        """Update the current state after a move."""
        self.turn_index += 1
        # Clear instance-level cache when position changes
        self._legal_moves_cache = None
        self._last_board_fen = None

    def reset_environ(self) -> None:
        """Reset the environment to initial state."""
        self.board.reset()
        self.turn_index = 0
        self._legal_moves_cache = None
        self._last_board_fen = None

    def push_move_object(self, move: chess.Move) -> None:
        """Push a move object directly (fastest)."""
        self.board.push(move)
        self.update_curr_state()

    def convert_san_to_move_object(self, san_move: str) -> chess.Move:
        """Convert SAN string to Move object for the current position."""
        return self.board.parse_san(san_move)

def test_cache_functionality():
    """Test if the cache is working correctly."""
    print("Testing Position Cache Functionality")
    print("=" * 50)
    
    # Initialize cache
    FixedEnviron.initialize_common_positions()
    
    # Create environment
    env = FixedEnviron()
    
    print(f"\n1. Testing starting position:")
    legal_moves1 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves1)} legal moves")
    FixedEnviron.print_cache_stats()
    
    print(f"\n2. Testing same position again (should hit cache):")
    legal_moves2 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves2)} legal moves")
    FixedEnviron.print_cache_stats()
    
    print(f"\n3. Testing after a move:")
    env.board.push_san('e4')
    env.update_curr_state()
    legal_moves3 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves3)} legal moves")
    FixedEnviron.print_cache_stats()
    
    print(f"\n4. Testing common response:")
    env.board.push_san('e5')
    env.update_curr_state()
    legal_moves4 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves4)} legal moves")
    FixedEnviron.print_cache_stats()
    
    print(f"\n5. Reset and test starting position again:")
    env.reset_environ()
    legal_moves5 = env.get_legal_moves_optimized()
    print(f"   Found {len(legal_moves5)} legal moves")
    FixedEnviron.print_cache_stats()

if __name__ == "__main__":
    test_cache_functionality()