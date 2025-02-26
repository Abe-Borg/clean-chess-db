# fully_optimized.py

from typing import Callable, List, Dict, Any, Optional, Union
import pandas as pd
import chess
from utils import game_settings, constants
from multiprocessing import Pool, cpu_count, shared_memory
import logging
import numpy as np
import os
import time

logger = logging.getLogger("fully_optimized")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.game_simulation_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Global process pool for reuse across multiple DataFrame processing tasks
_process_pool = None

def get_process_pool(num_processes=None):
    """Get or create a global process pool."""
    global _process_pool
    if _process_pool is None:
        if num_processes is None:
            num_processes = max(1, cpu_count())
        _process_pool = Pool(processes=num_processes)
    return _process_pool

def close_process_pool():
    """Close the global process pool."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.close()
        _process_pool.join()
        _process_pool = None

# -----------------------------------------------------------------------------
# Optimized Agent class
# -----------------------------------------------------------------------------

class OptimizedAgent:
    """Agent that selects moves using preloaded game data."""
    
    def __init__(self, color: str):
        self.color = color
    
    def choose_action(self, game_moves: Dict[str, str], environ_state: Dict[str, Any]) -> str:
        """Choose an action based on the current game state."""
        curr_turn = environ_state.get("curr_turn", "")
        if not environ_state.get('legal_moves', []):
            return ''
        
        # Get move directly from the dictionary
        return game_moves.get(curr_turn, '')

# -----------------------------------------------------------------------------
# Optimized Environment class
# -----------------------------------------------------------------------------

class OptimizedEnviron:
    """Optimized environment for chess game simulation."""
    
    # Class-level cache for turn list to avoid recreation
    _turn_list = None
    
    @classmethod
    def _get_turn_list(cls):
        """Get or create the turn list."""
        if cls._turn_list is None:
            max_turns = constants.max_num_turns_per_player * constants.num_players
            cls._turn_list = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' 
                             for i in range(max_turns)]
        return cls._turn_list
    
    def __init__(self):
        """Initialize the environment."""
        self.board = chess.Board()
        self.turn_list = self._get_turn_list()
        self.turn_index = 0
        self._legal_moves_cache = None
        self._last_board_fen = None
    
    def get_curr_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        return {
            'turn_index': self.turn_index,
            'curr_turn': self.turn_list[self.turn_index],
            'legal_moves': self.get_legal_moves()
        }
    
    def update_curr_state(self) -> None:
        """Update the environment state after a move."""
        self.turn_index += 1
        self._legal_moves_cache = None  # Invalidate cache
    
    def reset_environ(self) -> None:
        """Reset the environment to initial state."""
        self.board.reset()
        self.turn_index = 0
        self._legal_moves_cache = None
        self._last_board_fen = None
    
    def get_legal_moves(self) -> List[str]:
        """Get legal moves with caching for performance."""
        current_fen = self.board.fen()
        
        if self._legal_moves_cache is not None and self._last_board_fen == current_fen:
            return self._legal_moves_cache
        
        # Calculate and cache legal moves
        self._legal_moves_cache = [self.board.san(move) for move in self.board.legal_moves]
        self._last_board_fen = current_fen
        return self._legal_moves_cache

# -----------------------------------------------------------------------------
# Object Pool for reusing chess boards
# -----------------------------------------------------------------------------

class ChessBoardPool:
    """Pool of chess board objects for reuse."""
    
    def __init__(self, pool_size=100):
        """Initialize the pool with board objects."""
        self.pool = [chess.Board() for _ in range(pool_size)]
        self.in_use = set()
    
    def get_board(self):
        """Get a board from the pool or create a new one if needed."""
        if not self.pool:
            return chess.Board()  # Create new if pool is empty
        
        board = self.pool.pop()
        self.in_use.add(board)
        return board
    
    def return_board(self, board):
        """Return a board to the pool after use."""
        if board in self.in_use:
            self.in_use.remove(board)
            board.reset()  # Reset to initial state
            self.pool.append(board)

# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------

def play_games(chess_data: pd.DataFrame) -> List[str]:
    """Process all games in the provided DataFrame."""
    if chess_data.empty:
        return []
    
    # Check required columns
    if 'PlyCount' not in chess_data.columns:
        logger.critical("Missing required column: PlyCount")
        return list(chess_data.index)
    
    # Process games in parallel
    game_indices = list(chess_data.index)
    
    # Optimize batch size based on game complexity
    avg_ply_count = chess_data['PlyCount'].mean()
    optimal_batch_size = max(50, int(10000 / avg_ply_count))
    
    # Preprocess the DataFrame for faster worker access
    preprocessed_data = preprocess_dataframe(chess_data)
    
    return process_games_in_parallel(game_indices, worker_play_games, preprocessed_data, optimal_batch_size)

def preprocess_dataframe(chess_data: pd.DataFrame) -> Dict[str, Dict]:
    """Convert DataFrame to optimized dictionary structure."""
    preprocessed = {}
    
    # Convert to dictionary of dictionaries for fast access
    for game_id, row in chess_data.iterrows():
        game_dict = {
            'PlyCount': row['PlyCount'],
            'moves': {col: row[col] for col in row.index if col != 'PlyCount'}
        }
        preprocessed[game_id] = game_dict
    
    return preprocessed

def process_games_in_parallel(
    game_indices: List[str], 
    worker_function: Callable, 
    preprocessed_data: Dict[str, Dict],
    batch_size: int = 100
) -> List[str]:
    """Process games in parallel with optimized batching."""
    if not game_indices:
        return []
    
    # Determine optimal number of processes and batch size
    num_processes = max(1, min(cpu_count(), len(game_indices) // batch_size + 1))
    chunks = chunkify(game_indices, num_processes, batch_size)
    
    if not chunks:
        return []
    
    # Adjust process count if we have fewer chunks
    num_processes = min(num_processes, len(chunks))
    
    # Use the process pool
    pool = get_process_pool(num_processes)
    
    start_time = time.time()
    results = pool.starmap(worker_function, [(chunk, preprocessed_data) for chunk in chunks])
    elapsed = time.time() - start_time
    
    # Log performance metrics
    total_games = len(game_indices)
    games_per_second = total_games / elapsed
    print(f"Processed {total_games} games in {elapsed:.2f}s ({games_per_second:.2f} games/s)")
    
    # Flatten results
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

def worker_play_games(game_indices_chunk: List[str], preprocessed_data: Dict[str, Dict]) -> List[str]:
    """Worker function for processing games."""
    corrupted_games = []
    w_agent = OptimizedAgent('W')
    b_agent = OptimizedAgent('B')
    environ = OptimizedEnviron()
    
    for game_number in game_indices_chunk:
        try:
            game_data = preprocessed_data[game_number]
            result = play_one_game(game_number, game_data, w_agent, b_agent, environ)
        except Exception as e:
            logger.critical(f"Exception processing game {game_number}: {e}")
            result = game_number  # flag as corrupted

        if result is not None:
            corrupted_games.append(game_number)

        environ.reset_environ()
    
    return corrupted_games

def play_one_game(
    game_number: str, 
    game_data: Dict[str, Any], 
    w_agent: OptimizedAgent, 
    b_agent: OptimizedAgent, 
    environ: OptimizedEnviron
) -> Optional[str]:
    """Process a single game with optimized data access."""
    num_moves = game_data['PlyCount']
    game_moves = game_data['moves']
    
    # Main game loop
    while True:
        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        # Handle white's turn
        result = handle_agent_turn(w_agent, game_moves, curr_state, game_number, environ)
        if result:
            return result

        # Check game state after white's move
        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves or environ.board.is_game_over():
            break

        # Handle black's turn
        result = handle_agent_turn(b_agent, game_moves, curr_state, game_number, environ)
        if result:
            return result
        
        if environ.board.is_game_over():
            break
    
    return None  # Game processed successfully

def handle_agent_turn(
    agent: OptimizedAgent, 
    game_moves: Dict[str, str], 
    curr_state: Dict[str, Any], 
    game_number: str, 
    environ: OptimizedEnviron
) -> Optional[str]:
    """Handle a single agent turn with optimized access patterns."""
    curr_turn = curr_state['curr_turn']
    
    # Get move with optimized dictionary lookup
    chess_move = agent.choose_action(game_moves, curr_state)
    
    if chess_move == '' or pd.isna(chess_move):
        return None
    
    # Check if move is legal
    legal_moves = curr_state['legal_moves']
    if chess_move not in legal_moves:
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}")
        return game_number
    
    # Apply the move
    environ.board.push_san(chess_move)
    environ.update_curr_state()
    return None

def chunkify(lst, n, target_chunk_size=None):
    """Split a list into chunks with optimized size."""
    if not lst:
        return []
    
    # Determine chunk size based on target or divide evenly
    if target_chunk_size:
        # Use target chunk size but ensure we don't create too many chunks
        chunk_size = max(target_chunk_size, len(lst) // n)
        num_chunks = (len(lst) + chunk_size - 1) // chunk_size
    else:
        num_chunks = min(n, len(lst))
        chunk_size = len(lst) // num_chunks
    
    # Create chunks
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    
    return chunks

# -----------------------------------------------------------------------------
# Entry point for processing multiple DataFrames
# -----------------------------------------------------------------------------

def process_multiple_dataframes(file_paths, sample_size=None):
    """Process multiple DataFrames with a persistent process pool."""
    corrupted_games_total = 0
    total_games = 0
    start_time = time.time()
    
    try:
        # Create a process pool once for all DataFrames
        pool = get_process_pool()
        
        for file_path in file_paths:
            try:
                # Load the DataFrame
                print(f"\nProcessing: {file_path}")
                chess_data = pd.read_pickle(file_path, compression='zip')
                
                # Use a sample if specified
                if sample_size and sample_size < len(chess_data):
                    chess_data = chess_data.sample(sample_size, random_state=42)
                    print(f"Using sample of {sample_size} games")
                
                print(f"DataFrame size: {len(chess_data)} games")
                
                # Process the DataFrame
                corrupted_games = play_games(chess_data)
                
                # Update statistics
                corrupted_games_total += len(corrupted_games)
                total_games += len(chess_data)
                
                print(f"Corrupted games: {len(corrupted_games)}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                logger.critical(f'Error processing {file_path}: {e}')
    
    finally:
        # Close the process pool when done
        close_process_pool()
    
    # Print final statistics
    elapsed = time.time() - start_time
    print("\nProcessing complete!")
    print(f"Total games processed: {total_games}")
    print(f"Total corrupted games: {corrupted_games_total}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average performance: {total_games / elapsed:.2f} games/second")
    
    return {
        'total_games': total_games,
        'corrupted_games': corrupted_games_total,
        'elapsed_time': elapsed,
        'games_per_second': total_games / elapsed
    }

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils import game_settings
    
    # Default to processing part 1 if no arguments provided
    if len(sys.argv) > 1:
        num_parts = int(sys.argv[1])
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        num_parts = 1
        sample_size = None
    
    # Get file paths
    file_paths = []
    for part in range(1, num_parts + 1):
        attr_name = f'chess_games_filepath_part_{part}'
        if hasattr(game_settings, attr_name):
            file_path = getattr(game_settings, attr_name)
            file_paths.append(file_path)
    
    # Process the DataFrames
    results = process_multiple_dataframes(file_paths, sample_size)
    
    # Print final results
    print("\nPerformance Summary:")
    print(f"Games/second: {results['games_per_second']:.2f}")