from typing import Callable, List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import time
import chess
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count, shared_memory
import logging
import threading

import platform
import pickle

# Determine if we're running on Windows
IS_WINDOWS = platform.system() == 'Windows'

logger = logging.getLogger("fully_optimized_game_simulation")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.game_simulation_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def init_worker(i):
    """Function to initialize worker processes."""
    return f"Worker {i} initialized"

# Global adaptive chunker for tracking game processing time
class AdaptiveChunker:
    """Tracks game processing times and creates balanced chunks."""

    def __init__(self):
        self.game_processing_times = {}
        self.lock = threading.RLock()

    def update_processing_time(self, game_id: str, processing_time: float, num_moves: int):
        """Update the processing time for a game."""
        with self.lock:
            self.game_processing_times[game_id] = {
                'time': processing_time,
                'moves': num_moves,
                'moves_per_second': num_moves / processing_time if processing_time > 0 else 0
            }

    def estimate_processing_time(self, game_id: str, ply_count: int) -> float:
        """Estimate processing time based on historical data."""
        with self.lock:
            # If we've seen this game before, use its actual time
            if game_id in self.game_processing_times:
                return self.game_processing_times[game_id]['time']
            
            # Calculate average moves per second from historical data
            if self.game_processing_times:
                total_moves = sum(data['moves'] for data in self.game_processing_times.values())
                total_time = sum(data['time'] for data in self.game_processing_times.values())
                avg_moves_per_second = total_moves / total_time if total_time > 0 else 100
                
                # Estimate time based on move count
                return ply_count / avg_moves_per_second
            
        # Default heuristic if no historical data
        if ply_count <= 20:
            return ply_count * 0.01  # Opening phase (faster)
        elif ply_count <= 40:
            return 0.2 + (ply_count - 20) * 0.015  # Middle game
        else:
            return 0.2 + 0.3 + (ply_count - 40) * 0.02  # Endgame (slower)

    def create_balanced_chunks(self, game_indices: List[str], 
                              chess_data: pd.DataFrame, 
                              num_processes: int) -> List[List[str]]:
        """Create balanced chunks based on estimated processing times."""
        # Handle edge cases
        if not game_indices:
            return []
        
        # Estimate processing time for each game
        time_estimates = {}
        for game_id in game_indices:
            try:
                ply_count = chess_data.loc[game_id, 'PlyCount']
                time_estimates[game_id] = self.estimate_processing_time(game_id, ply_count)
            except (KeyError, TypeError):
                time_estimates[game_id] = 0.3  # Default estimate
        
        # Sort games by estimated time (longest first)
        sorted_games = sorted(game_indices, key=lambda g: time_estimates.get(g, 0), reverse=True)
        
        # Create balanced chunks using greedy algorithm
        chunks = [[] for _ in range(num_processes)]
        chunk_times = [0] * num_processes
        
        for game in sorted_games:
            # Assign to chunk with lowest total time
            min_idx = chunk_times.index(min(chunk_times))
            chunks[min_idx].append(game)
            chunk_times[min_idx] += time_estimates.get(game, 0)
        
        # Print statistics for monitoring
        if chunks:
            avg_time = sum(chunk_times) / len(chunks)
            max_dev = max(abs(t - avg_time) for t in chunk_times)
            imbalance = (max_dev / avg_time * 100) if avg_time > 0 else 0
            
            print(f"Chunking statistics:")
            print(f"  Total games: {len(game_indices)}")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Est. avg time/chunk: {avg_time:.2f}s")
            print(f"  Est. imbalance: {imbalance:.1f}%")
            print(f"  Games per chunk: {[len(c) for c in chunks]}")
        
        return [chunk for chunk in chunks if chunk]

# Create a single global instance
adaptive_chunker = AdaptiveChunker()

def create_shared_data(chess_data: pd.DataFrame) -> Dict:
    """Create shared memory representation of DataFrame, with Windows compatibility."""
    if IS_WINDOWS:
        # On Windows, use serialization instead of shared memory
        return {
            'windows_mode': True,
            'data': chess_data,  # Pass the DataFrame directly
        }
    else:
        # Unix-based systems can use shared memory
        # Extract game indices as a list
        game_indices = list(chess_data.index)

        # Convert to numpy array for shared memory
        indices_arr = np.array(game_indices, dtype=object)
        indices_shm = shared_memory.SharedMemory(create=True, size=indices_arr.nbytes)
        indices_shared = np.ndarray(indices_arr.shape, dtype=indices_arr.dtype, buffer=indices_shm.buf)
        indices_shared[:] = indices_arr[:]
        
        # Create shared memory for PlyCount
        ply_counts = chess_data['PlyCount'].values
        ply_shm = shared_memory.SharedMemory(create=True, size=ply_counts.nbytes)
        ply_shared = np.ndarray(ply_counts.shape, dtype=ply_counts.dtype, buffer=ply_shm.buf)
        ply_shared[:] = ply_counts[:]
        
        # Create shared memory for move columns
        move_columns = {}
        for col in chess_data.columns:
            if col != 'PlyCount':
                col_data = chess_data[col].values
                col_shm = shared_memory.SharedMemory(create=True, size=col_data.nbytes)
                col_shared = np.ndarray(col_data.shape, dtype=col_data.dtype, buffer=col_shm.buf)
                col_shared[:] = col_data[:]
                
                # Store shared memory metadata
                move_columns[col] = {
                    'shm_name': col_shm.name,
                    'shape': col_data.shape,
                    'dtype': str(col_data.dtype)
                }
        
        # Return all shared memory information
        return {
            'windows_mode': False,
            'indices': {
                'shm_name': indices_shm.name,
                'shape': indices_arr.shape,
                'dtype': str(indices_arr.dtype)
            },
            'ply_counts': {
                'shm_name': ply_shm.name,
                'shape': ply_counts.shape,
                'dtype': str(ply_counts.dtype)
            },
            'move_columns': move_columns
        }

def cleanup_shared_data(shared_data: Dict) -> None:
    """Clean up shared memory resources."""
    # Skip cleanup for Windows mode
    if shared_data.get('windows_mode', False):
        return

    try:
        # Clean up indices shared memory
        indices_shm = shared_memory.SharedMemory(name=shared_data['indices']['shm_name'])
        indices_shm.close()
        indices_shm.unlink()
        
        # Clean up ply_counts shared memory
        ply_shm = shared_memory.SharedMemory(name=shared_data['ply_counts']['shm_name'])
        ply_shm.close()
        ply_shm.unlink()
        
        # Clean up all move column shared memory
        for col_info in shared_data['move_columns'].values():
            col_shm = shared_memory.SharedMemory(name=col_info['shm_name'])
            col_shm.close()
            col_shm.unlink()
    except FileNotFoundError:
        # Handle the case where shared memory was already cleaned up
        print("Warning: Some shared memory segments were already cleaned up")

def play_games(chess_data: pd.DataFrame, pool = None) -> List[str]:
    """Process all games in the provided DataFrame using all optimizations."""
    # Handle empty DataFrame
    if chess_data.empty:
        return []

    # Check that required columns exist
    required_columns = ['PlyCount']
    missing_columns = [col for col in required_columns if col not in chess_data.columns]
    if missing_columns:
        logger.critical(f"Missing required columns: {missing_columns}")
        return list(chess_data.index)

    start_time = time.time()

    # Create shared memory representation
    shared_data = create_shared_data(chess_data)

    try:
        # Process games in parallel
        game_indices = list(chess_data.index)
        corrupted_games = process_games_in_parallel(
            game_indices, worker_play_games_optimized, shared_data, chess_data, pool)
        
        # Print timing information
        end_time = time.time()
        elapsed = end_time - start_time
        games_per_second = len(chess_data) / elapsed
        moves_per_second = chess_data['PlyCount'].sum() / elapsed
        
        print(f"Processed {len(chess_data)} games in {elapsed:.2f}s")
        print(f"Performance: {games_per_second:.2f} games/s, {moves_per_second:.2f} moves/s")
        
        # Print cache statistics from Environ
        Environ.print_cache_stats()
        
        return corrupted_games
    finally:
        # Always clean up shared memory
        cleanup_shared_data(shared_data)

def process_games_in_parallel(
    game_indices: List[str],
    worker_function: Callable[..., List],
    shared_data: Dict,
    chess_data: pd.DataFrame,
    pool = None
) -> List[str]:
    """Process games in parallel with optimized chunking."""
    # Handle the case when there are no games
    if not game_indices:
        return []

    # Get available CPU count and ensure it's at least 1
    num_processes = max(1, min(cpu_count(), len(game_indices)))

    # Use adaptive chunking based on game complexity
    chunks = adaptive_chunker.create_balanced_chunks(game_indices, chess_data, num_processes)

    # If chunking resulted in empty list, return empty result
    if not chunks:
        return []

    # Use provided pool or create a new one
    if pool is not None:
        # Use the persistent pool
        results = pool.starmap(worker_function, [(chunk, shared_data) for chunk in chunks])
    else:
        # Create a new pool
        with Pool(processes=len(chunks)) as pool:
            results = pool.starmap(worker_function, [(chunk, shared_data) for chunk in chunks])

    # Flatten results
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

def play_one_game_optimized(
    game_number: str,
    game_data: Dict[str, Any],
    w_agent: Agent,
    b_agent: Agent,
    environ: Environ
) -> Optional[str]:
    """Process a single game with optimized data access and minimal string operations."""
    num_moves = game_data['PlyCount']
    game_moves = game_data['moves']

    # Pre-convert all SAN moves to Move objects for faster processing
    move_objects_cache = {}
    
    # Loop until we reach the number of moves or the game is over
    while True:
        # Use optimized state retrieval
        curr_state, legal_moves = environ.get_curr_state_and_legal_moves()
        
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn_optimized(
            w_agent, game_moves, curr_state, legal_moves, 
            game_number, environ, move_objects_cache
        )
        if result is not None:
            return result  # Return the game_number for a corrupted game

        # Check state again after white's move
        curr_state, legal_moves = environ.get_curr_state_and_legal_moves()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn_optimized(
            b_agent, game_moves, curr_state, legal_moves,
            game_number, environ, move_objects_cache
        )
        if result is not None:
            return result  # Return the game_number for a corrupted game

        if environ.board.is_game_over():
            break
            
    return None  # Game processed normally

def handle_agent_turn_optimized(
    agent: Agent,
    game_moves: Dict[str, str],
    curr_state: dict,
    legal_moves: List[chess.Move],
    game_number: str,
    environ: Environ,
    move_objects_cache: Dict[str, chess.Move]
) -> Optional[str]:
    """Handle a single agent turn with optimized move processing."""
    curr_turn = curr_state['curr_turn']

    # Get move from game data (dictionary lookup)
    chess_move_san = game_moves.get(curr_turn, '')

    if chess_move_san == '' or pd.isna(chess_move_san):
        return None

    # Convert SAN to Move object (with caching)
    if chess_move_san not in move_objects_cache:
        try:
            move_obj = environ.convert_san_to_move_object(chess_move_san)
            move_objects_cache[chess_move_san] = move_obj
        except ValueError:
            logger.critical(f"Invalid move format '{chess_move_san}' for game {game_number}, turn {curr_turn}.")
            return game_number
    else:
        move_obj = move_objects_cache[chess_move_san]

    # Check if move is legal (compare Move objects directly - much faster than string comparison)
    if move_obj not in legal_moves:
        logger.critical(f"Invalid move '{chess_move_san}' for game {game_number}, turn {curr_turn}.")
        return game_number

    # Apply move directly as Move object (fastest method)
    environ.push_move_object(move_obj)
    return None

def worker_play_games_optimized(game_indices_chunk: List[str], shared_data: Dict) -> List[str]:
    """Optimized worker function that processes a chunk of games using shared data."""
    corrupted_games = []

    # Create agents and environment (reused across games)
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()

    # Handle Windows mode differently
    if shared_data.get('windows_mode', False):
        # For Windows, we have the DataFrame directly
        chess_data = shared_data['data']
        
        # Process each game in the chunk
        for game_number in game_indices_chunk:
            try:
                # Get data for this game
                row = chess_data.loc[game_number]
                ply_count = int(row['PlyCount'])
                
                # Extract moves from row
                moves = {}
                for col in chess_data.columns:
                    if col != 'PlyCount':
                        moves[col] = row[col]
                
                # Store in game data dictionary
                game_data = {
                    'PlyCount': ply_count,
                    'moves': moves
                }
                
                # Process the game with timing using optimized function
                start_time = time.time()
                try:
                    result = play_one_game_optimized(game_number, game_data, w_agent, b_agent, environ)
                except Exception as e:
                    logger.critical(f"Exception processing game {game_number}: {e}")
                    result = game_number  # flag as corrupted
                    
                if result is not None:
                    corrupted_games.append(game_number)
                    
                # Reset environment for next game
                environ.reset_environ()
                
                # Update processing time statistics
                end_time = time.time()
                processing_time = end_time - start_time
                adaptive_chunker.update_processing_time(game_number, processing_time, ply_count)
                
            except (KeyError, ValueError) as e:
                # Game not found or other error
                logger.critical(f"Error accessing game {game_number}: {e}")
                corrupted_games.append(game_number)
                
        return corrupted_games
        
    # Regular shared memory approach for non-Windows systems
    # Access shared memory for game indices
    indices_shm = shared_memory.SharedMemory(name=shared_data['indices']['shm_name'])
    indices_shape = tuple(shared_data['indices']['shape'])
    indices_dtype = np.dtype(shared_data['indices']['dtype'])
    game_indices_arr = np.ndarray(indices_shape, dtype=indices_dtype, buffer=indices_shm.buf)
    game_indices_list = game_indices_arr.tolist()

    # Access shared memory for ply counts
    ply_shm = shared_memory.SharedMemory(name=shared_data['ply_counts']['shm_name'])
    ply_shape = tuple(shared_data['ply_counts']['shape'])
    ply_dtype = np.dtype(shared_data['ply_counts']['dtype'])
    ply_counts = np.ndarray(ply_shape, dtype=ply_dtype, buffer=ply_shm.buf)

    # Access shared memory for move columns
    move_columns = {}
    move_shms = {}  # Keep references to shared memory objects
    for col, col_info in shared_data['move_columns'].items():
        shm = shared_memory.SharedMemory(name=col_info['shm_name'])
        move_shms[col] = shm  # Store reference to keep alive
        col_shape = tuple(col_info['shape'])
        col_dtype = np.dtype(col_info['dtype'])
        move_columns[col] = np.ndarray(col_shape, dtype=col_dtype, buffer=shm.buf)

    # Process each game in the chunk
    for game_number in game_indices_chunk:
        # Find index in the original array
        try:
            idx = game_indices_list.index(game_number)
            
            # Get data for this game
            ply_count = int(ply_counts[idx])
            moves = {col: move_columns[col][idx] for col in move_columns}
            
            # Store in game data dictionary
            game_data = {
                'PlyCount': ply_count,
                'moves': moves
            }
            
            # Process the game with timing using optimized function
            start_time = time.time()
            try:
                result = play_one_game_optimized(game_number, game_data, w_agent, b_agent, environ)
            except Exception as e:
                logger.critical(f"Exception processing game {game_number}: {e}")
                result = game_number  # flag as corrupted
                
            if result is not None:
                corrupted_games.append(game_number)
                
            # Reset environment for next game
            environ.reset_environ()
            
            # Update processing time statistics
            end_time = time.time()
            processing_time = end_time - start_time
            adaptive_chunker.update_processing_time(game_number, processing_time, ply_count)
            
        except ValueError:
            # Game not found in indices
            logger.critical(f"Game {game_number} not found in indices")
            corrupted_games.append(game_number)

    # Clean up shared memory access
    indices_shm.close()
    ply_shm.close()
    for shm in move_shms.values():
        shm.close()

    return corrupted_games

# Keep the old functions for backward compatibility
def worker_play_games(game_indices_chunk: List[str], shared_data: Dict) -> List[str]:
    """Legacy worker function - redirects to optimized version."""
    return worker_play_games_optimized(game_indices_chunk, shared_data)

def play_one_game(
    game_number: str,
    game_data: Dict[str, Any],
    w_agent: Agent,
    b_agent: Agent,
    environ: Environ
) -> Optional[str]:
    """Legacy function - redirects to optimized version."""
    return play_one_game_optimized(game_number, game_data, w_agent, b_agent, environ)

def handle_agent_turn(
    agent: Agent,
    game_moves: Dict[str, str],
    curr_state: dict,
    game_number: str,
    environ: Environ
) -> Optional[str]:
    """Legacy function for backward compatibility."""
    curr_turn = curr_state['curr_turn']
    chess_move = game_moves.get(curr_turn, '')

    if chess_move == '' or pd.isna(chess_move):
        return None

    # Get legal moves (with caching)
    legal_moves = curr_state['legal_moves']

    if chess_move not in legal_moves:
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}.")
        return game_number

    # Apply move and update environment
    environ.board.push_san(chess_move)
    environ.update_curr_state()
    return None

def warm_up_workers(pool, num_workers):
    """Initialize worker processes with warm-up tasks."""
    print("Warming up worker processes...")
    results = pool.map(init_worker, range(num_workers))
    for result in results:
        print(result)