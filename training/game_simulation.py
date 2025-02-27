# file: training/game_simulation.py

from typing import Callable, List, Dict, Any
import pandas as pd
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger("optimized_game_simulation")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.game_simulation_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def play_games(chess_data: pd.DataFrame) -> List[str]:
    """Process all games in the provided DataFrame.
    
    Args:
        chess_data: DataFrame containing chess games
        
    Returns:
        List of corrupted game identifiers
    """
    # Handle empty DataFrame
    if chess_data.empty:
        return []
    
    # Check that required columns exist
    required_columns = ['PlyCount']
    missing_columns = [col for col in required_columns if col not in chess_data.columns]
    if missing_columns:
        logger.critical(f"Missing required columns: {missing_columns}")
        # Return all game indices as corrupted if required columns are missing
        return list(chess_data.index)
    
    game_indices = list(chess_data.index)
    return process_games_in_parallel(game_indices, worker_play_games, chess_data)

def process_games_in_parallel(game_indices: List[str], worker_function: Callable[..., List], *args) -> List[str]:
    """Process games in parallel using multiprocessing.
    
    Args:
        game_indices: List of game identifiers to process
        worker_function: Function to process each chunk of games
        *args: Additional arguments to pass to the worker function
        
    Returns:
        List of corrupted game identifiers
    """
    # Handle the case when there are no games
    if not game_indices:
        return []
    
    # Get available CPU count and ensure it's at least 1
    num_processes = max(1, min(cpu_count(), len(game_indices)))
    
    # Optimize chunk size - aim for at least 100 games per process or more
    target_chunk_size = max(100, len(game_indices) // num_processes)
    chunks = chunkify(game_indices, num_processes, target_chunk_size)
    
    # If chunking resulted in empty list, return empty result
    if not chunks:
        return []
    
    # Use a smaller number of processes if we have fewer chunks
    num_processes = min(num_processes, len(chunks))
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

def worker_play_games(game_indices_chunk: List[str], chess_data: pd.DataFrame) -> List[str]:
    """Worker function that processes a chunk of games.
    
    OPTIMIZATION: Preload all game data into dictionaries to avoid DataFrame lookups
    """
    corrupted_games = []
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    # Preload game data for all games in this chunk
    # This is a major optimization to avoid repeated DataFrame lookups
    games_data = {}
    for game_number in game_indices_chunk:
        game_row = chess_data.loc[game_number]
        games_data[game_number] = {
            'PlyCount': game_row['PlyCount'],
            'moves': {col: game_row[col] for col in game_row.index if col != 'PlyCount'}
        }

    for game_number in game_indices_chunk:
        try:
            result = play_one_game(game_number, games_data[game_number], w_agent, b_agent, environ)
        except Exception as e:
            logger.critical(f"Exception processing game {game_number}: {e}")
            result = game_number  # flag as corrupted

        if result is not None:
            corrupted_games.append(game_number)

        environ.reset_environ()
    return corrupted_games

def play_one_game(game_number: str, game_data: Dict[str, Any], w_agent: Agent, b_agent: Agent, environ: Environ) -> str:
    """Process a single game with preloaded data.
    
    OPTIMIZATION: Using preloaded game data instead of DataFrame lookups
    """
    num_moves = game_data['PlyCount']
    game_moves = game_data['moves']
    
    # Loop until we reach the number of moves or the game is over.
    while True:
        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn(w_agent, game_moves, curr_state, game_number, environ)
        if result is not None:
            return result  # Return the game_number for a corrupted game

        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn(b_agent, game_moves, curr_state, game_number, environ)
        if result is not None:
            return result  # Return the game_number for a corrupted game

        if environ.board.is_game_over():
            break
            
    return None  # Game processed normally

def handle_agent_turn(agent: Agent, game_moves: Dict[str, str], curr_state: dict, game_number: str, environ: Environ) -> str:
    """Handle a single agent turn with optimized data access."""
    curr_turn = curr_state['curr_turn']
    
    # OPTIMIZATION: Using dictionary lookup instead of DataFrame.at
    chess_move = game_moves.get(curr_turn, '')
    
    if chess_move == '' or pd.isna(chess_move):
        return None
    
    legal_moves = environ.get_legal_moves()
    if chess_move not in legal_moves:
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. ")
        return game_number
    
    apply_move_and_update_state(chess_move, environ)
    return None

def apply_move_and_update_state(chess_move: str, environ) -> None:
    """Apply a move to the board and update the environment state."""
    environ.board.push_san(chess_move)
    environ.update_curr_state()

def chunkify(lst, n, target_chunk_size=None):
    """Split a list into chunks with optimized size.
    
    Args:
        lst: The list to split
        n: The target number of chunks
        target_chunk_size: Optional target size for each chunk
        
    Returns:
        A list of chunks
    """
    # Handle edge cases
    if not lst:
        return []
    
    if n <= 0:
        return [lst] if lst else []
    
    # If target chunk size is specified, use it to determine number of chunks
    if target_chunk_size:
        n = max(1, len(lst) // target_chunk_size)
    
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        if start < len(lst):  # Only add non-empty chunks
            chunks.append(lst[start:end])
        start = end
        
    return chunks
