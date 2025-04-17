# file: main/play_games.py

import numpy as np
import pandas as pd
import time
import sys
import os
from multiprocessing import Pool, cpu_count
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from training.game_simulation import play_games
from tqdm import tqdm
from training.game_simulation import process_games_in_parallel, worker_play_games, chunkify


logger = logging.getLogger("play_games")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.play_games_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def play_games_with_pool(chess_data, pool=None):
    """Wrapper that passes the pool to the actual play_games function"""
    # Handle empty DataFrame
    if chess_data.empty:
        return []
    
    # Check that required columns exist
    required_columns = ['PlyCount']
    missing_columns = [col for col in required_columns if col not in chess_data.columns]
    if missing_columns:
        # Return all game indices as corrupted if required columns are missing
        return list(chess_data.index)
    
    game_indices = list(chess_data.index)
    
    # Pass the pool to process_games_in_parallel
    return process_games_in_parallel_with_pool(game_indices, worker_play_games, chess_data, pool)


def process_games_in_parallel_with_pool(game_indices, worker_function, chess_data, external_pool=None):
    """Version of process_games_in_parallel that uses an external pool if provided"""
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
    
    # Use provided external pool if available
    if external_pool is not None:
        results = external_pool.starmap(worker_function, [(chunk, chess_data) for chunk in chunks])
    else:
        # Fall back to creating a new pool if none provided
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(worker_function, [(chunk, chess_data) for chunk in chunks])
    
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

# Implement worker warmup function
def warmup_worker(worker_id):
    """Initialize a worker process with common imports and operations"""
    # Import necessary modules
    import chess
    from agents.Agent import Agent
    from environment.Environ import Environ
    
    # Initialize common objects
    environ = Environ()
    w_agent = Agent('W')
    b_agent = Agent('B')
    
    # Perform some light calculations to ensure initialization
    board = chess.Board()
    _ = [board.san(move) for move in board.legal_moves]
    
    return f"Worker {worker_id} initialized"


if __name__ == '__main__':
    start_time = time.time()
    num_dataframes_to_process = 50 + 1 # add one more for the range function

    # Create a single persistent process pool
    num_workers = max(1, cpu_count() - 1)  # Leave one CPU for the main process
    with Pool(processes=num_workers) as pool:
        # Warm up the workers
        print("Initializing worker processes...")
        warmup_results = pool.map(warmup_worker, range(num_workers))
        for result in warmup_results:
            print(result)
            
        # Create a progress bar for the main loop
        for part in tqdm(range(1, num_dataframes_to_process), desc="Processing parts", unit="part"):
            # Dynamically retrieve the file path from game_settings.
            file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
            
            try:
                chess_data = pd.read_pickle(file_path, compression='zip')
                print(f"\nPart {part}: {len(chess_data)} games in dataframe.")
                
                # Use the persistent pool for this DataFrame
                corrupted_games = play_games_with_pool(chess_data, pool)
                
                print(f"Part {part}: {len(corrupted_games)} corrupted games detected.")
                # chess_data = chess_data.drop(corrupted_games)
                # chess_data.to_pickle(file_path, compression='zip')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                logger.critical(f'db cleanup interrupted because of:  {e}')
                logger.critical(traceback.format_exc())
                exit(1)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f'total time: {total_time} seconds')

