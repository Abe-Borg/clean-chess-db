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
from tqdm import tqdm

# Import from fully_optimized_game_simulation instead of game_simulation
from training.game_simulation import play_games, play_games_optimized, warm_up_workers

logger = logging.getLogger("play_games")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.play_games_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def create_persistent_pool(num_workers=None):
    """Create a persistent process pool with optimal worker count."""
    if num_workers is None:
        # Leave one CPU for the main process, but ensure at least 1 worker
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Creating persistent pool with {num_workers} workers...")
    pool = Pool(processes=num_workers)
    
    # Warm up the workers
    print("Warming up worker processes...")
    warm_up_workers(pool, num_workers)
    print("Worker pool ready!")
    
    return pool

if __name__ == '__main__':
    start_time = time.time()
    num_dataframes_to_process = 50 + 1 # add one more for the range function

    # Create a single persistent process pool for all processing
    persistent_pool = create_persistent_pool()

    # Create a single persistent process pool
    try:   
        # Create a progress bar for the main loop
        for part in tqdm(range(1, num_dataframes_to_process), desc="Processing parts", unit="part"):
            # Dynamically retrieve the file path from game_settings.
            file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
            
            try:
                part_start_time = time.time()
                chess_data = pd.read_pickle(file_path, compression='zip')
                print(f"\nPart {part}: {len(chess_data)} games in dataframe.")
                
                # Use the optimized play_games function that already accepts a pool parameter
                corrupted_games = play_games_optimized(chess_data, persistent_pool)

                part_end_time = time.time()
                part_elapsed = part_end_time - part_start_time

                print(f"Part {part}: {len(corrupted_games)} corrupted games detected.")
                print(f"Part {part}: Completed in {part_elapsed:.2f}s")
                
                # chess_data = chess_data.drop(corrupted_games)
                # chess_data.to_pickle(file_path, compression='zip')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                logger.critical(f'db cleanup interrupted because of:  {e}')
                logger.critical(traceback.format_exc())
                print(f'continuing to next part...')
                continue

    finally: 
        # Always clean up the pool
        print("\nCleaning up worker pool...")
        persistent_pool.close()  # No more tasks
        persistent_pool.join()   # Wait for workers to finish
        print("Worker pool cleaned up.")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f'\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)')

    print("\nOverall Performance Summary:")
    print(f"  Processed {num_dataframes_to_process-1} data files")
    print(f"  Average time per file: {total_time/(num_dataframes_to_process-1):.2f}s")
    print(f"  Pool creation overhead eliminated through reuse")