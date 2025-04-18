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
from training.game_simulation import play_games, warm_up_workers

logger = logging.getLogger("play_games")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.play_games_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

if __name__ == '__main__':
    start_time = time.time()
    num_dataframes_to_process = 50 + 1 # add one more for the range function

    # Create a single persistent process pool
    num_workers = max(1, cpu_count() - 1)  # Leave one CPU for the main process
    with Pool(processes=num_workers) as pool:
        # Warm up the workers using the function from fully_optimized_game_simulation
        warm_up_workers(pool, num_workers)
            
        # Create a progress bar for the main loop
        for part in tqdm(range(1, num_dataframes_to_process), desc="Processing parts", unit="part"):
            # Dynamically retrieve the file path from game_settings.
            file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
            
            try:
                chess_data = pd.read_pickle(file_path, compression='zip')
                print(f"\nPart {part}: {len(chess_data)} games in dataframe.")
                
                # Use the optimized play_games function that already accepts a pool parameter
                corrupted_games = play_games(chess_data, pool)
                
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