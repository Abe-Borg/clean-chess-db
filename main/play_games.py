# play_games.py

import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from training.game_simulation.py import play_games


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
    num_games_to_play = 2 + 1 # add one more for the range function

    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    # chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression='zip')
    # chess_data = chess_data.head(10000)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for part in range(1, num_games_to_play):
        # Dynamically retrieve the file path from game_settings.
        file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
        
        try:
            chess_data = pd.read_pickle(file_path, compression='zip')
            chess_data = chess_data.head(1000)
            corrupted_games = play_games(chess_data)           
            chess_data = chess_data.drop(corrupted_games)
            print(f"Part {part}: {len(corrupted_games)} corrupted games detected.")
            # chess_data.to_pickle(file_path, compression='zip')        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            logger.critical(f'db cleanup interrupted because of:  {e}')
            logger.critical(traceback.format_exc())
            exit(1)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f'total time: {total_time} seconds')