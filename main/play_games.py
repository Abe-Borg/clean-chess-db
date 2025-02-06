# play_games.py

import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from training.training_functions import play_games


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
    
    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression='zip')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    try:
        corrupted_games = play_games(chess_data.head(1000))
    except Exception as e:
        logger.critical(f'db cleanup interrupted because of:  {e}')
        logger.critical(traceback.format_exc())
        exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'corrupt games list: {len(corrupted_games)}')
    print('db cleanup is complete')
    print(f'total time: {total_time} seconds')

    chess_data = chess_data.drop(corrupted_games)
    chess_data.to_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')