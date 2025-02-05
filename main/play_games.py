# play_games.py

import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from training.training_functions import generate_q_est_df


logger = logging.getLogger("generate_q_est_df")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.generate_q_est_logger_filepath)
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
        estimated_q_values = generate_q_est_df(chess_data.head(1000))
    except Exception as e:
        logger.critical(f'q table generation interrupted because of:  {e}')
        logger.critical(traceback.format_exc())  # This will print the full traceback
        exit(1)

    print(estimated_q_values.iloc[:, :10])

    # estimated_q_values.to_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    
    end_time = time.time()
    total_time = end_time - start_time
    print('q table generation is complete')
    print(f'total time: {total_time} seconds')