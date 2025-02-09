# training_functions.py

from typing import Callable, List
import pandas as pd
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count
import logging

# Set up file-based logging (critical items only)
logger = logging.getLogger("training_functions")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.training_functions_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def play_games(chess_data):
    game_indices = list(chess_data.index)
    return process_games_in_parallel(game_indices, worker_play_games, chess_data)

def process_games_in_parallel(game_indices: List[str], worker_function: Callable[..., List], *args) -> List:
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

def worker_play_games(game_indices_chunk, chess_data):
    corrupted_games = []
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()

    for game_number in game_indices_chunk:
        result = play_one_game(game_number, chess_data, w_agent, b_agent, environ)
        # If a corrupted game is detected, 'result' is not an empty list.
        if result != '':
            corrupted_games.append(game_number)

        environ.reset_environ()
    return corrupted_games


def play_one_game(game_number, chess_data, w_agent, b_agent, environ):
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()

    while curr_state['turn_index'] < num_moves:
        if environ.board.is_game_over():
            break

        # result will be '' if game ended normally, otherwise it will be the game_number
        result = handle_agent_turn(
            agent=w_agent,
            chess_data=chess_data,
            curr_state=curr_state,
            game_number=game_number,
            environ=environ,
        )

        # if the result is not '', it means the game ended due to an invalid move
        if result != '': 
            return result # return the corrupted game_number
            
        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break 

        if environ.board.is_game_over():
            break

        result = handle_agent_turn(
            agent=b_agent,
            chess_data=chess_data,
            curr_state=curr_state,
            game_number=game_number,
            environ=environ,
        )

        if result != '': 
            return result # return the corrupted game_number

        curr_state = environ.get_curr_state()
        if environ.board.is_game_over():
            break 
            
    return '' # return '' if the game ended normally


def handle_agent_turn(agent, chess_data, curr_state, game_number, environ) -> str:
    curr_turn = curr_state['curr_turn']    
    chess_move = agent.choose_action(chess_data, curr_state, game_number)

    if chess_move == '' or pd.isna(chess_move):
        return ''
    
    if chess_move not in environ.get_legal_moves():
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. ")
        return game_number
    
    apply_move_and_update_state(chess_move, environ)
    return ''

def apply_move_and_update_state(chess_move: str, environ) -> None:
    environ.board.push_san(chess_move)
    environ.update_curr_state()

def chunkify(lst, n):
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
        
    return chunks



