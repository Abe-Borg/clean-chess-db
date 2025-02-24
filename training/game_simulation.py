# game_simulation.py.py

from typing import Callable, List
import pandas as pd
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger("game_simulation.py")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.game_simulation.py_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def play_games(chess_data: pd.DataFrame) -> List[str]:
    game_indices = list(chess_data.index)
    return process_games_in_parallel(game_indices, worker_play_games, chess_data)

def process_games_in_parallel(game_indices: List[str], worker_function: Callable[..., List], *args) -> List[str]:
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    
    corrupted_games_list = [game for sublist in results for game in sublist]
    return corrupted_games_list

def worker_play_games(game_indices_chunk: List[str], chess_data: pd.DataFrame) -> List[str]:
    corrupted_games = []
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()

    for game_number in game_indices_chunk:
        try:
            result = play_one_game(game_number, chess_data, w_agent, b_agent, environ)
        except Exception as e:
            logger.critical(f"Exception processing game {game_number}: {e}")
            result = game_number  # flag as corrupted

        if result is not None:
            corrupted_games.append(game_number)

        environ.reset_environ()
    return corrupted_games


def play_one_game(game_number: str, chess_data: pd.DataFrame, w_agent: Agent, b_agent: Agent, environ: Environ) -> str:
    num_moves = chess_data.at[game_number, 'PlyCount']
    # Loop until we reach the number of moves or the game is over.
    while True:
        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn(w_agent, chess_data, curr_state, game_number, environ)
        if result is not None:
            return result  # Return the game_number for a corrupted game

        curr_state = environ.get_curr_state()
        if curr_state['turn_index'] >= num_moves:
            break
        if environ.board.is_game_over():
            break

        result = handle_agent_turn(b_agent, chess_data, curr_state, game_number, environ)
        if result is not None:
            return result  # Return the game_number for a corrupted game

        if environ.board.is_game_over():
            break
            
    return None  # Game processed normally


def handle_agent_turn(agent: Agent, chess_data: pd.DataFrame, curr_state: dict, game_number: str, environ: Environ) -> str:
    curr_turn = curr_state['curr_turn']    
    chess_move = agent.choose_action(chess_data, curr_state, game_number)

    if chess_move == '' or pd.isna(chess_move):
        return None
    
    if chess_move not in environ.get_legal_moves():
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. ")
        return game_number
    
    apply_move_and_update_state(chess_move, environ)
    return None

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



