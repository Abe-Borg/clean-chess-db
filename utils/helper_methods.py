# helper_methods.py

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    curr_state = environ.get_curr_state() 
    chess_move: str = chess_agent.choose_action(curr_state)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move

def is_game_over(environ) -> bool:
    return environ.board.is_game_over() or (len(environ.get_legal_moves()) == 0)