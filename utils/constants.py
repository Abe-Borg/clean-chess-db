# constants.py

agent_vs_agent_num_games = 100

CHESS_MOVE_VALUES: dict[str, int] = {
        'new_move': 100,
        'capture': 150,
        'piece_development': 200,
        'check': 300,
        'promotion': 500,
        'promotion_queen': 900,
        'mate_score': 1_000
    }

training_sample_size = 50_000
max_num_turns_per_player = 200
max_turn_index = max_num_turns_per_player * 2 - 1
num_players = 2
initial_q_val = 1
chance_for_random_move = 0.05 
chess_engine_num_moves_to_return = 1
chess_engine_depth_limit = 1
default_learning_rate = 0.6
default_discount_factor = 0.35