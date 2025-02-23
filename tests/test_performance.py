# tests/test_performance.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
from training.training_functions import play_games

def test_play_games_performance(benchmark):
    # Create a synthetic DataFrame with a large number of valid games.
    num_games = 10000  # Adjust this number based on your performance criteria.
    data = {
        'W1': ['e4'] * num_games,
        'B1': [None] * num_games,  # Ensure required columns exist.
        'PlyCount': [1] * num_games,
    }
    df = pd.DataFrame(data, index=[f'Game {i}' for i in range(num_games)])
    
    # Benchmark the play_games function.
    corrupted_games = benchmark(play_games, df)
    
    # Since all games are valid, expect no corrupted games.
    assert corrupted_games == []
