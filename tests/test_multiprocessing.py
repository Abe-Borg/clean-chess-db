# Concurrency and Stress Multiprocessing Tests
# Testing concurrency at a larger scale for thousands of games.

# File: tests/test_multiprocessing.py

import pytest
import pandas as pd
import random
from training.game_simulation import process_games_in_parallel

def test_multiprocessing_with_large_number_of_games(benchmark):
    num_games = 100000  # Scaling for larger stress tests

    data = {
        'W1': ['e4'] * num_games,
        'B1': [None] * num_games,
        'PlyCount': [1] * num_games,
    }

    df = pd.DataFrame(data, index=[f'Game {i}' for i in range(num_games)])

    corrupted_games = benchmark(process_games_in_parallel, list(df.index), df)

    # Check if the multiprocessing function completes correctly under stress
    assert len(corrupted_games) == 0, "Some corrupted games were unexpectedly flagged."
