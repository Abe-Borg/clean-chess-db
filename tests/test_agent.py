# Additional Unit Tests (test_agent.py)
# file tests/test_agent.py


import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.Agent import Agent
import pytest

# Unit test for missing column handling in Agent
def test_agent_choose_action_missing_column():
    df = pd.DataFrame({'PlyCount': [1]}, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}

    # Expect a KeyError because 'W1' column is missing
    with pytest.raises(KeyError):
        agent.choose_action(df, environ_state, 'Game 1')

# Unit test for Agent's choose_action method
def test_agent_choose_action():
    df = pd.DataFrame({
        'W1': ['e4'],
        'PlyCount': [1]
    }, index=['Game 1'])
    agent = Agent('W')
    environ_state = {'curr_turn': 'W1', 'legal_moves': ['e4']}

    # Expect the agent to choose the move 'e4'
    move = agent.choose_action(df, environ_state, 'Game 1')
    assert move == 'e4', "Agent should choose the move 'e4'"



