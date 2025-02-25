#file tests/test_environ.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.Environ import Environ
import pytest
import chess

# Unit test to ensure Environ handles game reset correctly
def test_environ_reset_after_update():
    pass
