# CLAUDE.md - AI Assistant Guide for Clean Chess DB

## Project Overview

Clean Chess DB is a high-performance Python application that processes, validates, and cleans large chess game databases. It replays each game move-by-move using the `python-chess` library to detect corrupted, invalid, or illegal games, then removes them from the dataset. The application uses multiprocessing with shared memory to achieve high throughput (~65,000 moves/second per worker).

## Repository Structure

```
clean-chess-db/
├── agents/
│   └── Agent.py              # Replay agent that returns moves from historical game data
├── debug/                     # Log output files (auto-generated, gitignored implicitly)
├── environment/
│   └── Environ.py             # Chess board wrapper around python-chess with turn tracking
├── main/
│   └── play_games.py          # CLI entry point for batch processing data files
├── training/
│   └── game_simulation.py     # Core engine: multiprocessing, chunking, game validation
├── utils/
│   ├── constants.py           # Game constants (max turns, player count)
│   └── game_settings.py       # File paths for 100 data partitions and logger config
├── tests/
│   └── test_game_simulation.py  # Comprehensive pytest suite (25+ tests)
├── requirements.txt           # Pinned Python dependencies
├── .gitignore                 # Ignores chess_data/, venv/, *.pyc, *.ipynb, temp/
└── README.md                  # User-facing documentation
```

## Key Architecture

### Data Flow

1. **Input**: Pickle files with zlib compression (`chess_data/chess_games_part_N.pkl`, N=1..100)
2. **DataFrame schema**: Index = game_id, Columns = move keys (`W1`, `B1`, `W2`, `B2`, ... up to `W200`, `B200`) + `PlyCount`
3. **Processing**: Each game is replayed move-by-move; corrupted games are collected
4. **Output**: Corrupted games are dropped and the cleaned DataFrame is saved back to the same pickle file

### Core Components

- **`training/game_simulation.py`** - The heart of the application:
  - `play_games()`: Orchestrates multiprocessing, shared memory, and chunk distribution
  - `worker_process_games()`: Per-worker function that processes a chunk of games
  - `play_one_game()`: Validates a single game by replaying all moves
  - `AdaptiveChunker`: Distributes games across workers balanced by PlyCount
  - `create_shared_data()` / `cleanup_shared_data()`: Platform-aware shared memory (Unix uses `SharedMemory`, Windows passes DataFrame directly)

- **`environment/Environ.py`** - Chess board state manager:
  - Wraps `chess.Board()` from python-chess
  - Tracks turn index (0-399) and maps to turn identifiers (`W1`, `B1`, `W2`, `B2`, ...)
  - Class-level `_turn_list` cache avoids recreating the turn list per instance
  - `get_curr_state_and_legal_moves()` is the primary state access method

- **`agents/Agent.py`** - Simple replay agent:
  - Looks up the move for the current turn from the game's move dictionary
  - Instantiated as `Agent('W')` or `Agent('B')`

- **`main/play_games.py`** - CLI entry point:
  - Supports `--single N`, `--start N --end M`, or no args (processes all 100 files)
  - Creates a persistent `multiprocessing.Pool` reused across all file partitions
  - Reports per-file and aggregate statistics

- **`utils/game_settings.py`** - Central configuration:
  - Defines paths for all 100 data partitions via `chess_games_filepath_part_N` attributes
  - Logger file paths for each module
  - `PRINT_STEP_BY_STEP` debug flag

- **`utils/constants.py`** - Game constants:
  - `max_num_turns_per_player = 200`
  - `max_turn_index = 399`
  - `num_players = 2`

## Development Commands

### Running the application

```bash
# Process all 100 data files
python main/play_games.py

# Process a single file
python main/play_games.py --single 42

# Process a range of files
python main/play_games.py --start 1 --end 10
```

### Running tests

```bash
# Run all tests with verbose output
python -m pytest tests/test_game_simulation.py -v

# Run a specific test
python -m pytest tests/test_game_simulation.py -v -k "test_valid_short_game"

# Run with coverage
python -m pytest tests/test_game_simulation.py --cov=training --cov-report=term-missing
```

### Installing dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Code Conventions

### Style

- **Python 3.12+** required
- **PEP 8** naming: `snake_case` for functions/variables, `PascalCase` for classes
- **Type hints** on all function signatures using `typing` module (`Dict`, `List`, `Optional`, `Tuple`, `Union`)
- **Docstrings** follow Google style with `Args:`, `Returns:`, `Raises:` sections
- **Imports** are organized: stdlib first, then third-party, then local modules
- Files begin with a `# file: path/to/file.py` comment

### Logging

- Each module configures its own `logging.Logger` at `CRITICAL` level
- Loggers write to files in the `debug/` directory (not stdout)
- Only corruption events and fatal errors are logged
- User-facing output uses `print()` statements, not the logger

### Testing Patterns

- Tests are in `tests/test_game_simulation.py` using a single `TestGameSimulation` class
- **pytest fixtures** provide reusable components:
  - `agents_and_environ`: Fresh `Agent('W')`, `Agent('B')`, `Environ()` instances
  - `valid_short_game`: 4-ply test game data dict
  - `valid_scholars_mate`: 7-ply Scholar's mate data dict
- Game data in tests is a dict: `{'game_id': str, 'ply_count': int, 'moves': {turn_key: san_move}}`
- Valid games return `None` from `play_one_game()`; corrupted games return the `game_id`
- Integration tests use `play_games()` with constructed DataFrames
- Tests cover: valid games, corruption detection, edge cases, and integration scenarios

### Multiprocessing

- Worker count defaults to `cpu_count() - 1` (reserves one core for the main process)
- Unix systems use `multiprocessing.shared_memory.SharedMemory` for zero-copy data sharing
- Windows falls back to passing the DataFrame directly (pickle serialization)
- Workers reuse `Agent` and `Environ` instances across games within a chunk
- `Environ.reset_environ()` must be called between games

### Performance Considerations

- The `AdaptiveChunker` balances workload by sorting games by PlyCount (longest first) and assigning to the least-loaded chunk
- Calibrated at ~65,000 moves/second per worker
- Class-level caching of `_turn_list` in `Environ` avoids per-instance allocation
- `get_curr_state_and_legal_moves()` combines state + legal move retrieval in one call

## Data Format Details

### Move Keys

Turn identifiers follow the pattern `{color}{number}`:
- `W1` = White's 1st move, `B1` = Black's 1st move
- `W2` = White's 2nd move, `B2` = Black's 2nd move
- Up to `W200` / `B200` (400 half-moves maximum)

### Move Notation

All moves use Standard Algebraic Notation (SAN): `e4`, `Nf3`, `Qxf7#`, `O-O`, `e8=Q`, etc.

### Corruption Indicators

A game is flagged as corrupted if any of the following occur during replay:
- A move key is missing or the value is empty/NaN
- A move string is not valid SAN for the current board position
- A move is not in the legal moves list for the current position

## Important Notes for AI Assistants

- The `chess_data/` directory is gitignored and contains large binary pickle files. Do not attempt to read or generate these files.
- `sys.path.append` hacks exist in `main/play_games.py` and `tests/test_game_simulation.py` to allow running from the project root. This is the established import pattern; do not change it without also updating all entry points.
- The `game_settings.py` file uses `pathlib.Path` with relative `..` paths rooted from `utils/`. All file path configuration is centralized there.
- Shared memory cleanup is critical on Unix. `cleanup_shared_data()` must always run (it's in a `finally` block). Leaking shared memory segments will persist until system reboot.
- When adding new tests, follow the existing fixture-based pattern in `TestGameSimulation` and place them in the appropriate section (valid games, corruption detection, edge cases, or integration).
