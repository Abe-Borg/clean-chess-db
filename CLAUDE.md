# CLAUDE.md - AI Assistant Guide for Clean Chess DB

## Project Overview

Clean Chess DB is a high-performance Python application that processes, validates, and cleans large chess game databases. It replays each game move-by-move using the `python-chess` library to detect corrupted, invalid, or illegal games, then removes them from the dataset. The application uses multiprocessing (with shared memory on Unix) to achieve high throughput; the chunker is calibrated at ~65,000 moves/second per worker.

The folder names `agents/`, `environment/`, and `training/` are historical artifacts — the codebase is *not* an RL project. `Agent` is a deterministic SAN-replay shim, `Environ` is a thin `python-chess` wrapper, and `training/game_simulation.py` is the validation engine. Treat the names as legacy.

## Repository Structure

```
clean-chess-db/
├── agents/
│   └── Agent.py              # Replay agent that returns moves from historical game data
├── debug/                     # Log output files (CRITICAL-level only; usually empty)
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
│   └── test_game_simulation.py  # Comprehensive pytest suite (20+ tests)
├── chess_data/                # (gitignored) input/output pickle files; not present in fresh clones
├── requirements.txt           # Pinned Python dependencies (UTF-16 encoded — see notes)
├── .gitignore                 # Ignores chess_data/, venv/, *.pyc, src/__pycache__/, *.ipynb, temp/
├── Misc Code Testing.ipynb    # (gitignored) scratch notebook
└── README.md                  # User-facing documentation
```

## Key Architecture

### Data Flow

1. **Input**: Pickle files with `zip` compression (`chess_data/chess_games_part_N.pkl`, N=1..100). Loaded with `pd.read_pickle(path, compression='zip')`.
2. **DataFrame schema**: Index = game_id (str), Columns = move keys (`W1`, `B1`, `W2`, `B2`, ... up to `W200`, `B200`) + `PlyCount` (int).
3. **Processing**: Each game is replayed move-by-move; corrupted games are collected.
4. **Output**: Corrupted games are dropped and the cleaned DataFrame is **saved back to the same pickle file in place** (destructive). If non-destructive cleaning is needed, modify `main/play_games.py` to write to a separate path.

### Core Components

- **`training/game_simulation.py`** - The heart of the application:
  - `play_games(chess_data, pool=None)`: Orchestrates multiprocessing, shared memory, and chunk distribution. Accepts an optional persistent `Pool` for amortized startup cost.
  - `worker_process_games()`: Per-worker function that processes a chunk of games. Branches on `shared_data['windows_mode']`.
  - `play_one_game()`: Validates a single game by replaying all moves. Returns `game_id` if corrupted, `None` if valid.
  - `AdaptiveChunker`: Distributes games across workers balanced by PlyCount; uses a longest-processing-time greedy assignment. Calibration constant `calibrated_moves_per_second = 65000`.
  - `create_shared_data()` / `cleanup_shared_data()`: Platform-aware shared memory (Unix uses `SharedMemory`, Windows passes the DataFrame directly via spawn-pickle).
  - `warm_up_workers(pool, num_workers)`: Runs a no-op `init_worker` task on each worker so first-batch latency doesn't dominate small jobs.
  - Module-level `IS_WINDOWS = platform.system() == 'Windows'` gates the platform branches.

- **`environment/Environ.py`** - Chess board state manager:
  - Wraps `chess.Board()` from python-chess
  - Tracks turn index (0-399) and maps to turn identifiers (`W1`, `B1`, `W2`, `B2`, ...)
  - Class-level `_turn_list` cache avoids recreating the turn list per instance
  - `get_curr_state_and_legal_moves()` is the primary state access method

- **`agents/Agent.py`** - Simple replay agent:
  - Looks up the move for the current turn from the game's move dictionary
  - Instantiated as `Agent('W')` or `Agent('B')`

- **`main/play_games.py`** - CLI entry point:
  - Supports `--single N`, `--start N --end M`, `-h/--help`, or no args (processes all 100 files).
  - Creates a persistent `multiprocessing.Pool` (`cpu_count() - 1` workers) reused across all file partitions.
  - Calls `warm_up_workers()` immediately after pool creation.
  - Wraps the per-part loop in `tqdm` for a progress bar.
  - Catches `FileNotFoundError` per part and skips (logs critical, continues), so missing parts don't kill a multi-part run.
  - Pool cleanup is in a `finally` block to guarantee `close()` + `join()`.
  - Reports per-file and aggregate statistics (totals, corruption %, wall time, overall games/sec).

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
venv\Scripts\Activate.ps1   # Windows PowerShell (primary target)
# venv\Scripts\activate.bat # Windows CMD
# source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

**Note:** `requirements.txt` is currently saved as **UTF-16** (PowerShell's default redirect encoding). If `pip install -r requirements.txt` fails with a Unicode error, re-save the file as UTF-8 first. Don't "fix" this silently — it's a known artifact and the user is on Windows.

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
- Only corruption events and fatal errors are logged — under normal operation `debug/*.txt` stays empty
- User-facing output uses `print()` statements, not the logger
- Logger handlers are guarded with `if not logger.handlers:` to prevent duplicate handlers when modules are re-imported by spawned workers
- To get a paper trail of which game failed and why, bump the level to `INFO` or `DEBUG` in the relevant module

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
- Unix systems use `multiprocessing.shared_memory.SharedMemory` for zero-copy data sharing — one segment for the index array, one for `PlyCount`, and one per move column
- Windows uses `spawn`-based pool creation, so the DataFrame is passed directly (pickle serialization). The `windows_mode` flag in `shared_data` selects the right branch in `worker_process_games`
- Workers reuse `Agent` and `Environ` instances across games within a chunk
- `Environ.reset_environ()` must be called between games (resets `chess.Board()` and zeros the turn index)
- Any new entry point that calls `play_games()` from a script **must** be guarded with `if __name__ == '__main__':` because of Windows spawn semantics

### Performance Considerations

- The `AdaptiveChunker` balances workload by sorting games by PlyCount (longest first) and greedily assigning each game to the least-loaded chunk (longest-processing-time scheduling)
- Calibrated at ~65,000 moves/second per worker (`AdaptiveChunker.calibrated_moves_per_second`). This is *only* used to estimate per-game cost for chunking — being off by 2× still produces reasonable chunks
- Minimum chunk size scales with batch size: 5/20/100 games for tiny/small/large batches; chunk count is capped at `cpu_count() - 1`
- Class-level caching of `_turn_list` in `Environ` avoids per-instance allocation
- `get_curr_state_and_legal_moves()` combines state + legal move retrieval in one call to avoid duplicate `list(self.board.legal_moves)` calls
- Move objects are pushed via `push_move_object` (already-parsed `chess.Move`) rather than re-parsing SAN, to avoid double work after validation

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
- A move string is not valid SAN for the current board position (caught as an exception from `board.parse_san`)
- A move is not in the legal moves list for the current position
- Any other exception raised inside `worker_process_games` while iterating that game (the worker treats unhandled exceptions as corruption to keep the batch moving)

A whole DataFrame is treated as 100%-corrupt (every game ID returned) if it's missing the `PlyCount` column. This is the most common cause of "100% corruption" runs — verify the schema first.

## Important Notes for AI Assistants

- The `chess_data/` directory is gitignored and contains large binary pickle files. Do not attempt to read or generate these files.
- `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` hacks exist in `main/play_games.py` and `tests/test_game_simulation.py` to allow running from the project root. This is the established import pattern; do not change it without also updating all entry points.
- `utils/game_settings.py` uses `pathlib.Path` with relative `..` paths rooted from `utils/`. All file path configuration is centralized there. Adding a 101st part file means adding `chess_games_filepath_part_101 = ...` and bumping the CLI bounds in `main/play_games.py` (currently hard-coded to `1..100`).
- Shared memory cleanup is critical on Unix. `cleanup_shared_data()` must always run (it's in a `finally` block). Leaking shared memory segments will persist until system reboot. The cleanup intentionally swallows `FileNotFoundError` in case a worker exited and unlinked first.
- `play_games.py` **overwrites the input pickle in place** with the cleaned DataFrame. If a user asks to clean files non-destructively, change the `chess_data.to_pickle(file_path, ...)` line to write to a separate output path.
- Pickle compression is `'zip'`, not `'zlib'` or `'gzip'`. Match this when reading or writing or downstream loaders will choke.
- When adding new tests, follow the existing fixture-based pattern in `TestGameSimulation` and place them in the appropriate section (valid games, corruption detection, edge cases, or integration).
- The `Agent` class is essentially a vestigial dictionary lookup. If a future change replaces it with a real engine, the validation contract (`choose_action(moves, curr_turn) -> str` returning SAN) must be preserved.
- The README references the same architecture documented here — keep them in sync. The user's preference is "anytime the implementation changes, give me an updated readme."
