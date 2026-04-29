# Clean Chess DB

A high-throughput, multiprocessing toolkit for **validating and cleaning massive chess game databases**. Point it at a pickled DataFrame of games, and it will replay every move on a real chess board, flag anything illegal or malformed, and quietly drop the corrupted rows so you keep only games that actually make sense.

Built for cleaning datasets in the millions-of-games range, with adaptive load balancing across CPU cores and first-class Windows support.

---

## Table of Contents

1. [Why this exists](#why-this-exists)
2. [What it does](#what-it-does)
3. [Expected input data](#expected-input-data)
4. [Quick start](#quick-start)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project structure](#project-structure)
8. [How it works](#how-it-works)
9. [Performance](#performance)
10. [Testing](#testing)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)
13. [Roadmap / known limitations](#roadmap--known-limitations)
14. [License](#license)

---

## Why this exists

Public chess datasets aggregated from online platforms, PGN dumps, and historical archives are full of garbage: truncated games, illegal moves, mis-parsed SAN strings, mismatched ply counts, empty cells, NaNs in the middle of games, and so on. A serious training pipeline (for an engine, opening book, embedding model, whatever) chokes on this stuff.

`clean-chess-db` is a focused, no-frills *data scrubber*: it does one thing — verify that every recorded game is actually playable, end to end — and it does it fast enough to chew through a 100-part dataset of millions of games on a single workstation.

## What it does

Given a DataFrame of chess games, the pipeline:

1. **Replays each game move-by-move** on a real `python-chess` board.
2. **Validates each move** — checks that it parses as legal SAN and that the resulting move is in the current legal-move set.
3. **Catches every flavor of corruption**, including:
   - Illegal moves for the position
   - Invalid or unparseable SAN notation
   - Empty strings or `NaN` values where a move should be
   - Missing moves before the recorded `PlyCount`
   - Mismatched `PlyCount` vs. actual move data
4. **Returns the list of corrupted game IDs**, which the driver script then drops from the DataFrame and re-pickles.
5. **Reports throughput metrics** — games/second, moves/second, and corruption rate — for each part processed.

It does *not* train, evaluate, score, or analyze the games. The names `agents/`, `environment/`, and `training/` are historical artifacts from an earlier project — under the hood, `Agent` is just a deterministic move-replayer that looks up the next SAN string by turn key, and `training/game_simulation.py` is the validation engine.

## Expected input data

Each part file at `chess_data/chess_games_part_{1..100}.pkl` is a **zip-compressed pickled `pandas.DataFrame`** (read with `pd.read_pickle(path, compression='zip')`).

The DataFrame must have:

| Column | Type | Description |
| --- | --- | --- |
| index | str | Unique game ID — used as the corruption key returned to the caller |
| `PlyCount` | int | Total half-moves in the game (used for early termination and chunk balancing) |
| `W1`, `B1`, `W2`, `B2`, … | str (SAN) | One column per ply, holding the move in Standard Algebraic Notation. Empty cells past the actual end of the game are fine; empty cells *before* `PlyCount` are flagged as corruption. |

White/black turn keys follow the pattern `W{n}` and `B{n}`, where `n` is the move number for that color. The maximum supported is 200 plies per side (`max_num_turns_per_player = 200` in `utils/constants.py`), giving 400 total ply columns.

If you don't have data shaped this way already, the cleanest path is to PGN-parse upstream and write the result into a `pandas.DataFrame` matching the schema above before partitioning into 100 pickle files.

## Quick start

```bash
# Activate the virtualenv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Process every part file (chess_games_part_1.pkl … chess_games_part_100.pkl)
python main/play_games.py

# Process a single part
python main/play_games.py --single 42

# Process a contiguous range
python main/play_games.py --start 1 --end 10
```

For each part file, `play_games.py` loads it, validates every game in parallel, drops the corrupted rows, and **overwrites the part file in place** with the cleaned DataFrame.

## Installation

Tested on **Python 3.12** on **Windows 10/11**. Should also work on Linux/macOS (the codebase has explicit Unix-shared-memory branches), but the daily driver target is Windows.

```bash
# 1. Clone
git clone https://github.com/yourusername/clean-chess-db.git
cd clean-chess-db

# 2. Create + activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# venv\Scripts\activate.bat          # Windows CMD
# source venv/bin/activate           # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

The pinned core dependencies are:

- `python-chess==1.10.0` — board state and move legality
- `pandas==2.1.4` — DataFrame I/O
- `numpy==1.26.2` — shared-memory typed arrays (Unix path)
- `tqdm==4.66.4` — progress bars
- `psutil==5.9.6` — process/memory introspection

The full pin list (~57 packages, including the test/profiling/Jupyter ecosystem) is in `requirements.txt`. The file is currently saved as UTF-16 (Windows PowerShell default) — if `pip install -r requirements.txt` complains, re-save it as UTF-8 first.

> **Note:** `requirements.txt` includes the full development environment (Jupyter, matplotlib, plotly, pytest plugins, etc.). For a slim production install, you only need `chess`, `pandas`, `numpy`, `psutil`, and `tqdm`.

## Usage

### Command-line interface

```text
python main/play_games.py [-h] [--single N | --start N --end N]
```

| Flag | Meaning |
| --- | --- |
| *(no args)* | Process all 100 parts, in order |
| `--single N` | Process only part `N` (1–100) |
| `--start N --end N` | Process the inclusive range `[start, end]` (both 1–100, `start ≤ end`) |
| `-h`, `--help` | Show argparse help, including examples |

The driver:

1. Spins up a **persistent multiprocessing pool** (`cpu_count() - 1` workers) and **warms it up** so first-batch latency doesn't dominate small jobs.
2. Iterates the requested parts with a `tqdm` progress bar.
3. For each part: loads the pickle, calls `play_games(...)`, drops corrupted indices, re-pickles in place.
4. Prints a final summary (totals, corruption %, wall time, overall games/sec).

### Programmatic API

If you want to integrate the validator into another pipeline:

```python
import pandas as pd
from training.game_simulation import play_games

# Load games (must match the schema documented above)
chess_data = pd.read_pickle("chess_data/chess_games_part_1.pkl", compression="zip")

# Validate — returns a list of game IDs that failed validation
corrupted_ids = play_games(chess_data)

# Keep only the good ones
clean_data = chess_data.drop(corrupted_ids)

# Save back
clean_data.to_pickle("chess_data/chess_games_part_1.pkl", compression="zip")
```

For repeated calls (e.g., looping over many DataFrames), you can pass an existing `multiprocessing.Pool` to amortize worker startup:

```python
from multiprocessing import Pool, cpu_count
from training.game_simulation import play_games, warm_up_workers

with Pool(processes=max(1, cpu_count() - 1)) as pool:
    warm_up_workers(pool, pool._processes)
    for part_path in part_paths:
        df = pd.read_pickle(part_path, compression="zip")
        bad = play_games(df, pool=pool)
        df.drop(bad).to_pickle(part_path, compression="zip")
```

## Project structure

```
clean-chess-db/
├── agents/
│   └── Agent.py                # Deterministic SAN-replay agent (looks up next move by turn key)
├── environment/
│   └── Environ.py              # Thin python-chess wrapper: state, legal moves, push, reset
├── training/
│   └── game_simulation.py      # The actual validation engine — chunking, multiprocessing, replay
├── main/
│   └── play_games.py           # CLI driver: argparse, persistent pool, per-part orchestration
├── utils/
│   ├── constants.py            # max_num_turns_per_player, num_players, etc.
│   └── game_settings.py        # File paths for the 100 part files + log files
├── tests/
│   └── test_game_simulation.py # 20+ pytest cases covering valid games, corruption, edge cases
├── chess_data/                 # (gitignored) Input/output pickle files: chess_games_part_*.pkl
├── debug/                      # Logger output (CRITICAL-level only by default)
├── requirements.txt
├── .gitignore
└── README.md
```

## How it works

### The replay loop

For each game the worker:

1. Resets `Environ` (a `chess.Board()` plus a turn counter).
2. Loops:
   - Reads the current state and legal moves.
   - Stops if `turn_index >= PlyCount` or `board.is_game_over()`.
   - Pulls the SAN move for the current turn key (`W1`, `B1`, …) from the game data.
   - If the move cell is empty / NaN → flag and stop.
   - Parses the SAN with `board.parse_san(...)`. On exception → flag and stop.
   - Checks the parsed `chess.Move` is in the live legal-moves list. If not → flag and stop.
   - Pushes the move and increments the turn counter.
3. If the loop finishes cleanly, the game passes; otherwise the game ID is added to the corruption list.

### Adaptive load balancing

`AdaptiveChunker` (in `training/game_simulation.py`) doesn't just split games evenly across workers — it estimates each game's processing cost from its `PlyCount` (calibrated at ~65,000 moves/sec) and uses a longest-processing-time greedy heuristic to assemble chunks with roughly equal total work. This keeps long marathon games from stranding one worker at the end of a batch while the others sit idle.

The minimum chunk size scales with batch size (5/20/100 games for tiny/small/large batches), and the chunk count is capped at `cpu_count() - 1`.

### Windows vs. Unix multiprocessing

Python's `multiprocessing.shared_memory` works well on Linux/macOS but is awkward on Windows because `fork()` isn't available — every worker re-imports the world via `spawn`, which makes pickling DataFrames the path of least resistance. The code branches on `platform.system()`:

- **Windows path:** the DataFrame is passed to workers as-is via `Pool.starmap` (relying on `spawn` pickling).
- **Unix path:** the indices, `PlyCount` array, and each move column are written into named `SharedMemory` segments, and workers attach to them by name. The driver cleans up the segments in a `finally` block.

This is the main reason you'll see `if shared_data.get('windows_mode', False):` branches in `worker_process_games`.

### Logging

All four runtime loggers (`agent_logger_file.txt`, `environ_logger_file.txt`, `game_simulation_logger_file.txt`, `play_games_logger_file.txt` — paths configured in `utils/game_settings.py`) are at `CRITICAL` level by default, so under normal operation `debug/*.txt` stays empty. Bump the level to `INFO` or `DEBUG` in the relevant module if you want a paper trail of which game failed and why.

## Performance

On a typical multi-core workstation you should expect on the order of **tens of thousands of games/sec** and **hundreds of thousands of moves/sec** for clean batches, scaling roughly linearly with usable cores up to memory bandwidth limits. Actual numbers are printed at the end of every `play_games` call, e.g.:

```
Processed 250000 games in 12.43s (20112 games/s, 871234 moves/s)
Found 87 corrupted games (0.0%)
```

If you're tuning, the calibration constant `AdaptiveChunker.calibrated_moves_per_second = 65000` is what the chunker uses to estimate per-game cost. It's only a chunking heuristic, so being off by 2× is fine — but if your hardware is wildly faster or slower, adjusting it can improve load balance.

## Testing

The project ships with a comprehensive pytest suite in `tests/test_game_simulation.py` (20+ tests, ~23 KB).

```bash
python -m pytest                      # run all tests
python -m pytest -v                   # verbose
python -m pytest tests/test_game_simulation.py::test_name   # one test
python -m pytest --cov=. --cov-report=html                  # with coverage
```

Coverage groups:

- **Valid games:** 4-ply mini-games, full checkmates, long games, castling (king/queen side), en passant, pawn promotion.
- **Corrupted games:** illegal moves, unparseable SAN, empty/NaN move cells, wrong-piece moves, mismatched `PlyCount`, missing moves before termination, zero-ply rows.
- **Integration:** end-to-end `play_games(...)` on empty / all-valid / all-corrupted / mixed DataFrames.
- **Real-world scenarios:** captures, checks, queenside castling sequences pulled from real games.

## Configuration

`utils/game_settings.py` is the single source of truth for filesystem paths:

- `chess_games_filepath_part_1` … `chess_games_filepath_part_100` — input/output pickle paths under `chess_data/`.
- `agent_logger_filepath`, `environ_logger_filepath`, `game_simulation_logger_filepath`, `play_games_logger_filepath`, `helper_methods_logger_filepath` — log file destinations under `debug/`.
- `PRINT_STEP_BY_STEP = False` — verbose per-move tracing flag (off by default).

`utils/constants.py` holds the chess-side limits:

- `max_num_turns_per_player = 200`
- `max_turn_index = 399`
- `num_players = 2`

If you need more than 400 plies per game (extremely long endgames in some annotated databases can exceed this), bump `max_num_turns_per_player` and the turn-key generator in `Environ._get_turn_list` will adjust automatically.

## Troubleshooting

**`UnicodeError` when running `pip install -r requirements.txt`**
The file may be saved as UTF-16 (PowerShell's default redirect encoding). Open it in VS Code or Notepad++, save as UTF-8, retry.

**`FileNotFoundError` for a part file**
The driver logs and skips missing parts rather than failing the whole run. Confirm the file exists at the path defined in `utils/game_settings.py` and that compression is `zip`.

**Workers hang or never start on Windows**
Windows uses `spawn`, which means the entry point must be guarded with `if __name__ == '__main__':`. `main/play_games.py` already does this; if you're calling `play_games(...)` from your own script, do the same.

**100% corruption rate**
Almost always means the DataFrame is missing the `PlyCount` column (the validator returns *every* game ID as corrupted in that case) or the turn-key columns don't follow the `W1`/`B1`/… naming.

**`SharedMemory` `FileNotFoundError` cleanup warnings (Linux)**
These are non-fatal — the cleanup code intentionally swallows them in case workers exited and unlinked first.

## Roadmap / known limitations

- The `Agent` class is a vestigial replay shim — it doesn't actually choose anything, it just dictionary-looks-up the next SAN. Future work (if any) could replace it with a real engine integration without touching the validation contract.
- There's no schema validator — input DataFrames are trusted to have the right column shape. A pre-flight check would catch malformed inputs faster than letting the worker pool find out.
- Cleaned files **overwrite the originals in place**. If you want non-destructive cleaning, fork the `chess_data.to_pickle(...)` line in `main/play_games.py` to write to a new path.
- Only one CLI driver is shipped (`main/play_games.py`). For different I/O backends (Parquet, DuckDB, S3, etc.) you'd need to adapt the loader.

## License

Not yet declared. Add a `LICENSE` file (MIT, Apache 2.0, BSD-3-Clause are common for tooling projects) before publishing.

## Acknowledgements

- [`python-chess`](https://github.com/niklasf/python-chess) — does all the actual chess heavy lifting (board state, SAN parsing, legal-move generation). This project is a thin orchestration layer on top.
- The Python standard library `multiprocessing.shared_memory` for making the Unix fast-path possible without third-party IPC dependencies.
