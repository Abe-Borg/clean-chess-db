# file: main/play_games.py

import pandas as pd
import time
import sys
import os
from multiprocessing import Pool, cpu_count
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from tqdm import tqdm

# Import the refactored game simulation module
from training.game_simulation import play_games, warm_up_workers

# Logger setup
logger = logging.getLogger("play_games")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.play_games_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def create_persistent_pool(num_workers=None):
    """
    Create a persistent process pool with optimal worker count.
    
    Args:
        num_workers: Number of worker processes (default: CPU count - 1)
        
    Returns:
        Pool object
    """
    if num_workers is None:
        # Leave one CPU for the main process
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Creating persistent pool with {num_workers} workers...")
    pool = Pool(processes=num_workers)
    
    # Warm up the workers
    warm_up_workers(pool, num_workers)
    print("Worker pool ready!")
    
    return pool


def main():
    """Main function to process all chess game files."""
    start_time = time.time()
    
    # Configuration
    num_dataframes_to_process = 1
    total_games_processed = 0
    total_corrupted_games = 0
    
    # Create persistent process pool
    persistent_pool = create_persistent_pool()
    
    try:
        # Process each data file
        for part in tqdm(range(1, num_dataframes_to_process + 1), 
                        desc="Processing parts", unit="part"):
            
            # Get file path
            file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
            
            try:
                # Load data
                part_start_time = time.time()
                chess_data = pd.read_pickle(file_path, compression='zip')
                
                print(f"\nPart {part}: Loaded {len(chess_data)} games")
                
                # Process games
                corrupted_games = play_games(chess_data, persistent_pool)
                
                # Update statistics
                total_games_processed += len(chess_data)
                total_corrupted_games += len(corrupted_games)
                
                # Report results
                part_elapsed = time.time() - part_start_time
                print(f"Part {part}: {len(corrupted_games)} corrupted games "
                      f"({len(corrupted_games)/len(chess_data)*100:.1f}%)")
                print(f"Part {part}: Completed in {part_elapsed:.2f}s")
                
                # Remove corrupted games and save
                # if corrupted_games:
                #     chess_data = chess_data.drop(corrupted_games)
                #     chess_data.to_pickle(file_path, compression='zip')
                #     print(f"Part {part}: Removed corrupted games and saved")
                
            except FileNotFoundError:
                print(f"\nPart {part}: File not found - {file_path}")
                logger.critical(f"File not found: {file_path}")
                continue
                
            except Exception as e:
                print(f"\nPart {part}: Error processing - {e}")
                logger.critical(f"Error processing {file_path}: {e}")
                logger.critical(traceback.format_exc())
                continue
    
    finally:
        # Clean up pool
        print("\nCleaning up worker pool...")
        persistent_pool.close()
        persistent_pool.join()
        print("Worker pool cleaned up")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files processed: {num_dataframes_to_process}")
    print(f"Total games processed: {total_games_processed:,}")
    print(f"Total corrupted games: {total_corrupted_games:,} "
          f"({total_corrupted_games/total_games_processed*100:.2f}%)")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per file: {total_time/num_dataframes_to_process:.2f}s")
    
    if total_games_processed > 0:
        overall_games_per_sec = total_games_processed / total_time
        print(f"Overall performance: {overall_games_per_sec:.0f} games/s")


if __name__ == '__main__':
    main()