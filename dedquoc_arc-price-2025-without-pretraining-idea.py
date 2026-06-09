# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install numpy tqdm torch


import numpy as np

# Simulated ARC-AGI puzzles
puzzles = {
    "00d62c1b": {
        "train": [
            {"input": np.random.randint(0, 10, (3, 3)), "output": np.random.randint(0, 10, (3, 3))},
            {"input": np.random.randint(0, 10, (4, 4)), "output": np.random.randint(0, 10, (4, 4))},
        ],
        "test":  [{"input": np.random.randint(0, 10, (3, 3))}],
    },
    "0a9fa5e": {
        "train": [
            {"input": np.random.randint(0, 10, (2, 2)), "output": np.random.randint(0, 10, (2, 2))},
        ],
        "test": [{"input": np.random.randint(0, 10, (2, 2))}],
    },
}


import time
import random
from tqdm import tqdm

def simulate_compressarc(puzzle_id, puzzle_data, max_steps=2000, time_per_step=0.1):
    steps_taken = random.randint(int(max_steps * 0.7), max_steps)
    time_taken = steps_taken * time_per_step
    time.sleep(time_taken / 1000)  # simulate compute time

    # Simulate result (some puzzles "solve", others don't)
    solved = random.random() < 0.5  # 50% success rate for demo

    return {
        "puzzle_id": puzzle_id,
        "steps_taken": steps_taken,
        "time_taken": time_taken,
        "solved": solved
    }


from concurrent.futures import ThreadPoolExecutor

def run_demo_parallel(puzzles, max_workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for pid, pdata in puzzles.items():
            futures.append(executor.submit(simulate_compressarc, pid, pdata))

        for future in tqdm(futures, desc="Running Puzzles"):
            results.append(future.result())

    return results


def print_results(results):
    print("\nğŸ“Š Demo Results:")
    print("{:<10} {:<7} {:<12} {:<8}".format("Puzzle", "Solved", "Steps", "Time (s)"))
    for res in results:
        print("{:<10} {:<7} {:<12} {:<8.3f}".format(
            res['puzzle_id'],
            "âœ…" if res['solved'] else "â�Œ",
            res['steps_taken'],
            res['time_taken']
        ))


if __name__ == "__main__":
    print("ğŸ§  Starting Minimal CompressARC Demo (No Pretraining)")
    results = run_demo_parallel(puzzles, max_workers=2)
    print_results(results)

