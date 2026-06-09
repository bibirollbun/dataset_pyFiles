import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool, cpu_count


%%time
import glob
import random
import hashlib
import numpy as np
import pandas as pd
import time
import os

NEGATIVE_PART = -299
LARGEST_CHUNK = 600
SMALLEST_CHUNK = 350
TOTAL_REALIZATIONS = 10

def remove_chunk_opt(array, length):
    if length > len(array):
        raise ValueError("Length exceeds the size of the input array.")
    return array[:length], array[length:]

def add_chunk_to_rows(rows, input_array, chunk_length, file_name, initial_length):
    position = initial_length - len(input_array)
    chunk, shortened_array = remove_chunk_opt(input_array, chunk_length)
    chunk = chunk - chunk[-(LARGEST_CHUNK + NEGATIVE_PART)]
    padding_length = LARGEST_CHUNK - chunk_length
    padded_chunk = np.concatenate((np.full(padding_length, np.nan), chunk))
    full_id = f"{file_name}_{position}"
    hash_hex_id = f"g_{hashlib.md5(full_id.encode('utf-8')).hexdigest()[:10]}"
    row = [hash_hex_id] + padded_chunk.tolist()
    rows.append(row)
    return shortened_array

def process_folder_optimized(path_to_process, output_file_name=None, DO_PLOT=False, my_rnd=None, random_state=0):
    if my_rnd is None:
        my_rnd = random.Random()
    csv_files = glob.glob(f"{path_to_process}/*.csv")
    rows = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(),
                                df['VS_APPROX_adjusted'].max() + 1,
                                1)
        new_horizon_z = np.interp(new_vs_grid,
                                  df['VS_APPROX_adjusted'],
                                  df['HORIZON_Z_adjusted'])

        remaining_array = new_horizon_z
        initial_len = len(remaining_array)
        total_large = len(remaining_array) // LARGEST_CHUNK // 2
        for _ in range(total_large):
            remaining_array = add_chunk_to_rows(rows, remaining_array, LARGEST_CHUNK, file_name, initial_len)

        while len(remaining_array) >= LARGEST_CHUNK * 2.5:
            chunk_len = my_rnd.randint(SMALLEST_CHUNK, LARGEST_CHUNK)
            remaining_array = add_chunk_to_rows(rows, remaining_array, chunk_len, file_name, initial_len)

        remaining_len = len(remaining_array) // 3
        for _ in range(2):
            remaining_array = add_chunk_to_rows(rows, remaining_array, remaining_len, file_name, initial_len)

        remaining_array = add_chunk_to_rows(rows, remaining_array, len(remaining_array), file_name, initial_len)

    columns = ['geology_id'] + [NEGATIVE_PART + i for i in range(LARGEST_CHUNK)]
    total_df = pd.DataFrame(rows, columns=columns)
    for k in range(1, TOTAL_REALIZATIONS):
        for i in range(1, LARGEST_CHUNK + NEGATIVE_PART):
            total_df[f"r_{k}_pos_{i}"] = total_df[i]

    if output_file_name is not None:
        total_df.to_csv(output_file_name, encoding='utf-8', index=False)
    return total_df

##The above is the code provided by the organizer

def run_until_no_new_rows(path_to_process, max_run_seconds=6*3600, output_file_name_prefix='train_optimized'):
    my_rnd = random.Random(42)
    merged_df = pd.DataFrame()
    start_time = time.time()
    iteration = 0
    
    # create 4 kernel pool
    with Pool(processes=4) as pool:  # use pool
        while True:
            try:
                iteration += 1
                print(f"Iteration {iteration} is running...")
                
                # different seed
                seeds = [my_rnd.randint(0, 2**32-1) for _ in range(4)]
                
                # Prepare multi-process parameters (do not save files, return DataFrame directly)
                args_list = [(path_to_process, None, False, random.Random(seed)) for seed in seeds]
                
                # run multi process_folder_optimized
                results = pool.starmap(process_folder_optimized, args_list)
                # Merge all new rows in the result
                combined_new = pd.DataFrame()
                for new_df in results:
                    if merged_df.empty:
                        new_rows = new_df
                    else:
                        new_rows = new_df[~new_df['geology_id'].isin(merged_df['geology_id'])]
                    combined_new = pd.concat([combined_new, new_rows], ignore_index=True)
                
                # Update merge results
                if not combined_new.empty:
                    merged_df = pd.concat([merged_df, combined_new], ignore_index=True)
                
                print(f"Iteration {iteration}：Add unique rows = {len(combined_new)}，Total number of merged rows = {len(merged_df)}")
                
                if combined_new.empty:
                    print("success")
                    break
                if (time.time() - start_time) > max_run_seconds:
                    print(f"Time is {max_run_seconds/3600:.2f} h. Stop.")
                    break
            except Exception as e:
                print(f"Error: {e}. Skiped")
                continue

    final_output = f"{output_file_name_prefix}_final.csv"
    merged_df.to_csv(final_output, encoding='utf-8', index=False)
    print(f"Final result is in {final_output}")
    return merged_df

merged_result = run_until_no_new_rows(path_to_process='/kaggle/input/geology-forecast-challenge-open/data/train_raw',max_run_seconds=4*3600)

