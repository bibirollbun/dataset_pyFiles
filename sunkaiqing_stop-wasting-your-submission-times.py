import pandas as pd
import os
import re
from glob import glob
from tqdm import tqdm  
# 获取所有 date_id 的文件路径
test_files = sorted(glob(os.path.join("/kaggle/input/janestreet-updated-simulator-for-time-series-api/debug/test.parquet/", "date_id=*", "part-0.parquet")))
lags_files = sorted(glob(os.path.join("/kaggle/input/janestreet-updated-simulator-for-time-series-api/debug/lags.parquet/", "date_id=*", "part-0.parquet")))
test_files = sorted(test_files, key=lambda x: int(re.search(r"date_id=(\d+)", x).group(1)))
lags_files = sorted(lags_files, key=lambda x: int(re.search(r"date_id=(\d+)", x).group(1)))



date_id_pattern = re.compile(r"date_id=(\d+)")

# list for saving the prediction
results = []

prev_date_id = None  # Record for last date_id

for test_file in tqdm(test_files, desc="Processing Test Files"):
    # get current_date_id
    match = date_id_pattern.search(test_file)
    if not match:
        continue
    current_date_id = int(match.group(1))

    
    test_data = pl.read_parquet(test_file)

    # the predict function will be called on different time_id(time_id=0,1,2...)
    for time_id in tqdm(test_data["time_id"].unique(), desc=f"Processing date_id={current_date_id}", leave=False):
        time_batch = test_data.filter(pl.col("time_id") == time_id)

        if current_date_id != prev_date_id and time_id == 0:
            # ①If we go to the next date_id, we can get the full lags data when time_id==0.
            lags_file = next((f for f in lags_files if f"date_id={current_date_id}" in f), None)
            lags_data = pl.read_parquet(lags_file) if lags_file else None
            print(f"Processing new date_id: {current_date_id}, time_id: {time_id} with lags")
        else:
            # ②And we get None for the rest of the time_id(1,2,3...)
            lags_data = None
            print(f"Processing date_id: {current_date_id}, time_id: {time_id} without lags")

        # call predict func for every time_id batch
        prediction = predict(time_batch, lags_data)
        results.append(prediction)

    # Update prev_date_id
    prev_date_id = current_date_id

results_df = pl.concat(results)
print("Prediction summary:")
print(results_df.head())

