


import pandas as pd, numpy as np, os
from tqdm.notebook import tqdm
import polars as pl
import polars.selectors as cs


%%time 

train    = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
traind   = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

print(f"---> Original shape = {train.shape}")

sel_seq  = train["sequence_id"].unique()[0 : 3500]
seq      = sel_seq[0: 1750]
oth_cols = train.columns[9:]

train    = train.loc[train.sequence_id.isin(sel_seq)]
train.loc[train.sequence_id.isin(seq), oth_cols] = np.nan

print(f"---> Truncated shape = {train.shape}")


train.to_csv(f"test.csv", index = False)
traind.to_csv(f"test_demographics.csv", index = False)


%%time 

import kaggle_evaluation.cmi_inference_server

counter = 0

def predict(
    sequence     : pl.DataFrame, 
    demographics : pl.DataFrame,
) -> str:
    """
    Prediction function for Kaggle evaluation - uses a dummy submission to test my datasets
    Replace this with your submission script and test the timing using the cell magic command
    """
    
    global counter
    pred = 'Cheek - pinch skin'
    if counter % 100 == 0 :
        print(f"---> Final prediction = {pred} | {counter} ")
    counter = counter + 1
    
    return pred


%%time 

inference_server = \
kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/working/test.csv',
            '/kaggle/working/test_demographics.csv',
        )
    )

print()

