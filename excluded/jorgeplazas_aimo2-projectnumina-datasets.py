import numpy as np 
import pandas as pd 

import os


splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/AI-MO/NuminaMath-TIR/" + splits["train"])


df.info()


for c in df.columns:
    print(f"------------------\n{c}\n------------------\n")
    print(df[c][0])


df.to_parquet('NuminaMath-TIR.parquet')


df.to_csv('/kaggle/working/NuminaMath-TIR.csv', index=False)




