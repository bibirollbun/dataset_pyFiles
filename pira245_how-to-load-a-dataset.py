import numpy as np
import pandas as pd
import pathlib
from pathlib import Path


import kagglehub
from kagglehub import KaggleDatasetAdapter


gesture_cat_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"pira245/cmi-dataset","mapped_categorical_dataset.csv",)
gesture_cat_df.head(5)


gesture_cat_df.columns.to_list()

