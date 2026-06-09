import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plot


PATH_TO_TRAIN_DATASET = "/kaggle/input/playground-series-s5e7/train.csv"
PATH_TO_CANDIDATE_ONE = "/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv"
PATH_TO_CANDIDATE_TWO = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
PATH_TO_CANDIDATE_THREE = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv"

df_train = pd.read_csv(PATH_TO_TRAIN_DATASET)
df_candidate = pd.read_csv(PATH_TO_CANDIDATE_THREE)


print(f"Number of entries in candidate public dataset: {len(df_candidate)}")
print("Number of entries in the real original dataset: 2512")


df_candidate = df_candidate.iloc[:2512]


df_candidate.head()


stats = df_candidate.describe(percentiles=[.05, .25, .5, .75, .9, .95]).T
stats = stats[['mean', 'std', 'min', '5%', '25%', '50%', '75%', '90%', '95%']]
stats.style.background_gradient(cmap='Blues').format("{:.2f}")


df_candidate.info()


def get_uniques_nulls(df, colnames):
    nunq = df[colnames].nunique()
    nulls = df[colnames].isnull().sum()
    summary = pd.DataFrame([nunq, nulls], index=["Nunq", "Nulls"])
    return summary

# Usage
col_list = ["Time_spent_Alone", "Stage_fear", "Social_event_attendance", "Going_outside", "Drained_after_socializing", "Friends_circle_size", "Post_frequency"]
summary_df = get_uniques_nulls(df_candidate, col_list).T
summary_df.style.background_gradient(cmap="Blues")

