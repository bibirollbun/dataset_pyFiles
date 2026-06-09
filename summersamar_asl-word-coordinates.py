import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/asl-fingerspelling/train.csv')
train_df_sorted = train_df.sort_values(by='phrase',key=lambda x: x.str.len())
train_df_sorted


phrase = 'surprise az'
train_df_sorted['phrase'] = train_df_sorted['phrase'].str.strip()
train_df_sorted = train_df_sorted[train_df_sorted['phrase'] == phrase]
train_df_sorted


participant_136_df = pd.read_parquet('/kaggle/input/asl-fingerspelling/train_landmarks/1969985709.parquet')
participant_136_df_seq = participant_136_df[participant_136_df.index == 1595884623]
participant_136_df_seq.isna().sum()


participant_136_df_seq


participant_168_df = pd.read_parquet('/kaggle/input/asl-fingerspelling/train_landmarks/1557244878.parquet')
participant_168_df_seq = participant_168_df[participant_168_df.index == 331596218]
participant_168_df_seq.isna().sum()


participant_168_df_seq

