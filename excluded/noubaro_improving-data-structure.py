import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df_training = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv')
df_test = pd.read_csv('/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv',  header=None)

df_training.iloc[:, 1] = df_training.iloc[:, 1].astype(str).str.replace('[', '', regex=False)
df_training.iloc[:, 2048] = df_training.iloc[:, 2048].astype(str).str.replace(']', '', regex=False)
df_test.iloc[:, -1] = df_test.iloc[:, -1].astype(str).str.replace(']', '', regex=False)

df_samplesY = df_training.iloc[:96, -4:]
df_samplesX = df_training.iloc[:, :-4]



df_samplesY


df_samplesX


df_test




