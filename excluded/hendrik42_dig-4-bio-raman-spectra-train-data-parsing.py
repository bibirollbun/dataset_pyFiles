import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt


tmp_df = pd.read_csv("/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv")

train_target = tmp_df[['Analyte concentration', 'Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']].dropna().rename(columns={'Analyte concentration': 'sample_id'})
train_target['sample_id'] = train_target['sample_id'].str.strip()

train_spectra = pd.DataFrame(tmp_df[tmp_df.columns[0]].values, columns=['sample_id']).ffill()
train_spectra['measurement_id'] = np.tile([0, 1], 96)
train_spectra['sample_id'] = train_spectra['sample_id'].str.strip()
train_spectra['spectrum'] = tmp_df[tmp_df.columns[1:2049]].astype(str).agg(",".join, axis=1).apply(ast.literal_eval).apply(np.array)

train_data = pd.merge(train_spectra, train_target, on='sample_id', how='inner')


train_data


plt.plot(np.arange(2048), train_data['spectrum'] [1])




