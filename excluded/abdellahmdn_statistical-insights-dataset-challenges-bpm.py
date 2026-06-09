from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 


df= pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

# 1. Handle TrackDurationMs scaling
scaler = StandardScaler()
df['TrackDurationMs_scaled'] = scaler.fit_transform(df[['TrackDurationMs']])

# 2. Transform skewed features
df['InstrumentalScore_log'] = np.log1p(df['InstrumentalScore'])
df['VocalContent_sqrt'] = np.sqrt(df['VocalContent'])

# 3. Feature engineering (create ratios, interactions)
df['Rhythm_Energy_Ratio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
df['Mood_Acoustic_Interaction'] = df['MoodScore'] * df['AcousticQuality']




