# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
pd.read_csv("/kaggle/input/birdclef-25/taxonomy.csv").columns


import os
import numpy as np
import pandas as pd
import librosa
import scipy.signal
from tqdm.notebook import tqdm
from tensorflow.keras.models import load_model
import pickle

taxonomy_df = pd.read_csv("/kaggle/input/birdclef-25/taxonomy.csv")
species_cols = taxonomy_df['primary_label'].unique().tolist()

model = load_model("/kaggle/input/birdclef-25/final_cnn_model.keras")
with open("/kaggle/input/birdclef-25/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

test_audio_dir = "/kaggle/input/birdclef-2025/test_soundscapes"
file_list = [f for f in sorted(os.listdir(test_audio_dir)) if f.endswith('.ogg')]

pred_rows = []
for file_name in tqdm(file_list):
    file_path = os.path.join(test_audio_dir, file_name)
    y, sr = librosa.load(file_path, sr=32000)
    for i in range(0, len(y), sr*5):
        chunk = y[i:i+sr*5]
        if len(chunk) < sr*5:
            pad_width = sr*5 - len(chunk)
            chunk = np.pad(chunk, (0, pad_width))
        if len(chunk) >= 2048:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
        else:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40, n_fft=512)
        mfcc = mfcc.T
        if mfcc.shape[0] < 400:
            mfcc = np.pad(mfcc, ((0, 400-mfcc.shape[0]), (0,0)), mode='constant')
        else:
            mfcc = mfcc[:400, :]
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prob = model.predict(mfcc, verbose=0)[0]
        row_id = f"{file_name.split('.')[0]}_{i//sr+5}"
        prob_full = pd.Series(0, index=species_cols)
        for sp, p in zip(le.classes_, prob):
            if sp in prob_full.index:
                prob_full[sp] = p
        pred_rows.append([row_id] + prob_full.tolist())

submission_result = pd.DataFrame(pred_rows, columns=["row_id"] + list(species_cols))
submission_result.to_csv("/kaggle/working/submission.csv", index=False)

submission_result.head()




