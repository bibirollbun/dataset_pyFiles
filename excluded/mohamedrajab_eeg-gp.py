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


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 500)  

delta_wave = np.sin(2 * np.pi * 2 * t)  
theta_wave = np.sin(2 * np.pi * 6 * t) 
alpha_wave = np.sin(2 * np.pi * 10 * t) 
beta_wave = np.sin(2 * np.pi * 20 * t)  
gamma_wave = np.sin(2 * np.pi * 40 * t)

plt.figure(figsize=(10,6))
plt.plot(t, delta_wave, label="Delta (0.5-4 Hz)")
plt.plot(t, theta_wave, label="Theta (4-8 Hz)")
plt.plot(t, alpha_wave, label="Alpha (8-12 Hz)")
plt.plot(t, beta_wave, label="Beta (12-30 Hz)")
plt.plot(t, gamma_wave, label="Gamma (30+ Hz)")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Brain Wave Frequencies")
plt.legend()
plt.show()



import seaborn as sns
import pandas as pd

waves = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
frequencies = [4, 8, 12, 30, 40]
df = pd.DataFrame({"Brain Waves": waves, "Max Frequency (Hz)": frequencies})
plt.figure(figsize=(8,5))
sns.barplot(x="Brain Waves", y="Max Frequency (Hz)", data=df, palette="coolwarm")
plt.title("Brain Wave Frequency Ranges")
plt.show()



import scipy.signal as signal
eeg_signal = delta_wave + theta_wave + alpha_wave + beta_wave + gamma_wave
f, t, Sxx = signal.spectrogram(eeg_signal, fs=500)
plt.figure(figsize=(10,6))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Simulated EEG Waves')
plt.colorbar(label='Power')
plt.show()



import pandas as pd
import glob
from tqdm import tqdm
import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")


BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
SPEC_DIR = "/tmp/dataset/hms-hbac"
os.makedirs(SPEC_DIR+'/train_spectrograms', exist_ok=True)
os.makedirs(SPEC_DIR+'/test_spectrograms', exist_ok=True)


class config:
    BATCH_SIZE = 64
    FOLDS = 0
    MODEL = "tf_efficientnet_b0"
    SEED = 29


class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
label2name = dict(enumerate(class_names))
name2label = {v:k for k, v in label2name.items()}


train_eeg_path_list = glob.glob("/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/*")
train_spectrograms_path_list = glob.glob("/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/*")


train_eeg_dict = {path.split("/")[-1].split(".")[0]: path for path in train_eeg_path_list}
train_spectrogram_dict = {path.split("/")[-1].split(".")[0]: path for path in train_spectrograms_path_list}


train_df['eeg_path'] = train_df['eeg_id'].astype(str).map(train_eeg_dict)
train_df['spectrograms_path'] = train_df['spectrogram_id'].astype(str).map(train_spectrogram_dict)


train_df.head()


test_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")



test_eeg_path_list = glob.glob("/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/*")
test_spectrograms_path_list = glob.glob("/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/*")



test_eeg_dict = {path.split("/")[-1].split(".")[0]: path for path in test_eeg_path_list}
test_spectrogram_dict = {path.split("/")[-1].split(".")[0]: path for path in test_spectrograms_path_list}



test_df['eeg_path'] = test_df['eeg_id'].astype(str).map(test_eeg_dict)
test_df['spectrograms_path'] = test_df['spectrogram_id'].astype(str).map(test_spectrogram_dict)


test_df.head()


df_train = pl.read_csv(f'{BASE_PATH}/train.csv')
df_train = df_train.with_columns(
    eeg_path = f'{BASE_PATH}/train_eegs/' + pl.col('eeg_id').cast(pl.Utf8) + '.parquet',
    spec_path = f'{BASE_PATH}/train_spectrograms/' + pl.col('spectrogram_id').cast(pl.Utf8) + '.parquet',
    class_label = pl.col('expert_consensus').replace(name2label).str.to_integer()
)
df_train.head(9)


df_train.shape[0]


df_train.group_by('eeg_id').len().shape[0]


df_train.group_by('spectrogram_id').len().shape[0]


df_train.group_by('patient_id').len().shape[0]


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

histo = df_train.group_by('expert_consensus').len().sort('len', descending=True).to_pandas()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.3)
sns.histplot(data=histo,
             x='expert_consensus',
             weights='len',
             discrete=True,
             shrink=0.8,
             hue='expert_consensus',
             palette='pastel',
             legend=False)

plt.title('Histogram of expert consensus')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df_train.select('seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote').corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            xticklabels=['seizure','lpd','gpd','lrda','grda','other'],
            yticklabels=['seizure','lpd','gpd','lrda','grda','other'])
plt.title('Correlation Matrix between experts\' votes')
plt.show()


target_votes = df_train.group_by('expert_consensus').len()
total_votes = target_votes['len'].sum()
mean_votes = target_votes.with_columns(pl.col('len')/total_votes)

mean_votes


lol = df_train.group_by('eeg_id').first()
target_votes = lol.group_by('expert_consensus').len()
total_votes = target_votes['len'].sum()
mean_votes = target_votes.with_columns(pl.col('len')/total_votes)

mean_votes


lol = df_train.group_by('spectrogram_id').first()
target_votes = lol.group_by('expert_consensus').len()
total_votes = target_votes['len'].sum()
mean_votes = target_votes.with_columns(pl.col('len')/total_votes)

mean_votes


lol = df_train.group_by('patient_id').first()
target_votes = lol.group_by('expert_consensus').len()
total_votes = target_votes['len'].sum()
mean_votes = target_votes.with_columns(pl.col('len')/total_votes)

mean_votes


pqf_spec = pl.read_parquet('/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/1000086677.parquet')
pqf_spec


import polars as pl

pqf_eeg = pl.read_parquet('/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1002576868.parquet')
pqf_eeg




