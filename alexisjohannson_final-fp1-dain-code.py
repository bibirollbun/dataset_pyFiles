import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt


#code for GRDA DATA for 1 Participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_GRDA = pd.read_parquet(BASE_PATH + 'train_eegs/4030851372.parquet')
df_GRDA.head()


#code for GRDA Fp1 for 1 participant Spectrogram
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

channel_name = 'Fp1' 
signal = df_GRDA[channel_name].dropna().values 

fs = 200  

frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title(f'GRDA Spectrogram of {channel_name}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()


#Code for Seizure Data for 1 participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_seizure = pd.read_parquet(BASE_PATH + 'train_eegs/266631836.parquet')
df_seizure.head()


#code for Spectrogram of Seizure Fp1 Data of 1 Participant
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np


signal = df_seizure[channel_name].dropna().values  

fs = 200  

frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title(f'Seizure Spectrogram of {channel_name}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()      
plt.show()


#Code for LPD Data for 1 participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_LPD = pd.read_parquet(BASE_PATH + 'train_eegs/1317431280.parquet')
df_LPD.head()


#Code for Spectrogram of LPD Fp1 Data for 1 Participant 
import numpy as np

channel_name = 'Fp1' 
signal = df_LPD[channel_name].dropna().values  

fs = 200  

frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title(f'LPD Spectrogram of {channel_name}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()


#Code for GPD Data for 1 participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_GPD = pd.read_parquet(BASE_PATH + 'train_eegs/2846570074.parquet')
df_GPD.head()


#Code for Spectrogram of GPD Fp1 Data for 1 Participant 
import numpy as np

channel_name = 'Fp1'
signal = df_GPD[channel_name].dropna().values 

fs = 200  

frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title(f'GPD Spectrogram of {channel_name}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()


#Code for LRDA Data for 1 participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_LRDA = pd.read_parquet(BASE_PATH + 'train_eegs/2222924277.parquet')
df_LRDA.head()


#Code for Spectrogram of LRDA Fp1 Data for 1 Participant 
import numpy as np

channel_name = 'Fp1'
signal = df_LRDA[channel_name].dropna().values 

fs = 200  

frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
plt.title(f'LRDA Spectrogram of {channel_name}')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()


#Code for LRDA and Seizure Fp1 Data Table
import numpy as np
import pandas as pd

eeg_data = {
    '266631836': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/266631836.parquet', 'label': 'Seizure'},
    '2222924277': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/2222924277.parquet', 'label': 'LRDA'},
    '2894007647': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/2894007647.parquet', 'label': 'Seizure'},
    '722738444': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/722738444.parquet', 'label': 'LRDA'},
    '338161210': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/338161210.parquet', 'label': 'LRDA'},
    '2088807520': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/2088807520.parquet', 'label': 'Seizure'},
    '3030710864': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/3030710864.parquet', 'label': 'Seizure'},
    '3190279138': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/3190279138.parquet', 'label': 'Seizure'},
    '1844014178': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1844014178.parquet', 'label': 'LRDA'},
    '2622179549': {'path': '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/2622179549.parquet', 'label': 'LRDA'},
}

min_length = None
for pid, info in eeg_data.items():
    try:
        df = pd.read_parquet(info['path'])
        if 'Fp1' in df.columns:
            length = len(df['Fp1'])
            if min_length is None or length < min_length:
                min_length = length
    except Exception as e:
        print(f"Failed to load {pid}: {e}")

print(f"Minimum Fp1 length: {min_length}")
data_rows = []

for pid, info in eeg_data.items():
    try:
        df = pd.read_parquet(info['path'])
        if 'Fp1' not in df.columns:
            print(f"Skipping {pid} — no Fp1 column")
            continue

        fp1_values = df['Fp1'].values[:min_length]
        row = {
            'Patient ID': pid,
            'Label': info['label']
        }
        for t in range(min_length):
            row[f'Time {t}'] = fp1_values[t]

        data_rows.append(row)
    except Exception as e:
        print(f"Skipping {pid} due to error: {e}")

structured_df = pd.DataFrame(data_rows)

print(f"Final table shape: {structured_df.shape}")
structured_df.head()


#Code for PCA Table of Seizure and LRDA Fp1 Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features_only = structured_df.drop(columns=['Patient ID', 'Label']).fillna(structured_df.mean(numeric_only=True))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_only)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['Patient ID'] = structured_df['Patient ID'].values
pca_df = pca_df.set_index('Patient ID')
pca_df['Label'] = structured_df['Label'].values

pca_df


#Code for Spectrogram of PCA Fp1 Seizure v. LRDA data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Label',
    palette={'Seizure': 'red', 'LRDA': 'blue'},
    s=100,
    edgecolor='black'
)

plt.title('PCA of EEG Data (Fp1) - Seizure vs LRDA', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Label')
plt.tight_layout()
plt.show()


#Code for spectrogram that Classifies Fake Points into the PCA FP1 Data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(pca_df['Label']) 

X = pca_df[['PC1', 'PC2']].values
X = np.array(X).astype(float)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y_encoded)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min - 35, x_max, 300),
                     np.linspace(y_min - 30, y_max, 300))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

sns.set_style("white")
sns.scatterplot(
    x=X[:, 0], y=X[:, 1], hue=pca_df['Label'],
    palette={'Seizure': 'red', 'LRDA': 'blue'},
    edgecolor='black', s=100
)

n_random_points = 20
random_points = np.random.uniform(
    low=(x_min, y_min),
    high=(x_max, y_max),
    size=(n_random_points, 2)
)

random_preds = clf.predict(random_points)
predicted_labels = le.inverse_transform(random_preds)
color_map = {'Seizure': 'red', 'LRDA': 'blue'}

for i, (point, label) in enumerate(zip(random_points, predicted_labels)):
    plt.scatter(point[0], point[1], 
                color=color_map[label], 
                s=150, 
                edgecolor='black', 
                marker='X',
                label='Random Test Point' if i == 0 else ""
               )

plt.title("Decision Boundary: PCA Components (Seizure vs LRDA) Fp1", fontsize=16)
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


#We then wanted to look at the T3 Data only for LRDA and Seizure Participants
#The T3 data is in the other notebook

