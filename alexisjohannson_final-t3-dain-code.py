#Code for LRDA Data for 1 participant Table
import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt

BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_LRDA = pd.read_parquet(BASE_PATH + 'train_eegs/2222924277.parquet')
df_LRDA.head()


#Code for Spectrogram of LRDA T3 Data for 1 Participant
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

channel_name = 'T3' 
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


#Code for Seizure Data for 1 participant Table
BASE_PATH = '/kaggle/input/hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_seizure = pd.read_parquet(BASE_PATH + 'train_eegs/266631836.parquet')
df_seizure.head()


#Code for Spectrogram of Seizure T3 Data for 1 Participant
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

channel_name = 'T3' 
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


#Code for LRDA and Seizure T3 Data Table
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
        if 'T3' in df.columns:
            length = len(df['T3'])
            if min_length is None or length < min_length:
                min_length = length
    except Exception as e:
        print(f"Failed to load {pid}: {e}")

print(f"Minimum T3 length: {min_length}")

data_rows = []

for pid, info in eeg_data.items():
    try:
        df = pd.read_parquet(info['path'])
        if 'T3' not in df.columns:
            print(f"Skipping {pid} — no T3 column")
            continue

        T3_values = df['T3'].values[:min_length]
        row = {
            'Patient IDs': pid,
            'Labels': info['label']
        }
        for t in range(min_length):
            row[f'Time {t}'] = T3_values[t]

        data_rows.append(row)
    except Exception as e:
        print(f"Skipping {pid} due to error: {e}")

final_df = pd.DataFrame(data_rows)

print(f"Final table shape: {final_df.shape}")
final_df.head()


#Code for PCA Table of Seizure and LRDA T3 Data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features_only = final_df.drop(columns=['Patient IDs', 'Labels']).fillna(final_df.mean(numeric_only=True))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_only)

pca2 = PCA(n_components=2)
principal_components = pca2.fit_transform(scaled_features)

pca2_df = pd.DataFrame(principal_components, columns=['PC3', 'PC4'])
pca2_df['Patient IDs'] = final_df['Patient IDs'].values
pca2_df = pca2_df.set_index('Patient IDs')
pca2_df['Labels'] = final_df['Labels'].values

pca2_df


#Code for Spectrogram of PCA T3 Seizure v. LRDA data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca2_df,
    x='PC3',
    y='PC4',
    hue='Labels',
    palette={'Seizure': 'red', 'LRDA': 'blue'},
    s=100,
    edgecolor='black'
)

plt.title('PCA of EEG Data (T3) - Seizure vs LRDA', fontsize=16)
plt.xlabel('Principal Component 3', fontsize=12)
plt.ylabel('Principal Component 4', fontsize=12)
plt.legend(title= 'Labels')
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(pca2_df['Labels']) 

X = pca2_df[['PC3', 'PC4']].values
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
    x=X[:, 0], y=X[:, 1], hue=pca2_df['Labels'],
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

plt.title("Decision Boundary: PCA Components (Seizure vs LRDA) T3", fontsize=16)
plt.xlabel("PC3", fontsize=12)
plt.ylabel("PC4", fontsize=12)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()







