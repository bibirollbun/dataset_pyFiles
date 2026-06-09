import matplotlib.pyplot as plt
import numpy as np

labels = ['BirdCLEF-2025', 'BirdCLEF-2024-extra', 'BirdCLEF-2021', 'BirdCLEF-2024', 'BirdCLEF-2023', 'BirdCLEF-2020', 'BirdCLEF-2022']
sizes = [28552, 83, 62, 13, 10, 1, 0]

colors = plt.cm.Paired(np.linspace(0, 1, len(sizes)))

plt.figure(figsize=(10, 6))
bars = plt.barh(labels, sizes, color=colors, edgecolor='none')
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():,.0f}', va='center', ha='left', fontsize=12, fontweight='bold', color='black')

plt.title('# BirdCLEF of 2025', fontsize=16, fontweight='bold')

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_visible(False)



import numpy as np
import pandas as pd
! rm -rf /kaggle/working/*


cols = ['primary_label', 'secondary_labels', 'record_name', 'source']


train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
train

train['record'] = [filename.split('/')[1] for filename in train.filename]
train['record_name'] = [record.split('.')[0] for record in train.record]

train['secondary_labels'] = [eval(secondary_labels) for secondary_labels in train['secondary_labels']]
train['source'] = 'bc25'
train[cols]


unique_primary_labels = train.primary_label.unique()
len(unique_primary_labels), str(unique_primary_labels)


train_20 = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
train_20

train_20['record_name'] = [record.split('.')[0] for record in train_20.filename]
train_20['primary_label'] = train_20['ebird_code']
train_20['source'] = 'bc20'
train_20 = train_20[~train_20.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_20 = train_20[train_20.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_20[cols]


df = train_20[['species', 'ebird_code']].drop_duplicates().sort_values('species').reset_index(drop=True)
df

species2code = {species : code for species, code in zip(df.species, df.ebird_code)}
species2code

def process_secondary_labels(secondary_labels, species2code):
    secondary_labels = eval(secondary_labels)
    labels = []
    for secondary in secondary_labels:
        label = species2code.get(secondary.split('_')[1], '')
        if label != '':
            labels.append(label)
    return labels

train_20['secondary_labels'] = [process_secondary_labels(secondary_labels, species2code) for secondary_labels in train_20['secondary_labels']]
train_20 = train_20[~train_20.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_20 = train_20[train_20.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_20


train_21 = pd.read_csv('/kaggle/input/birdclef-2021/train_metadata.csv')
train_21
train_21['secondary_labels'] = [eval(secondary_labels) for secondary_labels in train_21['secondary_labels']]
train_21['record_name'] = [record.split('.')[0] for record in train_21.filename]
train_21['source'] = 'bc21'
train_21 = train_21[~train_21.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_21 = train_21[train_21.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_21[cols]


train_22 = pd.read_csv('/kaggle/input/birdclef-2022/train_metadata.csv')
train_22['secondary_labels'] = [eval(secondary_labels) for secondary_labels in train_22['secondary_labels']]
train_22['record_name'] = [record.split('/')[1].split('.')[0] for record in train_22.filename]
train_22['source'] = 'bc22'
train_22 = train_22[~train_22.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_22 = train_22[train_22.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_22[cols]


train_23 = pd.read_csv('/kaggle/input/birdclef-2023/train_metadata.csv')
train_23['secondary_labels'] = [eval(secondary_labels) for secondary_labels in train_23['secondary_labels']]
train_23['record_name'] = [record.split('/')[1].split('.')[0] for record in train_23.filename]
train_23['source'] = 'bc23'
train_23 = train_23[~train_23.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_23 = train_23[train_23.primary_label.isin(train.primary_label.unique())].reset_index(drop=True)
train_23[cols]


train_24 = pd.read_csv('/kaggle/input/birdclef-2024/train_metadata.csv')
train_24['secondary_labels'] = [eval(secondary_labels) for secondary_labels in train_24['secondary_labels']]
train_24['record_name'] = [record.split('/')[1].split('.')[0] for record in train_24.filename]
train_24['source'] = 'bc24'
train_24 = train_24[~train_24.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_24 = train_24[train_24.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_24[cols]


from glob import glob
file_paths = glob("/kaggle/input/birdclef2024-additional-mp3/additional_audio" + "/*/*")
len(file_paths), file_paths[:5]
train_additional = pd.DataFrame({
    'filename' : ['/'.join(filepath.split('/')[-2:]) for filepath in file_paths],
    'species' : [filepath.split('/')[-2] for filepath in file_paths],
    'record' : [filepath.split('/')[-1] for filepath in file_paths],
    'filepath': file_paths,
})
train_additional['source'] = 'bc00'
train_additional['record_name'] = [record.split('.')[0] for record in train_additional.record]
train_additional['primary_label'] = train_additional['species']
train_additional['secondary_labels'] = [[] for _ in train_additional['primary_label']]
train_additional = train_additional[~train_additional.record_name.isin(train.record_name.unique())].reset_index(drop=True)
train_additional = train_additional[train_additional.primary_label.isin(unique_primary_labels)].reset_index(drop=True)
train_additional[cols]


all_train = pd.concat([train_20[cols], 
                       train_21[cols], 
                       train_22[cols], 
                       train_23[cols],
                       train_24[cols],
                       train[cols],
                       train_additional[cols]
                       ]).reset_index(drop=True)
all_train


all_train = all_train.sort_values(['primary_label', 'source'], ascending=False).reset_index(drop=True)
all_train = all_train.drop_duplicates('record_name')
all_train['rank'] = all_train.groupby('primary_label').source.rank(method='first', ascending=False)
all_train


all_train.source.value_counts()


! mkdir /datasets


all_train.to_csv('/datasets/all_train.csv', index=False)


from pathlib import Path
from tqdm import tqdm
import librosa
sr = 32000

def load_audio(record_name, primary_label, source):
    if source == 'bc00':
        pathname = Path('/kaggle/input/birdclef2024-additional-mp3/additional_audio/') / primary_label / (record_name + '.mp3')
    elif source == 'bc20':
        pathname = Path('/kaggle/input/birdsong-recognition/train_audio/') / primary_label / (record_name + '.mp3')
    elif source == 'bc21':
        pathname = Path('/kaggle/input/birdclef-2021/train_short_audio/') / primary_label / (record_name + '.ogg')
    elif source == 'bc22':
        pathname = Path('/kaggle/input/birdclef-2022/train_audio/') / primary_label / (record_name + '.ogg')
    elif source == 'bc23':
        pathname = Path('/kaggle/input/birdclef-2023/train_audio/') / primary_label / (record_name + '.ogg')
    elif source == 'bc24':
        pathname = Path('/kaggle/input/birdclef-2024/train_audio/') / primary_label / (record_name + '.ogg')
    audio = librosa.load(pathname, sr=32000)[0].astype(np.float32)
    return audio


# import os
# lengths = []
# for record_name, primary_label, source in zip(tqdm(all_train.record_name), all_train.primary_label, all_train.source):
#     if source in ['bc00', 'bc20', 'bc21', 'bc22', 'bc23', 'bc24']:
#         audio = load_audio(record_name, primary_label, source)
#         lengths.append(len(audio))
#         save_path = Path('/datasets/') / primary_label
#         os.makedirs(save_path, exist_ok=True)
#         np.save(save_path / ('first_10_' + record_name), audio[: 10 * sr])
#         np.save(save_path / ('last_10_' + record_name), audio[-10 * sr : ])
#         print(record_name, primary_label, source)




