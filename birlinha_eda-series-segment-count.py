import os
import pandas as pd


os.listdir('/kaggle/input/rsna-intracranial-aneurysm-detection')


def create_iterator(df):
    return df[['SeriesInstanceUID']]


train_path = '/kaggle/input/rsna-intracranial-aneurysm-detection'
train_df = pd.read_csv(os.path.join(train_path,'train.csv'))

df_analysis = create_iterator(train_df)

for idx in tqdm(df_analysis.index, desc="loading..."):

    scan_id = df_analysis.at[idx,'SeriesInstanceUID']

    try:
        len_segmentations = len([f for f in os.listdir(os.path.join(train_path,'segmentations',scan_id))])
    except:
        len_segmentations = 0

    try:
        len_series = len([f for f in os.listdir(os.path.join(train_path,'series',scan_id))])
    except:
        len_series = 0
    
    df_analysis.loc[idx,'segmentations'] = len_segmentations
    df_analysis.loc[idx,'series'] = len_series


df_analysis


df_analysis['segmentations'].unique()


print(df_analysis.sort_values(by='segmentations').tail(1)['SeriesInstanceUID'].loc[2578])


os.listdir('/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.49718418682238683779854914910561017368')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(data=df_analysis, x='segmentations', palette='Set2')

plt.title('Frequency of Segmentations')
plt.xlabel('Segmentation Present (0 = No, 1 = Yes)')
plt.ylabel('Number of Series')
plt.xticks([0, 1], ['No', 'Yes'])  # Optional: label formatting
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 4))
sns.histplot(data=df_analysis, x='series', bins=30, kde=False, color='skyblue')

plt.title('Distribution of Series Values')
plt.xlabel('Series')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


