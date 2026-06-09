import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv").drop(columns = ['url', 'license'])
taxonomy_df = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")

train_df = pd.merge(
                train_df,
                taxonomy_df[['primary_label', 'class_name']],
                how = 'left',
                on = ['primary_label']
            )


len(train_df[train_df["secondary_labels"]!="['']"])/ len(train_df)


unique_animal_classses = taxonomy_df.class_name.unique()
for animal in unique_animal_classses:
    print(animal)
    data = train_df[train_df['class_name'] == animal]
    print(data.primary_label.value_counts())
    sns.scatterplot(data = data, x = 'latitude', y = 'longitude', hue = 'primary_label', legend=False)
    plt.show()


train_df.primary_label.value_counts().plot.hist(bins = 300)
print(train_df.primary_label.value_counts().describe())
print('classic')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
le = LabelEncoder().fit(train_df.primary_label)
train_idx, small_test_idx = train_test_split(
    np.arange(len(train_df)),
    train_size = 0.8,
    test_size = 0.2*0.2, 
    random_state = 32, 
    stratify = train_df['primary_label']
)


train_df['primary_label'] = le.transform(train_df['primary_label'])


train_df.iloc[small_test_idx].primary_label.value_counts().plot.hist(bins = 300)


train_df.iloc[train_idx].primary_label.value_counts().plot.hist(bins = 300)


train_df.groupby('class_name')['rating'].describe()


train_df[train_df['rating'] != 0].groupby('class_name')['rating'].describe()


taxonomy_2 = pd.read_excel('/kaggle/input/birdfams/AviList-v2025-11Jun-extended.xlsx', index_col = 'Sequence')



#families by audios
pd.merge(
    left = taxonomy_2, 
    right = train_df, 
    left_on = 'Scientific_name', 
    right_on = 'scientific_name').drop_duplicates(subset = ['filename'])['Family'].value_counts()


#families by species
pd.merge(
    left = taxonomy_2, 
    right = train_df, 
    left_on = 'Scientific_name', 
    right_on = 'scientific_name').drop_duplicates(subset = ['scientific_name'])['Family'].value_counts()

