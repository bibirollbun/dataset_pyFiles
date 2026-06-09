# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv')

df.info()


train = df[df['identity'].notnull()]
test = df[df['identity'].isna()]

print(train.shape) # (13074,8)
print(test.shape) # (2135,8)


# calculate number of identities
print("Total identities: ",len(train['identity'].unique()))


# check the number of images one identity could have
plt.figure(figsize=(12, 5))
sns.histplot(train['identity'].value_counts(), bins=30, kde=False)
plt.title("Distribution of Image Count per Identity")
plt.xlabel("Number of Images")
plt.ylabel("Number of Identities")
plt.show()


train['identity'].value_counts()


def summary_table(df):
    """
    summary table of species v.s. orientation
    """
    summary_df = df.groupby(['orientation','species'],dropna = False).size().unstack(fill_value = 0).sort_index()

    display(summary_table)
    return summary_df


summary_table(train) # missing species, and missing orientation founded


summary_table(test) # missing values under species


def image_show(image_id = None, species = None, orientation = None, source = df):
    """
    Display images.
    """
    root = '/kaggle/input/animal-clef-2025/'

    if source == 'train':
        df_used = train
    elif source =='test':
        df_used = test
    else:
        df_used = df.copy()
        
    if image_id:
        rows = df_used[df_used['image_id']==image_id]

    else:
        
        if species is not None:
            if isinstance(species, float) and np.isnan(species):
                df_used = df_used[df_used['species'].isna()]
            else:
                df_used = df_used[df_used['species'] == species]
        
        if orientation is not None:
            if isinstance(orientation, float) and np.isnan(orientation):
                df_used = df_used[df_used['orientation'].isna()]
            else:
                df_used = df_used[df_used['orientation'] == orientation]
                    
        rows = df_used.sample(n=min(16,len(df_used)))

    n_rows = 4
    n_cols = 4
    size = (224,224)
    selected_rows = rows.to_dict('records')

    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(n_rows, n_cols, i + 1)
        if i < len(selected_rows):
            row = selected_rows[i]
            full_path = os.path.join(root, row['path'])
            try:
                img = Image.open(full_path).convert('RGB')
                img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"[ERROR] Failed to load image at {full_path} – {e}")
                img = Image.new('RGB', size, (150, 150, 150))
            title = f"{row['identity']}\n{row['orientation']}"
        else:
            img = Image.new('RGB', size, (240, 240, 240))
            title = ""

        plt.imshow(img)
        plt.title(title, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    


image_show(species = np.nan,source = 'train') # missing species are comfirmed as salamander


# fill in the missing values for epscies column
condition = (df['species'].isna()) & (df['identity'].str.contains('salamander', case=False, na=False))
df.loc[condition, 'species'] = 'salamander'

train = df[df['identity'].notna()].reset_index(drop=True)
test = df[df['identity'].isna()].reset_index(drop=True)


# double check
summary_table(train)


summary_table(test)


# check species and id distribution

species = df['species'].unique()

for s in species:
    print(df[df['species']==s]['identity'].value_counts())


image_show(species = "lynx", orientation = "right",source = "train")


image_show(species = "lynx", orientation = "front",source = "train")


# Some images in the lynx category are labeled as unknown for orientation, 
# but this feature is clearly visible and should be corrected.
image_show(species = "lynx", orientation = "unknown",source = "train")


image_show(species = "lynx", orientation = "front",source = "test")


image_show(species = "lynx", orientation = "unknown",source = "test")


image_show(species = 'loggerhead turtle',orientation=np.nan, source='train')


image_show(species = 'loggerhead turtle',orientation=np.nan, source='test')

# Most NaN orientation images in the training set are full-body shots with unclear head features, 
# so we only infer orientations in the test set where head views are more visible.


image_show(species = 'loggerhead turtle',orientation='left', source='train')


# check for a single identity
import random

sample_id = random.choice(train['identity'].unique())

sample_rows = train[train['identity'] == sample_id].sample(n=min(10, train[train['identity'] == sample_id].shape[0]), random_state=42)


for idx, row in sample_rows.iterrows():
    image_path = os.path.join('/kaggle/input/animal-clef-2025', row['path'])
    image = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"ID: {row['identity']}\nOrientation: {row['orientation']}")
    plt.axis('off')
    plt.show()





train.to_csv('/kaggle/working/train.csv', index=False)
test.to_csv('/kaggle/working/test.csv', index=False)

