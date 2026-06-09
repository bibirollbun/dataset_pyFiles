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


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print(f'Training Data Shape: {train.shape}')
print(f'Testing Data Shape: {test.shape}')


train.head()


train.info()


for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    print(f'{col} uniques values: {train[col].unique()}')


train.drop('id', axis=1).describe()


train.isna().sum()


numeric_cols = train.drop('id', axis=1).select_dtypes(include='number').columns
numeric_cols


corr = train[numeric_cols].corr()

plt.figure(figsize=(18, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues')
plt.show()


fig, axes = plt.subplots(6, 3, figsize=(18, 24), constrained_layout=True)

for i, col in enumerate(numeric_cols):
    # Barplot: Fertilizer vs numeric col, colored by Soil Type
    sns.barplot(x='Fertilizer Name', y=col, data=train,
                hue='Soil Type', ax=axes[i, 0], palette='viridis')
    axes[i, 0].set_title(f'{col} by Fertilizer and Soil Type')
    axes[i, 0].tick_params(axis='x', rotation=45)
    if i == 0:
        axes[i, 0].legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[i, 0].legend_.remove()

    
    # Barplot: Fertilizer vs numeric col, colored by Crop Type
    sns.barplot(x='Fertilizer Name', y=col, data=train,
                hue='Crop Type', ax=axes[i, 1], palette='viridis')
    axes[i, 1].set_title(f'{col} by Fertilizer and Crop Type')
    axes[i, 1].tick_params(axis='x', rotation=45)
    if i == 0:
        axes[i, 1].legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[i, 1].legend_.remove()

    
    # Boxplot: Numeric col vs Fertilizer
    sns.boxplot(y=col, x='Fertilizer Name', data=train, ax=axes[i, 2], color='skyblue')
    axes[i, 2].set_title(f'{col} Distribution by Fertilizer')
    axes[i, 2].tick_params(axis='x', rotation=45)

plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 5))

sns.countplot(x='Soil Type', data=train, hue='Fertilizer Name', 
              palette='viridis', ax=axes[0])

axes[0].set_title('Fertilizer Distribution Across Soil Types')
axes[0].legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')


sns.countplot(x='Crop Type', data=train, hue='Fertilizer Name',
              palette='viridis', ax=axes[1])

axes[1].set_title('CountPlot of Crop Type')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()



def label_encoder(train, test):
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()
    
    train['Soil Type Label'] = le_soil.fit_transform(train['Soil Type'])
    train['Crop Type Label'] = le_crop.fit_transform(train['Crop Type'])
    train['Fertilizer Name Label'] = le_fert.fit_transform(train['Fertilizer Name'])


    test['Soil Type Label'] = le_soil.transform(test['Soil Type'])
    test['Crop Type Label'] = le_crop.transform(test['Crop Type'])

    return train, test


def total_nutrients(df):
    df['Total_Nutrients'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    return df


def n_k_p_ratios(df):
    df['NPK_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + df['Potassium'] + 1)
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
    df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1)
    df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1)

    return df


def water_balance(df):
    df['Water_Balance'] = df['Moisture'] + df['Humidity']
    return df


def nutrient_per_moisture(df):
    df['Nutrient_Per_Moisture'] = df['Total_Nutrients'] / (df['Moisture'] + 1)
    return df

def climate(df):
    df['Climate'] = (df['Temparature'] + df['Humidity']) / 2
    df['Temp_Humdidity_Ratio'] = df['Temparature'] / (df['Humidity'] + 1)
    return df



def soil_mean(df):
    for col in ['Nitrogen', 'Potassium', 'Phosphorous']:
        df[f'Soil_{col[0:2]}_Mean'] = df.groupby('Soil Type Label')[col].transform('mean')
        df[f'{col[0:2]}_vs_soil_avg'] = df[col] - df[f'Soil_{col[0:2]}_Mean']
    return df


def crop_mean(df):
    for col in ['Nitrogen', 'Potassium', 'Phosphorous']:
        df[f'Crop_{col[0:2]}_Mean'] = df.groupby('Crop Type Label')[col].transform('mean')
        df[f'{col[0:2]}_vs_crop_avg'] = df[col] - df[f'Crop_{col[0:2]}_Mean']

    return df


train, test = label_encoder(train, test)

train = total_nutrients(train)
test = total_nutrients(test)

train = n_k_p_ratios(train)
test = n_k_p_ratios(test)

train = water_balance(train)
test = water_balance(test)

train = nutrient_per_moisture(train)
test = nutrient_per_moisture(test)

train = climate(train)
test = climate(test)

train = soil_mean(train)
test = soil_mean(test)

train = crop_mean(train)
test = crop_mean(test)


train.head()


train.info()


test.info()




