import pandas as pd

# Load your data
df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')

# Check the first few rows
print(df.head())

# Overview of data types and missing values
print(df.info())

# Summary statistics
print(df.describe())



print(df.isnull().sum())



print(df['Soil Type'].value_counts())
print(df['Crop Type'].value_counts())
print(df['Fertilizer Name'].value_counts())


import matplotlib.pyplot as plt

num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
df[num_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()


df['Soil Type'].value_counts().plot(kind='bar', title='Soil Type Distribution')
plt.show()
df['Crop Type'].value_counts().plot(kind='bar', title='Crop Type Distribution')
plt.show()
df['Fertilizer Name'].value_counts().plot(kind='bar', title='Fertilizer Name Distribution')
plt.show()



import seaborn as sns

corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# Mean of nutrients per fertilizer
print(df.groupby('Fertilizer Name')[num_cols].mean())

# Average nutrient values per crop
print(df.groupby('Crop Type')[num_cols].mean())


for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()


print(pd.crosstab(df['Crop Type'], df['Fertilizer Name']))


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# For every pair of numerical features
for col1, col2 in tqdm(list(itertools.combinations(num_cols, 2)), desc="Plotting pairs"):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col1, y=col2, hue='Fertilizer Name', alpha=0.5)
    plt.title(f'{col1} vs {col2}')
    plt.tight_layout()
    plt.show()


