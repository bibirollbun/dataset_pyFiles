import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()


df.info()


# Changing rainfall from binary to text

df['rainfall_label'] = df['rainfall'].map({1: 'yes', 0:'no'})
df.head()


df.describe()


# Dropping id since we might not need it is only series of number from 0-2189, and we might not need it
df = df.drop(columns=['id'])
df.head()


import matplotlib.pyplot as plt
import seaborn as sns
import math

def numerical_analysis(df, plot_type='hist', bins=30):
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_cols = len(numerical_cols)
    
    # Define grid size
    cols = min(3, num_cols)  # Maximum of 3 columns per row
    rows = math.ceil(num_cols / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_cols > 1 else [axes]
    
    for i, col in enumerate(numerical_cols):
        if plot_type == 'hist':
            sns.histplot(df[col], bins=bins, kde=True, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


numerical_analysis(df, plot_type='hist')


numerical_analysis(df, plot_type='box')


plt.figure(figsize=(8, 5))
sns.countplot(x=df['rainfall_label'], palette='Paired')
plt.xlabel('Rainfall Category')
plt.ylabel('Count')
plt.title('Rainfall Counts')


num_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12,8))
sns.heatmap(num_df.corr(), annot=True, fmt='.2f')
plt.title('Feature Correlation')
plt.show()


df.head()


df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
df['temp_range'] = df['maxtemp'] - df['mintemp']
df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-6)
df['wind_power'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
df['humid_cloud'] = df['humidity'] * df['cloud']
df.head()


new_feature = ['day_sin', 'day_cos', 'temp_range', 'temp_dew_diff', 'cloud_sun_ratio',
              'wind_power', 'humid_cloud','rainfall']
plt.figure(figsize=(12,8))
sns.heatmap(df[new_feature].corr(), annot=True, fmt='.2f')
plt.title('New Feature Correlation')
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, fmt='.2f')
plt.title('All Feature Correlation')
plt.show()


numerical_analysis(df, plot_type='hist')


numerical_analysis(df, plot_type='box')

