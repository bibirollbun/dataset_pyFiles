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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



def boxplot_distribution(df, var):

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=var, y='Calories', data=df, color='skyblue')
    plt.title(f'Boxplot of Target by {var}')
    # rotate x labels for better readability
    plt.xticks(rotation=90)
    plt.show()


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')

train['set'] = 'train'
test['set'] = 'test'

df = pd.concat([train, test], axis=0, ignore_index=True)


var = 'Duration'
boxplot_distribution(df, var)



var = 'Heart_Rate'
df['Heart_Rate'] = df['Heart_Rate'].astype(int)
boxplot_distribution(df, var)


var = 'Body_Temp'
df['Body_Temp'] = df['Body_Temp'].round(1)
boxplot_distribution(df, var)


df["Duration_m_Heart_Rate"] = df["Duration"] * df["Heart_Rate"]

var = 'Duration_m_Heart_Rate'

# scatter plot calories vs age
plt.figure(figsize=(8, 5))
plt.scatter(df[var], df['Calories'])
plt.title(f'Calories vs {var}')
plt.xlabel(var)
plt.ylabel('Calories')

# plot line of best fit
sns.regplot(x=df[var], y=df['Calories'], scatter=False, color='red')

# print over the scatter plot the correlation coefficient
corr = df[var].corr(df['Calories'])
plt.text(0.15, 0.85, f'Correlation: {corr:.2f}', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
plt.show()


var = 'Duration_m_Heart_Rate'

# Map categorical values in the hue column to numeric values
hue_colors = df["Age"]

# Scatter plot calories vs var
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df[var], df['Calories'], c=hue_colors, cmap='viridis')
plt.title(f'Calories vs {var}')
plt.xlabel(var)
plt.ylabel('Calories')

# Add legend
handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title='Age',loc='upper left')
plt.show()


var = 'Duration_m_Heart_Rate'
hue_var = 'Body_Temp'

# Map categorical values in the hue column to numeric values
hue_colors = df[hue_var]

# Scatter plot calories vs var
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df[var], df['Calories'], c=hue_colors, cmap='viridis')
plt.title(f'Calories vs {var}')
plt.xlabel(var)
plt.ylabel('Calories')

# Add legend
handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title=hue_var, loc='upper left')
plt.show()


var = 'Duration_m_Heart_Rate'
hue_var = 'Sex'



df_plot = df.copy()
df_plot = df_plot[:]
df_plot.dropna(inplace=True)
# Map categorical values in the hue column to numeric values
hue_colors = df_plot[hue_var].map({"male": 0, "female": 1})
# Scatter plot calories vs var
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df_plot[var], df_plot['Calories'], c=hue_colors, cmap='viridis')
plt.title(f'Calories vs {var}')
plt.xlabel(var)
plt.ylabel('Calories')

# Add legend
handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title=hue_var)

plt.show()


for sex in ['male', 'female']:
    # X: array de entrada, y: variable dependiente
    df_plot = df.copy()
    df_plot = df_plot[df_plot['Sex'] == sex][['Duration_m_Heart_Rate', 'Calories']]
    df_plot.dropna(inplace=True)
    X = df_plot['Duration_m_Heart_Rate'].values
    y = df_plot['Calories'].values
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))

    residuals = y - y_pred  # o: df['y'] - y_pred
    residual_std = np.std(residuals).round(2)
    print(f"Standard deviation of residuals: {residual_std}")
    plt.figure(figsize=(8, 5))
    plt.scatter(X, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs Duration_m_Heart_Rate - sex {sex}: {residual_std}')

