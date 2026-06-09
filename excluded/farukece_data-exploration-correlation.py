import pandas as pd
import numpy as np 
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
x_numeric = 0


df_train.sample(8)


df_train['education'].unique()


# we may fill those by balance
df_train[df_train['education'] == 'unknown']


ed_bal = df_train[['education', 'balance']]


mapping = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0}
ed_bal['education'] = ed_bal['education'].map(mapping)


correlation = ed_bal.corr(numeric_only=True)

print(correlation)


x = df_train.drop(columns=['y'])
x_num = x.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
y = df_train['y']


# Compute the correlation matrix
corr_matrix = x_num.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

# Add titles and labels
plt.title("Correlation Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Merge y as a new column to x_num
corr_df = x_num.copy()
corr_df["y"] = y

corr_matrix = corr_df.corr()
corr_matrix = corr_matrix[['y']].drop(labels=['y'])

# Plot correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Numericv Correlation r. Target")
plt.tight_layout()
plt.show()


