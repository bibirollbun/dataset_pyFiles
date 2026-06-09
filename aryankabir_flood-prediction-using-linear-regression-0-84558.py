import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# to avoid warning
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s4e5/train.csv")
df.head()


# Shape of Data
df.shape


df.columns


# Information about dataset
df.info()


# Checking null values
df.isna().sum()


# Checking duplicate values
df.duplicated().sum()


df.drop('id',axis=1,inplace=True)


# Summary of Data
df.describe().T


# Define number of rows and columns for the grid
nrows = (len(df.columns) + 2) // 3  # 3 graphs per row
ncols = min(len(df.columns), 3)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

# Flatten axes if necessary
if nrows == 1:
    axes = [axes]

# Plot distribution for each feature
for i, col in enumerate(df.columns):
    row_index = i // ncols
    col_index = i % ncols
    ax = axes[row_index][col_index]
    sns.histplot(df[col],bins=40, kde=True, ax=ax)
    ax.set_title(col + ' Distribution')
    ax.set_ylabel('Frequency')
    ax.set_xlabel(col)

# Remove empty subplots if ncols * nrows > number of features
if nrows * ncols > len(df.columns):
    for i in range(len(df.columns), nrows * ncols):
        fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show();


# Define target column
target_column = 'FloodProbability'

# Define number of rows and columns for the grid
nrows = (len(df.columns) - 1 + 2) // 3  # 3 graphs per row
ncols = min(len(df.columns) - 1, 3)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

# Flatten axes if necessary
if nrows == 1:
    axes = [axes]

# Plot scatter plot for each feature with the target column
for i, col in enumerate(df.columns):
    if col == target_column:
        continue
    row_index = i // ncols
    col_index = i % ncols
    ax = axes[row_index][col_index]
    sns.scatterplot(x = df[col], y = df[target_column],ax=ax)
    ax.set_title(f'{col} vs {target_column}')
    ax.set_xlabel(col)
    ax.set_ylabel(target_column)

# Remove empty subplots if ncols * nrows > number of features
if nrows * ncols > len(df.columns) - 1:
    for i in range(len(df.columns) - 1, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show();


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True,fmt='.2f', cmap='coolwarm', vmax=1, vmin=-1)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title("Correlation Heatmap")
plt.show()
;


X = df.drop(columns= ['FloodProbability']) #features
y = df['FloodProbability'] #target


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.85,random_state=0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


lr.fit(X_train,y_train)


y_pred_lr = lr.predict(X_test)


from sklearn.metrics import mean_absolute_percentage_error,r2_score


mape = mean_absolute_percentage_error(y_test,y_pred_lr)
print("Error of Linear Regression Model = %.2f"%(mape*100),'%')
print("Accuracy of Linear Regression Model = %.2f"%((1 - mape)*100),'%')


r2 = r2_score(y_test,y_pred_lr)
print("R2 score of Linear Regression = %.2f"%(r2))


df2 = pd.read_csv("/kaggle/input/playground-series-s4e5/test.csv")
df2.head()


df2.shape


df2.info()


df2.isna().sum()


test_data = df2.drop(columns=['id'])


test_data.describe().T


# Define number of rows and columns for the grid
nrows = (len(test_data.columns) + 2) // 3  # 3 graphs per row
ncols = min(len(test_data.columns), 3)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

# Flatten axes if necessary
if nrows == 1:
    axes = [axes]

# Plot distribution for each feature
for i, col in enumerate(test_data.columns):
    row_index = i // ncols
    col_index = i % ncols
    ax = axes[row_index][col_index]
    sns.histplot(test_data[col],bins=40, kde=True, ax=ax)
    ax.set_title(col + ' Distribution')
    ax.set_ylabel('Frequency')
    ax.set_xlabel(col)

# Remove empty subplots if ncols * nrows > number of features
if nrows * ncols > len(test_data.columns):
    for i in range(len(test_data.columns), nrows * ncols):
        fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show();


plt.figure(figsize=(15, 10))
sns.heatmap(test_data.corr(), annot=True,fmt='.2f', cmap='coolwarm', vmax=1, vmin=-1)
plt.xticks(rotation=45)
;


pred_lr = lr.predict(test_data)


output = pd.DataFrame({
    'id' : df2.id,
    'FloodProbability' : pred_lr
})


output.to_csv('submission.csv',index=False)




