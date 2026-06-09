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


!pip install pycaret pandas scikit-learn matplotlib seaborn

import warnings


import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')
t = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df.info()


t.info()


t.isnull().sum()


print(set(df.columns) & set(t.columns))  # Check common columns



# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()
t.columns = t.columns.str.strip()

# Ensure index alignment
df = df.reset_index(drop=True)
t = t.reset_index(drop=True)

# Concatenate along columns
train = pd.concat([df, t], axis=1)

# Handle duplicate columns: Take first non-null value for each duplicate column
train = train.groupby(train.columns, axis=1).first()

# Verify result
print(train.info())



train.describe().T


train["rainfall"] = train["rainfall"].map({"yes": 1, "no": 0, 1: 1, 0: 0})




train.isnull().sum()


sns.countplot(x= train['rainfall'])


train = train.drop(columns = ['id'],axis =1)


fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 20))
axes = axes.flatten()

# Plot each column
for i, (col, values) in enumerate(train.items()):
    sns.histplot(values, ax=axes[i], kde=True)  # Replaces deprecated sns.distplot
    axes[i].set_title(col)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# create box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in train.items():
    sns.boxplot(y=col, data=train, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


!pip install --upgrade pycaret


!pip install pandas==1.5.3
import pandas as pd
print(pd.__version__)


from pycaret.classification import *  # Import everything from classification module

clf = setup(train, target='rainfall')




compare_models()


## select the best model

model = create_model('lr')


best_model = tune_model(model)


evaluate_model(best_model)


test = test.drop(columns =['id'], axis=1)


test.isnull().sum()


test["winddirection"] = test["winddirection"].fillna(test["winddirection"].mean())



# Get feature columns from train (excluding target 'rainfall')
train_features = train.drop(columns=["rainfall"], errors="ignore").columns

# Ensure test has only those columns, in the correct order
test = test[train_features]



predictions =best_model.predict(test)





submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission['rainfall'] = predictions




  # Reset index and ensure columns are correctly formatted
submission = submission.reset_index(drop=True)  # Reset index completely
submission.columns = submission.columns.astype(str)  # Ensure column names are strings
submission = submission.copy()  # Ensure no memory issues

    # Convert DataFrame to CSV safely
submission.to_csv('submission.csv', index=False)

    # Display submission head
print("\nSubmission File Head:")
print(submission.head())





