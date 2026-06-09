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


subs = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
train_fall = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_fall = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
#DATASETS SUMMARY
class get_summary:
    def __init__(self, x):
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame()
    def data_set(self):
        #checks for duplicate
        duplicate = self.x.duplicated().any()
        #drop duplicates 
        if duplicate == True:
            self.x.drop_duplicates(inplace=True)
            self.x.reset_index(drop=True)
        #checks for empty values
        null = self.x.isna().sum().any()
        #missing values
        total_missing = self.x.isnull().sum().sum()
        #data types
        data_type = self.x.dtypes
        #shape
        shapes = self.x.shape
        return f"Duplicate: {duplicate}\nNull: {null}\nMissing_value: {total_missing}\nTypes:\n{data_type}\nShape: {shapes}"
     #missing values
    def total_missing(self):
        missing_vals = self.x.isnull().sum()
        cols_with_missing = missing_vals[missing_vals > 0]
        return cols_with_missing.to_dict()
print(f"Training dataset:\n{get_summary(train_fall).data_set()}\nTest dataset:\n{get_summary(test_fall).data_set()}")
print(f"columns with missing values train\n{get_summary(train_fall).total_missing()}\ncolumns with missing values test\n{get_summary(test_fall).total_missing()}")


train_fall.describe().T


import seaborn as sns
import matplotlib.pyplot as plt

#visualizing th distribution of each columns
sns.set_style('darkgrid')
plot_cols = train_fall.columns.drop('id')
_rows = len(plot_cols)
plt.figure(figsize=(15, 3 * _rows))

for r, column in enumerate(plot_cols, 1):
    plt.subplot(_rows, 2, r)
    if train_fall[column].nunique() <= 10:
        sns.countplot(x=column, data=train_fall)
    else:
        sns.histplot(x=train_fall[column], kde=True, bins=10, color='m')
        
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler 

features = train_fall.drop(['id', 'rainfall'], axis=1)
y = train_fall.rainfall
scala = MinMaxScaler(feature_range=(0,1))

X = pd.DataFrame(scala.fit_transform(features))
X.head(3)


from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=31)


from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train, y_train)
preds = LDA_model.predict(X_val)
score = roc_auc_score(y_val, preds)
print(f"LDA_SCORE: {score:.3f}%")


from sklearn.gaussian_process import GaussianProcessClassifier

model_gaus = GaussianProcessClassifier(optimizer='fmin_l_bfgs_b',
                                       max_iter_predict=500,
                                       copy_X_train=True,
                                       random_state=31,
                                       n_jobs=-1)
model_gaus.fit(X_train, y_train)
preds_gaus = model_gaus.predict(X_val)
score_gaus = roc_auc_score(y_val, preds_gaus)
print(f"GAUS_SCORE: {score_gaus:.3f}%")


test_fall['winddirection'] = test_fall['winddirection'].fillna(test_fall['winddirection'].mean())
X_test = pd.DataFrame(scala.fit_transform(test_fall.drop('id', axis=1)))


prediction = model_gaus.predict(X_test)


submit = pd.DataFrame({'id': test_fall['id'], 
                      'rainfall': prediction})
submit.to_csv("submission.csv", index=False)
submit.head(2)

