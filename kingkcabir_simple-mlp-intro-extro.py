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


'''LOADING THE DATASET AND GETTING BASIC INFORMATION ABOUT THE DATASET IN 
   BOTH TRAIN AND TEST FILE.'''

#submission path
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
#train.csv path
df_1 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
#test.csv path
df_2 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

lent = '*'*40
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
        if not cols_with_missing.empty:
            return cols_with_missing.to_dict()
        else:
            return f"{'......No missing values detected......'}"
print(f"Training dataset:\n{get_summary(df_1).data_set()}\n{lent}\nTest dataset:\n{get_summary(df_2).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(df_1).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(df_2).total_missing()}")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


'''Both datasets contain missing values in each columns.
   next step is to fill up the missing columns.
   mean for numerical values and mode for categorical values'''
class fill_null_value(get_summary):
    def __init__(self, x):
        self.x = x.copy() #make a copy of incoming data
        self.x.drop('id', axis=1, inplace=True) #dropping the id column
    def fill_missing(self):
        for val in self.x.columns: #loop through columns
            if self.x[val].isnull().any():# check for empty values
                if self.x[val].dtype == 'float': 
                    self.x[val].fillna(self.x[val].mean(), inplace=True)
                elif self.x[val].dtype == 'object':
                    mode_ = self.x[val].mode()
                    if not mode_.empty:
                        self.x[val].fillna(mode_[0], inplace=True)

        return self.x

#calling the function to a new df_1 path
cleaned_df_1 = fill_null_value(df_1).fill_missing()
cleaned_df_1.isna().sum().any(), cleaned_df_1[:10]


#calling the function to a new df_2 path
cleaned_df_2 = fill_null_value(df_2).fill_missing()
cleaned_df_2.isna().sum().any()


#distribution of personality
for val in ['Personality']:
    counts = cleaned_df_1[val].value_counts()
    colors = plt.cm.tab20.colors[:len(counts)]
    
    plt.pie(counts,
            labels=counts.index,
            autopct='%1.2f%%',
            colors=colors)
    plt.title(f"Distribution by {val}")
    plt.tight_layout()
    plt.show()


#encoding train.csv
lab_enc = LabelEncoder()#encoder
encoded_cols = cleaned_df_1.drop('Personality', axis=1)#columns to encode
def encode_(data):
    for cols in data.columns:
        if data[cols].dtype == 'object':
            data[cols] = lab_enc.fit_transform(data[cols])
    return data

X = encode_(encoded_cols)
X.head(2)


#encoding test.csv
X_test = encode_(cleaned_df_2)
X_test.head(2)


y = lab_enc.fit_transform(cleaned_df_1['Personality'])

#splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)


#model
mlp_clf = MLPClassifier(solver='lbfgs',
                        random_state=12,
                        alpha=1e-5,
                        max_iter=15,
                        warm_start=True)
mlp_clf.fit(X_train, y_train)


preds = mlp_clf.predict_proba(X_val)
preds_labels = (preds[:, 1] > 0.5).astype(int)
acc_score = accuracy_score(y_val, preds_labels)
print(f"ACCURACY: {acc_score:.2f}%")


#prediction on test.csv
prediction = mlp_clf.predict_proba(X_test)
prediction_labels = (prediction[:, 1] > 0.5).astype(int)


#submission
submission = df_sub
submission['Personality'] = lab_enc.inverse_transform(prediction_labels)
submission.head(4)


submission.to_csv("submission.csv", index=False)

