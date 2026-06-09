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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 
import xgboost as xgb


fertile_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
fertile_trn = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
fertile_txt = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
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
            return f"{'No missing values detected'}"
print(f"Training dataset:\n{get_summary(fertile_trn).data_set()}\n{lent}\nTest dataset:\n{get_summary(fertile_txt).data_set()}")
print(f"{lent}\ncolumns with missing values train\n{lent}\n{get_summary(fertile_trn).total_missing()}\n{lent}\ncolumns with missing values test\n{lent}\n{get_summary(fertile_txt).total_missing()}")


fertile_sub.head(3)


fertile_trn[15:50]


fertile_trn.describe().T


plot_cols = fertile_trn.columns.drop(['id', 'Soil Type', 'Crop Type', 'Fertilizer Name'])
rows = len(plot_cols)
plt.figure(figsize=(15, 3 * rows))

for r, column in enumerate(plot_cols, 1):
    plt.subplot(rows, 2, r)
    if fertile_trn[column].nunique() <= 10:
        sns.countplot(x=column, data=fertile_trn)
    else:
        sns.histplot(x=fertile_trn[column], kde=True, bins=10, color='k')
        
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.show()


for val in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    counts = fertile_trn[val].value_counts()
    colors = plt.cm.tab20.colors[:len(counts)]
    
    plt.pie(counts,
            labels=counts.index,
            autopct='%1.2f%%',
            colors=colors)
    plt.title(f"Distribution by {val}")
    plt.tight_layout()
    plt.show()


#features
X = fertile_trn.drop(['id', 'Fertilizer Name'], axis=1)
#target
y = fertile_trn['Fertilizer Name']


enc = LabelEncoder()
X['Soil Type'] = enc.fit_transform(X['Soil Type'])
X['Crop Type'] = enc.fit_transform(X['Crop Type'])

X_txt = fertile_txt.drop('id', axis=1)
X_txt['Soil Type'] = enc.fit_transform(X_txt['Soil Type'])
X_txt['Crop Type'] = enc.fit_transform(X_txt['Crop Type'])


y = enc.fit_transform(y)
y


#scalling
scale = StandardScaler()
X = scale.fit_transform(X)
X_txt = scale.fit_transform(X_txt)


X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=31)                                                                         


def mapk(y_true, y_pred, k=3):
    N = len(y_true)
    scores = []
    for true, preds in zip(y_true, y_pred):
        score = 0.0
        found = False
        for i, p in enumerate(preds[:k], start=1):
            if p in true and not found:
                score = 1.0 / i
                found = True
                break
        scores.append(score)
    return np.mean(scores)


def map3_scorer(y_true, y_pred_probs):
    y_true_wrapped = [[int(lbl)] for lbl in y_true]
    top3 = np.argsort(-y_pred_probs, axis=1)[:, :3].tolist()
    return mapk(y_true_wrapped, top3, 3)


params = {'n_estimators': 319, 
          'max_depth': 9, 
          'learning_rate': 0.07726214802801594, 
          'subsample': 0.7870950567962739, 
          'colsample_bytree': 0.844347965836792, 
          'min_child_weight': 7, 
          'gamma': 0.5890318133775702, 
          'reg_alpha': 0.9406171225525337, 
          'reg_lambda': 1.6266675838412084,
          'random_state': 31,
          'tree_method': 'hist'  
         }
_model = xgb.XGBClassifier(**params)
_model.fit(X_train, y_train)


y_pred = _model.predict_proba(X_val)
score = map3_scorer(y_val, y_pred)
print(score)


predict = _model.predict_proba(X_txt)
predict


predictions = []
for item in predict:
    top3_indices = np.argsort(item)[::-1][:3]

    names = enc.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
    predictions.append(' '.join(names))   

submission_df = pd.DataFrame({
    'id': fertile_txt['id'],
    'Fertilizer Name': predictions
})
submission_df.to_csv('submission.csv', index=False)
submission_df.head()

