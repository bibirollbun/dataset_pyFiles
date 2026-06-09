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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency, pointbiserialr

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)



df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


df.describe()


df.duplicated().sum()


df.isnull().sum()


df.nunique()


target_col = ['Personality']
numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
nominal_cols = [
    'Stage_fear',
    'Drained_after_socializing'
]


num_imputer = SimpleImputer(strategy='median')
df_numeric = pd.DataFrame(
    num_imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)


df_nominal = df[nominal_cols].fillna('Missing').copy()
label_encoders = {}
for col in nominal_cols:
    le = LabelEncoder()
    df_nominal[col] = le.fit_transform(df_nominal[col])
    label_encoders[col] = le 


df = pd.concat([
    df_numeric,
    df_nominal,
    df['Personality']    
], axis=1)


df.head()


all_cols = numeric_cols + nominal_cols

n = len(all_cols)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
axes = axes.flatten()
blue_shade = '#004DAB'

for ax, col in zip(axes, all_cols):
    if col in numeric_cols:
        ax.hist(df[col], bins=30, color=blue_shade, edgecolor='black')
        ax.set_ylabel('Frequency')
    else:
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=blue_shade)
        ax.set_ylabel('Count')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel(col)
    ax.tick_params(axis='x', rotation=0)

for ax in axes[n:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()


df['y'] = le.fit_transform(df['Personality'].values.ravel())

cols = numeric_cols + nominal_cols
ncols = 2
nrows = (len(cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
axes = axes.flatten()
blue_shade = '#004DAB'
orange_shade = '#FFA500'

for ax, col in zip(axes, cols):
    if col in numeric_cols:
        counts = pd.crosstab(df[col].astype(int), df['y'])
        counts.plot(kind='bar', stacked=True, color=[blue_shade, orange_shade], ax=ax, legend=False)
    else:
        counts = pd.crosstab(df[col], df['y'])
        counts.plot(kind='bar', stacked=True, color=[blue_shade, orange_shade], ax=ax, legend=False)
    
    ax.set_ylim(0, counts.values.sum(axis=1).max() * 1.1)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, ['Introvert (0)', 'Extrovert (1)'], loc='upper right', title='Personality')
plt.tight_layout()
plt.show()


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
nominal_cols = ['Stage_fear','Drained_after_socializing']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for c in nominal_cols:
    df[c] = df[c].fillna('Missing')
le_target = LabelEncoder()
y = le_target.fit_transform(df['Personality'])
X = df[numeric_cols + nominal_cols]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
pre = ColumnTransformer([('num',StandardScaler(),numeric_cols),
                         ('cat',OneHotEncoder(handle_unknown='ignore'),nominal_cols)])
pipe = Pipeline([('prep',pre),
                 ('clf',LogisticRegression(max_iter=1000,random_state=42))])
param_grid = {'clf__C':[0.01,0.1,1,3,10,30]}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
grid = GridSearchCV(pipe,param_grid=param_grid,cv=cv,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

best_model = grid.best_estimator_
best_C = grid.best_params_['clf__C']
y_pred = best_model.predict(X_test)
print(best_C, accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=le_target.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
nominal_cols = ['Stage_fear','Drained_after_socializing']

train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
test_df[numeric_cols]  = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
for col in nominal_cols:
    train_df[col] = train_df[col].fillna('Missing')
    test_df[col]  = test_df[col].fillna('Missing')

target_le = LabelEncoder()
y_full = target_le.fit_transform(train_df['Personality'])
X_full = train_df[numeric_cols + nominal_cols]

pre = ColumnTransformer([('num',StandardScaler(),numeric_cols),
                         ('cat',OneHotEncoder(handle_unknown='ignore'),nominal_cols)])
final_pipe = Pipeline([('prep',pre),
                       ('clf',LogisticRegression(C=best_C,max_iter=1000,random_state=42))])
final_pipe.fit(X_full,y_full)

pred_num = final_pipe.predict(test_df[numeric_cols + nominal_cols])
pred_lbl = target_le.inverse_transform(pred_num)
submission = pd.DataFrame({'id':test_df['id'],'Personality':pred_lbl})
submission.to_csv('/kaggle/working/submission.csv',index=False)
print('/kaggle/working/submission.csv saved')


