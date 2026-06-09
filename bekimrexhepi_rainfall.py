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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df_train.info()
df_train.describe()


df_train.isnull().sum() > 0
df_train.head()


df_train.drop(['id'], axis=1, inplace=True)
df_train.shape


df_train.drop_duplicates(inplace=True)
df_train.shape


import matplotlib.pyplot as plt
import seaborn as sns


def boxplots_for_features(df):

    features = list(set(df.columns.to_list())- set(['rainfall', 'day']))


    for x in range(0, len(features), 2):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
        sns.boxplot(df_train[features[x]], ax=ax1)
        sns.boxplot(df_train[features[x+1]], ax=ax2)

boxplots_for_features(df_train)


# sns.countplot(x='rainfall',
#               hue=df_train['rainfall'],
#                 data=df_train, 
#                 palette='pastel', 
#                 legend=False)

sns.countplot(x='rainfall',
              hue=df_train['rainfall'],
                data=df_train, 
                palette='pastel')


def remove_outliers(df):
    print(df.shape)
    features = ['pressure', 'humidity', 'dewpoint']
    # features = list(set(df.columns.to_list())- set(['rainfall', 'day']))


    for feature in features:
        q_low, q_high = df[feature].quantile([0.01, 0.99])
        mask = (df[feature] < q_low) | (df[feature] > q_high)
        df = df[~mask]
        print(mask.count())
        
    print(df.shape)
    return df

df_train_2 = remove_outliers(df_train)

# df_train_2 = df_train.copy()



corr = df_train_2.corr(numeric_only=True)
threshold = 0.3

mask = (corr.abs() > threshold) & (corr.abs() < 1)
strong_corr = corr[mask]
sns.heatmap(strong_corr, annot=True, cmap='coolwarm')



df_train_2.info()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_columns = df_train_2.select_dtypes(['float64', 'int64']).columns.to_list()
numerical_columns = list(set(numerical_columns)-set(['rainfall']))

num_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num_cols', num_transformer, numerical_columns)
])


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = df_train_2[numerical_columns]
y = df_train_2['rainfall']

rainfall_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression())
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

scores = cross_val_score(rainfall_pipeline, X, y, cv=skf, scoring='roc_auc')
print("Accuracy per fold:", scores)
print("Mean accuracy:", scores.mean())




x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)

rainfall_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression())
])

rainfall_pipeline.fit(x_train, y_train)
y_pred_prob = rainfall_pipeline.predict_proba(x_val)[:,1]



from sklearn.metrics import roc_auc_score

roc_score = roc_auc_score(y_val, y_pred_prob)
print(roc_score)


import numpy as np


preprocessor = rainfall_pipeline.named_steps['preprocess']
model = rainfall_pipeline.named_steps['model']
features = preprocessor.get_feature_names_out()
coefficients = model.coef_[0]


importance_df = pd.DataFrame({
    'feature': features,
    'coeff': coefficients,
    'importance': np.abs(coefficients)
}).sort_values(by='importance', ascending=False)


sns.barplot(y='feature', x='importance', data=importance_df)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

X_test = df_test.copy()   
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

final_test_preds = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train = y.iloc[train_idx]

    rainfall_pipeline.fit(X_train, y_train)
    
    test_preds = rainfall_pipeline.predict_proba(X_test)[:, 1]
    final_test_preds += test_preds / skf.n_splits

print("Final shape:", final_test_preds.shape)



submission = pd.DataFrame({
    'id': df_test['id'],
    'rainfall': final_test_preds
})

submission.to_csv('submission.csv', index=False)

