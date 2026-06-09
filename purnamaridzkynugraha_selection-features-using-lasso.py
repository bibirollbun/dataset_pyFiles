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


!pip install jcopml


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe, cat_pipe


df=pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv', index_col='Batch_ID')
df.head()


X = df.select_dtypes(include='number').drop(columns=['T80'])
y = df['T80']

pipeline = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=5000, random_state=42))
pipeline.fit(X, y)

coef = pipeline.named_steps['lassocv'].coef_
selected_features = X.columns[coef != 0]
selected_features


X = df[selected_features]
y = df['T80']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape



from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

pipeline = Pipeline([
    ('algo', XGBRegressor())
])
model = RandomizedSearchCV(pipeline, rsp.xgb_params,n_iter=30, cv=3, n_jobs=-1, verbose=1, error_score='raise')

model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
from jcopml.tuning import random_search_params as rsp

preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling='standard'), X.select_dtypes(include=np.number).columns),
   
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBRegressor())
])

model = RandomizedSearchCV(pipeline,rsp.xgb_params, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))


!pip install rdkit


from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "MolLogP": Descriptors.MolLogP(mol),
        }
    else:
        return {k: None for k in ['MolWt', 'NumAromaticRings', 'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount', 'MolLogP']}

desc_df = df['Smiles'].apply(smiles_to_descriptors).apply(pd.Series)
desc_df['T80'] = df['T80']



desc_df


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.heatmap(desc_df.corr(numeric_only=True)[['T80']].sort_values(by='T80', ascending=False), annot=True)
plt.title('Correlation of SMILES Descriptors with T80')
plt.show()

# Scatter contoh
sns.pairplot(desc_df, vars=['MolWt', 'MolLogP', 'FractionCSP3'], hue=None, y_vars=['T80'])
plt.suptitle("Pairplot with T80", y=1.02)
plt.show()



desc_df.columns


df_nyoba = pd.concat([desc_df[['MolWt', 'HeavyAtomCount', 'MolLogP']], df[selected_features]], axis=1)




rsp.xgb_params


X = df_nyoba
y = df['T80']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
from jcopml.tuning import random_search_params as rsp

preprocessor = ColumnTransformer([
    ('numeric', num_pipe(scaling='standard'), X.select_dtypes(include=np.number).columns),
   
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBRegressor())
])

model = RandomizedSearchCV(pipeline,rsp.xgb_params, cv=3, n_jobs=-1, verbose=1 )
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))

