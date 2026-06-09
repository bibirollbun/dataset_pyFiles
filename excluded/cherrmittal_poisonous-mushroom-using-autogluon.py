# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install autogluon.tabular[all]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import autogluon.core as ag
print(f"AutoGluon Core Version: {ag.__version__}")

import sklearn
print(f"Scikit-learn Version: {sklearn.__version__}") # See what version AutoGluon chose

from autogluon.tabular import TabularDataset, TabularPredictor
print("AutoGluon TabularPredictor imported successfully!")


#train=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
#test=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
train_test = pd.concat([pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv'), pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')], axis=0, join='outer')
#first_1000_records_iloc = train.iloc[:9000]
train_test.drop(["stem-root", "spore-print-color"], axis=1, inplace=True)


#working on cap-diameter, stem-height, stem-width
list_columns = ['cap-diameter', 'stem-height', 'stem-width']
train_test[list_columns] = (train_test[list_columns] - train_test[list_columns].mean())/train_test[list_columns].std()

# working on cap-shape, cap-surface, cap-color
list_column = ['cap-shape', 'cap-surface', 'cap-color', 'gill-attachment', 'gill-color', 'stem-color', 'ring-type', 'habitat']
for column in list_column:
    train_test[column] = train_test[column].replace({'has f': 'f', 'has d': 'd', 'is a': 'a', 'does n': 'n', 'does w': 'w', 'is y': 'y', 'does f': 'f', 'is w': 'w', 'has g': 'g', 'is n': 'n'})
    counts = train_test[column].value_counts()
    values_to_replace = counts[counts<=500].index.tolist()
    train_test[column] = train_test[column].replace(values_to_replace, np.nan)
    train_test[column].fillna(train_test[column].mode()[0], inplace=True)
    #train_test[column] = train_test[column].replace('ext', train_test[column].mode()[0])


#working on does-bruise-or-bleed, has-ring
list_columns = ['does-bruise-or-bleed', 'has-ring']
for column in list_columns: 
    train_test[column].replace('does t', 't')
    counts = train_test[column].value_counts(dropna=False)
    values_to_replace = counts[counts<=100].index.tolist()
    train_test[column] = train_test[column].replace(values_to_replace,train_test[column].mode()[0])
    train_test[column] = train_test[column].replace({'t':True, 'f':False}).astype(bool)

#working on gill-spacing
train_test['gill-spacing'] = train_test['gill-spacing'].replace({'has f': 'f', 'does c': 'c', 'does f': 'f'})
gill_spacing_counts = train_test["gill-spacing"].value_counts()
values_to_replace = gill_spacing_counts[gill_spacing_counts <= 100].index.tolist()
train_test["gill-spacing"] = train_test["gill-spacing"].replace(values_to_replace, 'ext')
train_test["gill-spacing"].fillna('ext', inplace=True)


#working on veil-type
train_test['veil-type-u'] = (train_test['veil-type'] == 'u')
train_test.drop('veil-type', axis=1, inplace=True)

#working on veil-color
veil_color_counts = train_test["veil-color"].value_counts()
values_to_replace = veil_color_counts[veil_color_counts <= 100].index.tolist()
train_test["veil-color"] = train_test["veil-color"].replace(values_to_replace, 'ext')
train_test['veil-color'].fillna('ext', inplace=True)


# Get specific groups directly
Train = train_test[train_test['class'].isin(['e', 'p'])].copy()
Test = train_test[train_test['class'].isna()].copy()
#print(Train.shape)
#print(Test.shape)


from sklearn.model_selection import train_test_split

Train["class"] = Train["class"].replace({'e': True, 'p': False}).astype(bool)
y = Train["class"]

X_train, X_cv, y_train, y_cv = train_test_split(Train, y, test_size=0.2, random_state=42)
X_train.drop(["class", "id"], axis=1, inplace=True)
_ids = X_cv["id"]
X_cv.drop(["class", "id"], axis=1, inplace=True)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = X_train.select_dtypes(include=['object', 'category', 'boolean']).columns
numerical_features = X_train.select_dtypes(include=np.number).columns
#print(categorical_features)
#print(numerical_features)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features) # Pass numerical columns through
    ])
preprocessor.fit(X_train)
X_train_encoded = preprocessor.transform(X_train)
X_cv_encoded = preprocessor.transform(X_cv)



Train2 = train_test[train_test['class'].isin(['e', 'p'])].copy()
Test2 = train_test[train_test['class'].isna()].copy()
#X_train2, X_cv2 = train_test_split(Train2, test_size=0.2, random_state=42)
Train2.drop(['id'], axis=1, inplace=True)
id_field = Test2['id']
Test2.drop(['id'], axis=1, inplace=True)
train2_data = TabularDataset(Train2)
test2_data = TabularDataset(Test2)
#cv2_data = TabularDataset(X_cv2)
predictor = TabularPredictor(label='class', eval_metric='accuracy')
predictor.fit(Train2, presets='best_quality')

predictions = predictor.predict(test2_data)
#leaderboard = predictor.leaderboard(test2_data)
#print(leaderboard)
output = pd.DataFrame({'id': id_field, 'class': predictions})
output.to_csv('submission.csv', index=False)




