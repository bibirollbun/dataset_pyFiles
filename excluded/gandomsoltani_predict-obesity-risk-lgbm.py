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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())

train.isnull().sum()

train['NObeyesdad'].value_counts(normalize=True)


plt.figure(figsize=(6,4))
sns.countplot(data=train, x='NObeyesdad', order=train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Obesity Risk Categories')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=train, x='NObeyesdad', y='Age', order=train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Age Distribution by Obesity Category')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=train, x='NObeyesdad', hue='Gender', order=train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Obesity Category by Gender')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(train.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('NObeyesdad') 
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('id') 


numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])


preprocessor = ColumnTransformer(
transformers=[
('num', numeric_transformer, num_cols),
('cat', categorical_transformer, cat_cols)
],
remainder='drop'
)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['target'] = le.fit_transform(train['NObeyesdad'])
train = train.drop(columns=['NObeyesdad'])

le.classes_


from sklearn.model_selection import train_test_split

X = train.drop(columns=['id', 'target'], errors='ignore')
y = train['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


models = {
'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
'SVC': SVC(probability=True, random_state=42),
'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
'LightGBM': LGBMClassifier(random_state=42)
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


results = []
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    print(f'Evaluating: {name} ...')
    scores_acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    scores_f1 = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    results.append((name, scores_acc, scores_f1))
    print(f' Accuracy: {scores_acc.mean():.4f} ± {scores_acc.std():.4f}')
    print(f' F1_macro: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}')


best_model = LGBMClassifier(random_state=42)

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

final_pipe.fit(X, y)


test_pred = final_pipe.predict(test.drop(columns=['id']))
test_pred_labels = le.inverse_transform(test_pred)


submission = pd.DataFrame({
    'id': test['id'],
    'NObeyesdad': test_pred_labels
})

submission.to_csv('submission.csv', index=False)

