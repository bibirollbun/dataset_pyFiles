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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.head()


df_test.head()


df_train.info()
df_test.info()


df_train.isnull().sum()


df_test.isnull().sum()


print(df_train.shape,df_test.shape)


#PREPROCESSING
#X=features,y=target
X = df_train.drop(columns=['Personality','id'])
y = df_train['Personality']

#Colect features cat and num:
cat_cols = X.select_dtypes(include='object').columns.to_list()
num_cols = X.select_dtypes(exclude='object').columns.to_list()

preprocessor = ColumnTransformer(
    transformers = [
        ('cat',Pipeline([
            ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols),
        ('num', SimpleImputer(strategy='mean'), num_cols)
    ]
)


#Use Kfold to split data 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
oof_preds = np.zeros((len(X), len(y.unique())))

classes = np.sort(y.unique())
class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {i: c for c, i in class_to_index.items()}

y_encoded = y.map(class_to_index)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded), 1):
    print(f"Training Fold {fold}...")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold = y_encoded.iloc[train_idx]

    model = Pipeline([
        ('preprocessor',preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=5, random_state=56))
    ])

    model.fit(X_train_fold, y_train_fold)
    models.append(model)
    
    oof_preds[val_idx] = model.predict_proba(X_val_fold)


oof_labels = np.argmax(oof_preds, axis=1)
oof_true = y_encoded.values

oof_accuracy = accuracy_score(oof_true, oof_labels)
print(f"\nOOF CV Accuracy:{oof_accuracy:.4f}")


X_test = df_test.drop(columns=['id'])
test_probs = np.array([model.predict_proba(X_test) for model in models])
avg_test_probs = np.mean(test_probs, axis = 0)

final_test_preds = np.argmax(avg_test_probs, axis=1)
final_test_labels = [index_to_class[i] for i in final_test_preds]


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': final_test_labels
})

submission.to_csv('submission.csv', index=False)




