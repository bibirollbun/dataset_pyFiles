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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


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


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = df_test.copy()


model1 = LogisticRegression(max_iter=1000, random_state=42)
model2 = DecisionTreeClassifier(random_state=42)
model3 = SVC(probability=False, random_state=42)

voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('svm', model3)],
    voting='hard'
)

voting_clf = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', voting)
])

voting_clf.fit(X_train, y_train)
y_pred_train = voting_clf.predict(X_train)
y_pred_valid = voting_clf.predict(X_valid)
y_pred_test = voting_clf.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
valid_acc = accuracy_score(y_valid, y_pred_valid)

print(f"✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Validation Accuracy: {valid_acc:.4f}")


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': y_pred_test  # ← kết quả dự đoán từ mô hình
})

submission.to_csv('submission.csv', index=False)
print("✅ File submission.csv đã được tạo.")


print(submission)




