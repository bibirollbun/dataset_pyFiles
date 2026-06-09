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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
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


df_train.head()


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


model = GradientBoostingClassifier(random_state=42)

# Tạo pipeline
gb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', model)
])

# Grid các siêu tham số
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [2, 3, 4]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Huấn luyện mô hình
grid_search.fit(X_train, y_train)

# Dự đoán
best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_valid = best_model.predict(X_valid)
y_pred_test = best_model.predict(X_test)

# Đánh giá hiệu suất
train_acc = accuracy_score(y_train, y_pred_train)
valid_acc = accuracy_score(y_valid, y_pred_valid)

# Kết quả
print(f"✅ Best Params: {grid_search.best_params_}")
print(f"✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Valid Accuracy: {valid_acc:.4f}")


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': y_pred_test  # ← kết quả dự đoán từ mô hình
})

submission.to_csv('submission.csv', index=False)
print("✅ File submission.csv đã được tạo.")


print(submission)




