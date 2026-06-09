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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# 1. Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head(2)


train.shape


train.describe()


train.isnull().sum()


# Separate features and target
X = train.drop(['id', 'Personality'], axis=1)
#change personality column from string to number
y = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})  # Encode target
X_test = test.drop(['id'], axis=1)

# Handle categorical variables
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    X[col] = X[col].map({'No': 0, 'Yes': 1})
    X_test[col] = X_test[col].map({'No': 0, 'Yes': 1})
    
    # Impute missing values in categorical columns with mode
    #Nếu cột bị thiếu dữ liệu (NaN), ta dùng SimpleImputer để điền vào.
    #strategy='most_frequent': điền giá trị xuất hiện nhiều nhất (mode).
    imputer = SimpleImputer(strategy='most_frequent')
    #.ravel() giúp chuyển từ dạng 2D thành 1D Series (vì fit_transform trả về mảng 2D).
    X[col] = imputer.fit_transform(X[[col]]).ravel()
    X_test[col] = imputer.transform(X_test[[col]]).ravel()

# Handle numerical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']

#strategy='median': điền giá trị trung vị thay vì trung bình (ít bị ảnh hưởng bởi ngoại lệ).
imputer_num = SimpleImputer(strategy='median')
#Áp dụng imputer để điền missing values cho cả X và X_test trong các cột dạng số.
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])
X_test[numerical_cols] = imputer_num.transform(X_test[numerical_cols])


from sklearn.model_selection import GridSearchCV
# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X, y)
print("Best params:", grid.best_params_)

'''
# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Validate model
val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
'''


# Predict on test set
test_pred = grid.predict(X_test)
test_pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')


#importances = grid.feature_importances_
best_model = grid.best_estimator_
importances = best_model.feature_importances_
feature_names = X.columns
for name, importance in zip(feature_names, importances):
    print(f'{name}: {importance:.4f}')


# Create submission file
submissions = pd.DataFrame({'id': test['id'], 'Personality': test_pred_labels})
submissions.to_csv('submissions.csv', index=False)
print("Submission file created: submissions.csv")


print(submissions)

