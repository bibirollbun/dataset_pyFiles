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

import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report   
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train


train.dtypes


train.describe()


# Remove duplicates if any
train = train.drop_duplicates()


# Check missing values
train.isna().sum()


sns.pairplot(train, hue='y', diag_kind='kde', palette='coolwarm')
plt.suptitle("Scatter plots for all numeric features", y=1.02)
plt.show()


plt.figure()
sns.countplot(x="y", data=train)
plt.title("Paid vs No Paid")
plt.show()


corr = train.select_dtypes(include='number').corr()
corr_surv = corr[['y']].sort_values(by='y', ascending=False)

plt.figure(figsize=(4, 8))
sns.heatmap(corr_surv, annot=True, cmap='coolwarm')
plt.title('Correlation with y')
plt.show()




variance = train.select_dtypes(include=['number']).var().sort_values()
variance


def transformData(df):
    df = df.drop(["id"],axis=1)
    return df
train = transformData(train)
test = transformData(test)


#train_reduced = train.drop(["job", "month", "day"], axis=1)
#test_reduced =   test.drop(["job", "month", "day"], axis=1)
#train_encoded = pd.get_dummies(train_reduced, drop_first=True)
#test_encoded = pd.get_dummies(test_reduced, drop_first=True)
#For dummies


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

from sklearn.model_selection import cross_val_score

X_train = train.drop("y", axis=1).select_dtypes(include=['number'])
y_train = train["y"]
X_test = test.select_dtypes(include=['number']) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []
test_predictions = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    model.fit(X_train_scaled, y_train)
    
    test_predictions[name] = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "CV Accuracy Mean": cv_scores.mean(),
        "CV Accuracy Std": cv_scores.std()
    })

results_df = pd.DataFrame(results)
print(results_df)


y = train['y']
X = train.drop('y', axis=1).select_dtypes(include=['number'])
xgb = XGBClassifier(
                    eval_metric='logloss',
                    random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X, y)

print("XGBoost Optimization Results:")
print(f"Best Mean Accuracy (Grid Search CV): {grid_search.best_score_:.4f}")
print(f"Best Hyperparameters Found: {grid_search.best_params_}")

best_xgb_model = grid_search.best_estimator_


y_train = train['y']
X_train = train.drop(['y', 'id'], axis=1, errors='ignore').select_dtypes(include=['number'])
X_test = test.drop(['id'], axis=1, errors='ignore').select_dtypes(include=['number'])

gb_optimized = XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=300,
    subsample=0.7,
    eval_metric='logloss',
    random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_optimized, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Accuracy CV: {scores.mean():.4f}")

xgb_optimized.fit(X_train, y_train)





testId = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

predictions = xgb_optimized.predict(X_test)

submission_df = pd.DataFrame({
    'id': testId["id"],
    'y': predictions
})

submission_df['y'] = submission_df['y'].astype(int)
submission_df.to_csv('submission.csv', index=False)

