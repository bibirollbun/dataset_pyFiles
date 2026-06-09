# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
data.head()


test.head()


data.columns


test.columns


data.drop('id', axis= 1, inplace = True)
test_data = test.copy()
test.drop('id', axis= 1, inplace = True)
data.info()


test.info()


data.describe()


data.loan_intent.value_counts(), data.loan_grade.value_counts()              


data.isnull().sum()


test.isnull().sum()


num_cols = data.select_dtypes(exclude = 'object')
cat_cols = data.select_dtypes(include = 'object')


for i in num_cols:
    plt.hist(data[i])
    plt.title(i)
    plt.show()


from sklearn.preprocessing import StandardScaler, LabelEncoder


encoder = LabelEncoder()
for i in cat_cols:
    data[i] = encoder.fit_transform(data[i])
    test[i] = encoder.fit_transform(test[i])


fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(data.corr(), annot = True,ax= ax)
plt.show()


X = data.drop('loan_status',axis = 1)
y = data['loan_status']
scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)


from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}


# --- Evaluate each model ---
results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),   # ensures fair scaling for all models
        ('clf', model)
    ])
    
    # Use ROC AUC scoring (binary). For multiclass, use 'roc_auc_ovr'
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        scoring='roc_auc',
        cv=kf,
        n_jobs=-1
    )
    
    results[name] = cv_scores.mean()
    print(f"{name}: Mean AUC = {cv_scores.mean():.4f}")
   # print(classification_report(y_test, y_pred))

# --- Find the best model ---
best_model = max(results, key=results.get)
print("\n✅ Best Model:", best_model, "with Mean AUC:", results[best_model])


# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# Grid search
grid = GridSearchCV(estimator= XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), param_grid=param_grid, scoring="roc_auc", 
                    cv=3, verbose=2, n_jobs=-1)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)


# Evaluate on validation set
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)
print("Validation Accuracy:", accuracy_score(y_test, y_pred))


loan_status = best_xgb.predict_proba(test)[:, 1]


pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


submission = pd.DataFrame({
    "id": test_data.id,
    "loan_status": np.round(loan_status,1)
})
submission


submission.to_csv("submission.csv", index=False)




