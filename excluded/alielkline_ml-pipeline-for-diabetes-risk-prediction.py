import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
data.shape


data.head()


data.dtypes


data.isnull().sum()


categorical = []
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"----{col}----")
        print(data[col].value_counts())
        print()
        categorical.append(col)


data.describe()


data.info()


n = len(data.columns)
cols_per_row = 3
rows = math.ceil(n / cols_per_row)

plt.figure(figsize=(15, 4*rows))

for i, col in enumerate(data.columns):
    plt.subplot(rows, cols_per_row, i+1)
    plt.hist(data[col])
    plt.xlabel(col)
    plt.ylabel('Frequenc')
    plt.title(f"{col} Distribution")

plt.tight_layout()
plt.show()


numerical = data.select_dtypes(include=['int64', 'float64'])
categorical = data.select_dtypes(include=['object'])


plt.figure(figsize=(16, 12))
sns.heatmap(numerical.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


numerical.corr()['diagnosed_diabetes'].sort_values(ascending=False)


plt.figure(figsize=(15, 4*rows))

for i, col in enumerate(numerical.columns):
    plt.subplot(rows, cols_per_row, i+1)
    sns.boxplot(x='diagnosed_diabetes', y=col, data=numerical)
    plt.title(f"{col} vs Diagnosed Diabetes")
    
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 4*rows))

for i, col in enumerate(categorical.columns):
    plt.subplot(rows, cols_per_row, i+1)
    sns.countplot(data=data, x=col, hue='diagnosed_diabetes')
    plt.title(f"{col} vs Diagnosed Diabetes")
    
plt.tight_layout()
plt.show()


X = data.drop(['diagnosed_diabetes', 'id'], axis=1)
y = data['diagnosed_diabetes']

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
ordinal_features = ["income_level", "education_level"]
nominal_features = ["gender", "ethnicity", "smoking_status", "employment_status"]

numerical_features = [col for col in numerical_features 
                      if col not in ordinal_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("ord", OrdinalEncoder(), ordinal_features),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_features)
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.head()


def train_model(clf):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)

train_model(clf)


clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

train_model(clf)


xgb = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

train_model(xgb)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

train_data = lgb.Dataset(X_train_processed, label=y_train)
test_data = lgb.Dataset(X_test_processed, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

num_round = 1000
bst = lgb.train(
    params,
    train_data,
    num_round,
    valid_sets=[test_data],
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(100) 
    ]
)

Y_train = bst.predict(X_train_processed, num_iteration=bst.best_iteration)
Y_val = bst.predict(X_test_processed, num_iteration=bst.best_iteration)

y_train_class = (Y_train > 0.5).astype(int)
y_val_class = (Y_val > 0.5).astype(int)

print("Training ROC-AUC: ", roc_auc_score(y_train, Y_train))
print("Validation ROC-AUC: ", roc_auc_score(y_test, Y_val))


xgb = XGBClassifier(
        eval_metric='logloss',
        tree_method='hist',  
        random_state=42
    )


pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

param_grid = {
        "classifier__n_estimators": [400, 600, 800],
        "classifier__max_depth": [6, 8, 10],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__gamma": [0, 1, 5],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__reg_lambda": [1, 5, 10]
    }

search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,                 
        scoring="roc_auc",
        cv=3,                      
        verbose=0,
        n_jobs=-1                  
    )

search.fit(X, y)

print("\nğŸ”¥ Best Parameters:")
print(search.best_params_)

print("\nğŸ”¥ Best CV score:", search.best_score_)


best_xgb = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    colsample_bytree=0.8,
    gamma=0,
    subsample=0.8,
    reg_lambda=1,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42,
    use_label_encoder=False
)

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_xgb)
])

best_pipeline.fit(X_train, y_train)


pipeline = Pipeline([
    ("preprocess", preprocessor),   
    ("classifier", LGBMClassifier())
])


param_dist = {
    'classifier__num_leaves': np.arange(31, 150),
    'classifier__learning_rate': np.linspace(0.001, 0.2, 100),
    'classifier__n_estimators': np.arange(50, 600),
    'classifier__max_depth': np.arange(3, 15),
    'classifier__min_child_samples': np.arange(5, 50),
    'classifier__subsample': np.linspace(0.5, 1.0, 20),
    'classifier__colsample_bytree': np.linspace(0.5, 1.0, 20),
    "clf__verbose": [-1]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=3,
    verbose=0,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

y_val_pred = best_model.predict_proba(X_test)[:, 1]

print("Validation ROC-AUC:", roc_auc_score(y_test, y_val_pred))


best_params = {
    'subsample': 0.5,
    'num_leaves': 52,
    'n_estimators': 272,
    'min_child_samples': 36,
    'max_depth': 10,
    'learning_rate': 0.05527272727272727,
    'colsample_bytree': 0.5263157894736842,
    'verbose': -1
}

best_lgb = LGBMClassifier(**best_params)

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_lgb)
])

best_pipeline.fit(X_train, y_train)


test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test.head()


y_pred_test = best_pipeline.predict(test)
submission = pd.DataFrame({
    "id": test["id"],  
    "diagnosed_diabetes": y_pred_test  
})
submission.to_csv("submission.csv", index=False)

