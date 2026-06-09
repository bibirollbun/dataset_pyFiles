# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import optuna

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train_df.columns


train_df.describe()


train_df.head()


train_df['efs'].value_counts(normalize=True)


train_df.isnull().mean().sort_values(ascending=False).head(15)


plt.figure(figsize=(10, 6))

sns.histplot(data=train_df, x="age_at_hct", hue="efs", bins=30, kde=True, palette="husl", alpha=0.6)

plt.title('Distribution of Age at HCT by Event Status')
plt.xlabel('Age at HCT')
plt.ylabel('Count')
plt.legend(title='Event (efs)', labels=['Censored (0)', 'Event (1)'])
plt.show()


plt.figure(figsize=(10, 6))

sns.histplot(data=train_df, x="donor_age", hue="efs", bins=30, kde=True, palette="husl", alpha=0.6)

plt.title('Distribution of Donor Age by Event Status')
plt.xlabel('Donor Age')
plt.ylabel('Count')
plt.legend(title='Event (efs)', labels=['Censored (0)', 'Event (1)'])
plt.show()


plt.figure(figsize=(12, 6))


sns.countplot(y='race_group', data=train_df, hue='efs', order=train_df['race_group'].value_counts().index, palette='Set2')

plt.title('Event Counts by Race Group')
plt.xlabel('Count')
plt.ylabel('Race Group')
plt.legend(title='Event (efs)', labels=['Censored (0)', 'Event (1)'])
plt.show()


X = train_df.drop(columns="efs")
y = train_df["efs"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# Train 80% / Test 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


num_cols = [c for c in X_train.select_dtypes(include=['int64', 'float64']).columns if c not in ['ID', 'efs_time']]
cat_cols = [c for c in X_train.select_dtypes(include=['object']).columns]


plt.figure(figsize=(20,12))
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # see the missing, add it with means
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # see the missing, add it with "missing"
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# if column == numb >> num_pipeline
# if column == text >> cat_pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


print("\nProcessing Data...")
X_train_processed = preprocessor.fit_transform(X_train)


# expect it to have more column (since we use one-hot)

print("-" * 30)
print(f"Original Shape (Before):  {X_train.shape}")
print(f"Processed Shape (After): {X_train_processed.shape}")
print("-" * 30)


models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42))
    ]),
    
    "Gradient Boosting (sklearn)": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42))
    ]),
    
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    
    "LightGBM": Pipeline([
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



results = {}

print("Start Benchmarking Models...")

for name, model in models.items():
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        return_train_score=False
    )

    
    mean_score = cv_results['test_score'].mean()
    std_score = cv_results['test_score'].std()
    mean_time = cv_results['fit_time'].mean()
    
    results[name] = {
        "ROC-AUC": f"{mean_score:.4f} ± {std_score:.4f}",
        "Mean Score": mean_score,
        "Fit Time (s)": mean_time
    }
    
    
    print(f" {name:20s}: {mean_score:.4f} ± {std_score:.4f} (Time: {mean_time:.2f}s)")

results_df = pd.DataFrame(results).T.sort_values(by="Mean Score", ascending=False)

display(results_df.style.background_gradient(cmap='Greens', subset=['Mean Score']))


def to_string(x):
    return x.astype(str)

cat_pipeline_cb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('to_string', FunctionTransformer(to_string, validate=False)) 
])

preprocessor_cb = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline_cb, cat_cols)
])

cat_index = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

models["CatBoost"] = Pipeline([
    ("prep", preprocessor_cb),
    ("model", CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_index,
        verbose=0,
        random_state=42
    ))
])


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': 42,
        'verbose': 0,
        'loss_function': 'Logloss',
        'cat_features': cat_index
    }


    model = Pipeline([
        ("prep", preprocessor_cb),
        ("model", CatBoostClassifier(**params))
    ])

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()

print(" Starting Optuna Hyperparameter Search for CatBoost...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

print("\n" + "="*40)
print(f"Best ROC-AUC: {study.best_value:.4f}")
print("Best Parameters:")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")
    
print("="*40)


final_model = models["CatBoost"]

print(f"Retraining {final_model.steps[-1][0]} on full dataset...")

final_model.fit(X, y)


X_submission = test_df.drop(columns=['ID'])

print("Generating predictions...")
predictions = final_model.predict_proba(X_submission)[:, 1]


submission = pd.DataFrame({
    'ID': test_df['ID'],
    'prediction': predictions
})

submission.to_csv('submission.csv', index=False)

print(" Saved 'submission.csv' successfully!")
print(f"File shape: {submission.shape}")

display(submission.head())

