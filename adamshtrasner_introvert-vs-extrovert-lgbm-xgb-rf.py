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
import pandas as pd

# Visualizastion
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
sns.set_palette(sns.color_palette("deep", 10))
sns.set_style("whitegrid")

# Sklearn
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Hyperparameter Optimization
import optuna


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


df_train.head()


df_train.info()


msno.bar(df_train, figsize=(10, 6))


msno.bar(df_test, figsize=(10, 6))


categorical_columns = df_train.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Personality')

# for col in categorical_columns:
#     mode = df_train[col].mode()[0]
#     df_train[col].fillna(mode, inplace=True)
#     if col in df_test.columns:
#         df_test[col].fillna(mode, inplace=True)

mode_imputer = SimpleImputer(strategy='most_frequent')
df_train[categorical_columns] = mode_imputer.fit_transform(df_train[categorical_columns])
df_test[categorical_columns] = mode_imputer.transform(df_test[categorical_columns])


numeric_columns = df_train.select_dtypes(include=["float64"]).columns.tolist()

# for col in numeric_columns:
#     mean = df_train[col].mean()
#     df_train[col].fillna(mean, inplace=True)
#     if col in df_test.columns:
#         df_test[col].fillna(mean, inplace=True)

mean_imputer = SimpleImputer(strategy='mean')
df_train[numeric_columns] = mean_imputer.fit_transform(df_train[numeric_columns])
df_test[numeric_columns] = mean_imputer.transform(df_test[numeric_columns])

for col in numeric_columns:
    df_train[col] = df_train[col].astype(int)


sns.countplot(df_train, x="Personality")


sns.countplot(df_train, x="Personality", hue="Stage_fear")


df_train[df_train['Stage_fear'] == 'Yes']['Personality'].value_counts()


sns.countplot(df_train, x="Personality", hue="Drained_after_socializing")


df_train[df_train['Drained_after_socializing'] == 'Yes']['Personality'].value_counts()


plt.figure(figsize=(10, 5))
sns.countplot(df_train, x="Personality", hue="Time_spent_Alone")


plt.figure(figsize=(10, 5))
sns.countplot(df_train, x="Personality", hue="Social_event_attendance")


plt.figure(figsize=(10, 5))
sns.countplot(df_train, x="Personality", hue="Going_outside")


plt.figure(figsize=(10, 5))
sns.countplot(df_train, x="Personality", hue="Friends_circle_size")


plt.figure(figsize=(10, 5))
sns.countplot(df_train, x="Personality", hue="Post_frequency")


for col in categorical_columns:
    df_train[col] = df_train[col].map({"No": 0, "Yes": 1})
    df_test[col] = df_test[col].map({"No": 0, "Yes": 1})


df_train["Personality"] = df_train["Personality"].map({"Introvert": 1, "Extrovert": 0})


X = df_train.drop("Personality", axis=1)
y = df_train["Personality"]


# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, class_weight='balanced'))
    ]),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Naive Bayes": GaussianNB(),
    "MLP (Neural Net)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(max_iter=500))
    ]),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(class_weight='balanced', objective='binary')
}


# Perform cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


results = []

for name, model in models.items():
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1").mean()
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc").mean()

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "F1 Score": round(f1, 3),
        "ROC AUC": round(auc, 3)
    })


# Display results
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print(results_df)


# Hyperparameter optimization using optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "class_weight":'balanced',
        "objective":'binary' 
    }

    model = LGBMClassifier(**params, verbosity=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    return acc


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600)

# Show results
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = study.best_params
best_params['class_weight'] = 'balanced'
best_params['objective'] = 'binary'
best_params


lgbm = LGBMClassifier(
    **best_params, verbosity=-1
)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)


base_models = [
    ("lgbm", lgbm),
    ("xgb", xgb),
    ("rf", rf)
]

meta_model = LogisticRegression()


stacked_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        passthrough=True,
        n_jobs=-1
    ))
])


acc = cross_val_score(stacked_model, X, y, cv=skf, scoring="accuracy").mean()
f1 = cross_val_score(stacked_model, X, y, cv=skf, scoring="f1").mean()
auc = cross_val_score(stacked_model, X, y, cv=skf, scoring="roc_auc").mean()

print("Stacked Model Scores:\n")
print(f"Accuracy: {acc}")
print(f"F1-score: {f1}")
print(f"ROC AUC: {auc}")


stacked_model.fit(X, y)


predictions = stacked_model.predict(df_test)


reverse_mapping = {0: 'Extrovert', 1: 'Introvert'}
final_predictions_text = pd.Series(predictions).map(reverse_mapping)
submission_df = pd.DataFrame({'id': df_test.index, 'Personality': final_predictions_text})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print("Submission file head:")
print(submission_df.head())

