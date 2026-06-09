import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.columns # train data columns


train.head() # first 5 row of data


train.describe() # train data probabilistic values of columns


train.info()


def bar_plot(variable):

    var = train[variable]
    varValue = var.value_counts()

    plt.figure(figsize = (8,6))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print(varValue)


train["Stage_fear"].value_counts()


category_l = ["Stage_fear","Drained_after_socializing","Personality"]
for c in category_l:
    bar_plot(c)


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train[variable],bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


numericVar = ["Time_spent_Alone","Social_event_attendance","Going_outside","Friends_circle_size","Post_frequency"]
for n in numericVar:
    plot_hist(n)


train.isnull().sum()


test.isnull().sum()


from sklearn.impute import KNNImputer
train_knn = train.copy(deep=True)

knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")

train_knn['Time_spent_Alone'] = knn_imputer.fit_transform(train_knn[['Time_spent_Alone']])
train_knn['Social_event_attendance'] = knn_imputer.fit_transform(train_knn[['Social_event_attendance']])
train_knn['Going_outside'] = knn_imputer.fit_transform(train_knn[['Going_outside']])
train_knn['Friends_circle_size'] = knn_imputer.fit_transform(train_knn[['Friends_circle_size']])
train_knn['Post_frequency'] = knn_imputer.fit_transform(train_knn[['Post_frequency']])


test_knn = test.copy(deep=True)

knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")

test_knn['Time_spent_Alone'] = knn_imputer.fit_transform(test_knn[['Time_spent_Alone']])
test_knn['Social_event_attendance'] = knn_imputer.fit_transform(test_knn[['Social_event_attendance']])
test_knn['Going_outside'] = knn_imputer.fit_transform(test_knn[['Going_outside']])
test_knn['Friends_circle_size'] = knn_imputer.fit_transform(test_knn[['Friends_circle_size']])
test_knn['Post_frequency'] = knn_imputer.fit_transform(test_knn[['Post_frequency']])


test_knn.isnull().sum()


from sklearn.preprocessing import LabelEncoder


binary_cols = ["Stage_fear", "Drained_after_socializing"]

le = LabelEncoder()
for col in binary_cols:
    train_knn[col] = le.fit_transform(train_knn[col])
    test_knn[col] = le.transform(test_knn[col])


# Personality column do categorical: Introvert â†’ 0, Extrovert â†’ 1
train_knn["Personality"] = train_knn["Personality"].map({
    "Introvert": 0,
    "Extrovert": 1
})


train_knn.head()


# new feature
train_knn["Social_to_Alone_Ratio"] = train_knn["Social_event_attendance"] / (train_knn["Time_spent_Alone"] + 1)
test_knn["Social_to_Alone_Ratio"] = test_knn["Social_event_attendance"] / (test_knn["Time_spent_Alone"] + 1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Features and target variable
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_val, y_pred))


import lightgbm as lgb

# Features and target variable
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LGBM model
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# Predict ve evaluation
y_pred = lgbm_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# Features and target variable
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LGBM model
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgbm_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel("Importance Score")
plt.title("Feature Importance Graphic (LightGBM)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Features and target
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Objective function: F1-score maximize
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'None', 
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return np.mean(f1)

# Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameter
print("Best F1 Score:", study.best_value)
print("Best Parameters:", study.best_params)



# Train
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]
X_test = test_knn.drop(columns=["id"])

# Optuna best parameters
best_params = {
    'max_depth': 3,
    'num_leaves': 81,
    'learning_rate': 0.2858071780058402,
    'n_estimators': 377,
    'subsample': 0.5645469122782392,
    'colsample_bytree': 0.5186830585902273,
    'reg_alpha': 2.0268584191053747,
    'reg_lambda': 1.386118653633458,
    'objective': 'binary',
    'random_state': 42
}

# Model
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)

# Predict
test_preds = final_model.predict(X_test)

# Predict labels
pred_labels = ["Introvert" if p == 0 else "Extrovert" for p in test_preds]

# Submission
submission = pd.DataFrame({
    "id": test_knn["id"],
    "Personality": pred_labels
})

# to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv ready.")


def enrich_features(df):
    df = df.copy()
    
    # EtkileÅŸimli ve tÃ¼revsel Ã¶zellikler
    df["Interaction_1"] = df["Post_frequency"] * df["Going_outside"]
    df["Introversion_score"] = df["Time_spent_Alone"] + df["Drained_after_socializing"]
    df["Social_density"] = df["Friends_circle_size"] / (df["Social_event_attendance"] + 1)
    df["Online_vs_Offline_Ratio"] = df["Post_frequency"] / (df["Friends_circle_size"] + 1)
    df["Total_social_activity"] = df["Going_outside"] + df["Social_event_attendance"]
    df["Is_very_active_online"] = (df["Post_frequency"] > df["Post_frequency"].median()).astype(int)
    
    return df


train_knn = enrich_features(train_knn)
test_knn = enrich_features(test_knn)


train_knn.info()


train_knn.head()


# Features and target
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]

# Objective function: F1-score maximize
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'None', 
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return np.mean(f1)

# Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameter
print("Best F1 Score:", study.best_value)
print("Best Parameters:", study.best_params)



# Train
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]
X_test = test_knn.drop(columns=["id"])

# Optuna best parameters
best_params = {
    'max_depth': 5,
    'num_leaves': 71,
    'learning_rate': 0.29800282766059527,
    'n_estimators': 486,
    'subsample': 0.684163441604819,
    'colsample_bytree': 0.819455825732958,
    'reg_alpha': 3.936035666898007,
    'reg_lambda': 1.3863995079578715,
    'objective': 'binary',
    'random_state': 42
}

# Model
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)

# Predict
test_preds = final_model.predict(X_test)

# Predict labels
pred_labels = ["Introvert" if p == 0 else "Extrovert" for p in test_preds]

# Submission
submission = pd.DataFrame({
    "id": test_knn["id"],
    "Personality": pred_labels
})

# to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv ready.")


from xgboost import XGBClassifier

# Objective function: F1-score maximize
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'use_label_encoder' :'False',
        'eval_metric':'logloss',
        'random_state' :42
        
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
    return np.mean(f1)


# Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameter
print("Best F1 Score:", study.best_value)
print("Best Parameters:", study.best_params)


from xgboost import XGBClassifier

# Train
X = train_knn.drop(columns=["id", "Personality"])
y = train_knn["Personality"]
X_test = test_knn.drop(columns=["id"])

# Training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Optuna best parameters
best_params = {
    'n_estimators':181,
    'learning_rate':0.07972837093615673,
    'max_depth':10,
    'subsample':0.7782336561261523,
    'colsample_bytree':0.891909482135302,
    'reg_alpha':4.396985713962154,
    'reg_lambda':2.3272762781987626,
    'use_label_encoder':'False',
    'eval_metric':'logloss',
    'random_state':42
}

# Model
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# Predict
test_preds = final_model.predict(X_test)

# Predict labels
pred_labels = ["Introvert" if p == 0 else "Extrovert" for p in test_preds]

# Submission
submission = pd.DataFrame({
    "id": test_knn["id"],
    "Personality": pred_labels
})

# to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv ready.")




