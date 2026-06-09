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

import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns

import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col=0) 
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col=0)
train.head()


test.head()


print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


%%time
round(100*train["Personality"].value_counts(normalize=True), 2)


%%time
print('There are', sum(train.drop(columns=["Personality"]).duplicated()), 'duplicated observations in the train dataset')
print('There are', sum(test.duplicated()), 'duplicated observations in the test dataset')


%%time
to_consider = train.drop(columns=["Personality"], axis=1).columns.tolist()

train_dup = train.drop(columns=["Personality"], axis=1).drop_duplicates()
test_dup = test.drop_duplicates()
duplicates = pd.merge(train_dup, test_dup, on=to_consider)

print('There are', duplicates.shape[0], 'rows that appear in the train and test dataset.\n')


%%time
print("Missing values in the train dataset \n")
print(f"{train.isna().sum()}\n")

print("Missing values in the test dataset \n")
print(test.isna().sum())


sns.set(style="whitegrid")

# Target Balance
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x="Personality", palette="pastel")
plt.title("Distribution of Personality Types")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.show()


# Numerical features vs Personality
numerical_cols = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=train, x="Personality", y=col, palette="pastel")
    plt.title(f"{col} by Personality")
    plt.xlabel("")
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# Categorical features vs Personality
categorical_cols = ["Stage_fear", "Drained_after_socializing"]

plt.figure(figsize=(12, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=train, x=col, hue="Personality", palette="pastel")
    plt.title(f"{col} by Personality")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Personality")

plt.tight_layout()
plt.show()


X = train.drop(columns=["Personality"])
y = train["Personality"].map({"Extrovert": 0, "Introvert": 1})

del train
gc.collect()

# Handle categorical features (filling missing values)
cat_features = ["Stage_fear", "Drained_after_socializing"]
for col in cat_features:
    X[col] = X[col].fillna("missing").astype("category")
    test[col] = test[col].fillna("missing").astype("category")

# Handle numeric features & Fill in missing values with median
num_features = X.select_dtypes(include=["float", "int"]).columns.tolist()
X[num_features] = X[num_features].fillna(X[num_features].median())
test[num_features] = test[num_features].fillna(X[num_features].median())

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

logreg_params = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42
}

logreg_scores, logreg_oof_preds_df, logreg_test_preds_df = [], [], []
preprocessed_test = preprocessor.fit(X).transform(test)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    # Train logistic regression model
    model = LogisticRegression(**logreg_params)
    model.fit(X_train_processed, y_train)

    # Predict on validation fold (class predictions for scoring)
    md_pred = model.predict(X_valid_processed)
    score = accuracy_score(y_valid, md_pred)
    logreg_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    # OOF predictions 
    oof_preds = pd.DataFrame({
        "logreg_pred": md_pred,
        "y": y_valid.values,
        "fold": i + 1,
        "index": X_valid.index
    })
    logreg_oof_preds_df.append(oof_preds)

    # Predict on test set using probabilities (for averaging)
    test_preds = pd.DataFrame(
        model.predict_proba(preprocessed_test)[:, 1],
        columns=["logreg_pred"]
    )
    test_preds["fold"] = i + 1
    logreg_test_preds_df.append(test_preds)

logreg_mean = np.mean(logreg_scores)
logreg_sd = np.std(logreg_scores)
print(f"Logistic Regression CV accuracy: {logreg_mean:.4f} Â± {logreg_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(logreg_test_preds_df[i]["logreg_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_2.csv")


%%time

xgb_params = {
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.05,
    'gamma': 0.1,
    'min_child_weight': 10,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'n_jobs': -1
}

test_xgb = xgb.DMatrix(test, enable_categorical=True)
xgb_scores, xgb_oof_preds_df, xgb_test_preds_df = [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    md = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # Predict hard class labels on validation set
    md_pred = (md.predict(dvalid) > 0.5).astype(int)

    score = accuracy_score(y_test, md_pred)
    xgb_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    # OOF predictions (hard predictions)
    oof_preds = pd.DataFrame(
        md_pred,
        columns=["xgb_pred"]
    )
    oof_preds["y"] = y_test.values
    oof_preds["fold"] = i + 1
    oof_preds["index"] = X_test.index
    xgb_oof_preds_df.append(oof_preds)

    # Predict on test set (hard predictions)
    test_preds = pd.DataFrame(
        (md.predict(test_xgb) > 0.5).astype(int),
        columns=["xgb_pred"]
    )
    test_preds["fold"] = i + 1
    xgb_test_preds_df.append(test_preds)

xgb_mean = np.mean(xgb_scores)
xgb_sd = np.std(xgb_scores)
print(f"XGBoost CV accuracy: {xgb_mean:.4f} Â± {xgb_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(xgb_test_preds_df[i]["xgb_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_3.csv")


%%time

X_encoded = X.copy()
test_encoded = test.copy()

# Ordinal encode categorical features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded[cat_features] = encoder.fit_transform(X_encoded[cat_features])
test_encoded[cat_features] = encoder.transform(test_encoded[cat_features])

hgb_model = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=7,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

hgb_scores, hgb_oof_preds_df, hgb_test_preds_df = [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X_encoded, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_valid = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    hgb_model.fit(X_train, y_train)
    md_pred = hgb_model.predict(X_valid)

    score = accuracy_score(y_valid, md_pred)
    hgb_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    oof_preds = pd.DataFrame({
        "hgb_pred": md_pred,
        "y": y_valid.values,
        "fold": i + 1,
        "index": X_valid.index
    })
    hgb_oof_preds_df.append(oof_preds)

    test_preds = pd.DataFrame(
        hgb_model.predict(test_encoded),
        columns=["hgb_pred"]
    )
    test_preds["fold"] = i + 1
    hgb_test_preds_df.append(test_preds)

hgb_mean = np.mean(hgb_scores)
hgb_sd = np.std(hgb_scores)
print(f"HistGradientBoosting CV accuracy: {hgb_mean:.4f} Â± {hgb_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(hgb_test_preds_df[i]["hgb_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_15.csv")


%%time

lgb_params = {
    'learning_rate': 0.0951305706297959,
    'max_depth': 7,
    'reg_alpha': 1.803743818590143,
    'reg_lambda': 0.01128965915670642,
    'num_leaves': 87,
    'colsample_bytree': 0.7749196423063947,
    'verbose': -1,
    'n_jobs': -1,
    'device': 'cpu'
}

lgb_scores, lgb_oof_preds_df, lgb_test_preds_df = [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y.iloc[test_index]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, categorical_feature=cat_features)

    md = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    # Validation predictions (hard class labels)
    md_pred = (md.predict(X_test) > 0.5).astype(int)

    score = accuracy_score(y_test, md_pred)
    lgb_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    oof_preds = pd.DataFrame(
        md_pred,
        columns=["lgb_pred"]
    )
    oof_preds["y"] = y_test.values
    oof_preds["fold"] = i + 1
    oof_preds["index"] = X_test.index
    lgb_oof_preds_df.append(oof_preds)

    # Predict on test set (probabilities)
    test_preds = pd.DataFrame(
        md.predict(test),
        columns=["lgb_pred"]
    )
    test_preds["fold"] = i + 1
    lgb_test_preds_df.append(test_preds)

lgb_mean = np.mean(lgb_scores)
lgb_sd = np.std(lgb_scores)
print(f"LightGBM CV accuracy: {lgb_mean:.4f} Â± {lgb_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(lgb_test_preds_df[i]["lgb_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_4.csv")


%%time

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 10,
    'task_type': 'CPU'  # change to 'GPU' if you have GPU setup
}

catboost_scores, catboost_oof_preds_df, catboost_test_preds_df = [], [], []

# CatBoost Pool for test set
test_pool = Pool(test, cat_features=cat_features)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_test, y_test, cat_features=cat_features)

    md = CatBoostClassifier(**catboost_params)
    md.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Predict hard classes on validation set
    md_pred = md.predict(valid_pool).astype(int)
    score = accuracy_score(y_test, md_pred)
    catboost_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    oof_preds = pd.DataFrame(
        md_pred,
        columns=["catboost_pred"]
    )
    oof_preds["y"] = y_test.values
    oof_preds["fold"] = i + 1
    oof_preds["index"] = X_test.index
    catboost_oof_preds_df.append(oof_preds)

    # Predict on test set (probabilities)
    test_preds = pd.DataFrame(
        md.predict_proba(test_pool)[:, 1],
        columns=["catboost_pred"]
    )
    test_preds["fold"] = i + 1
    catboost_test_preds_df.append(test_preds)

catboost_mean = np.mean(catboost_scores)
catboost_sd = np.std(catboost_scores)
print(f"CatBoost CV accuracy: {catboost_mean:.4f} Â± {catboost_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(catboost_test_preds_df[i]["catboost_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_5.csv")


%%time
# Preprocess all data upfront
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test)

# Base models (no pipeline needed now)
logreg_base = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

xgb_base = xgb.XGBClassifier(
    device='cuda',
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree=0.7,
    subsample=0.7,
    reg_lambda=1.0,
    reg_alpha=0.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

lgb_base = lgb.LGBMClassifier(
    learning_rate=0.095,
    max_depth=7,
    reg_alpha=1.8,
    reg_lambda=0.01,
    num_leaves=87,
    colsample_bytree=0.77,
    n_jobs=-1,
    random_state=42
)

# Stacking Classifier with preprocessed features
stack_model = StackingClassifier(
    estimators=[
        ('logreg', logreg_base),
        ('xgb', xgb_base),
        ('lgb', lgb_base)
    ],
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

# Cross-validation
stack_scores = cross_val_score(stack_model, X_processed, y, cv=10, scoring='accuracy')
print(f"Stacking CV Accuracy: {stack_scores.mean():.4f} Â± {stack_scores.std():.4f}")


%%time
stack_model.fit(X_processed, y)
stack_test_preds = stack_model.predict_proba(test_processed)[:, 1]

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (stack_test_preds > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

print(submission.head())
submission.to_csv("sub_6.csv")


X = train.drop(columns=["Personality"])
y = train["Personality"].map({"Extrovert": 0, "Introvert": 1})

del train
gc.collect()

# Handle categorical features (leave missing values as-is)
cat_features = ["Stage_fear", "Drained_after_socializing"]
for col in cat_features:
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# Handle numeric features & fill in missing values with median
num_features = X.select_dtypes(include=["float", "int"]).columns.tolist()
X[num_features] = X[num_features].fillna(X[num_features].median())
test[num_features] = test[num_features].fillna(X[num_features].median())

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

logreg_params = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42
}

logreg_scores, logreg_oof_preds_df, logreg_test_preds_df = [], [], []
preprocessed_test = preprocessor.fit(X).transform(test)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    # Train logistic regression model
    model = LogisticRegression(**logreg_params)
    model.fit(X_train_processed, y_train)

    # Predict on validation fold (class predictions for scoring)
    md_pred = model.predict(X_valid_processed)
    score = accuracy_score(y_valid, md_pred)
    logreg_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    # OOF predictions 
    oof_preds = pd.DataFrame({
        "logreg_pred": md_pred,
        "y": y_valid.values,
        "fold": i + 1,
        "index": X_valid.index
    })
    logreg_oof_preds_df.append(oof_preds)

    # Predict on test set using probabilities (for averaging)
    test_preds = pd.DataFrame(
        model.predict_proba(preprocessed_test)[:, 1],
        columns=["logreg_pred"]
    )
    test_preds["fold"] = i + 1
    logreg_test_preds_df.append(test_preds)

logreg_mean = np.mean(logreg_scores)
logreg_sd = np.std(logreg_scores)
print(f"Logistic Regression CV accuracy: {logreg_mean:.4f} Â± {logreg_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(logreg_test_preds_df[i]["logreg_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_12.csv")


%%time

xgb_params = {
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.05,
    'gamma': 0.1,
    'min_child_weight': 10,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'n_jobs': -1
}

test_xgb = xgb.DMatrix(test, enable_categorical=True)
xgb_scores, xgb_oof_preds_df, xgb_test_preds_df = [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    md = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # Predict hard class labels on validation set
    md_pred = (md.predict(dvalid) > 0.5).astype(int)

    score = accuracy_score(y_test, md_pred)
    xgb_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    # OOF predictions (hard predictions)
    oof_preds = pd.DataFrame(
        md_pred,
        columns=["xgb_pred"]
    )
    oof_preds["y"] = y_test.values
    oof_preds["fold"] = i + 1
    oof_preds["index"] = X_test.index
    xgb_oof_preds_df.append(oof_preds)

    # Predict on test set (hard predictions)
    test_preds = pd.DataFrame(
        (md.predict(test_xgb) > 0.5).astype(int),
        columns=["xgb_pred"]
    )
    test_preds["fold"] = i + 1
    xgb_test_preds_df.append(test_preds)

xgb_mean = np.mean(xgb_scores)
xgb_sd = np.std(xgb_scores)
print(f"XGBoost CV accuracy: {xgb_mean:.4f} Â± {xgb_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(xgb_test_preds_df[i]["xgb_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_13.csv")


%%time

%%time

lgb_params = {
    'learning_rate': 0.0951305706297959,
    'max_depth': 7,
    'reg_alpha': 1.803743818590143,
    'reg_lambda': 0.01128965915670642,
    'num_leaves': 87,
    'colsample_bytree': 0.7749196423063947,
    'verbose': -1,
    'n_jobs': -1,
    'device': 'cpu'
}

lgb_scores, lgb_oof_preds_df, lgb_test_preds_df = [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"---- Working on Fold {i+1} ----")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Categorical features are passed as-is, including missing values
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, categorical_feature=cat_features)

    md = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    # Predict on validation fold
    md_pred = (md.predict(X_test) > 0.5).astype(int)
    score = accuracy_score(y_test, md_pred)
    lgb_scores.append(score)
    print(f"Fold {i+1} accuracy: {score:.4f}")

    oof_preds = pd.DataFrame({
        "lgb_pred": md_pred,
        "y": y_test.values,
        "fold": i + 1,
        "index": X_test.index
    })
    lgb_oof_preds_df.append(oof_preds)

    # Predict on test set using probabilities
    test_preds = pd.DataFrame(
        md.predict(test),
        columns=["lgb_pred"]
    )
    test_preds["fold"] = i + 1
    lgb_test_preds_df.append(test_preds)

lgb_mean = np.mean(lgb_scores)
lgb_sd = np.std(lgb_scores)
print(f"LightGBM CV accuracy: {lgb_mean:.4f} Â± {lgb_sd:.4f}")


%%time
preds = []
for i in range(10):
    preds.append(lgb_test_preds_df[i]["lgb_pred"])

submission = pd.read_csv("../input/playground-series-s5e7/sample_submission.csv", index_col=0)
submission["Personality"] = (np.mean(preds, axis=0) > 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({0: "Extrovert", 1: "Introvert"})

# Cleanup
del preds
gc.collect()

print(submission.head())
submission.to_csv("sub_14.csv")

