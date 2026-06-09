# !pip uninstall -y lightgbm
# !apt-get install -y libboost-all-dev
# !git clone --recursive https://github.com/microsoft/LightGBM
# !cd LightGBM && mkdir build && cd build && cmake -DUSE_GPU=1 .. && make -j4
# !cd LightGBM/python-package && pip install .



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import cudf
# import cupy as cp


# ignoring all the warnings
import warnings
import os

warnings.filterwarnings("ignore")

os.environ["PYTHONWARNINGS"] = "ignore"


raw_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
raw2_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


raw_df.head()


raw_df.describe(include = 'all')


raw_df.isna().sum()


raw_df.duplicated().sum()


raw_df.info()


raw2_df.head()


raw2_df.describe(include = 'all')


raw2_df.isna().sum()


raw2_df.duplicated().sum()


raw2_df.info()


train_df = raw_df.copy()
test_df = raw2_df.copy()


columns = train_df.select_dtypes(include = ['object']).columns

for col in columns:
    print(train_df[col].value_counts())
    print()


columns = test_df.select_dtypes(include = ['object']).columns

for col in columns:
    print(test_df[col].value_counts())
    print()


class_counts = train_df['y'].value_counts()
print(class_counts)
scale_pos_weight = class_counts[0] / class_counts[1]  # majority / minority
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


train_df = train_df.drop(columns = ['id', 'month', 'day'])
test_df = test_df.drop(columns = ['id', 'month', 'day'])


train_df_rf = train_df
test_df_rf = test_df


from sklearn.preprocessing import LabelEncoder

non_numeric_columns_train = train_df_rf.select_dtypes(include = ['object']).columns
non_numeric_columns_test = test_df_rf.select_dtypes(include = ['object']).columns

le = LabelEncoder()

for column in non_numeric_columns_train:
    temp_data = np.concatenate([train_df_rf[column].values, test_df_rf[column].values])
    le.fit(temp_data)
    train_df_rf[column] = le.transform(train_df[column])
    test_df_rf[column] = le.transform(test_df_rf[column])


train_corr_matrix = train_df_rf.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(train_corr_matrix, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlation Matrix - Train Dataset')
plt.show()


x = train_df_rf.drop(columns = 'y')
y = train_df_rf['y']


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x, y)


importances = rf.feature_importances_
feature_names = x.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Rank features by importance
feature_importance = feature_importance.sort_values(by = 'Importance', ascending=False).reset_index(drop = True)
feature_importance.head(10)


from sklearn.feature_selection import SelectKBest, f_classif

best_features = SelectKBest(score_func = f_classif, k = 'all')
fit = best_features.fit(x, y)


# Get the scores of the features
feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': fit.scores_})
feature_scores = feature_scores.sort_values(by = 'Score', ascending=False).reset_index(drop = True)
feature_scores.head(10)


fi_columns = feature_importance['Feature'].values
fi_importances = feature_importance['Importance'].values
fi_threshold = feature_importance['Importance'].median()
fi_selected_features = []

for i in range(len(feature_importance)):
    if fi_importances[i] >= fi_threshold:
        fi_selected_features.append(fi_columns[i])

fs_columns = feature_scores['Feature'].values
f_scores = feature_scores['Score'].values
fs_threshold = feature_scores['Score'].median()
fs_selected_features = []

for i in range(len(feature_scores)):
    if f_scores[i] >= fs_threshold:
        fs_selected_features.append(fs_columns[i])

print(len(fi_selected_features))
print(len(fs_selected_features))


# using correlation matrix to find the best features using median as threshold

cor = abs(train_corr_matrix['y'].values)
corr_cols = train_corr_matrix.index.to_numpy()
corr_threshold = train_corr_matrix['y'].median()
corr_selected_features = []

for i in range(len(cor)):
    if cor[i] >= corr_threshold:
        corr_selected_features.append(corr_cols[i])

len(corr_selected_features)
# corr_selected_features


corr_cols


# selecting the best features for model training

columns_selected = set(fs_selected_features) & set(fi_selected_features) & set(corr_selected_features)
columns_selected = list(columns_selected)
columns_selected


train_df_xg = train_df
test_df_xg = test_df

non_numeric_columns_train = train_df_xg.select_dtypes(include = ['object']).columns
non_numeric_columns_test = test_df_xg.select_dtypes(include = ['object']).columns

for col in non_numeric_columns_train:
    train_df_xg[col] = train_df_xg[col].astype('category')
    print(f"Converted {col}: {len(train_df_xg[col].cat.categories)} categories")

for col in non_numeric_columns_test:
    test_df_xg[col] = test_df_xg[col].astype('category')
    print(f"Converted {col}: {len(test_df_xg[col].cat.categories)} categories")


# shuffling the data
from sklearn.utils import shuffle
train_df = shuffle(train_df).reset_index(drop = True)
x_train = train_df_xg.drop(columns = 'y')
y_train = train_df_xg['y']

# splitting the train dataset into 2 parts one for model training and other for evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=42)


from sklearn.metrics import roc_auc_score, make_scorer
roc_auc = make_scorer(roc_auc_score, needs_proba=True)


import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from sklearn import metrics

# effective parameter grid
param_distributions = {
    'n_estimators': [300, 500, 700],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# XGBoost with class imbalance handling
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric = 'logloss',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    device='cuda',
    # random_state=42,
    enable_categorical = True
)

# using StratifiedKFold for better cross-validation
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_xgb = HalvingRandomSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    min_resources=10000, 
    scoring=roc_auc,
    cv=cv_strategy,
    verbose=1,
    n_jobs=-1
)

print("Starting XGBoost HalvingRandomSearchCV...")
random_search_xgb.fit(x_train, y_train)

# getting best model and parameters
best_model_xgb = random_search_xgb.best_estimator_
best_params_xgb = random_search_xgb.best_params_
best_cv_score_xgb = random_search_xgb.best_score_

print(f"Best CV AUC Score: {best_cv_score_xgb:.4f}")
print(f"Best Parameters: {best_params_xgb}")

# predictions
predictions_xgb = best_model_xgb.predict(x_val)
prediction_probs_xgb = best_model_xgb.predict_proba(x_val)[:, 1]


accuracy_xg = metrics.accuracy_score(y_val, predictions_xgb)
print(f"Accuracy of XGBoost {accuracy_xg}")


roc_xg = metrics.roc_auc_score(y_val, predictions_xgb)
print(f"ROC AUC score of XGBoost {roc_xg}")


from sklearn.metrics import classification_report

print(classification_report(y_val, predictions_xgb))


import lightgbm as lgb

# parameter grid for LightGBM
param_distributions = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [10, 15, -1],
    'learning_rate': [0.05, 0.03, 0.01],
    'num_leaves': [127, 255, 511],
    'min_gain_to_split': [0.0],
    'min_sum_hessian_in_leaf': [0.001, 0.01]
}

# LightGBM with class imbalance handling
lgb_model = lgb.LGBMClassifier(
    # scale_pos_weight=scale_pos_weight,
    boosting_type='gbdt',
    objective='binary',
    metric='logloss',
    device='gpu',
    is_unbalance=True
)

# using StratifiedKFold for better cross-validation
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_lgb = HalvingRandomSearchCV(
    estimator=lgb_model,
    param_distributions=param_distributions,
    min_resources=30000,
    scoring=roc_auc,
    cv=cv_strategy,
    n_jobs=-1,
    verbose=0,
    # random_state=15
)

print("Starting LightGBM HalvingRandomSearchCV...")

random_search_lgb.fit(x_train, y_train)

# getting best model and parameters
best_model_lgb = random_search_lgb.best_estimator_
best_params_lgb = random_search_lgb.best_params_
best_cv_score_lgb = random_search_lgb.best_score_

print(f"\n=== BEST MODEL RESULTS ===")
print(f"Best CV AUC Score: {best_cv_score_lgb:.4f}")
print(f"Best Parameters: {best_params_lgb}")

# predictions
predictions_lgb = best_model_lgb.predict(x_val)
prediction_probs_lgb = best_model_lgb.predict_proba(x_val)[:, 1]


accuracy_lgb = metrics.accuracy_score(y_val, predictions_lgb)
print(f"Accuracy of Light GBM {accuracy_lgb}")


roc_lgb = metrics.roc_auc_score(y_val, predictions_lgb)
print(f"ROC AUC score of Light GBM {roc_lgb}")


from sklearn.metrics import classification_report

print(classification_report(y_val, predictions_lgb))


from catboost import CatBoostClassifier

# parameter grid for CatBoost
param_distributions = {
    'iterations': [100, 200],
    'depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [128, 254],
    'bagging_temperature': [0]
}

# CatBoost with class imbalance
cat_model = CatBoostClassifier(
    scale_pos_weight=scale_pos_weight,
    objective='Logloss',
    eval_metric='Logloss',
    # cat_features=categorical_features,
    task_type='GPU',
    # random_state=42
)

# using StratifiedKFold for better cross-validation
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search_cat = HalvingRandomSearchCV(
    estimator=cat_model,
    param_distributions=param_distributions,
    min_resources=5000,
    max_resources=600000,
    scoring=roc_auc,
    cv=cv_strategy,
    # n_jobs=-1,
    verbose=1,
    # random_state=15
)

print("Starting CatBoost training...")
print("This will test 50 parameter combinations with 3-fold CV...")

random_search_cat.fit(x_train, y_train)

# getting best model and parameters
best_model_cat = random_search_cat.best_estimator_
best_params_cat = random_search_cat.best_params_
best_cv_score_cat = random_search_cat.best_score_

print(f"\n=== CATBOOST RESULTS ===")
print(f"Best CV AUC Score: {best_cv_score_cat:.4f}")
print(f"Best Parameters: {best_params_cat}")

# predictions
predictions_cat = best_model_cat.predict(x_val)
prediction_probs_cat = best_model_cat.predict_proba(x_val)[:, 1]


accuracy_cat = metrics.accuracy_score(y_val, predictions_cat)
print(f"Accuracy of Cat Boost {accuracy_cat}")


roc_cat = metrics.roc_auc_score(y_val, predictions_cat)
print(f"ROC AUC score of CatBoost {roc_cat}")


from sklearn.metrics import classification_report

print(classification_report(y_val, predictions_cat))


x_test = test_df_xg


predictions = (best_model_xgb.predict_proba(x_test)[:, 1]).tolist()
ids = raw2_df['id'].values


pred_df = pd.DataFrame()
pred_df['id'] = ids
pred_df['y'] = predictions


pred_df.head()


pred_df.to_csv('predicted_xgb.csv', index = False)


predictions = (best_model_lgb.predict_proba(x_test)[:, 1]).tolist()
ids = raw2_df['id'].values


pred_df['y'] = predictions
pred_df.head()


pred_df.to_csv('predicted_lgb.csv', index = False)


predictions = (best_model_cat.predict_proba(x_test)[:, 1]).tolist()
ids = raw2_df['id'].values


pred_df['y'] = predictions
pred_df.head()


pred_df.to_csv('predicted_cat.csv', index = False)

