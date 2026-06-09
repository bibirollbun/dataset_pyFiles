import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")


df.head()


df.isna().sum()


df['cb_person_default_on_file'].unique()


num_features=['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
cat_features=['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']


for feature in cat_features:
    plt.figure(figsize=(8, 5))  # Set figure size
    sns.histplot(df, x=feature, hue=feature, multiple="stack", shrink=0.8)  
    plt.title(f"Count of {feature}")  # Set title
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.show()


from sklearn.preprocessing import OneHotEncoder,StandardScaler


X=df.drop(columns=['id','loan_status'])


y_train=df['loan_status']


test=pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


test=test.drop(columns=['id'])


from sklearn.compose import ColumnTransformer
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Combine transformers in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# Transform the data
X_train_prepared=preprocessor.fit_transform(X)
X_test_prepared=preprocessor.transform(test)


# This optimizes model performance and prevents overfitting.
lgb_params = {
    'objective': 'binary',
    'n_estimators': 3000,
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.033,
    'num_leaves': 23,
    'max_depth': 14,
    'min_data_in_leaf': 25,
    'feature_fraction': 0.63,
    'bagging_fraction': 0.95,
    'bagging_freq': 3,
    'verbose': -1
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgbm_predictions = np.zeros(len(X_train_prepared))
lgbm_true_labels = np.zeros(len(X_train_prepared))
lgbm_test_predictions = np.zeros(len(X_test_prepared))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_prepared, y_train)):
    X_train_fold, X_val_fold = X_train_prepared[train_idx], X_train_prepared[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lgbm_model = LGBMClassifier(**lgb_params)
    lgbm_model.fit(X_train_fold, y_train_fold,
                   eval_set=[(X_val_fold, y_val_fold)],
                   eval_metric='auc',
                   callbacks=[early_stopping(stopping_rounds=100), log_evaluation(500)]
                  )

    lgbm_fold_preds = lgbm_model.predict_proba(X_val_fold)[:, 1]
    lgbm_fold_test_preds = lgbm_model.predict_proba(X_test_prepared)[:, 1]
    
    lgbm_predictions[val_idx] = lgbm_fold_preds
    lgbm_true_labels[val_idx] = y_val_fold
    lgbm_test_predictions += lgbm_fold_test_preds / n_splits


overall_metric_lgbm = roc_auc_score(lgbm_true_labels, lgbm_predictions)
print("Overall AUC (LGBMClassifier with StratifiedKFold):", overall_metric_lgbm)


catboost_params = {
    'depth': 7,
    'learning_rate': 0.19170080203172511,
    'bagging_temperature': 0.45469161043915605,
    'l2_leaf_reg': 5,
    'loss_function': 'Logloss',
    'iterations': 400,
    'grow_policy': 'Lossguide',
    'eval_metric': 'AUC',
    'random_seed': 42,
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

catboost_predictions = np.zeros(len(X_train_prepared))
catboost_true_labels = np.zeros(len(X_train_prepared))
catboost_test_predictions = np.zeros(len(X_test_prepared))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_prepared, y_train)):
    X_train_fold, X_val_fold = X_train_prepared[train_idx], X_train_prepared[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    catboost_model = CatBoostClassifier(**catboost_params)
    catboost_model.fit(X_train_fold, y_train_fold,
                       eval_set=(X_val_fold, y_val_fold),
                       early_stopping_rounds=200,
                       verbose=100)

    catboost_fold_preds = catboost_model.predict_proba(X_val_fold)[:, 1]
    catboost_fold_test_preds = catboost_model.predict_proba(X_test_prepared)[:, 1]
    
    catboost_predictions[val_idx] = catboost_fold_preds
    catboost_true_labels[val_idx] = y_val_fold
    catboost_test_predictions += catboost_fold_test_preds / n_splits

overall_metric_catboost = roc_auc_score(catboost_true_labels, catboost_predictions)
print("Overall AUC (CatBoostClassifier with StratifiedKFold):", overall_metric_catboost)



submission=pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")
submission['loan_status'] = lgbm_test_predictions*0.4 + catboost_test_predictions*0.6
submission.to_csv('loan.csv', index=False)

