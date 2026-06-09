import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


def create_features(df, is_train=True, target=None, encoders=None):
    df = df.copy()
    
    #Basic ratios
    df['balance_to_age'] = df['balance'] / (df['age'] + 1)
    df['duration_to_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['pdays_binary'] = (df['pdays'] > -1).astype(int)
    df['previous_contact'] = df['previous'].apply(lambda x: 1 if x > 0 else 0)
    df['duration_per_age'] = df['duration'] / (df['age'] + 1)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'] + 1)
    df['has_loan_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
    df['age_bin'] = pd.qcut(df['age'], q=5, labels=False, duplicates='drop')
    
    #Cyclical encoding
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_sin'] = np.sin(2 * np.pi * df['month'].map(month_map) / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'].map(month_map) / 12)
    
    #Polynomial
    num_cols = ['age', 'balance', 'duration', 'campaign']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[num_cols])
    for i, name in enumerate(poly.get_feature_names_out(num_cols)):
        df[f'poly_{name.replace(" ", "_")}'] = poly_features[:, i]
    
    #Target encoding
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 
                    'loan', 'contact', 'month', 'poutcome']
    if is_train:
        encoders = {cat: TargetEncoder().fit(df[cat], target) for cat in cat_features}
        for cat in cat_features:
            df[cat + '_target_enc'] = encoders[cat].transform(df[cat])
    else:
        for cat in cat_features:
            df[cat + '_target_enc'] = encoders[cat].transform(df[cat])
    
    return df, encoders

y = train['y']
train, encoders = create_features(train, is_train=True, target=y)
test, _ = create_features(test, is_train=False, encoders=encoders)
features = [col for col in train.columns if col not in ['id', 'y']]
cat_features = ['job', 'marital', 'education', 'default', 'housing', 
                'loan', 'contact', 'month', 'poutcome', 'has_loan_housing', 'age_bin']
num_features = [col for col in features if col not in cat_features]
poly_cols = [col for col in train.columns if col.startswith('poly_') and '_' in col[5:]]
num_features = list(dict.fromkeys([col for col in num_features if not col.startswith('poly_')] + poly_cols))


scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])

X_train, X_val, y_train, y_val = train_test_split(train[features], y, test_size=0.2, random_state=42, stratify=y)


catboost_params = {
    "iterations": 2453, "learning_rate": 0.04451431545395578, "depth": 10,
    "l2_leaf_reg": 6.627651844082078, "border_count": 253, 
    "random_strength": 0.3562390738100375, "bagging_temperature": 0.31542802019505106,
    "task_type": "GPU", "devices": "0:1", "verbose": 0, "random_seed": 42
}

xgboost_params = {
    "n_estimators": 1000, 
    "learning_rate": 0.05, 
    "max_depth": 8,
    "subsample": 0.8, 
    "colsample_bytree": 0.8, 
    "gamma": 0.1,
    "random_state": 42, 
    "tree_method": "hist",  
    "device": "cuda",     
    "enable_categorical": False
}

lightgbm_params = {
    "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 8,
    "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "device": "gpu", "verbose": -1
}


encoded_cat_features = [col for col in train.columns if col.endswith('_target_enc')]
features = [col for col in train.columns if col not in ['id', 'y'] and col not in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']]  # Exclude original categorical columns
catboost_cat_features = ['has_loan_housing', 'age_bin']  # Categorical features for CatBoost

missing_cat_features = [col for col in catboost_cat_features if col not in features]
if missing_cat_features:
    raise ValueError(f"Categorical features {missing_cat_features} not found in features list: {features}")

for col in catboost_cat_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(int)  # Convert to int for numeric compatibility
        X_val[col] = X_val[col].astype(int)
        test[col] = test[col].astype(int)

catboost = CatBoostClassifier(**catboost_params, cat_features=catboost_cat_features)  # CatBoost can still treat as categorical
xgboost = XGBClassifier(**xgboost_params)
lightgbm = LGBMClassifier(**lightgbm_params)


ensemble = VotingClassifier(
    estimators=[
        ('catboost', catboost),
        ('xgboost', xgboost),
        ('lightgbm', lightgbm)
    ],
    voting='soft',
    weights=[0.34, 0.33, 0.33]
)

ensemble.fit(X_train[features], y_train)

val_pred = ensemble.predict_proba(X_val[features])[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation ROC AUC: {val_auc:.4f}")
test_pred = ensemble.predict_proba(test[features])[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'y': test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


import os
print(os.listdir('/kaggle/working/')) 


submission.head()


submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved at:", os.path.abspath('/kaggle/working/submission.csv'))

