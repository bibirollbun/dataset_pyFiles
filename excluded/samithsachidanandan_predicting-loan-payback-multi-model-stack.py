import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier





train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train_ids = train['id']
test_ids = test['id']


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


train.head()


test.head()


train.shape


train.info()


train.dtypes


print("Target column statistics (loan_paid_back):")

train['loan_paid_back'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train.describe().T


def remove_outliers(train_df, test_df=None):
  
    train_df = train_df.copy()
    
   
    credit_upper = train_df['credit_score'].mean() + 3 * train_df['credit_score'].std()
    credit_lower = train_df['credit_score'].mean() - 3 * train_df['credit_score'].std()
    rate_upper = train_df['interest_rate'].mean() + 3 * train_df['interest_rate'].std()
    rate_lower = train_df['interest_rate'].mean() - 3 * train_df['interest_rate'].std()
    
  
    train_df['credit_score'] = np.clip(train_df['credit_score'], credit_lower, credit_upper)
    train_df['interest_rate'] = np.clip(train_df['interest_rate'], rate_lower, rate_upper)
    

    features = ['annual_income', 'debt_to_income_ratio', 'loan_amount']
    limits = {}
    
    for feature in features:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        limits[feature] = {
            'lower': Q1 - 1.5 * IQR,
            'upper': Q3 + 1.5 * IQR
        }
        train_df[feature] = np.clip(train_df[feature], limits[feature]['lower'], limits[feature]['upper'])
    

    if test_df is not None:
        test_df = test_df.copy()
        test_df['credit_score'] = np.clip(test_df['credit_score'], credit_lower, credit_upper)
        test_df['interest_rate'] = np.clip(test_df['interest_rate'], rate_lower, rate_upper)
        
        for feature in features:
            test_df[feature] = np.clip(test_df[feature], limits[feature]['lower'], limits[feature]['upper'])
        
        return train_df, test_df
    
    return train_df








def engineer_features(df):
   
    df = df.copy()
    

    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    
    df['total_debt'] = df['loan_amount'] * df['debt_to_income_ratio']
    df['monthly_debt'] = df['total_debt'] / 12
    df['remaining_income'] = df['monthly_income'] - df['monthly_debt']
    
    df['credit_efficiency'] = df['credit_score'] / (df['debt_to_income_ratio'] + 0.001)
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
    
  
    df['risk_score'] = (df['debt_to_income_ratio'] * df['interest_rate']) / (df['credit_score'] + 1)
    
 
    df['income_credit_interaction'] = df['annual_income'] * df['credit_score']
    df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_score']
    
   
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['debt_ratio_squared'] = df['debt_to_income_ratio'] ** 2
    df['income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    

    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['education_employment'] = df['education_level'] + '_' + df['employment_status']
    

    df['high_risk_flag'] = ((df['debt_to_income_ratio'] > 0.4) | 
                            (df['credit_score'] < 650) | 
                            (df['interest_rate'] > 15)).astype(int)
    
    df['excellent_credit_flag'] = (df['credit_score'] >= 750).astype(int)
    df['high_income_flag'] = (df['annual_income'] >= 50000).astype(int)
    df['has_advanced_degree'] = (df['education_level'].isin(["Master's", "PhD"])).astype(int)
    
    return df



train, test = remove_outliers(train, test)


train.columns


train_df = engineer_features(train)
test_df = engineer_features(test)


y_train = train_df['loan_paid_back']
X_train = train_df.drop('loan_paid_back', axis=1)

X_test = test_df.copy()


cols_to_drop = [col for col in X_train.columns if col.startswith('_')]
if cols_to_drop:
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
print(f"Dropped temporary columns: {cols_to_drop}")


numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()


categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()



print("*"*180)
print("Numeric:", numeric_cols)

print("*"*180)

print("Categorical:", categorical_cols)
print("*"*180)




preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
    ('scale', MinMaxScaler(), numeric_cols)
])



xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'device': 'cpu', #'cuda'
    'random_state': 42,
    'learning_rate': 0.010433357477511243,
    'n_estimators': 20000,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'enable_categorical': False,
    'scale_pos_weight': 0.8,
    "min_samples_split": 5,
    'lambda': 5.0, 
    'alpha': 2.5,
    'max_bin': 512
}


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1, 
    'n_jobs': -1, 
    
}

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'iterations': 5000,
    'random_seed': 42,
    'verbose': False
}

rf_params = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

et_params = ExtraTreesClassifier(
    n_estimators=800,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)


mlp_params = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

gbc_params = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    random_state=42
)


xgb_l2_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,        
    'n_estimators': 10000,        
    'max_depth': 3,               
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_lambda': 5,              
    'reg_alpha': 2,               
    'random_state': 42,
    'n_jobs': -1
}



print("Performing 5-fold CV for XGBoost (L1 Model 1)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
test_preds_xgb = np.zeros(len(X_test))
best_iters_xgb = []
cv_scores_xgb = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr_raw, X_val_raw = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    X_tr = preprocessor.fit_transform(X_tr_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test)

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )

    best_iters_xgb.append(model_xgb.best_iteration)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_val, iteration_range=(0, model_xgb.best_iteration + 1))[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test_trans, iteration_range=(0, model_xgb.best_iteration + 1))[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_xgb[val_idx])
    cv_scores_xgb.append(score)
    print(f"XGB Fold {fold} | AUC: {score:.5f}")

print("\nXGB Mean CV AUC:", np.mean(cv_scores_xgb))
print("Avg XGB Best Trees:", np.mean(best_iters_xgb))



print("\nPerforming 5-fold CV for LightGBM (L1 Model 2)")

oof_lgb = np.zeros(len(X_train))
test_preds_lgb = np.zeros(len(X_test))
best_iters_lgb = []
cv_scores_lgb = []



for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr_raw, X_val_raw = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    X_tr = preprocessor.fit_transform(X_tr_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test)

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val)

   
    model_lgb = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(stopping_rounds=300)]
    )

    best_iters_lgb.append(model_lgb.best_iteration)

    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += model_lgb.predict(X_test_trans, num_iteration=model_lgb.best_iteration) / skf.n_splits

    score = roc_auc_score(y_val, oof_lgb[val_idx])
    cv_scores_lgb.append(score)
    print(f"LGB Fold {fold} | Best Trees: {model_lgb.best_iteration} | AUC: {score:.5f}")

print("\nLGB Mean CV AUC:", np.mean(cv_scores_lgb))
print("Avg LGB Best Trees:", np.mean(best_iters_lgb))



print("\nPerforming 5-fold CV for CatBoost (L1 Model 3)")

oof_cat = np.zeros(len(X_train))
test_preds_cat = np.zeros(len(X_test))
cv_scores_cat = []
best_iters_cat = []



for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    X_tr_raw, X_val_raw = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

  
    X_tr = preprocessor.fit_transform(X_tr_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test)

    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

 
    best_iters_cat.append(model_cat.get_best_iteration())

    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test_trans)[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_cat[val_idx])
    cv_scores_cat.append(score)
    print(f"CAT Fold {fold} | AUC: {score:.5f}")

print("\nCAT Mean CV AUC:", np.mean(cv_scores_cat))
print("Avg CAT Best Iteration:", np.mean(best_iters_cat))



print("\n\nPerforming 5-fold CV for RandomForest (L1 Model 4)")

oof_rf = np.zeros(len(X_train))
test_preds_rf = np.zeros(len(X_test))

cv_scores_rf = []
best_iters_rf = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = preprocessor.fit_transform(X_train.iloc[train_idx])
    X_val = preprocessor.transform(X_train.iloc[val_idx])
    X_test_trans = preprocessor.transform(X_test)



    model_rf = RandomForestClassifier(**rf_params)
    model_rf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )

     
    best_iters_rf.append(model_rf.get_best_iteration())

    oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
    test_preds_rf += rf.predict_proba(X_test_trans)[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_rf[val_idx])
    cv_scores_rf.append(score)
    print(f"RF Fold {fold} | AUC: {score:.5f}")

print("\n RF Mean CV AUC:", np.mean(cv_scores_rf))
print("Avg RF Best Iteration:", np.mean(best_iters_rf))



print("\n\nPerforming 5-fold CV for ExtraTrees (L1 Model 5)")

oof_et = np.zeros(len(X_train))
test_preds_et = np.zeros(len(X_test))
cv_scores_et = []
best_iters_et = []


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = preprocessor.fit_transform(X_train.iloc[train_idx])
    X_val = preprocessor.transform(X_train.iloc[val_idx])
    X_test_trans = preprocessor.transform(X_test)

    model_et = ExtraTreesClassifier(**et_params)
    model_et.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )
    
    best_iters_et.append(model_et.get_best_iteration())
 

    oof_et[val_idx] = et.predict_proba(X_val)[:, 1]
    test_preds_et += et.predict_proba(X_test_trans)[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_et[val_idx])
    cv_scores_et.append(score)
    print(f"ET Fold {fold} | AUC: {score:.5f}")

print("\n ET Mean CV AUC:", np.mean(cv_scores_et))
print("Avg ET Best Iteration:", np.mean(best_iters_et))



print("\n\nPerforming 5-fold CV for MLPClassifier (L1 Model 6)")

oof_mlp = np.zeros(len(X_train))
test_preds_mlp = np.zeros(len(X_test))
cv_scores_mlp = []
best_iters_mlp = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = preprocessor.fit_transform(X_train.iloc[train_idx])
    X_val = preprocessor.transform(X_train.iloc[val_idx])
    X_test_trans = preprocessor.transform(X_test)

    model_mlp = MLPClassifier(**mlp_params)
    model_mlp.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )
    
    best_iters_mlp.append(model_mlp.get_best_iteration())

    oof_mlp[val_idx] = mlp.predict_proba(X_val)[:, 1]
    test_preds_mlp += mlp.predict_proba(X_test_trans)[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_mlp[val_idx])
    cv_scores_mlp.append(score)
    print(f"MLP Fold {fold} | AUC: {score:.5f}")

print("\n MLP Mean CV AUC:", np.mean(cv_scores_mlp))
print("Avg MLP Best Iteration:", np.mean(best_iters_mlp))



print("\n\nPerforming 5-fold CV for GradientBoostingClassifier (L1 Model 7)")

oof_gbc = np.zeros(len(X_train))
test_preds_gbc = np.zeros(len(X_test))

cv_scores_gbc = []
best_iters_gbc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = preprocessor.fit_transform(X_train.iloc[train_idx])
    X_val = preprocessor.transform(X_train.iloc[val_idx])
    X_test_trans = preprocessor.transform(X_test)

    model_gbc = GradientBoostingClassifier(**gbc_params)
    model_gbc.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )

    best_iters_gbc.append(model_gbc.get_best_iteration())

    oof_gbc[val_idx] = gbc.predict_proba(X_val)[:, 1]
    test_preds_gbc += gbc.predict_proba(X_test_trans)[:, 1] / skf.n_splits

    score = roc_auc_score(y_val, oof_gbc[val_idx])
    cv_scores_gbc.append(score)
    print(f"GBC Fold {fold} | AUC: {score:.5f}")

print("\n GBC Mean CV AUC:", np.mean(cv_scores_gbc))
print("Avg GBC Best Iteration:", np.mean(best_iters_gbc))


print("\nBuilding L2 training matrix (XGB + LGB + CAT + RF + ET + MLP + GBC)")

X_L2 = np.column_stack([
    oof_xgb,
    oof_lgb,
    oof_cat,
    oof_rf,
    oof_et,
    oof_mlp,
    #oof_gbc
])
y_L2 = y_train

X_test_L2 = np.column_stack([
    test_preds_xgb,
    test_preds_lgb,
    test_preds_cat,
    test_preds_rf,
    test_preds_et,
    test_preds_mlp,
    #test_preds_gbc
])



print("\nTraining Level-2 Meta-Model (XGBoost)")

oof_L2 = np.zeros(len(X_L2))
cv_scores_L2 = []

print("Performing 5-fold CV for L2 XGBoost meta-model")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_L2, y_L2), 1):
    X_tr, X_val = X_L2[train_idx], X_L2[val_idx]
    y_tr, y_val = y_L2.iloc[train_idx], y_L2.iloc[val_idx]

    model_L2 = xgb.XGBClassifier(**xgb_l2_params)
    model_L2.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=False
    )

    oof_L2[val_idx] = model_L2.predict_proba(X_val)[:, 1]

    score = roc_auc_score(y_val, oof_L2[val_idx])
    cv_scores_L2.append(score)
    print(f"L2 Fold {fold}: AUC = {score:.5f}")

print("\nL2 Mean CV AUC:", np.mean(cv_scores_L2))


print("\nTraining FINAL L2 XGBoost model on full data")

final_L2_model = xgb.XGBClassifier(**xgb_l2_params)
final_L2_model.fit(X_L2, y_L2)



print("\nGenerating final stacked predictions")

y_pred_proba = final_L2_model.predict_proba(X_test_L2)[:, 1]

submission['loan_paid_back'] = y_pred_proba
submission.to_csv('submission.csv', index=False)

print("Submission saved to submission.csv")
print(f"Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")







