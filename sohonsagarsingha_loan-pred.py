import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


SEED = 42
#dataset paths
TRAIN_CSV = "/kaggle/input/playground-series-s5e11/train.csv"
TEST_CSV  = "/kaggle/input/playground-series-s5e11/test.csv"
SUBMISSION_CSV = "/kaggle/input/playground-series-s5e11/sample_submission.csv"

#load dataset
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)


#feature extraction
use_features = ['annual_income', 'credit_score', 'loan_amount', 'interest_rate', 'debt_to_income_ratio'] 
target_col = 'loan_paid_back'

#subset
train_small = train[['id'] + use_features + [target_col]].copy()
test_small  = test[['id'] + use_features].copy()

#handling NaNs
imputer = SimpleImputer(strategy='median')
train_small[use_features] = imputer.fit_transform(train_small[use_features])
test_small[use_features]  = imputer.transform(test_small[use_features])

#optional scaling
#scaler = StandardScaler()
#train_small[use_features] = scaler.fit_transform(train_small[use_features])
#test_small[use_features]  = scaler.transform(test_small[use_features])

#prepare data for model
X = train_small[use_features].values
y = train_small[target_col].values
X_test = test_small[use_features].values
test_ids = test_small['id'].values


#OOF training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train_small))
oof_cat = np.zeros(len(train_small))
test_pred_lgb = np.zeros(len(test_small))
test_pred_cat = np.zeros(len(test_small))

lgb_params = {
    #'objective': 'binary',
    #'boosting_type': 'gbdt',
    #'n_estimators': 2000,
    #'learning_rate': 0.05,
    #'num_leaves': 64,
    #'random_state': SEED,
    #'n_jobs': -1
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': SEED,
    'metric': 'auc',
    'n_jobs': -1
}

cat_params = {
    #'iterations': 3000,
    #'learning_rate': 0.05,
    #'depth': 6,
    #'verbose': 0,
    #'random_seed': SEED
    'iterations': 3000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'verbose': 0,
    'random_seed': SEED
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{n_splits}")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    #LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    test_pred_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits

    #CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True, verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_pred_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits

    print(f"  LGB AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.4f} | CAT AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.4f}")


print("\nOverall AUCs:")
print("LightGBM:", roc_auc_score(y, oof_lgb))
print("CatBoost:", roc_auc_score(y, oof_cat))
oof_avg = 0.5 * oof_lgb + 0.5 * oof_cat
print("Average:", roc_auc_score(y, oof_avg))



#Feature Fusion
X_meta_train = np.hstack([X, oof_lgb.reshape(-1,1), oof_cat.reshape(-1,1)])
X_meta_test  = np.hstack([X_test, test_pred_lgb.reshape(-1,1), test_pred_cat.reshape(-1,1)])
feature_names = use_features + ['oof_pred_lgb', 'oof_pred_cat']

#Final LightGBM Model
final_lgb = lgb.LGBMClassifier(
    n_estimators=3300,
    learning_rate=0.001,
    num_leaves=63,
    random_state=SEED,
    n_jobs=-1
)

X_trf, X_valf, y_trf, y_valf = train_test_split(X_meta_train, y, test_size=0.15, random_state=SEED, stratify=y)
final_lgb.fit(X_trf, y_trf, eval_set=[(X_valf, y_valf)], callbacks=[lgb.early_stopping(50, verbose=False)])
val_preds_final = final_lgb.predict_proba(X_valf)[:, 1]
print("Final model validation AUC:", roc_auc_score(y_valf, val_preds_final))

test_preds_final = final_lgb.predict_proba(X_meta_test)[:, 1]


#SHAP
print("\nSHAP Explainability")
explainer = shap.TreeExplainer(final_lgb)

#Limit sample size for faster SHAP
sample_for_shap = X_meta_train if X_meta_train.shape[0] < 1000 else X_meta_train[np.random.RandomState(SEED).choice(X_meta_train.shape[0], 1000, replace=False)]
shap_values = explainer.shap_values(sample_for_shap)

#Summary plot
shap.summary_plot(shap_values, pd.DataFrame(sample_for_shap, columns=feature_names))
plt.show()

#Dependence plot for top feature
#mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
#top_feature = feature_names[np.argmax(mean_abs_shap)]
#print("Top feature by SHAP:", top_feature)
#shap.dependence_plot(top_feature, shap_values, pd.DataFrame(sample_for_shap, columns=feature_names))
#plt.show()


#Submission
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds_final
})
submission.to_csv('submission.csv', index=False)
print(f"\nSaved submission.")
df1 = pd.read_csv('submission.csv')
df1.head()













