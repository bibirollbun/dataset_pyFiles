import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import warnings
from sklearn.decomposition import PCA
import os
from itertools import combinations
from sklearn.linear_model import LogisticRegression
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv',index_col = 'id')
origin = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


TARGET = 'diagnosed_diabetes'
CATS = [cat for cat in train.columns if train[cat].dtype in [np.object_] and cat != TARGET]
NUMS = [col for col in train.columns if col not in CATS and col !=TARGET]


n_features = len(train.columns)-1
cols = train.drop('diagnosed_diabetes',axis=1).columns
fig, axs = plt.subplots(nrows = n_features,ncols=2,figsize = (12, 4 * n_features),dpi = 100)

for i, col in enumerate(cols):
    ## HISTOGRAMS
    sns.histplot(data=train, x=col, hue='diagnosed_diabetes', kde=True, ax=axs[i,0], multiple='dodge',palette='seismic')
    axs[i,0].set_title(f'HISTOGRAM_{col}')
    axs[i,0].grid(True, linestyle='--', alpha = 0.5)
    ## VIOLINPLOT
    sns.violinplot(data = train,x='diagnosed_diabetes', y = col, ax=axs[i,1],palette='seismic')
    axs[i,1].set_title(f'Diagnosed_diabetes_vs_{col}')
    axs[i,1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


le = LabelEncoder()

for col in CATS:
    train[col] = le.fit_transform(train[col])
    origin[col] = le.fit_transform(origin[col])
    test[col] = le.fit_transform(test[col])


fig = plt.figure(figsize=(15,15), dpi=100)
sns.heatmap(train.corr(), annot=True, fmt=".2f")
plt.title('Correlations')
plt.tight_layout()
plt.show()


X_pca = train.drop('diagnosed_diabetes', axis =1 ) ## needed drop target feature 
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_pca)
## EXPLAINED VARIANCE AFTER PCA
print(f'Explained variance: {pca.explained_variance_ratio_.sum()}')
df_pca = pd.DataFrame(X_pca,columns= ['PCA1','PCA2','PCA3'],index=train.index) ## Remember to define the index after dropping duplicates
df_pca = pd.concat([df_pca,train['diagnosed_diabetes']],axis=1)

fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111,projection='3d')

scatter = ax.scatter(
    df_pca['PCA1'],
    df_pca['PCA2'],
    df_pca['PCA3'],
    c = df_pca['diagnosed_diabetes'],
    cmap='rainbow',
    marker='o',
    alpha=0.6,
)
ax.set_title(f'3D PLOT')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.colorbar(scatter)
plt.show()



cols = [col for col in train.columns if col not in ['diagnosed_diabetes']]
new_cols = []

for col in cols:
    # mean
    mean_map = origin.groupby(col)['diagnosed_diabetes'].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name

    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    new_cols.append(new_mean_col_name)

    # count
    new_cnt_col_name = f"orig_cnt_{col}"
    cnt_map = origin.groupby(col).size().reset_index(name=new_cnt_col_name)

    train = train.merge(cnt_map, on=col, how='left')
    test = test.merge(cnt_map, on=col, how='left')
    new_cols.append(new_cnt_col_name)

for col in new_cols:
    if 'mean' in col:
        train[col] = train[col].fillna(origin['diagnosed_diabetes'].mean())
        test[col] = test[col].fillna(origin['diagnosed_diabetes'].mean())
    else:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


train['physical_activity_minutes_per_week'] = np.log1p(train['physical_activity_minutes_per_week'])
origin['physical_activity_minutes_per_week'] = np.log1p(origin['physical_activity_minutes_per_week'])
test['physical_activity_minutes_per_week'] =np.log1p(test['physical_activity_minutes_per_week'] )


X = train.drop(TARGET, axis = 1)
y = train[TARGET]
x_origin = origin.drop(TARGET,axis=1)
y_origin = origin[TARGET]


xgb_1_params = {
    'max_depth': 7,
    'colsample_bytree': 0.30306571599178395,
    'subsample': 0.8975815988996021,
    'learning_rate': 0.006234923329064926,
    'max_delta_step': 4,
    'reg_alpha': 0.48723878771443546,
    'reg_lambda': 1.9455005006285513,
    'random_state': 42,
    'device': "cuda",
    'tree_method': 'hist',
    'eval_metric': "auc",
    'objective': 'binary:logistic',
    'enable_categorical':True,
}

xgb_2_params = {
    'max_depth': 6,
    'colsample_bytree': 0.20392900126734176,
    'subsample': 0.8595787680081546,
    'learning_rate': 0.007819612534434382,
    'reg_alpha': 4.665093376582998,
    'reg_lambda': 7.147670343178915,
    'random_state': 42,
    'device': "cuda",
    'tree_method': 'hist',
    'eval_metric': "auc",
    'objective': 'binary:logistic',
    'enable_categorical':True
}

lgb_1_params = {
    'max_depth': 9,
    'num_leaves': 200,
    'colsample_bytree': 0.26380080769667474,
    'subsample': 0.6774064536899274,
    'learning_rate': 0.006664301757962708,
    'reg_alpha': 7.911556823105054,
    'reg_lambda': 3.060637528100754,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'metric': 'auc',
    'verbose': -1
}

xgb_3_params = {
    'colsample_bytree': 0.2133307353620508,
    'subsample': 0.8641359747737889,
    'learning_rate': 0.007451898019297675,
    'reg_alpha': 4.634879636776266, 
    'reg_lambda': 2.8555189043050673,
    'random_state': 42,
    'device': "cuda",
    'tree_method': 'hist',
    'eval_metric': "auc",
    'objective': 'binary:logistic',
    'enable_categorical':True
}


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
##  1
oof_xgb_1 = np.zeros(len(train))
pred_prob_xgb_1 = np.zeros(len(test))
##  2 
oof_xgb_2 = np.zeros(len(train))
pred_prob_xgb_2 = np.zeros(len(test))
##  3
oof_lgb_1 = np.zeros(len(train))
pred_prob_lgb_1 = np.zeros(len(test))
##  4
oof_xgb_3 = np.zeros(len(train))
pred_prob_xgb_3 = np.zeros(len(test))

correlations = []


for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print('#' * 15, i+1, '#' * 15)

    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # przygotowanie danych
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(test, enable_categorical=True)
    
    lgb_train = lgb.Dataset(x_train, label=y_train, categorical_feature='auto')
    lgb_valid = lgb.Dataset(x_valid, label=y_valid, reference=lgb_train, categorical_feature='auto')


    # =============== XGB 1 ===============
    model_1 = xgb.train(
        xgb_1_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    pred_proba = model_1.predict(dval, iteration_range=(0, model_1.best_iteration + 1))
    oof_xgb_1[valid_idx] = pred_proba
    pred_prob_xgb_1 += model_1.predict(dtest, iteration_range=(0, model_1.best_iteration + 1)) / skf.n_splits

    auc = roc_auc_score(y_valid, pred_proba)
    print(f"âœ… FOLD {i+1}: AUC XGB_1: {auc:.5f}")


    # =============== XGB 2 ===============
    model_2 = xgb.train(
        xgb_2_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    pred_proba = model_2.predict(dval, iteration_range=(0, model_2.best_iteration + 1))
    oof_xgb_2[valid_idx] = pred_proba
    pred_prob_xgb_2 += model_2.predict(dtest, iteration_range=(0, model_2.best_iteration + 1)) / skf.n_splits

    auc = roc_auc_score(y_valid, pred_proba)
    print(f"â˜‘ï¸� FOLD {i+1}: AUC XGB_2: {auc:.5f}")


    # =============== XGB 3 ===============
    model_3 = xgb.train(
        xgb_3_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    pred_proba = model_3.predict(dval, iteration_range=(0, model_3.best_iteration + 1))
    oof_xgb_3[valid_idx] = pred_proba
    pred_prob_xgb_3 += model_3.predict(dtest, iteration_range=(0, model_3.best_iteration + 1)) / skf.n_splits

    auc = roc_auc_score(y_valid, pred_proba)
    print(f"âœ”ï¸� FOLD {i+1}: AUC XGB_3: {auc:.5f}")


    # =============== LGB 1 ===============
    model_4 = lgb.train(
        lgb_1_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[early_stopping(stopping_rounds=100)]
    )

    pred_proba = model_4.predict(x_valid, num_iteration=model_4.best_iteration)
    oof_lgb_1[valid_idx] = pred_proba
    pred_prob_lgb_1 += model_4.predict(test, num_iteration=model_4.best_iteration) / skf.n_splits

    auc = roc_auc_score(y_valid, pred_proba)
    print(f"ğŸ’¡ FOLD {i+1}: AUC LGB_1: {auc:.5f}")


    # =============== KORELACJE â€” dynamiczne liczenie miÄ™dzy 4 modelami ===============

    oof_dict = {
        "XGB_1": oof_xgb_1,
        "XGB_2": oof_xgb_2,
        "XGB_3": oof_xgb_3,
        "LGB_1": oof_lgb_1
    }

    fold_corrs = []

    print(f"\nğŸ”— KORELACJE FOLD {i+1}:")
    for (name1, o1), (name2, o2) in combinations(oof_dict.items(), 2):
        corr = np.corrcoef(o1[valid_idx].ravel(), o2[valid_idx].ravel())[0, 1]
        fold_corrs.append(corr)
        print(f"Corr {name1} vs {name2}: {corr:.5f}")

    correlations.append(np.mean(fold_corrs))
    print(f"Åšrednia korelacja w foldzie: {np.mean(fold_corrs):.5f}\n")


print(f"\nğŸ“Š Mean correlation after {skf.n_splits} Folds: {np.mean(correlations):.5f}")


# XGB_1
auc = roc_auc_score(y, oof_xgb_1)
print(f'âœ… Final XGB_1 AUC Score: {auc:.5f}')

# XGB_2
auc = roc_auc_score(y, oof_xgb_2)
print(f'âœ… Final XGB_2 AUC Score: {auc:.5f}')

# XGB_3
auc = roc_auc_score(y, oof_xgb_3)
print(f'âœ… Final XGB_3 AUC Score: {auc:.5f}')

# LGB_1
auc = roc_auc_score(y, oof_lgb_1)
print(f'âœ… Final LGB_1 AUC Score: {auc:.5f}')


X_meta = np.column_stack([oof_xgb_1,oof_xgb_2,oof_xgb_3,oof_lgb_1])
x_test = np.column_stack([pred_prob_xgb_1,pred_prob_xgb_2,pred_prob_xgb_3,pred_prob_lgb_1])


def objective(trial):
    C = trial.suggest_float('C', 1e-3, 10.0)
    tol = trial.suggest_float('tol', 1e-5, 1e-1)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = trial.suggest_categorical('solver', ['saga', 'lbfgs', 'newton-cg', 'newton-cholesky'])
    max_iter = trial.suggest_int('max_iter', 1000, 3001)

    valid_combinations = {
        'saga': ['l1', 'l2'],
        'lbfgs': ['l2'],
        'newton-cg': ['l2'],
        'newton-cholesky': ['l2']
    }

    if penalty not in valid_combinations[solver]:
        raise optuna.exceptions.TrialPruned()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_meta))

    for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
        x_train, x_valid = X_meta[train_idx], X_meta[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        try:
            model_params = {
                'C': C,
                'tol': tol,
                'penalty': penalty,
                'solver': solver,
                'max_iter': max_iter,
                'fit_intercept': True
            }

            model = LogisticRegression(**model_params)
            model.fit(x_train, y_train)

            # â�— UÅ¼ywamy prawdopodobieÅ„stw do AUC, nie klas
            preds = model.predict_proba(x_valid)[:, 1]

            oof[valid_idx] = preds

            auc = roc_auc_score(y_valid, preds)
    

        except Exception as e:
            print(f"âš ï¸� Exception on fold {i+1}: {e}")
            return 0.0

    # â�— Finalny wynik takÅ¼e na AUC (OUT-OF-FOLD)
    overall_auc = roc_auc_score(y, oof)
    return overall_auc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial:")
print(study.best_trial.params)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_stack = np.zeros(len(X_meta))
test_preds_stack = []

print("\nStarting 10-Fold LR training (AUC)...")

for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
    
    lr_model = LogisticRegression(
        **study.best_trial.params,
        fit_intercept=True
    )

    x_train, x_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    lr_model.fit(x_train, y_train)

  
    preds_valid_proba = lr_model.predict_proba(x_valid)[:, 1]
    oof_stack[valid_idx] = preds_valid_proba

 
    preds_test_proba = lr_model.predict_proba(x_test)[:, 1]
    test_preds_stack.append(preds_test_proba)


    auc = roc_auc_score(y_valid, preds_valid_proba)
    print(f"âœ… FOLD {i+1}: AUC: {auc:.5f}")


final_auc = roc_auc_score(y, oof_stack)
print(f"\nğŸ�¯ Final OOF AUC (Stacking LR): {final_auc:.5f}")


final_test_preds = np.mean(test_preds_stack, axis=0)
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission['diagnosed_diabetes'] = final_test_preds
submission.to_csv('submission.csv',index = False)


submission.head()

