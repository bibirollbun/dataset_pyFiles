import xgboost
import lightgbm
import optuna
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, 
classification_report, 
confusion_matrix, 
roc_auc_score, 
make_scorer,
roc_curve,
f1_score)
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings('ignore')


dataTrain = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
dataTest = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

dfTrain = pd.DataFrame(dataTrain)
dfTest = pd.DataFrame(dataTest)

dfTrain.head()


print("-"*20)
print(dfTrain.info())
print("-"*20)
print(dfTrain.isnull().sum())
print("-"*20)
print(dfTrain.describe())


print("-"*20)
print(dfTest.info())
print("-"*20)
print(dfTest.isnull().sum())
print("-"*20)
print(dfTest.describe())


dfTrain['diagnosed_diabetes'] = pd.to_numeric(dfTrain['diagnosed_diabetes']).astype(int)


target = 'diagnosed_diabetes'

cat_features = dfTrain.select_dtypes(include = ['object', 'category']).columns.drop(target, errors = 'ignore')

c_features = len(cat_features)
cols = 2
rows = (c_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize = (12, rows * 3))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(
        x = col,
        data = dfTrain,
        ax = axes[i],
        #hue = target
        
    )
    axes[i].set_title(f"Count plot of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis = 'x', rotation = 45)
    #axes[i].legend(title = target)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


target = 'diagnosed_diabetes'

cat_features = dfTrain.select_dtypes(include = ['object', 'category']).columns.drop(target, errors = 'ignore')

c_features = len(cat_features)
cols = 2
rows = (c_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize = (12, rows * 3))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(
        x = col,
        data = dfTrain,
        ax = axes[i],
        hue = target
        
    )
    axes[i].set_title(f"Count plot of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis = 'x', rotation = 45)
    axes[i].legend(title = target)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


num_cols = dfTrain.select_dtypes(include=np.number).columns.drop(target, errors='ignore')

n_features = len(num_cols)
n_cols = 2
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows*3))
axes = np.array(axes).reshape(-1)  

for i, col in enumerate(num_cols):
    if i < len(axes):  
        sns.histplot(
            x=col,
            data=dfTrain,
            kde=True,
            ax=axes[i]
        )
        axes[i].set_title(f"Distribution plot of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)

for j in range(len(num_cols), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize = (12, 6))
sns.scatterplot(x = 'age', 
                y = 'physical_activity_minutes_per_week', 
                hue = 'diagnosed_diabetes',
                legend = 'auto',
               data = dfTrain)
plt.tight_layout()
plt.show()


box_cols = dfTrain.select_dtypes(include=np.number).columns.drop(target, errors='ignore')

plt.figure(figsize = (12,8))
sns.boxplot(data = dfTrain[box_cols])
plt.title("Checking for any outliers.")
plt.tick_params(axis = 'x', rotation = 45)
plt.tight_layout()
plt.show()


dfTrain.head()


plt.figure(figsize = (12,8))
sns.catplot(data = dfTrain, 
            x = 'age', 
            y = 'gender', 
            #hue = 'diagnosed_diabetes', 
            kind = 'box')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.violinplot(
    data = dfTrain,
    x = 'diet_score',
    y = 'employment_status',
    hue = 'diagnosed_diabetes',
    spilt = True,
    inner = 'quart',
    cut = 0
)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.violinplot(
    data = dfTrain,
    x = 'bmi',
    y = 'employment_status',
    hue = 'diagnosed_diabetes',
    spilt = True,
    inner = 'quart'
)
plt.tight_layout()
plt.show()


df_train_enc = dfTrain.copy()

target_col = target
df_train_enc[target_col] = pd.to_numeric(df_train_enc[target_col], errors='coerce')

cat_cols = df_train_enc.select_dtypes(include=['object', 'category']).columns.tolist()

smoothing = 20
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    enc_col = col + '_enc'
    df_train_enc[enc_col] = np.nan 

    for train_idx, val_idx in kf.split(df_train_enc):
        train_fold = df_train_enc.iloc[train_idx]
        val_fold = df_train_enc.iloc[val_idx]

        fold_global_mean = train_fold[target_col].mean()

        stats = train_fold.groupby(col)[target_col].agg(
            mean_target='mean',
            counts='count'
        )

        smooth = (stats['mean_target'] * stats['counts'] + fold_global_mean * smoothing) / (
            stats['counts'] + smoothing
        )

        df_train_enc.loc[val_idx, enc_col] = val_fold[col].map(smooth)

    global_mean_full = df_train_enc[target_col].mean()
    df_train_enc[enc_col].fillna(global_mean_full, inplace=True)

encoded_features = [col + '_enc' for col in cat_cols]

num_cols = df_train_enc.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in encoded_features + [target_col]]

final_df = df_train_enc[num_cols + encoded_features + [target_col]]

print("Final encoded dataset shape:", final_df.shape)
final_df.head()


global_mean_full = df_train_enc[target_col].mean()
encodings = {}

for col in cat_cols:
    stats = df_train_enc.groupby(col)[target_col].agg(
        mean_target='mean',
        counts='count'
    )

    smooth = (stats['mean_target'] * stats['counts'] + global_mean_full * smoothing) / (
        stats['counts'] + smoothing
    )

    encodings[col] = smooth  


df_test_enc = dfTest.copy()  

for col in cat_cols:
    enc_col = col + '_enc'
    mapping = encodings[col]

    df_test_enc[enc_col] = df_test_enc[col].map(mapping)

    df_test_enc[enc_col].fillna(global_mean_full, inplace=True)

encoded_features = [col + '_enc' for col in cat_cols]

num_cols_test = df_test_enc.select_dtypes(include=[np.number]).columns.tolist()
num_cols_test = [c for c in num_cols_test if c not in encoded_features + [target_col]]

if target_col in df_test_enc.columns:
    final_test_df = df_test_enc[num_cols_test + encoded_features + [target_col]]
else:
    final_test_df = df_test_enc[num_cols_test + encoded_features]

print("Final TEST encoded dataset shape:", final_test_df.shape)
final_test_df.head()


plt.figure(figsize=(12,8))
sns.heatmap(
    final_df.corr(),
    annot = True,
    fmt = '.2g',
    cmap = 'coolwarm'
)
plt.tight_layout()
plt.show()


X = final_df.drop(['id', 'diagnosed_diabetes'], axis = 1)
y = final_df['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(X, 
                                                  y, 
                                                  test_size = 0.2,
                                                  stratify = y,
                                                  random_state = 42)

neg = (y == 0).sum()
pos = (y == 1).sum()
spw = neg/pos

print("scale_pos_weight =", spw)


print("="*20)
print(X_train.shape)
print("="*20)
print(X_val.shape)


model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=1,
    class_weight = 'balanced'
)

model.fit(X_train, y_train)

model_train_preds = model.predict(X_train)
model_val_preds = model.predict(X_val)

model_train_acc = accuracy_score(y_train, model_train_preds)
model_val_acc = accuracy_score(y_val, model_val_preds)

model_roc_auc = roc_auc_score(y_val, model_val_preds)

print(f"Trainig Accuracy of HistGradientBoosting: {model_train_acc}")
print("-"*20)
print(f"Validation Accuracy of HistGradientBoosting: {model_val_acc}")
print("-"*20)
print(f"ROC-AUC Score of HistGradientBoosting: {model_roc_auc}")
print("="*20)
print(f"Classification report of HistGradientBoosting: \n {classification_report(y_val, model_val_preds)}")
print("="*20)
print(f"Confusion matrix of HistGradientBoosting: \n {confusion_matrix(y_val, model_val_preds)}")


'''

def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    lightmodel = LGBMClassifier(
        **params,
        device = 'gpu',
        gpu_device_id = 0,
        gpu_platform_id = 0,
        verbose = -1
    )
    lightmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = lightmodel.predict_proba(X_val)[:,1]
    
    return roc_auc_score(y_val, preds)


study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgbm, n_trials=50)   # increase to 100+ if needed

print("Best params:", study_lgb.best_params)
print("Best ROC-AUC:", study_lgb.best_value)

'''


'''

def objective_xgb(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
    }
    xgbmodel = XGBClassifier(
        **params,
        device = 'cuda',
        tree_method = 'hist'
    )
    xgbmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = xgbmodel.predict_proba(X_val)[:, 1]
    
    return roc_auc_score(y_val, preds)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=100)

print("Best params:", study_xgb.best_params)
print("Best ROC-AUC:", study_xgb.best_value)

'''


estimator = XGBClassifier(
    objective="binary:logistic",
    eval_metric = 'auc',
    device = 'gpu',
    tree_method = 'hist',
    random_state = 42
)

n_features_to_select = 15

selector = RFE(
    estimator = estimator,
    n_features_to_select = n_features_to_select,
    step = 1
)

selector.fit(X_train, y_train)

selected_mask = selector.support_
selected_features = X.columns[selected_mask]
feature_ranking = selector.ranking_

print("Selected features:")
print(selected_features)

rfe_df = pd.DataFrame({
    "feature": X.columns,
    "ranking": feature_ranking,
    "selected": selected_mask
})

# Sort by ranking (1 = selected)
rfe_df = rfe_df.sort_values(by="ranking", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(rfe_df["feature"], rfe_df["ranking"])
plt.gca().invert_yaxis()  
plt.xlabel("RFE Ranking (1 = Selected)")
plt.title("RFE Feature Ranking")
plt.tight_layout()
plt.show()

scorer = make_scorer(roc_auc_score, needs_proba=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
print("Selected features:", X.columns[rfecv.support_])


scores = rfecv.cv_results_["mean_test_score"]

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(scores) + 1), 
    scores
)
plt.xlabel("Number of features selected")
plt.ylabel("Mean CV ROC-AUC")
plt.title("RFECV - Number of Features vs ROC-AUC")
plt.grid(True)
plt.tight_layout()
plt.show()


selected_features = X.columns[rfecv.support_]

print(len(selected_features))
print(selected_features.tolist())


X_train_sel = X_train.loc[:, selected_features]
X_val_sel = X_val.loc[:, selected_features]


xgb = XGBClassifier(
    random_state = 42,
    device = 'cuda',
    tree_method = 'hist',          
    n_estimators = 1982,
    learning_rate = 0.05442455156738541,
    max_depth = 4,
    min_child_weight = 15,
    subsample = 0.5924908500012145,
    colsample_bytree = 0.9076162209956082,
    gamma = 3.27299133245244,
    alpha = 0.2748389974509704,
    scale_pos_weight = spw,
    eval_metric = 'auc',
    grow_policy='lossguide'
)

xgb.fit(X_train_sel, y_train)

xg_train_preds = xgb.predict(X_train_sel)   
xg_val_preds   = xgb.predict(X_val_sel)     

xg_val_proba = xgb.predict_proba(X_val_sel)[:, 1]

xg_train_acc = accuracy_score(y_train, xg_train_preds)
xg_val_acc   = accuracy_score(y_val, xg_val_preds)
xg_roc_auc   = roc_auc_score(y_val, xg_val_proba)

print(f"Trainig Accuracy XGBoost: {xg_train_acc}")
print("-"*20)
print(f"Validation Accuracy XGBoost: {xg_val_acc}")
print("-"*20)
print(f"ROC-AUC Score XGBoost: {xg_roc_auc}")
print("="*20)
print("Classification report of XGBoost: \n",
      classification_report(y_val, xg_val_preds))
print("="*20)
print("Confusion matrix of XGBoost: \n",
      confusion_matrix(y_val, xg_val_preds))


0.7273941803894466
0.727683458367908
0.7273646835852845
0.727683453587438


lgbm = LGBMClassifier(
    verbose = -1,
    random_state = 42,
    device = 'gpu',
    gpu_device_id = 0,
    gpu_platform_id = 0,
    n_estimators = 1453,
    learning_rate = 0.08, #0.07
    num_leaves = 289, 
    max_depth = 4, 
    min_child_samples = 42,
    subsample = 0.61,
    colsample_bytree = 0.58,
    reg_lambda = 0.05,
    min_split_gain = 0.46,
    scale_pos_weight=spw,
    eval_metric = 'auc'
)
lgbm.fit(X_train_sel, y_train)

lgbm_train_preds = lgbm.predict(X_train_sel)
lgbm_val_preds = lgbm.predict(X_val_sel)

lgbm_train_acc = accuracy_score(y_train, lgbm_train_preds)
lgbm_val_acc = accuracy_score(y_val, lgbm_val_preds)

lgbm_val_proba = lgbm.predict_proba(X_val_sel)[:, 1]

threshold = 0.50799745
preds = (lgbm_val_proba >= threshold).astype(int)

lgbm_roc_auc = roc_auc_score(y_val, lgbm_val_proba)

print(f"Trainig Accuracy of LightGBM: {lgbm_train_acc}")
print("-"*20)
print(f"Validation Accuracy of LightGBM: {lgbm_val_acc}")
print("-"*20)
print(f"ROC-AUC Score of LightGBM: {lgbm_roc_auc}")
print("="*20)
print(f"Classification report of LightGBM: \n {classification_report(y_val, preds)}")
print("="*20)
print(f"Confusion matrix of LightGBM: \n {confusion_matrix(y_val, preds)}")


xg_val_proba = xgb.predict_proba(X_val_sel)[:, 1]

print("ROC-AUC (unchanged by threshold):", roc_auc_score(y_val, xg_val_proba))


fpr, tpr, thresholds = roc_curve(y_val, xg_val_proba)

# Youden's J = TPR - FPR
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thr = thresholds[best_idx]

print("Best threshold by TPR-FPR:", best_thr)

# Apply this threshold
xg_val_preds_opt = (xg_val_proba >= best_thr).astype(int)

print("Classification report (tuned threshold):")
print(classification_report(y_val, xg_val_preds_opt))

print("Confusion matrix (tuned threshold):")
print(confusion_matrix(y_val, xg_val_preds_opt))

print("F1 (class 1) at tuned threshold:",
      f1_score(y_val, xg_val_preds_opt, pos_label=1))


thr_grid = np.linspace(0.1, 0.9, 81)   # thresholds from 0.10 to 0.90
f1_scores = []

for thr in thr_grid:
    preds = (xg_val_proba >= thr).astype(int)
    f1_scores.append(f1_score(y_val, preds, pos_label=1))

best_idx_f1 = np.argmax(f1_scores)
best_thr_f1 = thr_grid[best_idx_f1]

print("Best threshold by F1(class 1):", best_thr_f1)
print("Best F1(class 1):", f1_scores[best_idx_f1])

xg_val_preds_f1 = (xg_val_proba >= best_thr_f1).astype(int)
print("Classification report (F1-optimized threshold):")
print(classification_report(y_val, xg_val_preds_f1))
print("Confusion matrix:")
print(confusion_matrix(y_val, xg_val_preds_f1))


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color = 'blue', label = f'AUC = {lgbm_roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color = 'grey', linestyle = '--')
plt.title("ROC-AUC Curve")
plt.xlabel("False postive rate")
plt.ylabel("True postive rate")
plt.legend()
plt.show()


test_ids = dfTest['id']
X_test_sel = dfTest[selected_features]

test_preds = lgbm.predict_proba(X_test_sel)[:, 1]

submission = pd.DataFrame(
    {
        "id" : test_ids,
        'diagnosed_diabetes' : test_preds
    }
)

submission.to_csv("submission.csv", index = False)
print("submission file saved!")

