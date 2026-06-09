import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import early_stopping
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import cupy as cp
from tqdm import tqdm
import gc
import os


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col = 'id')


train.head()


for i,z in zip(train.columns, test.columns):
    print(f'Column_train_name: {i}, Unique_values: {train[i].nunique()}')
    print(f'Column_test_name: {z}, Unique_values: {test[z].nunique()}')


CAT = [i for i in train.columns if train[i].dtype in [object, bool]]
CAT


le = LabelEncoder()
for i in CAT:
    train[i] = le.fit_transform(train[i])
    test[i] = le.transform(test[i])


train.isnull().sum()


test.isnull().sum()


n_rows = len(train.columns[:-1])
n_cols = 2 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))

if n_rows == 1:
    axes = axes[np.newaxis, :] 

for i, col in enumerate(train.columns[:-1]):
    # train
    sns.histplot(train[col], kde=False, ax=axes[i,0], color='blue')
    axes[i,0].set_title(f'{col} - train')

    # test
    sns.histplot(test[col], kde=False, ax=axes[i,1], color='red')
    axes[i,1].set_title(f'{col} - test')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))
sns.heatmap(train.corr(), 
            annot=True,      
            fmt=".2f",       
            cmap="coolwarm", 
            cbar=True)
plt.title("Correlation Matrix")
plt.show()


for col in CAT:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


X = train.drop('accident_risk',axis = 1)
y = train['accident_risk']
X_test = test.copy()


##--------------------- PARAMS -----------------
xgb_params = {
    'learning_rate'       : 0.01166263009723703,
    'max_depth'            : 9, 
    'subsample'            : 0.8675575671387363, 
    'colsample_bytree'     : 0.7538206746710048,
    'reg_alpha'            : 0.49050499542803777, 
    'reg_lambda'           : 4.985648886263337, 
    'min_child_weight'     : 7,
    'n_estimators'         : 10000,
    'random_state'         : 42,
    'verbosity'            : 0,
    'objective'            : 'reg:squarederror',
    'enable_categorical'   : True,
    'device'               : 'cuda',
    'early_stopping_rounds': 150
}

cat_params = {
    'learning_rate'  : 0.013103028545812951,
    'depth'          : 8,
    'subsample'      : 0.8740009991386357, 
    'l2_leaf_reg'    : 6.217811804745148,
    'bootstrap_type' : 'Bernoulli',
    'iterations'     : 10000,
    'loss_function'  : 'RMSE',
    'eval_metric'    : 'RMSE',
    'task_type'      : 'GPU',
    'random_seed'    : 42,
    'verbose'        : False,
    'cat_features'   : CAT,
}

lgb_params = {
    'learning_rate'   : 0.012935612522642423,
    'max_depth'       : 9, 
    'subsample'       : 0.7145308562281946, 
    'subsample_bytree': 0.826166159182878, 
    'reg_alpha'       : 0.3814277252801652, 
    'reg_lambda'      : 4.648259170622134,
    'boosting_type'   : 'gbdt',
    'n_estimators'    : 10000,
    'random_state'    : 42,
    'verbosity'       : -1,
    "device_type"     : 'cpu',
    'objective'       : 'regression',
    'metric'          : 'rmse',

}

## ---------------------- KFOLD -------------------
kf = KFold(n_splits=5, random_state=42, shuffle=True)
## ---------------------- OOFs --------------------

oof_1 = np.zeros(len(train))
test_oof_1 = np.zeros(len(test))

oof_2 = np.zeros(len(train))
test_oof_2 = np.zeros(len(test))

oof_3 = np.zeros(len(train))
test_oof_3 = np.zeros(len(test))

## ---------------------- TRAIN LOOP --------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'\nFOLD: {fold+1}\n')
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test.copy()

    ## ------------ XGB --------------------------
    model_1 = XGBRegressor(**xgb_params)    
    model_1.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=False)
    oof_1[val_idx] = model_1.predict(X_val, iteration_range=(0, model_1.best_iteration))
    y_test = model_1.predict(X_test, iteration_range=(0, model_1.best_iteration))
    test_oof_1 += y_test / kf.n_splits 
    print(f'XGB_RMSE:  {mean_squared_error(y_val, oof_1[val_idx], squared=False)}')

    ## ------------- CAT -------------------------
    model_2 = CatBoostRegressor(**cat_params)
    model_2.fit(
            X_train,y_train,
            eval_set=[(X_val,y_val)],
            early_stopping_rounds=150,
            verbose=False
        )
    oof_2[val_idx] = model_2.predict(X_val)
    y_test = model_2.predict(X_test)
    test_oof_2 += y_test / kf.n_splits
    print(f'CAT_RMSE:  {mean_squared_error(y_val, oof_2[val_idx], squared=False)}')

    ## ------------- LGB -----------------------
    model_3  = LGBMRegressor(**lgb_params)
    model_3.fit(X_train,y_train,
                 eval_set = [(X_val,y_val)],
                 callbacks = [early_stopping(stopping_rounds = 150, verbose = False)])
    oof_3[val_idx] = model_3.predict(X_val,num_iteration=model_3.best_iteration_)
    y_test = model_3.predict(X_test, num_iteration=model_3.best_iteration_)
    test_oof_3 += y_test / kf.n_splits 
    print(f'LGB_RMSE:  {mean_squared_error(y_val, oof_3[val_idx], squared=False)}')

print(f'\n--------- SUMMARY -----------')
print(f'XGB - RMSE: {mean_squared_error(y, oof_1, squared=False)}')
print(f'CAT - RMSE: {mean_squared_error(y, oof_2, squared=False)}')
print(f'LGB - RMSE: {mean_squared_error(y, oof_3, squared=False)}')


oof_1 = cp.asarray(oof_1)
oof_2 = cp.asarray(oof_2)
oof_3 = cp.asarray(oof_3)

true = cp.asarray(y)

USE_NEGATIVE_WGT = True
MAX_MODELS = 1000
TOL = 1e-5
files = ['model_1','model_2','model_3']


oof_all = cp.stack([
    oof_1, oof_2, oof_3
], axis=-1)

n_models = oof_all.shape[1]


def rmse_cupy(y_true, y_pred):
    return cp.sqrt(cp.mean((y_true - y_pred) ** 2))


rmse_scores = []
for i in range(n_models):
    score = rmse_cupy(true, oof_all[:, i])
    rmse_scores.append(score)


best_index = int(cp.argmin(cp.array(rmse_scores)))

best_score = rmse_scores[best_index]
print(f'0 We begin with best single model RMSE {best_score:0.5f} from "{files[best_index]}"')


start = -0.5 if USE_NEGATIVE_WGT else 0.02
ww = cp.arange(start, 0.51, 0.005)
models = [best_index]
weights = []
metrics = [best_score]
ensemble = oof_all[:, best_index].copy()
old_best_score = best_score


for kk in range(1_000_000):
    best_candidate_score = 1e9  
    best_candidate_index = -1
    best_weight = 0
    best_combination = None

    candidates = list(range(n_models))

    with tqdm(total=len(candidates) * len(ww), desc=f"Iteracja {kk+1}") as pbar:
        for i in candidates:
            for w in ww:
                combined = (1 - w) * ensemble + w * oof_all[:, i]
                score = rmse_cupy(true, combined)
                if score < best_candidate_score:  
                    best_candidate_score = score
                    best_candidate_index = i
                    best_weight = float(w)
                    best_combination = combined
                pbar.update(1)

    if best_candidate_index == -1 or old_best_score - best_candidate_score < TOL:
        print(f'=> We reached tolerance {TOL}')
        break

    print(f'{kk+1} New best RMSE {best_candidate_score:.5f} adding "{files[best_candidate_index]}" with weight {best_weight:.3f}')
    ensemble = best_combination
    old_best_score = best_candidate_score
    models.append(best_candidate_index)
    weights.append(best_weight)
    metrics.append(best_candidate_score)

    if len(models) >= MAX_MODELS:
        print(f'=> We reached {MAX_MODELS} models')
        break

    del combined, best_combination
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

# Summary
print("\nğŸ“Œ Final ensemble:")
for i, idx in enumerate(models[1:], 1):
    print(f"{i}. {files[idx]} with weight {weights[i-1]:.3f}")
print(f"\nğŸ“‰ Final RMSE score: {metrics[-1]:.5f}")



wgt = np.array([1])
for w in weights:
    wgt = wgt*(1-w)
    wgt = np.concatenate([wgt,np.array([w])])
    
rows = []
t = 0
for m,w,s in zip(models,wgt,metrics):
    name = files[m]
    dd = {}
    dd['weight'] = w
    dd['model'] = name
    rows.append(dd)
    t += float( f'{w:.3f}' )

# DISPLAY WEIGHT PER MODEL
df = pd.DataFrame(rows)
df = df.groupby('model').agg('sum').reset_index().sort_values('weight',ascending=False)
df = df.reset_index(drop=True)
df


oof_dict = {}

for name in files:
    try:
        var_name = name.replace('model_', 'oof_')
        var = eval(var_name)
        oof_dict[name] = var.get() if hasattr(var, "get") else var
    except NameError:
        print(f"âš ï¸� Variable '{var_name}' not found â€” skipped.")

preds_oof = sum(w * oof_dict[m] for m, w in zip(df['model'], df['weight']))


print(f'âœ… Final  RMSE Score: {mean_squared_error(y,preds_oof, squared=False)}')


df['model'] = df['model'].str.replace('model_', '', regex=False)
pred_dict = {}
for model_name in df['model'].unique():
    var_name = f'test_oof_{model_name}'
    try:
        var = eval(var_name)
        pred_dict[model_name] = var.get() if hasattr(var, "get") else var
    except NameError:
        print(f"âš ï¸� Variable '{var_name}' not found â€” skipped.")


preds = sum(w * pred_dict[m] for m, w in zip(df['model'], df['weight']))


submission = pd.DataFrame({
    'id': test.index,
    'accident_risk': preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")
submission.head()

