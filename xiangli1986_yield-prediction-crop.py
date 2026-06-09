import pandas as pd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from itertools import combinations

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from itertools import product
from tqdm import tqdm


df_train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
df_test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
df_train.head()


print(df_train.isnull().sum())
print(df_test.isnull().sum())


# æ•°å€¼å�‹åˆ—ï¼ˆä¸�åŒ…æ‹¬ç›®æ ‡åˆ—ï¼‰
numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.drop('yield')

for col in numerical_cols:
    print(f"\nğŸ“Š åˆ†æ��æ•°å€¼ç‰¹å¾�: {col}")
    
    # 1. æ��è¿°æ€§ç»Ÿè®¡
    print(df_train[col].describe())
    print('Skewness:', df_train[col].skew())
    print('Kurtosis:', df_train[col].kurt())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1è¡Œ2åˆ—ï¼Œæ•´ä½“å›¾å®½12é«˜4
    sns.histplot(df_train[col], kde=True, bins=30, ax=axes[0])
    axes[0].set_title(f'Distribution of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Count')

    # 3. ä¸�ç›®æ ‡å�˜é‡�çš„å…³ç³»å›¾
    sns.scatterplot(x=df_train[col], y=df_train['yield'], ax=axes[1])
    axes[1].set_title(f'{col} vs Yield')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Yield')

    plt.tight_layout()
    plt.show()


df_numeric = df_train.drop(columns=['field_id'])

correlation = df_numeric.corr()

print(correlation['yield'].sort_values(ascending=False))



# 2. åŸºç¡€ç‰¹å¾�
features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']

# 3. åˆ›å»ºæ´¾ç”Ÿç‰¹å¾�å‡½æ•°
def create_features(df):
    df_new = df.copy()
    new_cols = {}

    for col in features:
        new_cols[f'{col}_sq'] = df_new[col] ** 2
        new_cols[f'{col}_sqrt'] = np.sqrt(df_new[col])
        new_cols[f'{col}_log'] = np.log1p(df_new[col])
        new_cols[f'{col}_inv'] = 1 / (df_new[col] + 1e-5)

    for f1, f2 in combinations(features, 2):
        new_cols[f'{f1}_plus_{f2}'] = df_new[f1] + df_new[f2]
        new_cols[f'{f1}_minus_{f2}'] = df_new[f1] - df_new[f2]
        new_cols[f'{f1}_times_{f2}'] = df_new[f1] * df_new[f2]
        new_cols[f'{f1}_div_{f2}'] = df_new[f1] / (df_new[f2] + 1e-5)

    new_features_df = pd.DataFrame(new_cols)
    df_new = pd.concat([df_new, new_features_df], axis=1)
    return df_new

print("Creating features for train...")
df_train = create_features(df_train)
print("Creating features for test...")
df_test = create_features(df_test)


drop_cols = ['field_id', 'yield'] + [c for c in df_train.columns if c.endswith('_missing')]
feature_cols = [c for c in df_train.columns if c not in drop_cols]

X = df_train[feature_cols].astype(float)
y = df_train['yield']
X_test = df_test[feature_cols].astype(float)

X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_holdout.shape}")



# å�‚æ•°æ�œç´¢ç©ºé—´
#param_grid = {
#    'max_depth': [2, 4, 6, 8],
#    'learning_rate': [0.01, 0.03, 0.1],
#    'n_estimators': [200, 300, 500],
#    'subsample': [0.6,0.7, 0.8],
#    'colsample_bytree': [0.6,0.7, 0.8]
#}

#param_combinations = list(product(*param_grid.values()))
#param_names = list(param_grid.keys())

#best_val_rmse = float('inf')
#best_params = None

#kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combo in tqdm(param_combinations, desc="Grid Search"):
#    params = dict(zip(param_names, combo))
#    params.update({
#        'random_state': 42,
#        'n_jobs': -1,
#        'verbosity': 0,
#        'early_stopping_rounds': 30  # æ—©å�œæ”¾è¿™é‡Œ
#    })

#    val_preds = np.zeros(len(X_val))

#    for train_idx, valid_idx in kf.split(X_train):
#        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]#

#        model = XGBRegressor(**params)
#        model.fit(
#            X_tr, y_tr,
#            eval_set=[(X_va, y_va)],
#            verbose=False
#        )
#        val_preds += model.predict(X_val) / kf.get_n_splits()

#    val_rmse = mean_squared_error(y_val, val_preds, squared=False)

#    if val_rmse < best_val_rmse:
#        best_val_rmse = val_rmse
#        best_params = params

#print(f"\nâœ… Best params: {best_params}")
#print(f"âœ… Best validation RMSE: {best_val_rmse:.4f}")

# ç”¨è®­ç»ƒé›†è®­ç»ƒæœ€ç»ˆæ¨¡å�‹ï¼ŒéªŒè¯�é›†ç”¨äº�æ—©å�œ
#final_model = XGBRegressor(**best_params)
#final_model.fit(
#    X_train, y_train,
#    eval_set=[(X_val, y_val)],
#    verbose=True
#)

# æµ‹è¯•é›†è¯„ä¼°
#test_preds = final_model.predict(X_holdout)
#test_rmse = mean_squared_error(y_holdout, test_preds, squared=False)
#print(f"ğŸ“Œ Test RMSE (check overfitting): {test_rmse:.4f}")



#import pandas as pd
#import numpy as np
#from itertools import product
#from sklearn.model_selection import train_test_split, KFold
#from sklearn.metrics import mean_squared_error
#from catboost import CatBoostRegressor
#from tqdm import tqdm

# å�‡è®¾æ•°æ�®å·²ç»�åŠ è½½ä¸º dfï¼Œç›®æ ‡å�˜é‡�ä¸º 'y'
# ç‰¹å¾�ä¸�ç›®æ ‡

# æ•°æ�®é›†åˆ’åˆ†ï¼šè®­ç»ƒã€�éªŒè¯�ã€�æµ‹è¯• = 0.8, 0.1, 0.1

# CatBoost å�‚æ•°æ�œç´¢ç©ºé—´
#param_grid_cat = {
#    'depth': [4, 6, 8],
#    'learning_rate': [0.01, 0.03, 0.1],
#    'iterations': [200, 300, 500],
#    'subsample': [0.6, 0.7, 0.8],
#    'l2_leaf_reg': [1, 3, 5]
#}

#param_combinations = list(product(*param_grid_cat.values()))
#param_names = list(param_grid_cat.keys())

#best_val_rmse = float('inf')
#best_params = None

#kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combo in tqdm(param_combinations, desc="Grid Search"):
#    params = dict(zip(param_names, combo))
#    params.update({
#        'random_seed': 42,
#        'verbose': 0,
#        'early_stopping_rounds': 30
#    })

#    val_preds = np.zeros(len(X_val))

#    for train_idx, valid_idx in kf.split(X_train):
#        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

#        model = CatBoostRegressor(**params)
#        model.fit(
#            X_tr, y_tr,
#            eval_set=(X_va, y_va),
#            use_best_model=True
#        )
#        val_preds += model.predict(X_val) / kf.get_n_splits()

#    val_rmse = mean_squared_error(y_val, val_preds, squared=False)

#    if val_rmse < best_val_rmse:
#        best_val_rmse = val_rmse
#        best_params = params

#print(f"\nâœ… Best params: {best_params}")
#print(f"âœ… Best validation RMSE: {best_val_rmse:.4f}")

# ç”¨æ•´ä¸ªè®­ç»ƒé›†è®­ç»ƒæœ€ç»ˆæ¨¡å�‹ï¼ˆæ—©å�œä»�ç”¨éªŒè¯�é›†ï¼‰
#final_model = CatBoostRegressor(**best_params)
#final_model.fit(
#    X_train, y_train,
#    eval_set=(X_val, y_val),
#    use_best_model=True,
#    verbose=True
#)

# åœ¨æµ‹è¯•é›†ä¸Šè¯„ä¼°è¿‡æ‹Ÿå�ˆæƒ…å†µ
#test_preds = final_model.predict(X_holdout)
#test_rmse = mean_squared_error(y_holdout, test_preds, squared=False)
#print(f"ğŸ“Œ Test RMSE (check overfitting): {test_rmse:.4f}")



#import pandas as pd
#import numpy as np
#from itertools import product
#from sklearn.model_selection import train_test_split, KFold
#from sklearn.metrics import mean_squared_error
#from lightgbm import LGBMRegressor
#from tqdm import tqdm
#import lightgbm



# LightGBM å�‚æ•°æ�œç´¢ç©ºé—´
#param_grid_lgb = {
#    'num_leaves': [31, 63],
#    'learning_rate': [0.01, 0.03, 0.1],
#    'n_estimators': [200, 300, 500],
#    'subsample': [0.6, 0.8],
#    'colsample_bytree': [0.6, 0.8],
#    'num_leaves': [31, 63]
#}

#param_combinations = list(product(*param_grid_lgb.values()))
#param_names = list(param_grid_lgb.keys())

#best_val_rmse = float('inf')
#best_params = None

#kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combo in tqdm(param_combinations, desc="Grid Search"):
#    params = dict(zip(param_names, combo))
#    params.update({
#        'random_state': 42
#    })

#    val_preds = np.zeros(len(X_val))

#    for train_idx, valid_idx in kf.split(X_train):
#        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

#        model = LGBMRegressor(**params)
#        model.fit(
#            X_tr, y_tr,
#            eval_set=[(X_va, y_va)],
#            callbacks=[lightgbm.early_stopping(stopping_rounds=30, verbose=False)]
#        )
#        val_preds += model.predict(X_val) / kf.get_n_splits()

#    val_rmse = mean_squared_error(y_val, val_preds, squared=False)

#    if val_rmse < best_val_rmse:
#        best_val_rmse = val_rmse
#        best_params = params

#print(f"\nâœ… Best params: {best_params}")
#print(f"âœ… Best validation RMSE: {best_val_rmse:.4f}")

# ç”¨æ•´ä¸ªè®­ç»ƒé›†è®­ç»ƒæœ€ç»ˆæ¨¡å�‹ï¼ˆæ—©å�œä»�ç”¨éªŒè¯�é›†ï¼‰
#final_model = LGBMRegressor(**best_params)
#final_model.fit(
#    X_train, y_train,
#    eval_set=[(X_val, y_val)],
#    callbacks=[lightgbm.early_stopping(stopping_rounds=30), lightgbm.log_evaluation(1)]
#)

# åœ¨æµ‹è¯•é›†ä¸Šè¯„ä¼°è¿‡æ‹Ÿå�ˆæƒ…å†µ
#test_preds = final_model.predict(X_holdout)
#test_rmse = mean_squared_error(y_holdout, test_preds, squared=False)
#print(f"ğŸ“Œ Test RMSE (check overfitting): {test_rmse:.4f}")



#import numpy as np
#from sklearn.ensemble import HistGradientBoostingRegressor
#from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error
#from tqdm import tqdm

#param_grid_hgb = {
#    'max_iter': [100, 200, 300],
#    'max_depth': [3, 5, 7],
#    'learning_rate': [0.01, 0.03, 0.1],
#    'min_samples_leaf': [10, 20, 30]
#}

#param_combinations = list(product(*param_grid_hgb.values()))
#param_names = list(param_grid_hgb.keys())

#best_val_rmse = float('inf')
#best_params = None

#kf = KFold(n_splits=5, shuffle=True, random_state=42)

#for combo in tqdm(param_combinations, desc="Grid Search"):
#    params = dict(zip(param_names, combo))
    # åŠ å…¥æ—©å�œå’ŒéªŒè¯�æ¯”ä¾‹å�‚æ•°#
#    params.update({
#        'early_stopping': True,
#        'validation_fraction': 0.1,
#        'n_iter_no_change': 30,
#        'random_state': 42,
#        'verbose': 0
#    })

#    val_preds = np.zeros(len(X_val))

#    for train_idx, valid_idx in kf.split(X_train):
#        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
#        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

#        model = HistGradientBoostingRegressor(**params)
#        model.fit(X_tr, y_tr)  # å�ªèƒ½ä¼ è®­ç»ƒé›†ï¼Œè‡ªåŠ¨å†…éƒ¨åˆ’åˆ†éªŒè¯�é›†æ—©å�œ

#        val_preds += model.predict(X_val) / kf.get_n_splits()

#    val_rmse = mean_squared_error(y_val, val_preds, squared=False)

#    if val_rmse < best_val_rmse:
#        best_val_rmse = val_rmse
#        best_params = params

#print(f"\nâœ… Best params: {best_params}")
#print(f"âœ… Best validation RMSE: {best_val_rmse:.4f}")

# è®­ç»ƒæœ€ç»ˆæ¨¡å�‹
#X_combined = pd.concat([X_train, X_val])
#y_combined = pd.concat([y_train, y_val])

#final_model = HistGradientBoostingRegressor(**best_params)
#final_model.fit(X_combined, y_combined)

#test_preds = final_model.predict(X_holdout)
#test_rmse = mean_squared_error(y_holdout, test_preds, squared=False)
#print(f"ğŸ“Œ Test RMSE (check overfitting): {test_rmse:.4f}")



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')



# æ‹†åˆ† holdout éªŒè¯�é›†ï¼ˆæœ€ç»ˆæ¨¡å�‹è¯„ä¼°ç”¨ï¼‰

# ----------------------------
# æ¨¡å�‹å®šä¹‰ï¼ˆä½¿ç”¨æœ€ä¼˜å�‚æ•°ï¼‰
# ----------------------------
models = {
    "XGB": XGBRegressor(
        max_depth=8,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30
    ),
    "CatBoost": CatBoostRegressor(
        depth=4,
        learning_rate=0.1,
        iterations=200,
        subsample=0.6,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=30
    ),
    "LightGBM": LGBMRegressor(
        num_leaves=31,
        learning_rate=0.01,
        n_estimators=200,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=42
    ),
    "HistGB": HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=3,
        learning_rate=0.01,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        verbose=0
    )
}

# ----------------------------
# æ¨¡å�‹äº¤å�‰éªŒè¯�ï¼ˆ5æŠ˜ï¼‰
# ----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = {name: np.zeros(X_temp.shape[0]) for name in models}
test_preds = {name: np.zeros(X_test.shape[0]) for name in models}

for name, model in models.items():
    print(f"Training {name}")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_temp, y_temp)):
        X_tr, X_va = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
        y_tr, y_va = y_temp.iloc[train_idx], y_temp.iloc[val_idx]

        if name == "HistGB":
            model.fit(X_tr, y_tr)
        elif name == "LightGBM":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(0)
                ]
            )
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        oof_preds[name][val_idx] = model.predict(X_va)
        test_preds[name] += model.predict(X_test) / cv.n_splits

# ----------------------------
# è��å�ˆæ¨¡å�‹ï¼šRidge å›�å½’
# ----------------------------
oof_stack = np.vstack([oof_preds[name] for name in models]).T
test_stack = np.vstack([test_preds[name] for name in models]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_stack, y_temp)

stacked_preds = meta_model.predict(test_stack)
holdout_stack = np.column_stack([model.predict(X_holdout) for model in models.values()])
final_holdout_preds = meta_model.predict(holdout_stack)

# ----------------------------
# è¾“å‡ºæœ€ç»ˆè¯„åˆ†
# ----------------------------
rmse = mean_squared_error(y_holdout, final_holdout_preds, squared=False)
print(f"Final RMSE on holdout set: {rmse:.4f}")



# é¢„æµ‹æµ‹è¯•é›†ï¼Œæ¯�ä¸ªåŸºæ¨¡å�‹åˆ†åˆ«é¢„æµ‹
base_test_preds = []
for name, model in models.items():
    pred = model.predict(X_test)
    base_test_preds.append(pred)

# å †å� åŸºæ¨¡å�‹é¢„æµ‹ç»“æ�œä½œä¸ºè��å�ˆæ¨¡å�‹çš„è¾“å…¥
X_test_meta = np.column_stack(base_test_preds)

# ä½¿ç”¨è��å�ˆæ¨¡å�‹é¢„æµ‹æœ€ç»ˆç»“æ�œ
final_preds = meta_model.predict(X_test_meta)

# ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
submission = df_test[['field_id']].copy()
submission['yield'] = final_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission saved to /kaggle/working/submission.csv")
import os

submission_path = '/kaggle/working/submission.csv'
submission_dir = os.path.dirname(submission_path)
submission_file = os.path.basename(submission_path)

# é��å�†ç›®å½•ï¼Œåˆ é™¤é™¤äº†æ��äº¤æ–‡ä»¶å¤–çš„æ‰€æœ‰æ–‡ä»¶å’Œæ–‡ä»¶å¤¹
for f in os.listdir(submission_dir):
    file_path = os.path.join(submission_dir, f)
    if f != submission_file:
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # åˆ é™¤æ–‡ä»¶æˆ–é“¾æ�¥
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)  # åˆ é™¤æ–‡ä»¶å¤¹å�Šå†…å®¹
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

print(f"âœ… Only {submission_file} kept, others deleted.")


import os

submission_path = '/kaggle/working/submission.csv'
if os.path.exists(submission_path):
    print("âœ… Submission file exists:", submission_path)
else:
    print("â�Œ Submission file NOT found:", submission_path)





