
import gc, time, warnings, math, itertools
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor

warnings.simplefilter('ignore')

# --------------------------------------------------------------------------
# 1. LOAD
PATH = "/kaggle/input/playground-series-s5e5/"
train  = pd.read_csv(PATH + "train.csv")
test   = pd.read_csv(PATH + "test.csv")
sub    = pd.read_csv(PATH + "sample_submission.csv")

# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
CONT = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
def feat_eng(df: pd.DataFrame, fit_le=False):
    out = df.copy()
    
    # --- Sex label ---
    if fit_le:
        out['Sex'] = LabelEncoder().fit_transform(out['Sex'])
    else:
        out['Sex'] = lbl.transform(out['Sex'])
    
    # --- Basic physio/BMI ---
    out['BMI']           = out['Weight']/((out['Height']/100)**2 + 1e-3)
    out['HR_max']        = 220 - out['Age']           # crude cardio formula
    out['HR_reserve']    = out['Heart_Rate']/out['HR_max']
    out['Temp_dev']      = out['Body_Temp']-37.0
    out['Effort_Index']  = out['Duration']*out['Heart_Rate']*out['Body_Temp']
    out['Intensity']     = out['Duration']*out['Heart_Rate']
    
    # --- Squares ---
    for f in CONT:
        out[f'{f}_sq'] = out[f]**2
    
    # --- Selected pairwise products / ratios (avoid blow-up) ---
    pairs = [('Duration','Heart_Rate'),
             ('Age','Heart_Rate'),
             ('Body_Temp','Heart_Rate'),
             ('Duration','Body_Temp'),
             ('BMI','Duration'),
             ('BMI','Heart_Rate')]
    for a,b in pairs:
        out[f'{a}_x_{b}']   = out[a]*out[b]
        out[f'{a}_div_{b}'] = out[a]/(out[b]+1e-3)
    
    return out

lbl = LabelEncoder(); lbl.fit(train['Sex'])
train_fe = feat_eng(train, fit_le=True)
test_fe  = feat_eng(test,  fit_le=False)

FEATURES = [c for c in train_fe.columns if c not in ['id','Calories']]
X, y  = train_fe[FEATURES], np.log1p(train_fe['Calories'])
X_te  = test_fe[FEATURES]

# --------------------------------------------------------------------------
# 3. MODEL DEFINITIONS
SEEDS       = [42, 2024, 1337]
FOLDS       = 5
cat_params  = dict(loss_function='RMSE', eval_metric='RMSE', iterations=3000,
                   learning_rate=0.025, depth=8, l2_leaf_reg=4,
                   random_seed=None, verbose=False, early_stopping_rounds=150,
                   cat_features=['Sex'])
lgb_params  = dict(objective='rmse', n_estimators=3000, learning_rate=0.03,
                   max_depth=-1, num_leaves=63, subsample=0.85,
                   colsample_bytree=0.75, random_state=None, n_jobs=-1)
xgb_params  = dict(objective='reg:squarederror', eval_metric='rmse',
                   tree_method='hist', n_estimators=3000, learning_rate=0.03,
                   max_depth=8, subsample=0.85, colsample_bytree=0.75,
                   gamma=0.01, random_state=None)

models_cfg = {'Cat': (CatBoostRegressor, cat_params),
              'LGB': (LGBMRegressor,      lgb_params),
              'XGB': (XGBRegressor,       xgb_params)}

# --------------------------------------------------------------------------
# 4. TRAIN / PREDICT / CV
oof_log = pd.DataFrame(index=train.index)
test_pred_log_accum = pd.DataFrame()

cv_scores = {}

for mdl_name, (cls, base_params) in models_cfg.items():
    fold_preds_oof = np.zeros(len(train))
    fold_preds_test = np.zeros(len(test))
    fold_scores = []
    
    for seed in SEEDS:
        params = base_params.copy(); params['random_state'] = seed
        kf     = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        
        for f, (tr, va) in enumerate(kf.split(X)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            
            model = cls(**params)
            if mdl_name == 'Cat':
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            elif mdl_name == 'LGB':
                model.fit(X_tr, y_tr,
                          eval_set=[(X_va, y_va)],
                          callbacks=[early_stopping(stopping_rounds=150,
                                                    verbose=False)])
            else:  # XGB
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          verbose=False, early_stopping_rounds=150)
            
            oof            = model.predict(X_va)
            fold_preds_oof[va] += oof
            fold_score      = math.sqrt(mean_squared_log_error(
                                np.expm1(y_va), np.expm1(np.maximum(oof,0))))
            fold_scores.append(fold_score)
            
            fold_preds_test += model.predict(X_te) / (FOLDS*len(SEEDS))
            
            del model; gc.collect()
    
    # Average WHERE duplicates happened (each val seen len(SEEDS) times)
    fold_preds_oof /= len(SEEDS)
    oof_log[mdl_name] = fold_preds_oof
    test_pred_log_accum[mdl_name] = fold_preds_test
    
    cv_scores[mdl_name] = np.mean(fold_scores)
    print(f"{mdl_name}: CV RMSLE {cv_scores[mdl_name]:.5f}  (seeds mean)")

# --------------------------------------------------------------------------
# 5. RIDGE STACK
ridge = Ridge(alpha=1.0)
ridge.fit(oof_log, y)
stack_oof_log   = ridge.predict(oof_log)
stack_test_log  = ridge.predict(test_pred_log_accum)

stack_score = math.sqrt(mean_squared_log_error(
                    np.expm1(y), np.expm1(np.maximum(stack_oof_log,0))))
print(f"\nSTACK OOF RMSLE: {stack_score:.5f}")

# --------------------------------------------------------------------------
# 6. WEIGHTED BLEND  (ridge-stack + best single)
best_single = min(cv_scores, key=cv_scores.get)
w_best      = 0.25          # tweakable
final_test_log = (1-w_best)*stack_test_log + w_best*test_pred_log_accum[best_single]

final_preds = np.clip(np.expm1(final_test_log), 1, 314)
sub['Calories'] = final_preds
sub.to_csv("submission.csv", index=False)

print("\n✅ submission.csv written.  Preview:")
print(sub.head())
# ==========================================================================



