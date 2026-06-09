import numpy as np
import pandas as pd


# Reading Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


FOLDS = 5
SEED = 42
VAL_SIZE = 0.2


# Checking for missing values
print('------> Train')
print(train_df.isnull().sum())
print('')
print('------> Test')
print(test_df.isnull().sum())


# Checking for data types
print('------> Train')
print(train_df.dtypes)
print('')
print('------> Test')
print(test_df.dtypes)


def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Rhythm_Energy'] = df['RhythmScore'] * df['Energy']
    df['Rhythm_Loudness'] = df['RhythmScore'] * df['AudioLoudness']
    df['Duration_Minutes'] = df['TrackDurationMs'] / 60000  
    df['Duration_Energy_Ratio'] = df['TrackDurationMs'] / (df['Energy'] * 10000 + 1)  
    df['RhythmScore_Squared'] = df['RhythmScore'] ** 2
    df['Energy_Squared'] = df['Energy'] ** 2
    df['Log_Duration'] = np.log1p(df['TrackDurationMs']) 
    df['Acoustic_Instrumental_Ratio'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 0.01) 
    df['Vocal_Energy'] = df['VocalContent'] * df['Energy']
    df['Live_Energy'] = df['LivePerformanceLikelihood'] * df['Energy']
    df['Mood_Rhythm'] = df['MoodScore'] * df['RhythmScore']
    df['Audio_Intensity'] = (df['Energy'] * np.abs(df['AudioLoudness'])) / 10  
    df['Performance_Character'] = (df['LivePerformanceLikelihood'] + df['MoodScore']) / 2
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 0.01)
    df['Rhythm_Duration_Density'] = df['RhythmScore'] / df['Duration_Minutes']

    return df

train = feature_eng(train_df)
test = feature_eng(test_df)


import itertools
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor


# Partitioning into feature matrix X and target vector y 
X = train.drop(['id', 'BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']

# Dropping id from test dataset
test.drop(['id'], axis=1, inplace=True)


# Splitting the dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, random_state=SEED, stratify=y)


# # -------------------------
# # Scaling
# # -------------------------
# scaler = StandardScaler()

# X_train = pd.DataFrame(
#     scaler.fit_transform(X_train),
#     columns=X_train.columns,
#     index=X_train.index
# )
# X_val = pd.DataFrame(
#     scaler.transform(X_val),
#     columns=X_val.columns,
#     index=X_val.index
# )
# test = pd.DataFrame(
#     scaler.transform(test),
#     columns=test.columns,
#     index=test.index
# )


# Settings for each model
regressors = {
    "LGBM(gbdt)_1": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="gbdt", max_depth=6, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "LGBM(gbdt)_2": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="gbdt", max_depth=32, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "LGBM(gbdt)_3": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="gbdt", max_depth=64, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "LGBM(goss)_1": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="goss", max_depth=3, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "LGBM(goss)_2": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="goss", max_depth=32, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "LGBM(goss)_3": lambda: lgb.LGBMRegressor(n_estimators=1000, boosting="goss", max_depth=64, learning_rate=0.05, random_state=SEED, n_jobs=-1),
    "XGB_1": lambda: xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=SEED, verbosity=0, n_jobs=4),
    "XGB_2": lambda: xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05, random_state=SEED, verbosity=0, n_jobs=4),
    "XGB_3": lambda: xgb.XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.05, random_state=SEED, verbosity=0, n_jobs=4),
    "CatBoost": lambda: CatBoostRegressor(iterations=1000, learning_rate=0.05, random_state=SEED, verbose=0),
    "HGBR": lambda: HistGradientBoostingRegressor(max_iter=2000, learning_rate=0.05, random_state=SEED),
    # "AdaBoost": lambda: AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=SEED),
    # "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
}


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def train_kfold_get_models(name, estimator_factory, X_tr, y_tr, n_splits=FOLDS, random_state=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = []
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_tr_fold = X_tr.iloc[tr_idx]
        y_tr_fold = y_tr.iloc[tr_idx]
        X_val_fold = X_tr.iloc[val_idx]
        y_val_fold = y_tr.iloc[val_idx]
        
        model = estimator_factory()

        if "LGBM" in name:
            model.fit(
                X_tr_fold, y_tr_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        elif  "XGB" in name:
            model.fit(
                X_tr_fold, y_tr_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=100,
                verbose=False
            )
        elif name == "CatBoost":
            model.fit(
                X_tr_fold, y_tr_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                early_stopping_rounds=100,
                use_best_model=True,
                verbose=False
            )
        else:
            model.fit(X_tr_fold, y_tr_fold)
        
        preds_val_fold = model.predict(X_val_fold)
        f_rmse = rmse(y_val_fold, preds_val_fold)
        fold_scores.append(f_rmse)
        models.append(model)

        print(f"  Fold {fold_idx+1}/{n_splits} done, fold RMSE = {f_rmse:.5f}")

    return models, fold_scores


def ensemble_predict(models, X):
    preds = []
    for m in models:
        preds.append(m.predict(X))
    preds = np.vstack(preds)  # shape: (n_models, n_samples)
    return preds.mean(axis=0)



# -------------------------
# Trainig for each model
# -------------------------
results = {} 
for name, factory in regressors.items():
    print(f"\nTraining {name} with {FOLDS}-fold CV on training set...")
    models, fold_scores = train_kfold_get_models(name, factory, X_train, y_train, n_splits=FOLDS)
    val_pred = ensemble_predict(models, X_val)
    val_rmse = rmse(y_val, val_pred)
    print(f"{name} ensemble on validation RMSE = {val_rmse:.5f} (fold mean RMSE = {np.mean(fold_scores):.5f})")
    results[name] = {
        "models": models,
        "fold_rmse": fold_scores,
        "val_pred": val_pred,
        "val_rmse": val_rmse
    }


# --------------------------------------------------
# Searching for combinations of models (simple average)
# --------------------------------------------------
model_names = list(results.keys())
best_combo = None
best_rmse = float("inf")
combo_records = []

# evaluate combos of size 1..len(model_names)
for r in range(1, len(model_names)+1):
    for combo in itertools.combinations(model_names, r):
        preds = np.vstack([results[m]["val_pred"] for m in combo])
        combo_pred = preds.mean(axis=0)
        combo_rmse = rmse(y_val, combo_pred)
        combo_records.append((combo, combo_rmse))
        print(f"Combo {combo} -> RMSE on validation = {combo_rmse:.5f}")
        if combo_rmse < best_rmse:
            best_rmse = combo_rmse
            best_combo = combo

print("\n=== Best combo ===")
print(best_combo, "RMSE =", best_rmse)


# -----------------------------------------------------------------
# Creating a final model with the best combination
# -----------------------------------------------------------------
print("\nRetraining best combo on full training (train+val) and predicting test set...")
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

final_models_by_method = {}
for name in best_combo:
    print(f" Retraining {name} on full data with {FOLDS}-fold...")
    factory = regressors[name]
    models_full, fold_scores_full = train_kfold_get_models(name, factory, X_full, y_full, n_splits=FOLDS)
    final_models_by_method[name] = models_full


# --------------------------------------------------
# Inference with the best combination models
# --------------------------------------------------
per_method_test_preds = []
for name, models in final_models_by_method.items():
    p = ensemble_predict(models, test)
    per_method_test_preds.append(p)
per_method_test_preds = np.vstack(per_method_test_preds)  # shape (n_methods, n_test)
final_test_pred = per_method_test_preds.mean(axis=0)


# --------------------------------------------------
# Creating submission file
# --------------------------------------------------
submission = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': final_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Complete!!!!")


submission

