import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]  
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[numerical_features])
poly_test = poly.transform(test[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

train = pd.concat([train.reset_index(drop=True), poly_train_df], axis=1)
test = pd.concat([test.reset_index(drop=True), poly_test_df], axis=1)


def drop_duplicate_columns(df):
    hashes = df.apply(lambda col: pd.util.hash_pandas_object(col, index=False).sum())
    duplicated = hashes.duplicated(keep='first')
    return df.loc[:, ~duplicated]


train_cleaned=drop_duplicate_columns(train)
test_cleaned=drop_duplicate_columns(test)


train=train_cleaned
test=test_cleaned


X = train.drop(columns=['id', 'Calories'])
y = np.log1p(train['Calories'])  
X_test = test.drop(columns=['id'])


FEATURES = X.columns.tolist()


print(FEATURES)
print(len(FEATURES))


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Base Models
base_models = {
    'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000,
                             learning_rate=0.02, gamma=0.01, max_delta_step=2, early_stopping_rounds=100,
                             eval_metric='rmse', enable_categorical=True, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
                               subsample=0.9, random_state=42, verbose=-1)
}

results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in base_models}

meta_train = np.zeros((len(train), len(base_models)))
meta_test = np.zeros((len(test), len(base_models)))

# Layer 1: Base Model Training
for model_idx, (name, model) in enumerate(base_models.items()):
    print(f"\n=== Training {name} (Layer 1) ===")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        x_train = x_train.loc[:, ~x_train.columns.duplicated()]
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
        x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

        start = time.time()

        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)

        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)

        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS
        meta_train[valid_idx, model_idx] = oof_pred
        meta_test[:, model_idx] += test_pred / FOLDS

        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)

        print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")

print("\n=== Layer 1 Model Comparison ===")
for name in base_models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")




from lightgbm import early_stopping


import numpy as np
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import time

# Assuming meta_train, y, meta_test, test, and kf are defined
# Example: from sklearn.model_selection import KFold
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

meta_models = {
    'Meta_CatBoost': CatBoostRegressor(verbose=100, random_seed=42, early_stopping_rounds=100),
    'Meta_XGBoost': XGBRegressor(max_depth=5, n_estimators=1000, learning_rate=0.05, random_state=42,
                                 early_stopping_rounds=50, eval_metric='rmse', verbosity=1)
}

# Initialize arrays for predictions
final_preds = np.zeros((len(test), len(meta_models)))
meta_oof_preds = {name: np.zeros(len(meta_train)) for name in meta_models}

print("\n=== Training Meta Models (Layer 2) ===")
for model_idx, (name, model) in enumerate(meta_models.items()):
    print(f"\nTraining {name}")
    start = time.time()

    for i, (train_idx, valid_idx) in enumerate(kf.split(meta_train, y)):
        print(f"Fold {i+1}")
        X_meta_train_fold, y_train_fold = meta_train[train_idx], y[train_idx]
        X_meta_valid_fold, y_valid_fold = meta_train[valid_idx], y[valid_idx]

        # Debugging: Check shapes and data validity
        print(f"Train fold shape: {X_meta_train_fold.shape}, {y_train_fold.shape}")
        print(f"Valid fold shape: {X_meta_valid_fold.shape}, {y_valid_fold.shape}")
        if len(X_meta_valid_fold) == 0 or len(y_valid_fold) == 0:
            raise ValueError(f"Empty validation set in fold {i+1}")

        # Ensure data is numeric and has no NaNs
        if np.any(np.isnan(X_meta_train_fold)) or np.any(np.isnan(X_meta_valid_fold)) or \
           np.any(np.isnan(y_train_fold)) or np.any(np.isnan(y_valid_fold)):
            raise ValueError(f"NaN values detected in fold {i+1} data")

        # Model training
        if 'XGBoost' in name:
            model.fit(X_meta_train_fold, y_train_fold,
                      eval_set=[(X_meta_valid_fold, y_valid_fold)],
                      verbose=100)
        elif 'CatBoost' in name:
            model.fit(X_meta_train_fold, y_train_fold,
                      eval_set=(X_meta_valid_fold, y_valid_fold))

        # Out-of-fold predictions
        meta_oof = model.predict(X_meta_valid_fold)
        meta_oof_preds[name][valid_idx] = meta_oof

        # RMSLE calculation with safety checks
        try:
            y_true_exp = np.expm1(y_valid_fold)
            y_pred_exp = np.expm1(meta_oof)
            if np.any(y_true_exp <= 0) or np.any(y_pred_exp <= 0):
                raise ValueError(f"Non-positive values in expm1 output for fold {i+1}")
            rmsle = np.sqrt(mean_squared_log_error(y_true_exp, y_pred_exp))
            print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        except Exception as e:
            print(f"Error in RMSLE calculation for fold {i+1}: {str(e)}")
            raise

    # Train on full meta_train for final predictions
    if 'XGBoost' in name:
        # Disable early stopping for final fit
        model.set_params(early_stopping_rounds=None)
        model.fit(meta_train, y, verbose=100)
    else:
        model.fit(meta_train, y)
    
    final_preds[:, model_idx] = model.predict(meta_test)
    print(f"{name} training time: {time.time() - start:.1f} sec")


# Final blend
blended_final = np.mean(final_preds, axis=1)
submission['Calories'] = np.clip(np.expm1(blended_final), 1, 314)
submission.to_csv('submission_layer2.csv', index=False)

print("\nSubmission Head:")
print(submission.head())
print(f"\nFinal Predict Mean: {submission['Calories'].mean():.2f}")
print(f"Final Predict Median: {submission['Calories'].median():.2f}")


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
models = {
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=2000, learning_rate=0.02,
                            gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
                            enable_categorical=True, random_state=42),
    'CatBoost': CatBoostRegressor(verbose=100, random_seed=42, cat_features=['Sex'], early_stopping_rounds=100),
    'LightGBM': LGBMRegressor(n_estimators=2000, learning_rate=0.02, max_depth=10, colsample_bytree=0.7,
                              subsample=0.9, random_state=42, verbose=-1)
}

results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]
        
        x_train = x_train.loc[:, ~x_train.columns.duplicated()]
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
        x_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

        start = time.time()
        
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)

        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        
        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS
        
        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)
        
        print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")


print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error

oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
test_preds = {name: np.expm1(results[name]['pred']) for name in results}
y_true = np.expm1(y)

def rmsle_loss(weights):
    blended = (
        weights[0] * oof_preds['CatBoost'] +
        weights[1] * oof_preds['XGBoost'] +
        weights[2] * oof_preds['LightGBM']
    )
    return np.sqrt(mean_squared_log_error(y_true, blended))

initial_weights = [1/3, 1/3, 1/3]
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * 3

res = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"\n✅ Optimized Weights:")
print(f"CatBoost = {best_weights[0]:.4f}")
print(f"XGBoost  = {best_weights[1]:.4f}")
print(f"LightGBM = {best_weights[2]:.4f}")

blended_preds = (
    best_weights[0] * test_preds['CatBoost'] +
    best_weights[1] * test_preds['XGBoost'] +
    best_weights[2] * test_preds['LightGBM']
)

blended_preds = np.clip(blended_preds, 1, 314)

submission['Calories'] = blended_preds
submission.to_csv('submission.csv', index=False)

print("\nSubmission_2 Head:")
print(submission.head())

print(f"\nPredict Mean: {blended_preds.mean():.2f}")
print(f"Predict Median: {np.median(blended_preds):.2f}")



import pandas as pd
import numpy as np

df1 = pd.read_csv("/kaggle/input/caloriecast-adaptive-ensemble-engine-for-s5e5/submission.csv")
df2 = pd.read_csv("/kaggle/input/ensemble-of-solutions/submission.csv")
df3 = pd.read_csv("/kaggle/input/ps-s5e5-log-blended-cat-xgboost-with-50-fold-cv/ensemble_submission.csv")


ground_truth = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")  

ground_truth['Calories'] = (0.4 * df1['Calories']) + (0.3 * df2['Calories'])+(.3 * df3['Calories'])
ground_truth.to_csv('submission.csv', index=False)



submission = pd.read_csv("submission.csv")
print(submission.describe())
print(submission.head(5))  





