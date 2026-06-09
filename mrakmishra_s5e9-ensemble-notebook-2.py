import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")



print("== Checking the Dataset == \n")
print("Shape: ", train.shape,"\n")
print("Datatype:\n" + train.dtypes.value_counts().to_string(), "\n")
print("Total duplicates: ",train.duplicated().sum(),"\n")
print("Total Null Values:\n" + train[train.columns[train.isnull().any()]].isnull().sum().to_string())


#understanding distributions of the features 
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(train.columns[:9]):  # first 9 columns
    sns.histplot(train[col], ax=axes[i], kde=True, color="skyblue")
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# Outlier Detection
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(train.columns[:9]):  # first 9 columns
    sns.boxplot(train[col], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


sns.heatmap(train.corr(), cmap="coolwarm", annot=True, fmt=".2f")


#Target Distribution
sns.histplot(train["BeatsPerMinute"])


# -------------------------------------
# 3. Exploring Outlier Removal Functions
# ------------------------------------
def outlier_removal_iqr(X):
    X = X.copy()
    for col in X.columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

def outlier_removal_std(X):
    X = X.copy()
    for col in X.columns:
        mean, std = X[col].mean(), X[col].std()
        lower, upper = mean - 3 * std, mean + 3 * std
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

X = train.drop(["BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test
# Apply IQR to all features except VocalContent
removed_X = outlier_removal_iqr(X.drop("VocalContent", axis=1))

# Apply STD only to VocalContent
removed_X_std = outlier_removal_std(X[["VocalContent"]])

# Keep only rows present in BOTH
common_idx = removed_X.index.intersection(removed_X_std.index)

# Merge them back safely
outlier_removed_df = pd.concat([removed_X.loc[common_idx], removed_X_std.loc[common_idx]], axis=1)

# Match y
y = y.loc[common_idx]
print(f"Shape before outlier removal: {X.shape}")
print(f"Shape after outlier removal: {outlier_removed_df.shape}")


# ---------------------------------------------------------
# 5. As Distribution We Know There Are Some Skewed Features 
#    So Tying Which Method Works using(log and sqrt)
# ---------------------------------------------------------
for col in outlier_removed_df.columns:
    print(f"{col}: "
          f"Original={outlier_removed_df[col].skew():.2f}, "
          f"log1p={np.log1p(outlier_removed_df[col]).skew():.2f}, "
          f"sqrt={np.sqrt(outlier_removed_df[col]).skew():.2f}")


# --------------------------------------------
# 6. Applying Transformation On Skewed Columns
# --------------------------------------------
def apply_transformations(X, scaler=None, fit_scaler=True):
    X = X.copy()

    # From skewness analysis
    log_transform_cols = ["RhythmScore", "AcousticQuality"]
    sqrt_transform_cols = ["InstrumentalScore", "LivePerformanceLikelihood", "VocalContent"]

    for col in log_transform_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    for col in sqrt_transform_cols:
        if col in X.columns:
            X[col] = np.sqrt(X[col])

    scale_cols = ["TrackDurationMs", "AudioLoudness"]
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        X[scale_cols] = scaler.transform(X[scale_cols])

    return X, scaler

# Apply to training and test
X, scaler = apply_transformations(outlier_removed_df, fit_scaler=True)
X_test, _ = apply_transformations(X_test, scaler=scaler, fit_scaler=False)
X.dropna(inplace=True)
X_test.dropna(inplace=True)
X_test = X_test[X.columns]


# LightGBM Model
def train_optuna_lightgbm(X, y, X_test):
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'feature_pre_filter': False,
        'learning_rate': 0.001502328415098844,
        'num_leaves': 79,
        'max_depth': 14,
        'feature_fraction': 0.8933016300882094,
        'bagging_fraction': 0.9754103048412501,
        'bagging_freq': 7,
        'min_child_samples': 40,
        'lambda_l1': 7.10897934678165e-07,
        'lambda_l2': 7.81564014894075e-08,
        'n_jobs': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"LGB Fold {fold + 1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(0)
            ]
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(fold_rmse)
        print(f"LGB Fold {fold + 1} RMSE: {fold_rmse:.5f}")
    
    cv_score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"LightGBM CV RMSE: {cv_score:.5f}")
    
    return oof_preds, test_preds, cv_score


def train_xgboost(X, y, X_test):
   best_params = {
       'objective': 'reg:squarederror',
       'eval_metric': 'rmse',
       'random_state': 42,
       'learning_rate': 0.05,
       'max_depth': 3,
       'min_child_weight': 1,
       'subsample': 0.9,
       'colsample_bytree': 0.9,
       'reg_alpha': 0.01,
       'reg_lambda': 1,
       'n_jobs': -1
  }

   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   oof_preds = np.zeros(len(X))
   test_preds = np.zeros(len(X_test))
   fold_scores = []
    
   for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
       print(f"XGB Fold {fold + 1}")
       X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
       model = xgb.XGBRegressor(**best_params, n_estimators=10000)
        
       model.fit(
           X_train, y_train,
           eval_set=[(X_val, y_val)],
           early_stopping_rounds=200,
           verbose=False
      )
        
       val_pred = model.predict(X_val)
       test_pred = model.predict(X_test)
        
       oof_preds[val_idx] = val_pred
       test_preds += test_pred / 5
        
       fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
       fold_scores.append(fold_rmse)
       print(f"XGB Fold {fold + 1} RMSE: {fold_rmse:.5f}")
    
   cv_score = np.sqrt(mean_squared_error(y, oof_preds))
   print(f"XGBoost CV RMSE: {cv_score:.5f}")
    
   return oof_preds, test_preds, cv_score


def train_rf(X, y, X_test):
    # Define model parameters
    rf_params = {
        'n_estimators': 400,
        'max_depth': 6,
        'max_features': 0.7806627262109607,
        'min_samples_split': 10,
        'min_samples_leaf': 8,
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"RF Fold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / kf.get_n_splits()

        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(fold_rmse)
        print(f"RF Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    cv_score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Random Forest CV RMSE: {cv_score:.5f}")

    return oof_preds, test_preds, cv_score



print("=== TRAINING LIGHTGBM ===")
lgb_oof, lgb_test, lgb_score = train_optuna_lightgbm(X, y, X_test)

# print("\n=== TRAINING XGBOOST ===")
# xgb_oof, xgb_test, xgb_score = train_xgboost(X, y, X_test)

# print("\n=== TRAINING NEURAL NETWORK ===")
# ann_oof, ann_test, ann_score = train_ann(X, y, X_test)


print("\n=== TRAINING XGBOOST ===")
xgb_oof, xgb_test, xgb_score = train_xgboost(X, y, X_test)


print("=== TRAINING RF ===")
rf_oof, rf_test, rf_score = train_rf(X, y, X_test)



print("=== TRAINING RANDOM FOREST ===")
rf_oof, rf_test, rf_score = train_rf(X, y, X_test)

# Total score (optional, for reference)
total_score = lgb_score + xgb_score + rf_score

# Ensemble weights based on inverse RMSE
lgb_weight = (1 / lgb_score) / ((1 / lgb_score) + (1 / xgb_score) + (1 / rf_score))
xgb_weight = (1 / xgb_score) / ((1 / lgb_score) + (1 / xgb_score) + (1 / rf_score))
rf_weight  = (1 / rf_score)  / ((1 / lgb_score) + (1 / xgb_score) + (1 / rf_score))

print(f"\n=== ENSEMBLE WEIGHTS ===")
print(f"LightGBM weight: {lgb_weight:.4f} (score: {lgb_score:.5f})")
print(f"XGBoost weight: {xgb_weight:.4f} (score: {xgb_score:.5f})")
print(f"Random Forest weight: {rf_weight:.4f} (score: {rf_score:.5f})")

# Create ensemble predictions
ensemble_oof = (
    lgb_weight * lgb_oof +
    xgb_weight * xgb_oof +
    rf_weight  * rf_oof
)

ensemble_test = (
    lgb_weight * lgb_test +
    xgb_weight * xgb_test +
    rf_weight  * rf_test
)

ensemble_score = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"\nEnsemble CV RMSE: {ensemble_score:.5f}")

# Create submission
submission = pd.DataFrame({
    'id': test.index,
    'BeatsPerMinute': ensemble_test
})

submission.to_csv('ensemble_submission.csv', index=False)

print(f"\n=== FINAL RESULTS ===")
print(f"LightGBM: {lgb_score:.5f}")
print(f"XGBoost: {xgb_score:.5f}")
print(f"Random Forest: {rf_score:.5f}")
print(f"Ensemble: {ensemble_score:.5f}")
print("\nEnsemble model completed!")



submission.head(25)


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# def create_ann_model(input_dim):
#     model = Sequential([
#         Dense(512, activation='relu', input_shape=(input_dim,)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(256, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(128, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
        
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),
        
#         Dense(32, activation='relu'),
#         Dropout(0.1),
        
#         Dense(1, activation='linear')
#     ])
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='mse',
#         metrics=['mae']
#     )
#     return model

# def train_ann(X, y, X_test):
#     # Scale the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_test_scaled = scaler.transform(X_test)
    
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X))
#     test_preds = np.zeros(len(X_test))
#     fold_scores = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"ANN Fold {fold + 1}")
#         X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         model = create_ann_model(X_train.shape[1])
        
#         callbacks = [
#             EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss'),
#             ReduceLROnPlateau(patience=20, factor=0.5, min_lr=1e-6, monitor='val_loss')
#         ]
        
#         model.fit(
#             X_train, y_train,
#             validation_data=(X_val, y_val),
#             epochs=300,
#             batch_size=512,
#             callbacks=callbacks,
#             verbose=0
#         )
        
#         val_pred = model.predict(X_val, verbose=0).flatten()
#         test_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
#         oof_preds[val_idx] = val_pred
#         test_preds += test_pred / 5
        
#         fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
#         fold_scores.append(fold_rmse)
#         print(f"ANN Fold {fold + 1} RMSE: {fold_rmse:.5f}")
    
#     cv_score = np.sqrt(mean_squared_error(y, oof_preds))
#     print(f"Neural Network CV RMSE: {cv_score:.5f}")
    
#     return oof_preds, test_preds, cv_score

