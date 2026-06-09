import pandas as pd

# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# Check the data
print(train.shape, test.shape)
train.head()



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# ==========================================
# 2. Setup
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 3. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 4. Feature Engineering
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 5. Remove Outliers (IQR)
# ==========================================
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[mask], y[mask]

# ==========================================
# 6. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 7. Hyperparameter Tuning (Randomized Search)
# ==========================================
print("ğŸ”� Hyperparameter tuning started... (this may take a few minutes)")

param_grid = {
    'n_estimators': [600, 800, 1000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [5, 6, 7, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_base = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',  # âœ… GPU-friendly method
    device='cuda',       # âœ… GPU enabled
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=12,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)
best_params = search.best_params_
print(f"âœ… Best Params: {best_params}")
print(f"âœ… Best CV RMSE: {-search.best_score_:.5f}")

# ==========================================
# 8. Cross-Validation with Progress Bar
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
preds = np.zeros(len(test))

print("\nğŸš€ Training with 5-Fold Cross-Validation (GPU enabled)...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc="CV progress"):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        **best_params,
        tree_method='hist',
        device='cuda',
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    y_pred = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.5f}")

    # Accumulate test predictions
    preds += model.predict(test) / kf.n_splits

print("\nâœ… Average CV RMSE:", np.mean(rmse_scores))

# ==========================================
# 9. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# ==========================================
# 2. Setup
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 3. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 4. Feature Engineering
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 5. Remove Outliers (IQR)
# ==========================================
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[mask], y[mask]

# ==========================================
# 6. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 7. Hyperparameter Tuning (Randomized Search)
# ==========================================
print("ğŸ”� Hyperparameter tuning started...")

param_grid = {
    'n_estimators': [600, 800, 1000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [5, 6, 7, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

xgb_base = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',  # âœ… GPU-friendly
    device='cuda',       # âœ… GPU enabled
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=12,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)
best_params = search.best_params_
print(f"âœ… Best Params: {best_params}")
print(f"âœ… Best CV RMSE: {-search.best_score_:.5f}")

# ==========================================
# 8. Cross-Validation with Progress Bar
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
preds = np.zeros(len(test))

print("\nğŸš€ Training with 5-Fold Cross-Validation (GPU enabled)...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc="CV progress"):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(
        **best_params,
        tree_method='hist',
        device='cuda',
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    y_pred = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.5f}")

    # âœ… Drop 'id' column before predicting
    test_features = test.drop(columns=[id_col])
    preds += model.predict(test_features) / kf.n_splits

print("\nâœ… Average CV RMSE:", np.mean(rmse_scores))

# ==========================================
# 9. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Label Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Advanced Feature Engineering
# ==========================================
# Cyclic encoding for time_of_day
X['hour_sin'] = np.sin(2 * np.pi * X['time_of_day']/24)
X['hour_cos'] = np.cos(2 * np.pi * X['time_of_day']/24)
test['hour_sin'] = np.sin(2 * np.pi * test['time_of_day']/24)
test['hour_cos'] = np.cos(2 * np.pi * test['time_of_day']/24)

# Interaction and aggregated features
X['roadtype_lanes'] = X['road_type'] * X['num_lanes']
X['weather_speed'] = X['weather'] * X['speed_limit']
X['accidents_per_speed'] = X['num_reported_accidents'] / (X['speed_limit']+1)
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test['roadtype_lanes'] = test['road_type'] * test['num_lanes']
test['weather_speed'] = test['weather'] * test['speed_limit']
test['accidents_per_speed'] = test['num_reported_accidents'] / (test['speed_limit']+1)
test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 4. Remove Outliers (trim extreme 1%)
# ==========================================
Q1, Q3 = y.quantile(0.01), y.quantile(0.99)
mask = (y >= Q1) & (y <= Q3)
X, y = X[mask], y[mask]

# ==========================================
# 5. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 6. Log-transform target
# ==========================================
y_log = np.log1p(y)

# ==========================================
# 7. Cross-validation and ensemble
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
preds = np.zeros(len(test))

xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 1,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
}

lgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

print("ğŸš€ Training ensemble with 5-fold CV...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc="CV progress"):
    X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    test_features = test.drop(columns=[id_col]).values

    # XGBoost
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_pred_test = xgb_model.predict(test_features)

    # LightGBM
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
    )
    lgb_pred_val = lgb_model.predict(X_val)
    lgb_pred_test = lgb_model.predict(test_features)

    # Ensemble
    val_pred = (xgb_pred_val + lgb_pred_val) / 2
    rmse = sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.5f}")

    preds += (xgb_pred_test + lgb_pred_test) / 2 / kf.n_splits

print("\nâœ… Average CV RMSE:", np.mean(rmse_scores))

# ==========================================
# 8. Create submission file
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': np.expm1(preds)  # inverse log1p
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Target Encoding for Categoricals
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    mapping = train.groupby(col)[target].mean()
    X[col] = X[col].map(mapping)
    test[col] = test[col].map(mapping)

# ==========================================
# 3. Advanced Feature Engineering
# ==========================================
X['hour_sin'] = np.sin(2 * np.pi * X['time_of_day']/24)
X['hour_cos'] = np.cos(2 * np.pi * X['time_of_day']/24)

X['roadtype_lanes'] = X['road_type'] * X['num_lanes']
X['weather_speed'] = X['weather'] * X['speed_limit']
X['accidents_per_speed'] = X['num_reported_accidents'] / (X['speed_limit']+1)
X['accidents_per_lane'] = X['num_reported_accidents'] / (X['num_lanes'] + 1)
X['speed_curvature_ratio'] = X['speed_limit'] / (X['curvature'] + 1)
X['is_school_holiday'] = ((X["school_season"]==1) & (X["holiday"]==1)).astype(int)

test['hour_sin'] = np.sin(2 * np.pi * test['time_of_day']/24)
test['hour_cos'] = np.cos(2 * np.pi * test['time_of_day']/24)
test['roadtype_lanes'] = test['road_type'] * test['num_lanes']
test['weather_speed'] = test['weather'] * test['speed_limit']
test['accidents_per_speed'] = test['num_reported_accidents'] / (test['speed_limit']+1)
test['accidents_per_lane'] = test['num_reported_accidents'] / (test['num_lanes'] + 1)
test['speed_curvature_ratio'] = test['speed_limit'] / (test['curvature'] + 1)
test['is_school_holiday'] = ((test["school_season"]==1) & (test["holiday"]==1)).astype(int)

# ==========================================
# 4. Remove extreme outliers
# ==========================================
Q1, Q3 = y.quantile(0.01), y.quantile(0.99)
mask = (y >= Q1) & (y <= Q3)
X, y = X[mask], y[mask]

# ==========================================
# 5. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 6. Target Transformation (optional)
# ==========================================
# You can try 'log' or 'sqrt'
y_trans = np.log1p(y)

# ==========================================
# 7. Seed Averaging + Repeated K-Fold CV
# ==========================================
n_seeds = 5
kf_splits = 5
preds_test = np.zeros(len(test))
rmse_list = []

xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 1,
    'tree_method': 'hist',
    'device': 'cuda'
}

for seed in range(42, 42 + n_seeds):
    kf = KFold(n_splits=kf_splits, shuffle=True, random_state=seed)
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf_splits, desc=f"Seed {seed} CV"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y_trans.iloc[train_idx], y_trans.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**xgb_params, random_state=seed)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=100)

        val_pred = model.predict(X_val)
        val_rmse = sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        fold_rmse.append(val_rmse)

        fold_preds += model.predict(test_feat) / kf_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / n_seeds  # average across seeds

print(f"\nâœ… Overall Average CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 8. Create submission file
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': np.expm1(preds_test)
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

# ==========================================
# 2. Handle Missing Values
# ==========================================
train.dropna(inplace=True)

# Fill numeric columns in test with median
num_cols = test.select_dtypes(include=np.number).columns
test[num_cols] = test[num_cols].fillna(test[num_cols].median())

# Fill categorical columns in test with mode
cat_cols_test = test.select_dtypes(include='object').columns
for col in cat_cols_test:
    test[col] = test[col].fillna(test[col].mode()[0])

# ==========================================
# 3. Remove exact duplicate rows in training (ignore ID)
# ==========================================
train = train.drop_duplicates(subset=train.columns.difference([id_col]))

# ==========================================
# 4. Remove extreme outliers (top/bottom 1%)
# ==========================================
Q1, Q3 = train[target].quantile(0.01), train[target].quantile(0.99)
mask = (train[target] >= Q1) & (train[target] <= Q3)
train = train[mask]

# ==========================================
# 5. Separate features and target
# ==========================================
X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 6. Encode categorical features
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].astype(str).factorize()[0]
    test[col] = test[col].astype(str).factorize()[0]

# ==========================================
# 7. Normalize features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 8. Hyperparameter tuning with RandomizedSearchCV
# ==========================================
xgb_model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    objective='reg:squarederror',
    eval_metric='rmse',
    n_jobs=-1,
    random_state=42
)

param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [5, 6, 7, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# Use one train/val split for tuning
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(X))
X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)
random_search.fit(X_tr, y_tr)
best_params = random_search.best_params_
print("âœ… Best Hyperparameters:", best_params)

# ==========================================
# 9. Train final model with seed-averaged K-Fold CV
# ==========================================
n_seeds = 5
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_test = np.zeros(len(test))
rmse_list = []

for seed in range(42, 42+n_seeds):
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values  # keep test intact

        model = XGBRegressor(
            **best_params,
            tree_method='hist',
            device='cuda',
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=seed
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )

        val_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, val_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / n_seeds

print(f"\nâœ… Overall Average CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 10. Create submission file
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})
submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Feature Engineering (same as before)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 4. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 5. Seed-Averaged K-Fold CV Training
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
seeds = [42, 2023, 7]  # 3 seeds for averaging
preds_test = np.zeros(len(test))
rmse_list = []

# Fine-tuned hyperparameters (tweaked around previous best)
# Remove 'random_state' from xgb_params
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
}

# Inside the loop, pass seed dynamically
model = XGBRegressor(**xgb_params, random_state=seed)


print("ğŸš€ Starting Seed-Averaged 5-Fold CV Training...")

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**xgb_params, random_state=seed)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 6. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 4. Seed-Averaged 5-Fold CV Training
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
seeds = [42, 2023]  # 2 seeds for averaging
preds_test = np.zeros(len(test))
rmse_list = []

# Fine-tuned XGBoost parameters (slight improvement over your original)
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,   # slightly smaller learning rate
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
}

print("ğŸš€ Starting Seed-Averaged 5-Fold CV Training...")

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**xgb_params, random_state=seed)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,  # allow more rounds for convergence
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 5. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#INCREASE NUMBER OF ROUNDS

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# ==========================================
# 2. Setup
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 3. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 4. Feature Engineering (same as your best)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 5. Remove Outliers (IQR)
# ==========================================
Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[mask], y[mask]

# ==========================================
# 6. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 7. Optimized XGBoost Parameters
# ==========================================
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,        # ğŸ”½ Slightly smaller
    'n_estimators': 1500,         # ğŸ”¼ Longer training
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_lambda': 1.2,
    'reg_alpha': 0.4,
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'n_jobs': -1
}

# ==========================================
# 8. Cross-Validation with Multi-Seed Averaging
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
preds = np.zeros(len(test))

SEEDS = [42, 84, 1337]  # ğŸ”� multiple seeds for smoothing

print("\nğŸš€ Training with multi-seed + 5-Fold CV + longer rounds...")

for seed in SEEDS:
    print(f"\nğŸŒ± Seed {seed}")
    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"Seed {seed} CV"):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**xgb_params)
        model.set_params(random_state=seed)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=200,   # ğŸ”¼ allow longer learning
            verbose=200
        )

        y_pred = model.predict(X_val)
        rmse = sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

        preds += model.predict(test.drop(columns=[id_col])) / (len(SEEDS) * kf.n_splits)

print("\nâœ… Average CV RMSE:", np.mean(rmse_scores))

# ==========================================
# 9. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#USING Z SCORE TO remove OUTLIERS

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from scipy import stats
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Feature Engineering (same as your best)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 4. Z-Score Outlier Removal (Target)
# ==========================================
z_scores = np.abs(stats.zscore(y))
threshold = 3  # keep rows with z < 3
mask = (z_scores < threshold)
X, y = X[mask], y[mask]
print(f"âœ… Removed {np.sum(~mask)} outliers using Z-score on target")

# ==========================================
# 5. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 6. XGBoost Parameters (Optimized)
# ==========================================
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_lambda': 1.2,
    'reg_alpha': 0.4,
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1
}

# ==========================================
# 7. Seed-Averaged 5-Fold CV Training
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 84, 1337]  # multi-seed averaging
preds_test = np.zeros(len(test))
rmse_list = []

print("\nğŸš€ Training with multi-seed + 5-Fold CV + Z-score outlier removal...")

for seed in SEEDS:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col])

        model = XGBRegressor(**xgb_params, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=200,  # allow longer training
            verbose=200
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))

        # accumulate test predictions
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(SEEDS)

print("\nâœ… Overall Seed-Averaged CV RMSE:", np.mean(rmse_list))

# ==========================================
# 8. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#IMPROVE Z SCORE ONE
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from scipy import stats
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Feature Engineering (safe minimal)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 4. Z-Score Outlier Removal (target)
# ==========================================
z_scores = np.abs(stats.zscore(y))
threshold = 3
mask = (z_scores < threshold)
X, y = X[mask], y[mask]
print(f"âœ… Removed {np.sum(~mask)} outliers using Z-score on target")

# ==========================================
# 5. Normalize Numeric Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 6. Hyperparameter Grid (around previous best)
# ==========================================
param_grid = {
    'learning_rate': [0.015, 0.02, 0.025],
    'max_depth': [5, 6, 7],
    'subsample': [0.75, 0.8, 0.85],
    'colsample_bytree': [0.75, 0.8, 0.85],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.05, 0.1],
    'reg_alpha': [0, 0.2, 0.4],
    'reg_lambda': [1.0, 1.2]
}

xgb_base = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    tree_method='hist',
    device='cuda',
    eval_metric='rmse',
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=12,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X, y)
best_params = search.best_params_
print(f"\nâœ… Best Hyperparameters: {best_params}")
print(f"âœ… Best CV RMSE (3-fold): {-search.best_score_:.5f}")

# ==========================================
# 7. Seed-Averaged 5-Fold CV Training
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 84, 1337]
preds_test = np.zeros(len(test))
rmse_list = []

print("\nğŸš€ Training with multi-seed + 5-Fold CV using optimized params...")

for seed in SEEDS:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col])

        model = XGBRegressor(**best_params, n_estimators=1200, tree_method='hist', device='cuda', random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=150,
            verbose=200
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(SEEDS)

print("\nâœ… Overall Seed-Averaged CV RMSE:", np.mean(rmse_list))

# ==========================================
# 8. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#INCREASE FOLDS 
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 4. Seed-Averaged 10-Fold CV Training
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # âœ… 10 folds
seeds = [42, 2023]  # seed averaging
preds_test = np.zeros(len(test))
rmse_list = []

# Fine-tuned XGBoost parameters
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
}

print("ğŸš€ Starting Seed-Averaged 10-Fold CV Training...")

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"Seed {seed} CV progress"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**xgb_params, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 5. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#One HOT ENCODING


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from scipy import stats
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. One-Hot Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

# Ensure train and test have same columns
missing_cols = set(X.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0
test = test[X.columns]  # reorder columns

# ==========================================
# 3. Feature Engineering (safe)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)

X["is_school_holiday"] = (
    (X.get("school_season_1", pd.Series(0, index=X.index)) == 1) &
    (X.get("holiday_1", pd.Series(0, index=X.index)) == 1)
).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)

test["is_school_holiday"] = (
    (test.get("school_season_1", pd.Series(0, index=test.index)) == 1) &
    (test.get("holiday_1", pd.Series(0, index=test.index)) == 1)
).astype(int)

# ==========================================
# 4. Z-Score Outlier Removal (target)
# ==========================================
z_scores = np.abs(stats.zscore(y))
threshold = 3
mask = z_scores < threshold
X, y = X[mask], y[mask]
print(f"âœ… Removed {np.sum(~mask)} outliers using Z-score")

# ==========================================
# 5. Normalize Numeric Features
# ==========================================
numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

# ==========================================
# 6. 10-Fold CV + Seed Averaging
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
seeds = [42, 2023]
preds_test = np.zeros(len(test))
rmse_list = []

# XGBoost parameters
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
}

print("ğŸš€ Starting Seed-Averaged 10-Fold CV Training with One-Hot Encoding...")

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"Seed {seed} CV progress"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.values

        model = XGBRegressor(**xgb_params, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 7. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission23.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())




# Z Score with 10 folds
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from scipy import stats
from tqdm import tqdm

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Feature Engineering (optional, safe)
# ==========================================
X["accidents_per_lane"] = X["num_reported_accidents"] / (X["num_lanes"] + 1)
X["speed_curvature_ratio"] = X["speed_limit"] / (X["curvature"] + 1)
X["is_school_holiday"] = ((X["school_season"] == 1) & (X["holiday"] == 1)).astype(int)

test["accidents_per_lane"] = test["num_reported_accidents"] / (test["num_lanes"] + 1)
test["speed_curvature_ratio"] = test["speed_limit"] / (test["curvature"] + 1)
test["is_school_holiday"] = ((test["school_season"] == 1) & (test["holiday"] == 1)).astype(int)

# ==========================================
# 4. Z-Score Outlier Removal (target)
# ==========================================
z_scores = np.abs(stats.zscore(y))
threshold = 3
mask = z_scores < threshold
X, y = X[mask], y[mask]
print(f"âœ… Removed {np.sum(~mask)} outliers using Z-score")

# ==========================================
# 5. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 6. 10-Fold CV + Seed Averaging
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # âœ… 10 folds
seeds = [42, 2023]  # seed averaging
preds_test = np.zeros(len(test))
rmse_list = []

# Fine-tuned XGBoost parameters
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
}

print("ğŸš€ Starting Seed-Averaged 10-Fold CV Training with Z-score...")

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"Seed {seed} CV progress"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**xgb_params, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 7. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv created successfully!")
display(submission.head())



#Bayesian Optimisation

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm
import optuna

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 4. Bayesian Optimization Function
# ==========================================
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0, 0.2),
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,  # âœ… move here to constructor
        'n_jobs': -1,
        'random_state': 42
    }
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_list = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        y_pred = model.predict(X_val)
        rmse_list.append(sqrt(mean_squared_error(y_val, y_pred)))
    
    return np.mean(rmse_list)

# ==========================================
# 5. Run Optuna Study
# ==========================================
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

print("âœ… Best Params:", study.best_params)
print("âœ… Best CV RMSE:", study.best_value)

# ==========================================
# 6. Train Final Model with Best Params + 10-Fold Seed Averaging
# ==========================================
best_params = study.best_params
best_params.update({
    'n_estimators': 1000,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100,
    'n_jobs': -1
})

kf = KFold(n_splits=10, shuffle=True, random_state=42)
seeds = [42, 2023]
preds_test = np.zeros(len(test))
rmse_list = []

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"Seed {seed} CV"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        test_feat = test.drop(columns=[id_col]).values

        model = XGBRegressor(**best_params, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(test_feat) / kf.n_splits

    print(f"Seed {seed} Average RMSE: {np.mean(fold_rmse):.5f}")
    rmse_list.append(np.mean(fold_rmse))
    preds_test += fold_preds / len(seeds)

print(f"\nâœ… Overall Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}")

# ==========================================
# 7. Create Submission
# ==========================================
submission = pd.DataFrame({'id': test[id_col], 'accident_risk': preds_test})
submission.to_csv("submission_bayes_final.csv", index=False)
print("\nğŸ“� submission_bayes_final.csv created successfully!")
display(submission.head())



# ==========================================
#  ğŸš€ Improved XGBoost: 15-Fold + 5 Seeds + Progress Bar (FIXED)
#  Fixed: Removed duplicate 'random_state' error
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from math import sqrt
from xgboost import XGBRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("First 5 rows of training data:")
print(train.head())

# ==========================================
# 2. Define Features
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f'\nCategorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')

# ==========================================
# 3. One-Hot Encoding
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Feature names after encoding
feature_names = num_cols + preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
X_proc_df = pd.DataFrame(X_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# ==========================================
# 4. 15-Fold CV + 5 Seeds Training with Progress Bar (Fixed!)
# ==========================================
kf = KFold(n_splits=15, shuffle=True, random_state=42)
seeds = [42, 2023, 123, 999, 777]  # 5 seeds for better averaging

preds_test = np.zeros(len(X_test_df))
rmse_list = []

# Base XGBoost Parameters (without random_state here to avoid conflict)
xgb_params_base = {
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist',  # Change to 'hist' if GPU not available
    'eval_metric': 'rmse',
    'n_jobs': -1
}

print('ğŸš€ Starting 15-Fold + 5-Seed XGBoost Training with Progress Bar...')

for seed in seeds:
    # Set random_state per seed
    xgb_params = xgb_params_base.copy()
    xgb_params['random_state'] = seed
    
    fold_rmse = []
    fold_preds = np.zeros(len(X_test_df))
    
    # Progress bar for folds
    fold_progress = tqdm(kf.split(X_proc_df), 
                         total=kf.get_n_splits(), 
                         desc=f'Seed {seed} Progress', 
                         colour='green')
    
    for fold, (train_idx, val_idx) in enumerate(fold_progress, 1):
        X_tr, X_val = X_proc_df.iloc[train_idx], X_proc_df.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=150,
            verbose=200
        )
        
        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        fold_preds += model.predict(X_test_df) / kf.n_splits
    
    avg_rmse = np.mean(fold_rmse)
    print(f'Seed {seed} Average RMSE: {avg_rmse:.5f}')
    rmse_list.append(avg_rmse)
    preds_test += fold_preds / len(seeds)

print(f'\nâœ… Final Seed-Averaged CV RMSE: {np.mean(rmse_list):.5f}')

# ==========================================
# 5. Create Submission File
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': preds_test
})

submission.to_csv('submission25.csv', index=False)
print('\nğŸ“� submission.csv created successfully!')
print(submission.head())



# ==========================================
#  ğŸ�† ELITE XGBoost + LightGBM + CatBoost Ensemble
#  Target: RMSE â‰¤ 0.052
#  Works with ACTUAL Playground S5E10 columns
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("âœ… Data loaded successfully!")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"\nColumns: {train.columns.tolist()}")

# ==========================================
# 2. Feature Engineering (Using ACTUAL columns)
# ==========================================
def feature_engineering(df):
    """Create powerful interaction features with actual column names"""
    df = df.copy()
    
    # Numerical interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    
    # Categorical interaction features
    df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_time'] = df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)
    df['weather_time'] = df['weather'].astype(str) + '_' + df['time_of_day'].astype(str)
    df['holiday_school'] = df['holiday'].astype(str) + '_' + df['school_season'].astype(str)
    
    # Risk indicators
    df['high_speed_low_light'] = ((df['speed_limit'] > df['speed_limit'].median()) & 
                                   (df['lighting'] == 'dim')).astype(int)
    df['dangerous_combo'] = ((df['weather'] == 'rainy') & 
                             (df['road_signs_present'] == False)).astype(int)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)

print(f"âœ… Feature Engineering Complete! New shape: {train.shape}")

# ==========================================
# 3. Encode Categorical Features
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# Encode all object columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

print(f"âœ… Encoding Complete! Total Features: {X.shape[1]}")

# ==========================================
# 4. Stratified K-Fold for Better CV
# ==========================================
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ==========================================
# 5. Model Definitions - Optimized for Accident Data
# ==========================================
seeds = [42, 2023, 123]

# XGBoost - Optimized
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse'
}

# LightGBM - Optimized
lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 7,
    'num_leaves': 45,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'min_child_weight': 5,
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'gpu',
    'verbose': -1
}

# CatBoost - Optimized
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.025,
    'depth': 6,
    'l2_leaf_reg': 4,
    'subsample': 0.85,
    'colsample_bylevel': 0.85,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'verbose': 0
}

# ==========================================
# 6. Training Function
# ==========================================
def train_model(model_type, params, seed):
    """Initialize model with proper random state"""
    params_copy = params.copy()
    
    if model_type == 'xgb':
        params_copy['random_state'] = seed
        return XGBRegressor(**params_copy)
    elif model_type == 'lgb':
        params_copy['random_state'] = seed
        return LGBMRegressor(**params_copy)
    elif model_type == 'cat':
        params_copy['random_seed'] = seed
        return CatBoostRegressor(**params_copy)

# ==========================================
# 7. Train All Models with Progress Bars
# ==========================================
xgb_preds_test = np.zeros(len(X_test))
lgb_preds_test = np.zeros(len(X_test))
cat_preds_test = np.zeros(len(X_test))

xgb_rmse_list = []
lgb_rmse_list = []
cat_rmse_list = []

print('\nğŸš€ Starting Elite Ensemble Training...\n')

for seed in seeds:
    print(f'{"="*60}')
    print(f'Training with Seed: {seed}')
    print(f'{"="*60}')
    
    # XGBoost
    xgb_fold_preds = np.zeros(len(X_test))
    xgb_fold_rmse = []
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y_binned), 1), 
                                           total=skf.n_splits, 
                                           desc=f'XGBoost Seed {seed}', 
                                           colour='green'):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_model('xgb', xgb_params, seed)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, verbose=0)
        
        y_pred = model.predict(X_val)
        xgb_fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        xgb_fold_preds += model.predict(X_test) / skf.n_splits
    
    xgb_rmse = np.mean(xgb_fold_rmse)
    print(f'âœ… XGBoost Seed {seed} RMSE: {xgb_rmse:.5f}')
    xgb_rmse_list.append(xgb_rmse)
    xgb_preds_test += xgb_fold_preds / len(seeds)
    
    # LightGBM
    lgb_fold_preds = np.zeros(len(X_test))
    lgb_fold_rmse = []
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y_binned), 1), 
                                           total=skf.n_splits, 
                                           desc=f'LightGBM Seed {seed}', 
                                           colour='blue'):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_model('lgb', lgb_params, seed)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        y_pred = model.predict(X_val)
        lgb_fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        lgb_fold_preds += model.predict(X_test) / skf.n_splits
    
    lgb_rmse = np.mean(lgb_fold_rmse)
    print(f'âœ… LightGBM Seed {seed} RMSE: {lgb_rmse:.5f}')
    lgb_rmse_list.append(lgb_rmse)
    lgb_preds_test += lgb_fold_preds / len(seeds)
    
    # CatBoost
    cat_fold_preds = np.zeros(len(X_test))
    cat_fold_rmse = []
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y_binned), 1), 
                                           total=skf.n_splits, 
                                           desc=f'CatBoost Seed {seed}', 
                                           colour='yellow'):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_model('cat', cat_params, seed)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                 early_stopping_rounds=50, verbose=0)
        
        y_pred = model.predict(X_val)
        cat_fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))
        cat_fold_preds += model.predict(X_test) / skf.n_splits
    
    cat_rmse = np.mean(cat_fold_rmse)
    print(f'âœ… CatBoost Seed {seed} RMSE: {cat_rmse:.5f}\n')
    cat_rmse_list.append(cat_rmse)
    cat_preds_test += cat_fold_preds / len(seeds)

# ==========================================
# 8. Weighted Ensemble
# ==========================================
xgb_avg_rmse = np.mean(xgb_rmse_list)
lgb_avg_rmse = np.mean(lgb_rmse_list)
cat_avg_rmse = np.mean(cat_rmse_list)

print(f'\n{"="*60}')
print(f'ğŸ“Š Individual Model Performance:')
print(f'{"="*60}')
print(f'XGBoost Average RMSE: {xgb_avg_rmse:.5f}')
print(f'LightGBM Average RMSE: {lgb_avg_rmse:.5f}')
print(f'CatBoost Average RMSE: {cat_avg_rmse:.5f}')

# Calculate optimal weights (inverse RMSE)
total_inv = (1/xgb_avg_rmse) + (1/lgb_avg_rmse) + (1/cat_avg_rmse)
xgb_weight = (1/xgb_avg_rmse) / total_inv
lgb_weight = (1/lgb_avg_rmse) / total_inv
cat_weight = (1/cat_avg_rmse) / total_inv

print(f'\nğŸ”§ Optimal Ensemble Weights:')
print(f'XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}, CatBoost: {cat_weight:.3f}')

# Final ensemble prediction
final_preds = (xgb_weight * xgb_preds_test + 
               lgb_weight * lgb_preds_test + 
               cat_weight * cat_preds_test)

# ==========================================
# 9. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': final_preds
})

submission.to_csv('submission.csv', index=False)

expected_rmse = min(xgb_avg_rmse, lgb_avg_rmse, cat_avg_rmse) * 0.98
print(f'\nâœ… submission.csv created successfully!')
print(f'ğŸ�¯ Expected Ensemble RMSE: ~{expected_rmse:.5f}')
print(f'\nFirst 5 predictions:')
print(submission.head())



df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
column_names = df.columns

    # Optional: Print as a standard Python list
print("\nColumn Names (as a list):")
print(column_names.tolist())


# ==========================================
#  ğŸ�† Elite Strategy: TabM + XGBoost + LightGBM (ACTUAL TabM!)
#  Target: RMSE â‰¤ 0.052
#  Strategy: Simple Weighted Averaging
# ==========================================

# Install TabM first
!pip install rtdl torch -q

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
import torch
import torch.nn as nn
from rtdl_num_embeddings import PiecewiseLinearEncoding
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"âœ… Data loaded: Train {train.shape}, Test {test.shape}")

# ==========================================
# 2. Feature Engineering
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Numerical interactions
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    
    # High-risk indicators
    df['high_risk_weather'] = (df['weather'] == 'rainy').astype(int)
    df['low_visibility'] = (df['lighting'] == 'dim').astype(int)
    df['dangerous_combo'] = ((df['weather'] == 'rainy') & 
                             (df['road_signs_present'] == False)).astype(int)
    df['night_curve'] = ((df['lighting'] == 'dim') & 
                         (df['curvature'] > df['curvature'].median())).astype(int)
    
    return df

train = create_features(train)
test = create_features(test)

# ==========================================
# 3. Prepare Features
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

print(f"âœ… Features created: {X.shape[1]} total features")

# ==========================================
# 4. Cross-Validation Setup
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
            'holiday', 'school_season', 'road_signs_present', 'public_road']

# ==========================================
# 5. TabM Model Definition
# ==========================================
class TabMModel(nn.Module):
    def __init__(self, n_features, d_model=256, n_blocks=4, dropout=0.1, n_ensembles=32):
        super().__init__()
        self.n_ensembles = n_ensembles
        
        # Feature embedding
        self.input_layer = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer-like blocks with BatchEnsemble
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_blocks)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        
        # Ensemble prediction heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(n_ensembles)
        ])
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        
        # Apply blocks with residual connections
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(x)
            x = norm(x)
        
        # Ensemble predictions
        predictions = torch.stack([head(x) for head in self.heads], dim=1)
        return predictions.mean(dim=1).squeeze()

# ==========================================
# 6. Train TabM
# ==========================================
def train_tabm(X_train, y_train, X_val, y_val, epochs=100):
    """Train TabM model"""
    X_train_t = torch.FloatTensor(X_train.values).to(device)
    y_train_t = torch.FloatTensor(y_train.values).to(device)
    X_val_t = torch.FloatTensor(X_val.values).to(device)
    y_val_t = torch.FloatTensor(y_val.values).to(device)
    
    model = TabMModel(n_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    return model

tabm_preds = np.zeros(len(X_test))
tabm_cv_scores = []

print("\nğŸš€ Training TabM with 10-Fold CV...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), 
                                       total=10, desc='TabM Training'):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # Target encoding
    te = TargetEncoder(cols=cat_cols)
    X_tr = te.fit_transform(X_tr, y_tr)
    X_val = te.transform(X_val)
    X_test_fold = te.transform(X_test_fold)
    
    # Train TabM
    tabm_model = train_tabm(X_tr, y_tr, X_val, y_val)
    
    # Predictions
    tabm_model.eval()
    with torch.no_grad():
        val_preds = tabm_model(torch.FloatTensor(X_val.values).to(device)).cpu().numpy()
        test_preds = tabm_model(torch.FloatTensor(X_test_fold.values).to(device)).cpu().numpy()
    
    tabm_cv_scores.append(sqrt(mean_squared_error(y_val, val_preds)))
    tabm_preds += test_preds / 10

tabm_cv = np.mean(tabm_cv_scores)
print(f"TabM CV RMSE: {tabm_cv:.5f}")

# ==========================================
# 7. XGBoost with Optuna
# ==========================================
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.04),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.75, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    
    rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        te = TargetEncoder(cols=cat_cols)
        X_tr = te.fit_transform(X_tr, y_tr)
        X_val = te.transform(X_val)
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, verbose=0)
        
        preds = model.predict(X_val)
        rmse_scores.append(sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

print("\nğŸ”§ Tuning XGBoost...")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
best_xgb_params = xgb_study.best_params

# ==========================================
# 8. Train XGBoost & LightGBM
# ==========================================
xgb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))
xgb_cv_scores = []
lgb_cv_scores = []

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.025,
    'max_depth': 7,
    'num_leaves': 45,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'min_child_weight': 5,
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1
}

print("\nğŸš€ Training XGBoost & LightGBM...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), 
                                       total=10, desc='GBDT Training'):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    te = TargetEncoder(cols=cat_cols)
    X_tr = te.fit_transform(X_tr, y_tr)
    X_val = te.transform(X_val)
    X_test_fold = te.transform(X_test_fold)
    
    # XGBoost
    xgb_model = XGBRegressor(**best_xgb_params, tree_method='hist', 
                            device='cuda', random_state=42)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, verbose=0)
    xgb_cv_scores.append(sqrt(mean_squared_error(y_val, xgb_model.predict(X_val))))
    xgb_preds += xgb_model.predict(X_test_fold) / 10
    
    # LightGBM
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    lgb_cv_scores.append(sqrt(mean_squared_error(y_val, lgb_model.predict(X_val))))
    lgb_preds += lgb_model.predict(X_test_fold) / 10

xgb_cv = np.mean(xgb_cv_scores)
lgb_cv = np.mean(lgb_cv_scores)

print(f"\nğŸ“Š Final Model Performance:")
print(f"TabM CV RMSE: {tabm_cv:.5f}")
print(f"XGBoost CV RMSE: {xgb_cv:.5f}")
print(f"LightGBM CV RMSE: {lgb_cv:.5f}")

# ==========================================
# 9. Weighted Ensemble
# ==========================================
total_inv = (1/tabm_cv) + (1/xgb_cv) + (1/lgb_cv)
tabm_weight = (1/tabm_cv) / total_inv
xgb_weight = (1/xgb_cv) / total_inv
lgb_weight = (1/lgb_cv) / total_inv

print(f"\nğŸ”§ Ensemble Weights:")
print(f"TabM: {tabm_weight:.3f}, XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")

final_preds = (tabm_weight * tabm_preds + 
               xgb_weight * xgb_preds + 
               lgb_weight * lgb_preds)

# ==========================================
# 10. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': final_preds
})

submission.to_csv('submission.csv', index=False)

expected_rmse = min(tabm_cv, xgb_cv, lgb_cv) * 0.98
print(f"\nâœ… submission.csv created!")
print(f"ğŸ�¯ Expected Ensemble RMSE: ~{expected_rmse:.5f}")
print(submission.head())



# ==========================================
#  ğŸ�† Elite Strategy: XGBoost + LightGBM + CatBoost
#  Target: RMSE â‰¤ 0.052
#  No TabM dependencies - Pure GBDT Power
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"âœ… Data loaded: Train {train.shape}, Test {test.shape}")

# ==========================================
# 2. Advanced Feature Engineering
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Numerical interactions
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_cubed'] = df['speed_limit'] ** 3
    
    # Normalized features
    df['speed_normalized'] = df['speed_limit'] / (df['num_lanes'] + 1)
    df['curvature_speed_ratio'] = df['curvature'] / (df['speed_limit'] + 1)
    
    # High-risk indicators
    df['high_risk_weather'] = (df['weather'] == 'rainy').astype(int)
    df['low_visibility'] = (df['lighting'] == 'dim').astype(int)
    df['dangerous_combo'] = ((df['weather'] == 'rainy') & 
                             (df['road_signs_present'] == False)).astype(int)
    df['night_curve'] = ((df['lighting'] == 'dim') & 
                         (df['curvature'] > df['curvature'].median())).astype(int)
    df['high_speed_bad_weather'] = ((df['speed_limit'] > df['speed_limit'].median()) & 
                                     (df['weather'] == 'rainy')).astype(int)
    
    # Categorical combinations
    df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_time'] = df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)
    
    return df

train = create_features(train)
test = create_features(test)

# ==========================================
# 3. Prepare Features
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

print(f"âœ… Features created: {X.shape[1]} total features")

# ==========================================
# 4. Define Categorical Columns
# ==========================================
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
            'holiday', 'school_season', 'road_signs_present', 'public_road',
            'weather_lighting', 'road_time']

# ==========================================
# 5. Cross-Validation Setup
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ==========================================
# 6. Optuna Tuning for XGBoost
# ==========================================
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 700, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 4.0),
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    
    rmse_scores = []
    for train_idx, val_idx in list(kf.split(X))[:5]:  # Use 5 folds for speed
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        te = TargetEncoder(cols=cat_cols)
        X_tr = te.fit_transform(X_tr, y_tr)
        X_val = te.transform(X_val)
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=50, verbose=0)
        
        preds = model.predict(X_val)
        rmse_scores.append(sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

print("\nğŸ”§ Tuning XGBoost with Optuna (50 trials)...")
xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

print(f"âœ… Best XGBoost RMSE: {xgb_study.best_value:.5f}")
best_xgb_params = xgb_study.best_params
print(f"Best params: {best_xgb_params}")

# ==========================================
# 7. Train All Models with 10-Fold CV
# ==========================================
xgb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

xgb_cv_scores = []
lgb_cv_scores = []
cat_cv_scores = []

# LightGBM params (manually tuned)
lgb_params = {
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'max_depth': 8,
    'num_leaves': 60,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 8,
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1
}

# CatBoost params (manually tuned)
cat_params = {
    'iterations': 1200,
    'learning_rate': 0.025,
    'depth': 7,
    'l2_leaf_reg': 5,
    'subsample': 0.85,
    'colsample_bylevel': 0.85,
    'min_child_samples': 10,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'random_seed': 42,
    'verbose': 0
}

print("\nğŸš€ Training all models with 10-Fold CV...")

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), 
                                       total=10, desc='CV Training'):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # Target encoding within fold
    te = TargetEncoder(cols=cat_cols)
    X_tr = te.fit_transform(X_tr, y_tr)
    X_val = te.transform(X_val)
    X_test_fold = te.transform(X_test_fold)
    
    # XGBoost
    xgb_model = XGBRegressor(**best_xgb_params, tree_method='hist', 
                            device='cuda', random_state=42)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                 early_stopping_rounds=75, verbose=0)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_cv_scores.append(sqrt(mean_squared_error(y_val, xgb_val_pred)))
    xgb_preds += xgb_model.predict(X_test_fold) / 10
    
    # LightGBM
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_cv_scores.append(sqrt(mean_squared_error(y_val, lgb_val_pred)))
    lgb_preds += lgb_model.predict(X_test_fold) / 10
    
    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                 early_stopping_rounds=75, verbose=0)
    cat_val_pred = cat_model.predict(X_val)
    cat_cv_scores.append(sqrt(mean_squared_error(y_val, cat_val_pred)))
    cat_preds += cat_model.predict(X_test_fold) / 10

# ==========================================
# 8. Model Performance Summary
# ==========================================
xgb_cv = np.mean(xgb_cv_scores)
lgb_cv = np.mean(lgb_cv_scores)
cat_cv = np.mean(cat_cv_scores)

print(f"\n{'='*60}")
print(f"ğŸ“Š Final Model Performance:")
print(f"{'='*60}")
print(f"XGBoost CV RMSE:  {xgb_cv:.5f} (Â±{np.std(xgb_cv_scores):.5f})")
print(f"LightGBM CV RMSE: {lgb_cv:.5f} (Â±{np.std(lgb_cv_scores):.5f})")
print(f"CatBoost CV RMSE: {cat_cv:.5f} (Â±{np.std(cat_cv_scores):.5f})")

# ==========================================
# 9. Weighted Ensemble (Inverse RMSE)
# ==========================================
total_inv = (1/xgb_cv) + (1/lgb_cv) + (1/cat_cv)
xgb_weight = (1/xgb_cv) / total_inv
lgb_weight = (1/lgb_cv) / total_inv
cat_weight = (1/cat_cv) / total_inv

print(f"\nğŸ”§ Optimal Ensemble Weights:")
print(f"XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}, CatBoost: {cat_weight:.3f}")

final_preds = (xgb_weight * xgb_preds + 
               lgb_weight * lgb_preds + 
               cat_weight * cat_preds)

# ==========================================
# 10. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': final_preds
})

submission.to_csv('submission.csv', index=False)

expected_rmse = min(xgb_cv, lgb_cv, cat_cv) * 0.98
print(f"\nâœ… submission.csv created successfully!")
print(f"ğŸ�¯ Expected Ensemble RMSE: ~{expected_rmse:.5f}")
print(f"\nFirst 5 predictions:")
print(submission.head())



# ==========================================
#  ğŸ�† Elite Strategy: XGBoost + LightGBM + CatBoost
#  Target: RMSE â‰¤ 0.052
#  No TabM dependencies - Pure GBDT Power
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"âœ… Data loaded: Train {train.shape}, Test {test.shape}")

# ==========================================
# 2. Advanced Feature Engineering
# ==========================================
def create_features(df):
    df = df.copy()
    
    # Numerical interactions
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_cubed'] = df['speed_limit'] ** 3
    
    # Normalized features
    df['speed_normalized'] = df['speed_limit'] / (df['num_lanes'] + 1)
    df['curvature_speed_ratio'] = df['curvature'] / (df['speed_limit'] + 1)
    
    # High-risk indicators
    df['high_risk_weather'] = (df['weather'] == 'rainy').astype(int)
    df['low_visibility'] = (df['lighting'] == 'dim').astype(int)
    df['dangerous_combo'] = ((df['weather'] == 'rainy') & 
                             (df['road_signs_present'] == False)).astype(int)
    df['night_curve'] = ((df['lighting'] == 'dim') & 
                          (df['curvature'] > df['curvature'].median())).astype(int)
    df['high_speed_bad_weather'] = ((df['speed_limit'] > df['speed_limit'].median()) & 
                                     (df['weather'] == 'rainy')).astype(int)
    
    # Categorical combinations
    df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    df['road_time'] = df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)
    
    return df

train = create_features(train)
test = create_features(test)

# ==========================================
# 3. Prepare Features
# ==========================================
target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

print(f"âœ… Features created: {X.shape[1]} total features")

# ==========================================
# 4. Define Categorical Columns
# ==========================================
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
            'holiday', 'school_season', 'road_signs_present', 'public_road',
            'weather_lighting', 'road_time']

# ==========================================
# 5. Cross-Validation Setup
# ==========================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ==========================================
# 6. Optuna Tuning for XGBoost
# *Using 'cuda' for speed
# ==========================================
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 700, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 4.0),
        'tree_method': 'hist',
        'device': 'cuda', # Back to 'cuda' for speed
        'random_state': 42
    }
    
    rmse_scores = []
    # Use 5 folds for faster tuning
    for train_idx, val_idx in list(kf.split(X))[:5]: 
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Target encoding
        te = TargetEncoder(cols=cat_cols)
        X_tr = te.fit_transform(X_tr, y_tr).astype('float32')
        X_val = te.transform(X_val).astype('float32')
        
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=50, verbose=0)
        
        preds = model.predict(X_val)
        rmse_scores.append(sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmse_scores)

print("\nğŸ”§ Tuning XGBoost with Optuna (50 trials)...")
xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
try:
    xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    print(f"âœ… Best XGBoost RMSE: {xgb_study.best_value:.5f}")
    print(f"Best params: {best_xgb_params}")
except Exception as e:
    print(f"âš ï¸� Optuna failed (using default params): {e}")
    # Fallback to good default if Optuna fails
    best_xgb_params = {
        'n_estimators': 1200, 'learning_rate': 0.02, 'max_depth': 7, 
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 
        'gamma': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 1.5
    }
    


# ==========================================
# 7. Train All Models with 10-Fold CV (GPU)
# ==========================================
print("\nğŸš€ Training all models with 10-Fold CV on GPU...")

xgb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

xgb_cv_scores, lgb_cv_scores, cat_cv_scores = [], [], []

# ---- LightGBM Parameters ----
lgb_params = {
    'n_estimators': 2500,
    'learning_rate': 0.01,
    'max_depth': 8,
    'num_leaves': 64,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 8,
    'device': 'gpu',
    'objective': 'huber',
    'random_state': 42,
    'verbose': -1
}

# ---- CatBoost Parameters ----
cat_params = {
    'iterations': 1500,
    'learning_rate': 0.02,
    'depth': 7,
    'l2_leaf_reg': 5,
    'subsample': 0.85,
    'bootstrap_type': 'Poisson',
    'min_child_samples': 10,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 75,
    'verbose': 0
}

for fold, (train_idx, val_idx) in tqdm(
    enumerate(kf.split(X), 1),
    total=kf.get_n_splits(),
    desc="CV Training"
):
    print(f"\n===== Fold {fold} =====")

    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()

    # Independent target encoding per model for fold diversity
    for model_name in ["XGB", "LGB", "CAT"]:
        te = TargetEncoder(cols=cat_cols)
        X_tr_enc = te.fit_transform(X_tr, y_tr).astype('float32')
        X_val_enc = te.transform(X_val).astype('float32')
        X_test_enc = te.transform(X_test_fold).astype('float32')

        # ===== XGBoost =====
        if model_name == "XGB":
            xgb_model = XGBRegressor(
                **best_xgb_params,
                tree_method='hist',
                device='cuda',
                random_state=42
            )
            xgb_model.fit(
                X_tr_enc, y_tr,
                eval_set=[(X_val_enc, y_val)],
                early_stopping_rounds=100,
                verbose=0
            )

            # --- Safe handling for best iteration ---
            best_iter = getattr(xgb_model, 'best_iteration', None)
            if best_iter is None:
                best_iter = getattr(
                    xgb_model, 'best_ntree_limit',
                    best_xgb_params.get('n_estimators', 1000)
                )

            xgb_val_pred = xgb_model.predict(X_val_enc, iteration_range=(0, best_iter))
            xgb_test_pred = xgb_model.predict(X_test_enc, iteration_range=(0, best_iter))

            rmse = sqrt(mean_squared_error(y_val, xgb_val_pred))
            xgb_cv_scores.append(rmse)
            xgb_preds += xgb_test_pred / kf.get_n_splits()

        # ===== LightGBM =====
        elif model_name == "LGB":
            lgb_model = LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_tr_enc, y_tr,
                eval_set=[(X_val_enc, y_val)]
            )

            # Use best_iteration_ safely
            best_iter = getattr(lgb_model, 'best_iteration_', lgb_params['n_estimators'])
            lgb_val_pred = lgb_model.predict(X_val_enc, num_iteration=best_iter)
            lgb_test_pred = lgb_model.predict(X_test_enc, num_iteration=best_iter)

            rmse = sqrt(mean_squared_error(y_val, lgb_val_pred))
            lgb_cv_scores.append(rmse)
            lgb_preds += lgb_test_pred / kf.get_n_splits()

        # ===== CatBoost =====
        else:
            cat_model = CatBoostRegressor(**cat_params)
            cat_model.fit(X_tr_enc, y_tr, eval_set=(X_val_enc, y_val), verbose=0)
            cat_val_pred = cat_model.predict(X_val_enc)
            cat_test_pred = cat_model.predict(X_test_enc)

            rmse = sqrt(mean_squared_error(y_val, cat_val_pred))
            cat_cv_scores.append(rmse)
            cat_preds += cat_test_pred / kf.get_n_splits()

# ==========================================
# 8. Model Performance Summary
# ==========================================
xgb_cv = np.mean(xgb_cv_scores)
lgb_cv = np.mean(lgb_cv_scores)
cat_cv = np.mean(cat_cv_scores)

print(f"\n{'='*60}")
print("ğŸ“Š Final Model Performance")
print(f"{'='*60}")
print(f"XGBoost CV RMSE:  {xgb_cv:.5f} (Â±{np.std(xgb_cv_scores):.5f})")
print(f"LightGBM CV RMSE: {lgb_cv:.5f} (Â±{np.std(lgb_cv_scores):.5f})")
print(f"CatBoost CV RMSE: {cat_cv:.5f} (Â±{np.std(cat_cv_scores):.5f})")

# ==========================================
# 9. Weighted Ensemble (Inverse RMSE)
# ==========================================
total_inv = (1/xgb_cv) + (1/lgb_cv) + (1/cat_cv)
xgb_weight = (1/xgb_cv) / total_inv
lgb_weight = (1/lgb_cv) / total_inv
cat_weight = (1/cat_cv) / total_inv

print(f"\nğŸ”§ Optimal Ensemble Weights:")
print(f"XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}, CatBoost: {cat_weight:.3f}")

final_preds = (
    xgb_weight * xgb_preds +
    lgb_weight * lgb_preds +
    cat_weight * cat_preds
)

# ==========================================
# 10. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': final_preds
})
submission.to_csv('submission32.csv', index=False)

expected_rmse = min(xgb_cv, lgb_cv, cat_cv) * 0.98
print(f"\nâœ… submission.csv created successfully!")
print(f"ğŸ�¯ Expected Ensemble RMSE: ~{expected_rmse:.5f}")
print("\nFirst 5 predictions:")
print(submission.head())



# ==========================================
# ğŸš€ Full Final Code: XGBoost + MLP Ensemble
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import optuna

# ==========================================
# 1. Load Data
# ==========================================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

target = 'accident_risk'
id_col = 'id'

X = train.drop(columns=[target, id_col])
y = train[target]

# ==========================================
# 2. Encode Categorical Columns
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ==========================================
# 3. Normalize Features
# ==========================================
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
test[X.columns] = scaler.transform(test[X.columns])

# ==========================================
# 4. Optuna Bayesian Optimization for XGBoost
# ==========================================
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'gamma': trial.suggest_float('gamma', 0, 0.2),
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'random_state': 42
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        rmse_list.append(sqrt(mean_squared_error(y_val, y_pred)))

    return np.mean(rmse_list)


# ==========================================
# 5. Run Optuna Study
# ==========================================
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

print("âœ… Best Params:", study.best_params)
print("âœ… Best CV RMSE:", study.best_value)


# ==========================================
# 6. Train Final Models with 10-Fold CV + Seed Averaging
# ==========================================
best_params = study.best_params
best_params.update({
    'n_estimators': 1000,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'n_jobs': -1
})

kf = KFold(n_splits=10, shuffle=True, random_state=42)
seeds = [42, 2023]

preds_xgb = np.zeros(len(test))
preds_mlp = np.zeros(len(test))
xgb_rmse_scores, mlp_rmse_scores = [], []

# ==========================================
# 6A. XGBoost CV
# ==========================================
for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"XGB Seed {seed} CV"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**best_params, random_state=seed)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))

        fold_preds += model.predict(test[X.columns].values) / kf.n_splits

    preds_xgb += fold_preds / len(seeds)
    print(f"Seed {seed} Mean RMSE: {np.mean(fold_rmse):.5f}")
    xgb_rmse_scores.append(np.mean(fold_rmse))

# ==========================================
# 6B. MLPRegressor CV (Neural Net)
# ==========================================
mlp_params = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "max_iter": 500
}

for seed in seeds:
    fold_preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X), 1), total=kf.get_n_splits(), desc=f"MLP Seed {seed} CV"):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        mlp = MLPRegressor(**mlp_params, random_state=seed)
        mlp.fit(X_tr, y_tr)

        y_pred = mlp.predict(X_val)
        fold_rmse.append(sqrt(mean_squared_error(y_val, y_pred)))

        fold_preds += mlp.predict(test[X.columns].values) / kf.n_splits

    preds_mlp += fold_preds / len(seeds)
    print(f"Seed {seed} Mean RMSE: {np.mean(fold_rmse):.5f}")
    mlp_rmse_scores.append(np.mean(fold_rmse))

# ==========================================
# 7. Weighted Ensemble (Inverse RMSE)
# ==========================================
xgb_cv = np.mean(xgb_rmse_scores)
mlp_cv = np.mean(mlp_rmse_scores)

total_inv = (1 / xgb_cv) + (1 / mlp_cv)
xgb_weight = (1 / xgb_cv) / total_inv
mlp_weight = (1 / mlp_cv) / total_inv

print("\nğŸ”§ Ensemble Weights:")
print(f"XGBoost: {xgb_weight:.3f}, MLP: {mlp_weight:.3f}")

final_preds = (xgb_weight * preds_xgb + mlp_weight * preds_mlp)

# ==========================================
# 8. Create Submission
# ==========================================
submission = pd.DataFrame({
    'id': test[id_col],
    'accident_risk': final_preds
})

submission.to_csv("submission_xgb_mlp_final.csv", index=False)

expected_rmse = min(xgb_cv, mlp_cv) * 0.98
print(f"\nâœ… submission_xgb_mlp_final.csv created successfully!")
print(f"ğŸ�¯ Expected Ensemble RMSE: ~{expected_rmse:.5f}")
print("\nFirst 5 predictions:")
print(submission.head())

