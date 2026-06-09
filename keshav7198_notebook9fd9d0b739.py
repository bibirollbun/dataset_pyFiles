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




import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib


# ---------------------- CONFIG ----------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"   # update if needed
TEST_PATH  = "/kaggle/input/playground-series-s5e10/test.csv"    # update if needed
TARGET     = "accident_risk"
ID_COL     = "id"

N_SPLITS   = 5
SEEDS      = [42, 2023]            # seed averaging
EARLY_STOP = 100
USE_GPU    = True                 # set True if you have GPU & XGBoost with GPU support
MAX_ONEHOT_UNIQUE = 12             # max unique values for forcing OHE (besides forced OHE cols)
FORCED_OHE_COLS = ["lighting", "weather", "time_of_day"]  # keep these one-hot encoded
# ----------------------------------------------------



def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test

def build_preprocessor(X, forced_ohe):
    """
    Build ColumnTransformer:
      - numeric: median imputer
      - forced one-hot for small nominal features
      - ordinal for remaining categorical
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Keep forced OHE columns only if they actually exist in data
    ohe_cols = [c for c in forced_ohe if c in cat_cols]
    # remaining categorical (exclude forced ohe)
    other_cat = [c for c in cat_cols if c not in ohe_cols]

    # For other_cat, if cardinality very small, you may still prefer OHE - but we'll ordinal encode
    # to keep the pipeline compact; trees handle ordinal encoding fine.
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if ohe_cols:
        transformers.append(("ohe", ohe_pipe, ohe_cols))
    if other_cat:
        transformers.append(("ord", ord_pipe, other_cat))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    return preprocessor, numeric_cols, ohe_cols, other_cat


def xgb_default_params(use_gpu=False):
    return {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "predictor": "gpu_predictor" if use_gpu else "cpu_predictor",
        "n_jobs": -1,
        "verbosity": 1
    }

def fit_and_predict(train, test):
    X_all = train.drop(columns=[TARGET, ID_COL])
    y_all = train[TARGET].copy()
    X_test_raw = test.drop(columns=[ID_COL]).copy()
    test_ids = test[ID_COL].copy()

    preprocessor, numeric_cols, ohe_cols, other_cat = build_preprocessor(X_all, FORCED_OHE_COLS)
    print(f"Numeric cols: {len(numeric_cols)}, forcing OHE for: {ohe_cols}, ordinal for: {other_cat}")

    # Prepare CV split (fixed KFold so folds are consistent)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    preds_test = np.zeros((len(SEEDS), len(X_test_raw)), dtype=float)
    seed_scores = []

    for s_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ===")
        fold_test_preds = np.zeros(len(X_test_raw), dtype=float)
        fold_scores = []

        params = xgb_default_params(USE_GPU)
        params["random_state"] = seed

        # For each fold, fit preprocessor on training fold only (avoids encoding leakage)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), start=1):
            print(f"Seed {seed} - Fold {fold_idx}/{N_SPLITS}")
            X_tr_raw = X_all.iloc[train_idx].reset_index(drop=True)
            y_tr = y_all.iloc[train_idx].reset_index(drop=True)
            X_val_raw = X_all.iloc[val_idx].reset_index(drop=True)
            y_val = y_all.iloc[val_idx].reset_index(drop=True)

            # Fit preprocessor on fold training data
            preprocessor.fit(X_tr_raw)

            X_tr = preprocessor.transform(X_tr_raw)
            X_val = preprocessor.transform(X_val_raw)
            X_test = preprocessor.transform(X_test_raw)

            model = XGBRegressor(**params)

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=EARLY_STOP,
                verbose=100
            )

            y_val_pred = model.predict(X_val)
            rmse = sqrt(mean_squared_error(y_val, y_val_pred))
            fold_scores.append(rmse)
            print(f"Fold {fold_idx} RMSE: {rmse:.6f}")

            # accumulate test predictions averaged across folds
            fold_test_preds += model.predict(X_test) / N_SPLITS

        seed_mean = float(np.mean(fold_scores))
        seed_scores.append(seed_mean)
        preds_test[s_idx, :] = fold_test_preds
        print(f"Seed {seed} average RMSE: {seed_mean:.6f}")

    # average predictions across seeds
    final_preds = preds_test.mean(axis=0)
    print("\nSeed-averaged CV RMSEs:", seed_scores, "-> mean:", np.mean(seed_scores))
    return test_ids, final_preds, preprocessor



def save_submission(test_ids, preds):
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

def main():
    print("Loading data...")
    train, test = load_data()
    print("Train shape:", train.shape, "Test shape:", test.shape)

    test_ids, preds, preprocessor = fit_and_predict(train, test)
    save_submission(test_ids, preds)

    # Save the last fitted preprocessor for reuse (note: it's fitted on last fold's train)
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("Saved preprocessor.joblib")

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path2="/kaggle/input/playground-series-s5e10/train.csv"
path1="/kaggle/input/playground-series-s5e10/test.csv"
newdf=pd.read_csv(path2)

x2=newdf.iloc[:,:14].values
y2=newdf.iloc[:,-1].values
y2old=y2
print(x2)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])],remainder='passthrough')
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[6])],remainder='passthrough')
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[9])],remainder='passthrough')

x2=np.array(ct.fit_transform(x2))


print(x2)


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# load dataframe
df = pd.read_csv(path2)   # or however you load it

# 1) Convert boolean-like columns to numeric (works for booleans or "TRUE"/"FALSE" strings)
bool_candidates = ['road_signs', 'public_road', 'holiday', 'school_season']  # update if names differ

# If some columns are actual bool dtype:
for col in bool_candidates:
    if col in df.columns:
        # handle boolean dtype or string "TRUE"/"FALSE" (case-insensitive)
        df[col] = df[col].replace({'TRUE':1, 'True':1, 'FALSE':0, 'False':0}).astype(int)

# 2) split features/target
X = df.drop(columns=['accident_risk'])   # change target name if different
y = df['accident_risk']

# 3) OneHotEncode multiple categorical columns at once
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']  # update these names if needed
# use drop='first' to prevent dummy variable trap or keep all if tree-based models (no need to drop)
ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'   # keep all other (already numeric) columns as-is
)

X_trans = ct.fit_transform(X)   # numpy array
print("Transformed shape:", X_trans.shape)



print(X_trans)


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
y_pred=lin_reg.predict(x)


##transforming the linear model into polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)       ##this method transform the features into degree of features according to polynomial function
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
y_poly_pred=lin_reg2.predict(x_poly)


# stack_mlp_xgb.py
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# -------------------------
# USER SETTINGS - edit
# -------------------------
path_train = "/kaggle/input/playground-series-s5e10/train.csv"
path_test  = "/kaggle/input/playground-series-s5e10/test.csv"    # may or may not contain target
target_col = "accident_risk"
id_col     = "id"

bool_candidates = ['road_signs', 'public_road', 'holiday', 'school_season']
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# CV / training settings
N_SPLITS = 5
SEED = 42
BATCH_SIZE = 2048
EPOCHS = 200
PATIENCE = 10

# -------------------------
# Repro
# -------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Utilities
# -------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def convert_bool_like(df, bool_cols):
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({'TRUE':1, 'True':1, 'true':1,
                                       'FALSE':0, 'False':0, 'false':0}).astype(int)
    return df

# -------------------------
# 1) Load
# -------------------------
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test) if os.path.exists(path_test) else None

X = train_df.drop(columns=[target_col])
y = train_df[target_col].values.astype(float)

if test_df is not None:
    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(float)
    else:
        X_test = test_df.copy()
        y_test = None
else:
    X_test = None
    y_test = None

# -------------------------
# 2) Convert boolean-like cols
# -------------------------
X = convert_bool_like(X, bool_candidates)
if X_test is not None:
    X_test = convert_bool_like(X_test, bool_candidates)

# -------------------------
# 3) Build preprocessor
# -------------------------
all_features = list(X.columns)
numeric_cols = [c for c in all_features if c not in categorical_cols and c != id_col]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
], remainder="drop")

preprocessor.fit(X)
X_enc = preprocessor.transform(X)
X_enc = X_enc.astype(np.float32)
if X_test is not None:
    X_test_enc = preprocessor.transform(X_test).astype(np.float32)
else:
    X_test_enc = None

print("X_enc shape:", X_enc.shape, "X_test_enc shape:", None if X_test_enc is None else X_test_enc.shape)

# -------------------------
# 4) Define MLP builder
# -------------------------
def build_mlp(input_dim, hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model

# optional GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# -------------------------
# 5) Create OOF arrays and per-fold test preds
# -------------------------
n_samples = X_enc.shape[0]
oof_mlp = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)

test_preds_mlp = None
test_preds_xgb = None
if X_test_enc is not None:
    test_preds_mlp = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32)
    test_preds_xgb = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold = 0
for train_idx, val_idx in kf.split(X_enc):
    fold += 1
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # ----- MLP (Keras) -----
    mlp = build_mlp(input_dim=X_enc.shape[1], hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=PATIENCE, restore_best_weights=True, mode="min", verbose=0),
    ]
    # Fit
    mlp.fit(X_tr, y_tr, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    # Predict val
    val_pred_mlp = mlp.predict(X_val, batch_size=1024).ravel()
    oof_mlp[val_idx] = val_pred_mlp
    print("MLP fold RMSE (val):", rmse(y_val, val_pred_mlp))

    # Predict test (store per-fold)
    if X_test_enc is not None:
        test_preds_mlp[:, fold-1] = mlp.predict(X_test_enc, batch_size=1024).ravel()

    # ----- XGBoost -----
    xgb = XGBRegressor(objective="reg:squarederror",
                       n_estimators=800,
                       learning_rate=0.03,
                       max_depth=6,
                       subsample=0.9,
                       colsample_bytree=0.8,
                       reg_alpha=0.1,
                       reg_lambda=5,
                       random_state=SEED,
                       n_jobs=-1,
                       verbosity=0)
   xgb.set_params(early_stopping_rounds=30)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = xgb.predict(X_val)
    oof_xgb[val_idx] = val_pred_xgb
    print("XGB fold RMSE (val):", rmse(y_val, val_pred_xgb))

    if X_test_enc is not None:
        test_preds_xgb[:, fold-1] = xgb.predict(X_test_enc)

    # optional: free memory
    del mlp, xgb
    tf.keras.backend.clear_session()

# -------------------------
# 6) CV results for base models
# -------------------------
cv_rmse_mlp = rmse(y, oof_mlp)
cv_rmse_xgb = rmse(y, oof_xgb)
print(f"\nCV RMSE MLP (OOF): {cv_rmse_mlp:.5f}")
print(f"CV RMSE XGB (OOF): {cv_rmse_xgb:.5f}")

# -------------------------
# 7) Train meta-model on OOF preds
# -------------------------
stacked_oof = np.vstack([oof_mlp, oof_xgb]).T
meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stacked_oof, y)
print("Meta coefficients:", meta.coef_, "intercept:", meta.intercept_)

# -------------------------
# 8) Retrain base models on FULL training set and predict test
# -------------------------
# MLP full
mlp_full = build_mlp(input_dim=X_enc.shape[1], hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3)
cb_full = [tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=PATIENCE, restore_best_weights=True, mode="min", verbose=0)]
# use a small validation split from train for early stopping
mlp_full.fit(X_enc, y, validation_split=0.05, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_full, verbose=0)
if X_test_enc is not None:
    mlp_test_pred = mlp_full.predict(X_test_enc, batch_size=1024).ravel()
else:
    mlp_test_pred = None

# XGB full
xgb_full = XGBRegressor(objective="reg:squarederror",
                       n_estimators=1000,
                       learning_rate=0.03,
                       max_depth=6,
                       subsample=0.9,
                       colsample_bytree=0.8,
                       reg_alpha=0.1,
                       reg_lambda=5,
                       random_state=SEED,
                       n_jobs=-1,
                       verbosity=0)
xgb_full.fit(X_enc, y, verbose=False)
if X_test_enc is not None:
    xgb_test_pred = xgb_full.predict(X_test_enc)
else:
    xgb_test_pred = None

# Save full models
try:
    joblib.dump(xgb_full, "xgb_full.joblib")
    mlp_full.save("mlp_full.keras")
    joblib.dump(meta, "stack_meta_ridge.joblib")
    print("Saved xgb_full.joblib, mlp_full.keras and stack_meta_ridge.joblib")
except Exception as e:
    print("Could not save models:", e)

# -------------------------
# 9) Final stacked predictions and evaluation
# -------------------------
if X_test_enc is not None:
    stacked_test = np.vstack([mlp_test_pred, xgb_test_pred]).T
    final_test_pred = meta.predict(stacked_test)

    if y_test is not None:
        final_rmse = rmse(y_test, final_test_pred)
        print(f"\nFinal stacked RMSE on test: {final_rmse:.5f}")
        print(f"Base MLP test RMSE: {rmse(y_test, mlp_test_pred):.5f}")
        print(f"Base XGB  test RMSE: {rmse(y_test, xgb_test_pred):.5f}")
    else:
        print("\nNo test target provided â€” final predictions produced for test set.")

    # Save predictions
    out = pd.DataFrame()
    if id_col in X_test.columns:
        out[id_col] = X_test[id_col].values
    else:
        out["index"] = X_test.index
    out["predicted_accident_risk"] = final_test_pred
    out.to_csv("stacked_mlp_xgb_preds.csv", index=False)
    print("Saved stacked predictions to 'stacked_mlp_xgb_preds.csv'")

else:
    print("\nNo test set provided; stacked OOF predictions are available for training evaluation.")
    # You can inspect stacked_oof and meta predictions on train
    train_meta_pred = meta.predict(stacked_oof)
    print("Stacked RMSE on train (meta on OOF):", rmse(y, train_meta_pred))

# -------------------------
# Done
# -------------------------



# stack_mlp_xgb_rounded.py
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# -------------------------
# USER SETTINGS - edit
# -------------------------
path_train = "/kaggle/input/playground-series-s5e10/train.csv"
path_test  = "/kaggle/input/playground-series-s5e10/test.csv"    # may or may not contain target
target_col = "accident_risk"
id_col     = "id"

bool_candidates = ['road_signs', 'public_road', 'holiday', 'school_season']
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# CV / training settings
N_SPLITS = 5
SEED = 42
BATCH_SIZE = 2048
EPOCHS = 200
PATIENCE = 10

# -------------------------
# Repro
# -------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Utilities
# -------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def convert_bool_like(df, bool_cols):
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({'TRUE':1, 'True':1, 'true':1,
                                       'FALSE':0, 'False':0, 'false':0}).astype(int)
    return df

# -------------------------
# 1) Load
# -------------------------
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test) if os.path.exists(path_test) else None

X = train_df.drop(columns=[target_col])
y = train_df[target_col].values.astype(float)

if test_df is not None:
    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(float)
    else:
        X_test = test_df.copy()
        y_test = None
else:
    X_test = None
    y_test = None

# -------------------------
# 2) Convert boolean-like cols
# -------------------------
X = convert_bool_like(X, bool_candidates)
if X_test is not None:
    X_test = convert_bool_like(X_test, bool_candidates)

# -------------------------
# 3) Build preprocessor
# -------------------------
all_features = list(X.columns)
numeric_cols = [c for c in all_features if c not in categorical_cols and c != id_col]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
], remainder="drop")

preprocessor.fit(X)
X_enc = preprocessor.transform(X)
X_enc = X_enc.astype(np.float32)
if X_test is not None:
    X_test_enc = preprocessor.transform(X_test).astype(np.float32)
else:
    X_test_enc = None

print("X_enc shape:", X_enc.shape, "X_test_enc shape:", None if X_test_enc is None else X_test_enc.shape)

# -------------------------
# 4) Define MLP builder
# -------------------------
def build_mlp(input_dim, hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model

# optional GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# -------------------------
# 5) Create OOF arrays and per-fold test preds
# -------------------------
n_samples = X_enc.shape[0]
oof_mlp = np.zeros(n_samples, dtype=np.float32)
oof_xgb = np.zeros(n_samples, dtype=np.float32)

test_preds_mlp = None
test_preds_xgb = None
if X_test_enc is not None:
    test_preds_mlp = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32)
    test_preds_xgb = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold = 0
for train_idx, val_idx in kf.split(X_enc):
    fold += 1
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # ----- MLP (Keras) -----
    mlp = build_mlp(input_dim=X_enc.shape[1], hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=PATIENCE, restore_best_weights=True, mode="min", verbose=0),
    ]
    # Fit
    mlp.fit(X_tr, y_tr, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    # Predict val
    val_pred_mlp = mlp.predict(X_val, batch_size=1024).ravel()
    oof_mlp[val_idx] = val_pred_mlp
    print("MLP fold RMSE (val):", rmse(y_val, val_pred_mlp))

    # Predict test (store per-fold)
    if X_test_enc is not None:
        test_preds_mlp[:, fold-1] = mlp.predict(X_test_enc, batch_size=1024).ravel()

    # ----- XGBoost -----
    # pass early_stopping_rounds to constructor (avoid deprecation warning)
    xgb = XGBRegressor(objective="reg:squarederror",
                       n_estimators=800,
                       learning_rate=0.03,
                       max_depth=6,
                       subsample=0.9,
                       colsample_bytree=0.8,
                       reg_alpha=0.1,
                       reg_lambda=5,
                       random_state=SEED,
                       n_jobs=-1,
                       verbosity=0,
                       early_stopping_rounds=30)

    # use eval_set in fit for early stopping (constructor param avoids the deprecation warning)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_xgb = xgb.predict(X_val)
    oof_xgb[val_idx] = val_pred_xgb
    print("XGB fold RMSE (val):", rmse(y_val, val_pred_xgb))

    if X_test_enc is not None:
        test_preds_xgb[:, fold-1] = xgb.predict(X_test_enc)

    # optional: free memory
    del mlp, xgb
    tf.keras.backend.clear_session()

# -------------------------
# 6) CV results for base models
# -------------------------
cv_rmse_mlp = rmse(y, oof_mlp)
cv_rmse_xgb = rmse(y, oof_xgb)
print(f"\nCV RMSE MLP (OOF): {cv_rmse_mlp:.5f}")
print(f"CV RMSE XGB (OOF): {cv_rmse_xgb:.5f}")

# -------------------------
# 7) Train meta-model on OOF preds
# -------------------------
stacked_oof = np.vstack([oof_mlp, oof_xgb]).T
meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stacked_oof, y)
print("Meta coefficients:", meta.coef_, "intercept:", meta.intercept_)

# -------------------------
# 8) Retrain base models on FULL training set and predict test
# -------------------------
# MLP full
mlp_full = build_mlp(input_dim=X_enc.shape[1], hidden_units=[512,256,128], dropout_rate=0.15, lr=1e-3)
cb_full = [tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=PATIENCE, restore_best_weights=True, mode="min", verbose=0)]
# use a small validation split from train for early stopping
mlp_full.fit(X_enc, y, validation_split=0.05, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_full, verbose=0)
if X_test_enc is not None:
    mlp_test_pred = mlp_full.predict(X_test_enc, batch_size=1024).ravel()
else:
    mlp_test_pred = None

# XGB full
xgb_full = XGBRegressor(objective="reg:squarederror",
                       n_estimators=1000,
                       learning_rate=0.03,
                       max_depth=6,
                       subsample=0.9,
                       colsample_bytree=0.8,
                       reg_alpha=0.1,
                       reg_lambda=5,
                       random_state=SEED,
                       n_jobs=-1,
                       verbosity=0)
xgb_full.fit(X_enc, y, verbose=False)
if X_test_enc is not None:
    xgb_test_pred = xgb_full.predict(X_test_enc)
else:
    xgb_test_pred = None

# Save full models
try:
    joblib.dump(xgb_full, "xgb_full.joblib")
    mlp_full.save("mlp_full.keras")
    joblib.dump(meta, "stack_meta_ridge.joblib")
    print("Saved xgb_full.joblib, mlp_full.keras and stack_meta_ridge.joblib")
except Exception as e:
    print("Could not save models:", e)

# -------------------------
# 9) Final stacked predictions and evaluation
# -------------------------
if X_test_enc is not None:
    stacked_test = np.vstack([mlp_test_pred, xgb_test_pred]).T
    final_test_pred = meta.predict(stacked_test)

    # ----- ROUND final predictions to 3 decimal places -----
    final_test_pred_rounded = np.round(final_test_pred, 3)

    if y_test is not None:
        final_rmse = rmse(y_test, final_test_pred)
        print(f"\nFinal stacked RMSE on test: {final_rmse:.5f}")
        print(f"Base MLP test RMSE: {rmse(y_test, mlp_test_pred):.5f}")
        print(f"Base XGB  test RMSE: {rmse(y_test, xgb_test_pred):.5f}")
    else:
        print("\nNo test target provided â€” final predictions produced for test set.")

    # Save predictions (rounded to 3 decimals)
    out = pd.DataFrame()
    if id_col in X_test.columns:
        out[id_col] = X_test[id_col].values
    else:
        out["index"] = X_test.index
    out["predicted_accident_risk"] = final_test_pred_rounded
    out.to_csv("stacked_mlp_xgb_preds.csv", index=False)
    print("Saved stacked predictions to 'stacked_mlp_xgb_preds.csv' (rounded to 3 decimals)")

else:
    print("\nNo test set provided; stacked OOF predictions are available for training evaluation.")
    # You can inspect stacked_oof and meta predictions on train
    train_meta_pred = meta.predict(stacked_oof)
    print("Stacked RMSE on train (meta on OOF):", rmse(y, train_meta_pred))

# -------------------------
# Done
# -------------------------



# stack_optuna_tuning.py
import os
import numpy as np
import pandas as pd
import joblib
import optuna
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# -------------------------
# USER SETTINGS - edit
# -------------------------
path_train = "/kaggle/input/playground-series-s5e10/train.csv"
path_test  = "/kaggle/input/playground-series-s5e10/test.csv"
target_col = "accident_risk"
id_col     = "id"

bool_candidates = ['road_signs', 'public_road', 'holiday', 'school_season']
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# CV / training settings
N_SPLITS = 5
SEED = 42
BATCH_SIZE_DEFAULT = 2048
EPOCHS_DEFAULT = 200
PATIENCE = 10

# Optuna settings (adjust to available compute)
N_TRIALS_XGB = 40   # try 40 (increase for better search)
N_TRIALS_MLP = 30   # try 30 (increase for better search)

# -------------------------
# reproducibility
# -------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# utilities
# -------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def convert_bool_like(df, bool_cols):
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                'TRUE':1, 'True':1, 'true':1,
                'FALSE':0, 'False':0, 'false':0
            }).astype(int)
    return df

# -------------------------
# 1) Load data
# -------------------------
train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test) if os.path.exists(path_test) else None

X = train_df.drop(columns=[target_col])
y = train_df[target_col].values.astype(float)

if test_df is not None:
    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(float)
    else:
        X_test = test_df.copy()
        y_test = None
else:
    X_test = None
    y_test = None

# -------------------------
# 2) Convert booleans
# -------------------------
X = convert_bool_like(X, bool_candidates)
if X_test is not None:
    X_test = convert_bool_like(X_test, bool_candidates)

# -------------------------
# 3) Preprocessor
# -------------------------
all_features = list(X.columns)
numeric_cols = [c for c in all_features if c not in categorical_cols and c != id_col]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
], remainder="drop")

preprocessor.fit(X)
X_enc = preprocessor.transform(X).astype(np.float32)
X_test_enc = preprocessor.transform(X_test).astype(np.float32) if X_test is not None else None

print("Processed shapes:", X_enc.shape, None if X_test_enc is None else X_test_enc.shape)

# -------------------------
# KFold for consistency
# -------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_indices = list(kf.split(X_enc))  # list of (train_idx, val_idx)

# -------------------------
# 4) Optuna objective for XGBoost
# -------------------------
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',           # fast; if GPU: 'gpu_hist'
        'predictor': 'auto',
        'n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 800, 1200]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': SEED,
        'n_jobs': -1,
        # set early stopping param here to avoid deprecation warning
        'early_stopping_rounds': 50
    }

    # Use 'gpu_hist' if GPU available
    try:
        physical_gpus = tf.config.list_physical_devices('GPU')
        if physical_gpus:
            params['tree_method'] = 'gpu_hist'
    except Exception:
        pass

    rmses = []
    for train_idx, val_idx in fold_indices:
        X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**params)
        # fit with eval_set to enable early stopping
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        rmses.append(rmse(y_val, pred))
        # allow trial pruning (optional) - skip for simplicity

    return float(np.mean(rmses))

# Run Optuna for XGB
study_xgb = optuna.create_study(direction='minimize', study_name='xgb_study')
print("Starting XGBoost tuning with Optuna...")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, show_progress_bar=True)
print("XGB best value (CV RMSE):", study_xgb.best_value)
print("XGB best params:", study_xgb.best_params)

# Extract best XGB params and create final estimators
best_xgb_params = study_xgb.best_params.copy()
# convert categorical sampling keys into params expected by constructor
# keep early_stopping_rounds and other constant settings
best_xgb = XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist',
    n_estimators=best_xgb_params.get('n_estimators'),
    learning_rate=best_xgb_params.get('learning_rate'),
    max_depth=best_xgb_params.get('max_depth'),
    subsample=best_xgb_params.get('subsample'),
    colsample_bytree=best_xgb_params.get('colsample_bytree'),
    min_child_weight=best_xgb_params.get('min_child_weight'),
    reg_alpha=best_xgb_params.get('reg_alpha'),
    reg_lambda=best_xgb_params.get('reg_lambda'),
    gamma=best_xgb_params.get('gamma'),
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbosity=0
)

# -------------------------
# 5) Optuna objective for MLP (Keras)
# -------------------------
def build_mlp_model(input_dim, n_layers, units, dropout, lr):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model

def objective_mlp(trial):
    # sample hyperparams
    n_layers = trial.suggest_int('n_layers', 1, 4)
    units = trial.suggest_categorical('units', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    epochs = 100  # we'll use early stopping

    rmses = []
    for train_idx, val_idx in fold_indices:
        X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_mlp_model(input_dim=X_enc.shape[1], n_layers=n_layers, units=units, dropout=dropout, lr=lr)
        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, mode='min', restore_best_weights=True, verbose=0)]
        # Fit quietly
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
        pred = model.predict(X_val, batch_size=1024).ravel()
        rmses.append(rmse(y_val, pred))
        # clear session to free GPU memory
        tf.keras.backend.clear_session()

    return float(np.mean(rmses))

# Run Optuna for MLP
study_mlp = optuna.create_study(direction='minimize', study_name='mlp_study')
print("Starting MLP tuning with Optuna...")
study_mlp.optimize(objective_mlp, n_trials=N_TRIALS_MLP, show_progress_bar=True)
print("MLP best value (CV RMSE):", study_mlp.best_value)
print("MLP best params:", study_mlp.best_params)

# Build final MLP using best params
best_mlp_params = study_mlp.best_params.copy()
best_mlp = build_mlp_model(
    input_dim=X_enc.shape[1],
    n_layers=best_mlp_params.get('n_layers'),
    units=best_mlp_params.get('units'),
    dropout=best_mlp_params.get('dropout'),
    lr=best_mlp_params.get('lr')
)

# -------------------------
# 6) Create OOF preds using tuned models (same KFold)
# -------------------------
n = X_enc.shape[0]
oof_mlp = np.zeros(n, dtype=np.float32)
oof_xgb = np.zeros(n, dtype=np.float32)

test_preds_mlp = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32) if X_test_enc is not None else None
test_preds_xgb = np.zeros((X_test_enc.shape[0], N_SPLITS), dtype=np.float32) if X_test_enc is not None else None

fold = 0
for train_idx, val_idx in fold_indices:
    fold += 1
    print(f"OOF Fold {fold}/{N_SPLITS}")
    X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Train MLP for this fold
    mlp_fold = build_mlp_model(
        input_dim=X_enc.shape[1],
        n_layers=best_mlp_params['n_layers'],
        units=best_mlp_params['units'],
        dropout=best_mlp_params['dropout'],
        lr=best_mlp_params['lr']
    )
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, mode='min', restore_best_weights=True, verbose=0)]
    mlp_fold.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS_DEFAULT, batch_size=best_mlp_params.get('batch_size', BATCH_SIZE_DEFAULT), callbacks=cb, verbose=0)
    pred_val_mlp = mlp_fold.predict(X_val, batch_size=1024).ravel()
    oof_mlp[val_idx] = pred_val_mlp
    if X_test_enc is not None:
        test_preds_mlp[:, fold-1] = mlp_fold.predict(X_test_enc, batch_size=1024).ravel()
    tf.keras.backend.clear_session()

    # Train XGB for this fold (use best params)
    xgb_fold = XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist',
        n_estimators=best_xgb.get_params()['n_estimators'],
        learning_rate=best_xgb.get_params()['learning_rate'],
        max_depth=best_xgb.get_params()['max_depth'],
        subsample=best_xgb.get_params()['subsample'],
        colsample_bytree=best_xgb.get_params()['colsample_bytree'],
        min_child_weight=best_xgb.get_params()['min_child_weight'],
        reg_alpha=best_xgb.get_params()['reg_alpha'],
        reg_lambda=best_xgb.get_params()['reg_lambda'],
        gamma=best_xgb.get_params()['gamma'],
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_fold.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred_val_xgb = xgb_fold.predict(X_val)
    oof_xgb[val_idx] = pred_val_xgb
    if X_test_enc is not None:
        test_preds_xgb[:, fold-1] = xgb_fold.predict(X_test_enc)

# CV RMSEs
cv_mlp = rmse(y, oof_mlp)
cv_xgb = rmse(y, oof_xgb)
print(f"CV RMSE after Optuna - MLP: {cv_mlp:.5f}, XGB: {cv_xgb:.5f}")

# -------------------------
# 7) Meta model and final stacking
# -------------------------
stacked_oof = np.vstack([oof_mlp, oof_xgb]).T
meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta.fit(stacked_oof, y)

# Retrain base models on full training set with best params
# MLP full
best_batch = best_mlp_params.get('batch_size', BATCH_SIZE_DEFAULT)
best_mlp_full = build_mlp_model(
    input_dim=X_enc.shape[1],
    n_layers=best_mlp_params['n_layers'],
    units=best_mlp_params['units'],
    dropout=best_mlp_params['dropout'],
    lr=best_mlp_params['lr']
)
cb_full = [tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True, mode='min', verbose=0)]
best_mlp_full.fit(X_enc, y, validation_split=0.05, epochs=EPOCHS_DEFAULT, batch_size=best_batch, callbacks=cb_full, verbose=0)
mlp_final_pred = best_mlp_full.predict(X_test_enc, batch_size=1024).ravel() if X_test_enc is not None else None

# XGB full
best_xgb_full = XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist',
    n_estimators=best_xgb.get_params()['n_estimators'],
    learning_rate=best_xgb.get_params()['learning_rate'],
    max_depth=best_xgb.get_params()['max_depth'],
    subsample=best_xgb.get_params()['subsample'],
    colsample_bytree=best_xgb.get_params()['colsample_bytree'],
    min_child_weight=best_xgb.get_params()['min_child_weight'],
    reg_alpha=best_xgb.get_params()['reg_alpha'],
    reg_lambda=best_xgb.get_params()['reg_lambda'],
    gamma=best_xgb.get_params()['gamma'],
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbosity=0
)
best_xgb_full.fit(X_enc, y, verbose=False)
xgb_final_pred = best_xgb_full.predict(X_test_enc) if X_test_enc is not None else None

# Stack final preds
if X_test_enc is not None:
    stacked_test = np.vstack([mlp_final_pred, xgb_final_pred]).T
    final_pred = meta.predict(stacked_test)
    final_pred_rounded = np.round(final_pred, 3)

    # Evaluate if y_test exists
    if y_test is not None:
        print("Final stacked test RMSE:", rmse(y_test, final_pred))
    # Save CSV with 3 decimals (and formatted strings if you want exact 3 digits)
    out = pd.DataFrame()
    out[id_col if id_col in X_test.columns else 'index'] = (X_test[id_col].values if id_col in X_test.columns else X_test.index)
    # numeric rounded
    out['predicted_accident_risk'] = final_pred_rounded
    out.to_csv("stacked_optuna_preds_3dec.csv", index=False)
    print("Saved predictions to stacked_optuna_preds_3dec.csv (rounded to 3 decimals)")

# -------------------------
# Save models and studies
# -------------------------
joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(meta, "stack_meta_ridge.joblib")
joblib.dump(best_xgb_full, "xgb_optuna_full.joblib")
best_mlp_full.save("mlp_optuna_full.keras")
study_xgb.trials_dataframe().to_csv("optuna_xgb_trials.csv", index=False)
study_mlp.trials_dataframe().to_csv("optuna_mlp_trials.csv", index=False)
joblib.dump(study_xgb, "study_xgb.pkl")
joblib.dump(study_mlp, "study_mlp.pkl")
print("Saved models and Optuna study artifacts.")


