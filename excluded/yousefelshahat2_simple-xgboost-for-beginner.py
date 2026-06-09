pip install --upgrade xgboost scikit-learn


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as RMSE


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.
    
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test, new_num


# Identify feature
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test, new_num = create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)


print(df.columns.tolist())


df.head()


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# === Prepare data ===
X = df.drop(columns=target)
y = df[target].values

# === Custom KFold ===
n_splits = 8
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# === XGBoost parameters ===
xgb_params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'rmse',
    'random_state': 42,
    'max_bin': 512,
    'min_child_weight': 3,
    'max_delta_step': 1,
    'max_depth': 11,
    'learning_rate': 0.010453775390437146,
    'subsample': 0.8162196077561874,
    'colsample_bytree': 0.8057453252225478,
    'gamma': 0.011515371568909936,
    'reg_alpha': 0.1153674139991063,
    'reg_lambda': 0.4029264986439234,
    'colsample_bylevel': 0.8675078626084138,
    'colsample_bynode': 0.8804930677965951,
    'scale_pos_weight': 0.3615894752587659,
}

# === OOF storage ===
oof_preds = np.zeros(len(X))
best_iterations = []

# === Train XGBoost with custom folds ===
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    evals = [(dtrain, 'train'), (dval, 'valid')]
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=1000
    )
    
    oof_preds[val_idx] = model.predict(dval)
    best_iterations.append(model.best_iteration)

# === XGBoost OOF MSE ===
xgb_mse = RMSE(y, oof_preds)
print(f"XGBoost OOF MSE: {xgb_mse:.7f}")

# === Train Linear Regression on OOF predictions ===
lr = LinearRegression()
lr.fit(oof_preds.reshape(-1, 1), y)

# Predict using the Linear Regression on the same OOF
lr_preds = lr.predict(oof_preds.reshape(-1, 1))
lr_mse = RMSE(y, lr_preds)
print(f"Linear Regression on OOF MSE: {lr_mse:.7f}")


# Average of best iterations from all folds
avg_best_iteration = int(np.mean(best_iterations))
xgb_params["n_estimators"] = avg_best_iteration + 10


# # Prepare DMatrix for XGBoost
# dtrain = xgb.DMatrix(df.drop(columns=target), label=df[target], enable_categorical=True)

# # Define XGBoost parameters
# xgb_params  = {
#     'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'rmse',
#     'random_state': 42,'max_bin': 512, 'min_child_weight': 3,
#     'max_delta_step': 1, 'max_depth': 11, 'learning_rate': 0.010453775390437146,
#     'subsample': 0.8162196077561874,'colsample_bytree': 0.8057453252225478,
#     'gamma': 0.011515371568909936,'reg_alpha': 0.1153674139991063,
#     'reg_lambda': 0.4029264986439234,'colsample_bylevel': 0.8675078626084138,
#     'colsample_bynode': 0.8804930677965951,'scale_pos_weight': 0.3615894752587659,
# }

# # Run cross-validation
# cv_results = xgb.cv(
#     params=xgb_params,
#     dtrain=dtrain,
#     nfold=5,
#     num_boost_round=2000,
#     metrics='rmse',
#     verbose_eval=100,
#     early_stopping_rounds=50
# )

# # Display last few CV results
# print(cv_results.tail())

# # Extract best boosting round
# best_round = cv_results['test-rmse-mean'].idxmin()
# best_rmse = cv_results['test-rmse-mean'][best_round]
# print(f"Best round: {best_round}, Best CV RMSE: {best_rmse:.7f}")


# # putting the n_estimator at the average early stopping point to avoid overfitting
# last_round = len(cv_results) - 1
# xgb_params["n_estimators"] = last_round + 10


# Prepare training data
X_train = df.drop(columns=target)
y_train = df[target]

# Train XGBoost model
model = XGBRegressor(**xgb_params, enable_categorical=True)
model.fit(X_train, y_train)

# Predict on test set
pred_xgb = model.predict(df_test.drop(columns="id"))
pred = lr.predict(pred_xgb.reshape(-1, 1))

# Prepare submission
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)

