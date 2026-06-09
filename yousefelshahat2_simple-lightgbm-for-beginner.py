pip install --upgrade lightgbm scikit-learn


import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor


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
            for q in [5, 10]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test


# Identify feature types
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category","bool"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test= create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present"]
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)

cat = [c for c in cat if c not in remove]


# Prepare LightGBM dataset
lgb_train = lgb.Dataset(df.drop(columns=target), label=df[target], categorical_feature=cat)

# Define LightGBM parameters
lgb_params = {'objective': 'regression', 'n_estimators':2000,
              'metric': 'rmse', 'boosting_type': 'gbdt',
              'learning_rate': 0.005, 'max_depth': -1,
              'num_leaves': 190, 'min_child_weight': 2,
              'subsample': 0.85, 'colsample_bytree': 0.8,
              'reg_alpha': 0.1, 'reg_lambda': 0.46,
              'verbose': -1, 'seed': 42, 'device': 'GPU'}
# Run cross-validation
cv_results = lgb.cv(
    params=lgb_params,
    train_set=lgb_train,
    num_boost_round=2000,
    nfold=5,
    stratified=False,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Display last few CV results
print(cv_results['valid rmse-mean'][-5:])

# Extract best boosting round
best_round = len(cv_results['valid rmse-mean'])
best_rmse = cv_results['valid rmse-mean'][-1]
print(f"Best round: {best_round}, Best CV RMSE: {best_rmse:.5f}")


# Best iteration (1-based index)
best_iteration = len(cv_results['valid rmse-mean'])
lgb_params["n_estimators"] = best_iteration


# Prepare training data
X_train = df.drop(columns=target)
y_train = df[target]

# Train XGBoost model
model = LGBMRegressor(**lgb_params, enable_categorical=True)
model.fit(X_train, y_train)

# Predict on test set
pred = model.predict(df_test.drop(columns = "id"))

# Prepare submission
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)

