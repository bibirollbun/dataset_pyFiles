import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Merge train and extra data
df = pd.concat([train, extra], ignore_index=True)
test_ids = test["id"]
df.drop(columns=["id"], inplace=True, errors='ignore')
test.drop(columns=["id"], inplace=True, errors='ignore')


# Define columns
cat_cols = ["Brand", "Material", "Size", "Style", "Color", "Laptop Compartment", "Waterproof"]
num_cols = ["Compartments", "Weight Capacity (kg)"]


# Convert boolean to string
for col in ["Laptop Compartment", "Waterproof"]:
    df[col] = df[col].astype(str)
    test[col] = test[col].astype(str)


# Create simplified fingerprint (faster than full fingerprint)
# Use only the most important features
df['fingerprint'] = df['Brand'].astype(str) + '_' + df['Style'].astype(str) + '_' + df['Size'].astype(str)
test['fingerprint'] = test['Brand'].astype(str) + '_' + test['Style'].astype(str) + '_' + test['Size'].astype(str)


# Feature engineering - traditional
df['Brand_Style'] = df['Brand'].astype(str) + '_' + df['Style'].astype(str)
test['Brand_Style'] = test['Brand'].astype(str) + '_' + test['Style'].astype(str)
cat_cols.append('Brand_Style')


df['Material_Size'] = df['Material'].astype(str) + '_' + df['Size'].astype(str)
test['Material_Size'] = test['Material'].astype(str) + '_' + test['Size'].astype(str)
cat_cols.append('Material_Size')

df['Log_Weight_Capacity'] = np.log1p(df['Weight Capacity (kg)'])
test['Log_Weight_Capacity'] = np.log1p(test['Weight Capacity (kg)'])
num_cols.append('Log_Weight_Capacity')


# Instead of KNN, use group statistics which is much faster
# Group statistics for fingerprint
fingerprint_stats = df.groupby('fingerprint')['Price'].agg(['mean', 'std', 'count']).reset_index()
fingerprint_stats.columns = ['fingerprint', 'fingerprint_mean_price', 'fingerprint_std_price', 'fingerprint_count']

# Merge back to the datasets
df = df.merge(fingerprint_stats, on='fingerprint', how='left')
test = test.merge(fingerprint_stats, on='fingerprint', how='left')


# Fill NaN values in case there's no exact match
df['fingerprint_mean_price'].fillna(df['Price'].mean(), inplace=True)
df['fingerprint_std_price'].fillna(df['Price'].std(), inplace=True)
df['fingerprint_count'].fillna(0, inplace=True)

test['fingerprint_mean_price'].fillna(df['Price'].mean(), inplace=True)
test['fingerprint_std_price'].fillna(df['Price'].std(), inplace=True)
test['fingerprint_count'].fillna(0, inplace=True)


# Create more groupby statistics for other combinations
# These can capture the patterns without expensive KNN computation
for combo in [['Brand', 'Style'], ['Material', 'Size'], ['Color', 'Size']]:
    combo_name = '_'.join(combo)
    stats = df.groupby(combo)['Price'].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    # Rename columns
    new_cols = [f"{combo_name}_mean", f"{combo_name}_median", f"{combo_name}_std", f"{combo_name}_count"]
    stats.columns = combo + new_cols
    
    # Merge with training and test data
    df = df.merge(stats, on=combo, how='left')
    test = test.merge(stats, on=combo, how='left')
    
    # Add features to numerical columns
    num_cols.extend(new_cols)


# Handle missing values
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])


num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])


# Encode categorical variables
def encode_categorical(df, train_df=None):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        if train_df is not None:
            le.fit(train_df[col].dropna().astype(str))
        else:
            le.fit(df[col].dropna().astype(str))
        df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        encoders[col] = le
    return df, encoders


df, encoders = encode_categorical(df)
test, _ = encode_categorical(test, train_df=df)


# Add fingerprint mean price as a direct feature - this is our strongest signal
df['target_proxy'] = df['fingerprint_mean_price']
test['target_proxy'] = test['fingerprint_mean_price']
num_cols.append('target_proxy')


# Prepare data
X = df.drop(columns=["Price", "fingerprint"])
y = df["Price"]
X_test = test.drop(columns=["fingerprint"])


# LightGBM model with improved parameters for speed
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1000,  # Reduced number of estimators
    "learning_rate": 0.03,  # Increased learning rate for faster convergence
    "num_leaves": 50,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": 42,
    "force_col_wise": True,  # Improve speed
}


# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))
val_rmse = []


for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train LightGBM
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric="rmse", 
        callbacks=[lgb.early_stopping(30)]  # Reduced early stopping rounds
    )
    
    # Make predictions on validation set
    val_preds = model.predict(X_val)
    val_rmse.append(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    # Predictions on test
    test_preds += model.predict(X_test) / kf.n_splits

print(f"Mean CV RMSE: {np.mean(val_rmse):.4f}")


# Post-process predictions
test_preds = np.clip(test_preds, y.min(), y.max())

# Create submission
submission = pd.DataFrame({"id": test_ids, "Price": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




