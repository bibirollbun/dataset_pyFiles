import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import optuna
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import xgboost as xgb


train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


train_df.head()


test_df.head()


print(list(train_df.columns))


test_df.columns


# Summary Statistics
train_df["label"].describe()


# Check Null Values
train_df.isnull().sum().sum()





# Check if inf or inf is exist
print("Any Infs?", np.isinf(train_df.to_numpy()).sum(), np.isinf(test_df.to_numpy()).sum())


count_inf = pd.DataFrame({
    "train_inf": np.isinf(train_df).sum(),
    "test_inf" : np.isinf(test_df).sum()
})


count_inf = count_inf[(count_inf["train_inf"] > 0) | (count_inf["test_inf"] > 0)]


print("Columns That contain Inf Values..... ")
print(f"There are ==> {len(count_inf)} Columns Contain inf value")
count_inf





# Drop unimportant Woman

X = train_df.drop(columns= ["label"] + list(count_inf.index))
y = train_df["label"]


X_test = test_df.drop(columns= ["label"] + list(count_inf.index) )


scaler = StandardScaler()


for col in X.columns:
    X[col] = scaler.fit_transform(X[[col]])


X


params = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": False,
    'subsample': 0.8, 
    'single_precision_histogram': True
}


X_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


batch_size = 20000  # Adjust based on your RAM (50k works for 16GB RAM)
num_rounds_per_batch = 10  # Trees added per batch
model = None


import gc

for i in range(0, len(X_train), batch_size):
    # Load data batch
    X_batch = X_train[i:i+batch_size].astype(np.float32)
    y_batch = y_train[i:i+batch_size].astype(np.float32)
    
    # Explicit garbage collection
    gc.collect()
    
    # Train with warm start
    dmatrix = xgb.DMatrix(X_batch, y_batch)
    model = xgb.train(
        params,
        dtrain=dmatrix,
        num_boost_round=num_rounds_per_batch,
        xgb_model=model,  # Warm start magic here
        verbose_eval=False
    )
    
    # Memory cleanup
    del X_batch, y_batch, dmatrix
    gc.collect()
    
    print(f"Processed {min(i+batch_size, len(X_train))}/{len(X_train)} rows")

# 3. Memory-Safe Prediction
def predict_in_batches(model, X, batch_size=50000):
    predictions = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size].astype(np.float32)
        dmatrix = xgb.DMatrix(X_batch)
        batch_preds = model.predict(dmatrix)
        predictions.append(batch_preds)
        
        # Cleanup
        del X_batch, dmatrix
        gc.collect()
        
    return np.concatenate(predictions)


import shap

# Sample 10,000 rows
sample_idx = np.random.choice(len(X_train), 10000, replace=False)
X_sample = X_train.iloc[sample_idx]

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# SHAP summary plot
shap.summary_plot(shap_values, X_sample, max_display=30)

# Top features by mean absolute SHAP value
shap_df = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

top_shap_features = shap_df.head(100)['feature'].values



top_shap_features


X_train = X_train[top_shap_features]

X_train.shape[1]

# X_train = X_train.loc[:, (X_train != 0).any(axis=0)]


x_test = x_test[top_shap_features]

x_test.shape[1]


y_pred = predict_in_batches(model, x_test)


corr, _ = pearsonr(y_test, y_pred)
print(f"\nPearson Correlation: {corr:.4f}")


# Switch to LightGBM (often better for feature-rich datasets)
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    boosting_type='goss',  # Better for high dimensionality
    num_leaves=127,        # More complex relationships
    max_depth=-1,          # Unlimited depth
    n_estimators=2000,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    importance_type='gain',
    device='cpu'           # Enable GPU acceleration
)

# Train in batches
for i in range(0, len(X_train), 50000):
    model.fit(
        X_train[i:i+50000], 
        y_train[i:i+50000],
        init_model=model if i > 0 else None
    )


def predict_in_batches(model, X, batch_size=50000, verbose=True):
    """Make predictions in memory-safe batches"""
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size].astype(np.float32)
        batch_preds = model.predict(batch, num_iteration=model.best_iteration_)
        predictions.append(batch_preds)
        
        if verbose:
            print(f"Predicted batch {i//batch_size + 1}/{(len(X)-1)//batch_size + 1}")
    
    return np.concatenate(predictions)

# Usage
y_pred = predict_in_batches(model, x_test, batch_size=50000)


corr, _ = pearsonr(y_test, y_pred)
print(f"\nPearson Correlation: {corr:.4f}")


X_test_scaled = scaler.fit_transform(X_test)

X_test.shape
# X_train.shape
# X_test = X_test.drop(columns=X_test.select_dtypes(include=['object', 'category']).columns)


y_pred_1 = predict_in_batches(model, X_test, batch_size=50000)


# Generate row IDs (starting at 1)
row_ids = range(1, len(y_pred_1) + 1)

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': row_ids,
    'prediction': y_pred_1
})

# Save to CSV without index
submission.to_csv('submission.csv', index=False)

print("Submission file saved: submission.csv")





df_test = test_df.drop(columns= ["label"] + list(count_inf.index))


df_test.columns


for col in df_test.columns:
    df_test[col] = scaler.fit_transform(df_test[[col]])





y_pred = predict_in_batches(model, _test, batch_size=50000)











params = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": False,
}


model = XGBRegressor(**params)


pca = PCA(n_components=100)


X_train = pca.fit_transform(X_train)


model.fit




