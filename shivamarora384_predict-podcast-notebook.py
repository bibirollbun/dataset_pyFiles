






# ğŸ“š Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ğŸ“‚ Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# ğŸ�¯ Target - NO LOG TRANSFORM (try raw minutes first)
target = "Listening_Time_minutes"
train.dropna(subset=[target], inplace=True)
y = train[target].values  # Use raw minutes

# ğŸ› ï¸� Simple Feature Engineering (avoid leakage)
features = [col for col in train.columns if col not in ['id', target]]
numeric_cols = train[features].select_dtypes(include=np.number).columns

# Only keep features available in real-world scenarios
safe_features = []
for col in features:
    if col in test.columns:  # Only use features present in test data
        safe_features.append(col)
features = safe_features

# âš¡ Fast Preprocessing
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
cat_cols = train[features].select_dtypes(include='object').columns
if len(cat_cols) > 0:
    train[cat_cols] = encoder.fit_transform(train[cat_cols])
    test[cat_cols] = encoder.transform(test[cat_cols])

# Fill NA (use train stats only)
train.fillna(train[numeric_cols].median(), inplace=True)
test.fillna(train[numeric_cols].median(), inplace=True)

# ğŸ�† Robust Model Training
model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

# âœ… K-Fold Validation (to check real performance)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(train):
    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    scores.append(rmse)
    print(f"Fold RMSE: {rmse:.2f} minutes")

print(f"\nğŸ�† Average Validation RMSE: {np.mean(scores):.2f} minutes")

# ğŸ”® Predict
test_preds = model.predict(test[features])

# ğŸ“� Save submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission[target] = test_preds
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission created!")




