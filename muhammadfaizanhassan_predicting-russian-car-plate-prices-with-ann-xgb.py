import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import sys

# adjust path for supplemental files
sys.path.append('/kaggle/input/russian-car-plates-prices-prediction')
from supplemental_english import REGION_CODES, GOVERNMENT_CODES

# for inline plotting
%matplotlib inline


BASE_PATH = '/kaggle/input/russian-car-plates-prices-prediction/'
train = pd.read_csv(BASE_PATH + 'train.csv')
test  = pd.read_csv(BASE_PATH + 'test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


def extract_plate_features(df):
    df = df.copy()
    df['plate_str'] = df['plate'].astype(str)
    # region code: last 2–3 digits
    df['region_code'] = df['plate_str'].str.extract(r'(\d{2,3})$')[0]
    # prefix letters: first 1–3 characters
    df['prefix'] = df['plate_str'].str.extract(r'^([A-ZА-Я]{1,3})')[0]
    # numeric block: three digits
    df['number'] = df['plate_str'].str.extract(r'([0-9]{3})')[0].astype(int)
    # date parts
    df['date'] = pd.to_datetime(df['date'])
    df['year']    = df['date'].dt.year
    df['month']   = df['date'].dt.month
    df['day']     = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    # government‐plate flag
    df['is_gov'] = df['prefix'].isin(GOVERNMENT_CODES).astype(int)
    # numeric region code directly
    df['region_num'] = pd.to_numeric(df['region_code'], errors='coerce').fillna(0).astype(int)
    return df

train = extract_plate_features(train)
test  = extract_plate_features(test)


le_pref = LabelEncoder()
le_reg  = LabelEncoder()

train['pref_enc'] = le_pref.fit_transform(train['prefix'])
test ['pref_enc'] = le_pref.transform(test['prefix'])

train['reg_enc'] = le_reg.fit_transform(train['region_code'])
test ['reg_enc'] = le_reg.transform(test['region_code'])


FEATURES = [
    'pref_enc','number','reg_enc',
    'region_num','is_gov',
    'year','month','day','weekday'
]
X = train[FEATURES]
y = train['price']
X_test = test[FEATURES]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train MLP (ANN) with increased layers and neurons
mlp = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation='relu',
    learning_rate_init=1e-3,
    max_iter=1000,
    alpha=0.0001,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_tr, y_tr)


# Plot the MLP loss curve
plt.figure(figsize=(6,4))
plt.plot(mlp.loss_curve_, label='Train Loss')
plt.title('MLP Loss Curve with Increased Layers & Neurons')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_absolute_error

# Validate MLP performance
mlp_val_preds = mlp.predict(X_val)
mlp_mae = mean_absolute_error(y_val, mlp_val_preds)
print(f"MLP Validation MAE: {mlp_mae:.2f}")


# Train XGBoost with advanced hyperparameters
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,  # More trees for higher complexity
    learning_rate=0.005,  # Smaller learning rate for better precision
    max_depth=12,  # Increased depth for more complex trees
    min_child_weight=10,  # Reduces overfitting by controlling complexity
    gamma=0.1,  # Minimum loss reduction for further splits
    colsample_bytree=0.85,  # Sampling of features
    subsample=0.85,  # Sampling of training data
    reg_alpha=0.5,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    random_state=42
)
xgb_model.fit(
    X_tr, y_tr,
    early_stopping_rounds=50,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# Plot feature importance for XGBoost
plt.figure(figsize=(6,4))
plt.barh(FEATURES, xgb_model.feature_importances_)
plt.title('XGBoost Feature Importance with Increased Complexity')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


# Validate XGBoost performance
xgb_val_preds = xgb_model.predict(X_val)
xgb_mae = mean_absolute_error(y_val, xgb_val_preds)
print(f"XGBoost Validation MAE: {xgb_mae:.2f}")


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression  # Also needed if not already imported


# Ensemble using Stacking Regressor (combining ANN and XGBoost)
stacking_model = StackingRegressor(
    estimators=[('mlp', mlp), ('xgb', xgb_model)],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_tr, y_tr)

# Predict using the stacked model
stacking_preds = stacking_model.predict(X_test)



print(stacking_preds)


# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'price': stacking_preds
})
display(submission.head())

# Save submission
submission.to_csv('submission.csv', index=False)
print("Saved -> final_submission.csv")

