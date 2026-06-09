# Import required files and libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


# Read data files
train_df=pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test_df=pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

train_df.head()


# Check if we need to handle missing values
missing_values_count = train_df.isnull().sum()
filtered_missing = missing_values_count[missing_values_count > 0]
filtered_missing.head()

# Delete columns will a lot of missing values
train_df = train_df.drop(columns=["sale_nbr","subdivision","submarket"])
test_df = test_df.drop(columns=["sale_nbr","subdivision","submarket"])


# Adding new data points
train_df['sale_date'] = pd.to_datetime(train_df['sale_date'])
train_df['sale_year'] = train_df['sale_date'].dt.year
train_df['sale_month'] = train_df['sale_date'].dt.month
train_df['age'] = train_df['sale_year'] - train_df['year_built']
train_df['renovated'] = np.where(train_df['year_reno'] > 0, 1, 0)
train_df['years_since_reno'] = np.where(train_df['renovated'], train_df['sale_year'] - train_df['year_reno'], 0)
train_df['total_baths'] = train_df['bath_full'] + 0.75*train_df['bath_3qtr'] + 0.5*train_df['bath_half']
train_df['total_value'] = train_df['land_val'] + train_df['imp_val']
train_df['living_area'] = train_df['sqft'] + train_df['sqft_fbsmt']

test_df['sale_date'] = pd.to_datetime(test_df['sale_date'])
test_df['sale_year'] = test_df['sale_date'].dt.year
test_df['sale_month'] = test_df['sale_date'].dt.month
test_df['age'] = test_df['sale_year'] - test_df['year_built']
test_df['renovated'] = np.where(test_df['year_reno'] > 0, 1, 0)
test_df['years_since_reno'] = np.where(test_df['renovated'], test_df['sale_year'] - test_df['year_reno'], 0)
test_df['total_baths'] = test_df['bath_full'] + 0.75*test_df['bath_3qtr'] + 0.5*test_df['bath_half']
test_df['total_value'] = test_df['land_val'] + test_df['imp_val']
test_df['living_area'] = test_df['sqft'] + test_df['sqft_fbsmt']

train_df = train_df.drop(columns=['sale_date', 'id'])
test_df = test_df.drop(columns=['sale_date', 'id'])


# Use ordinal encoder to encode category columns
cat_cols_train = train_df.select_dtypes(include=['object']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_df[cat_cols_train] = ordinal_encoder.fit_transform(train_df[cat_cols_train].astype(str))
test_df[cat_cols_train] = ordinal_encoder.transform(test_df[cat_cols_train].astype(str))


# Derive X and Y values
X = train_df.drop(columns=['sale_price'])
y = train_df['sale_price']


# correlation heatmap
correlation_matrix = train_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# Calculate winkler score
def winkler_score(y_true, lower, upper, alpha=.1):
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)
    return width + (2/alpha) * (below + above)


# Train model
def train_model(model, X_train, y_train, X_val):
    model.fit(X_train, y_train)
    lower = model["lower"].predict(X_val)
    upper = model["upper"].predict(X_val)
    return lower, upper


# Model configurations
models = {
    "lower": lgb.LGBMRegressor(
        objective="quantile", 
        alpha=0.05,
        device="gpu",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        random_state=42
    ),
    "upper": lgb.LGBMRegressor(
        objective="quantile", 
        alpha=0.95,
        device="gpu",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        random_state=42
    )
}


# Finding optimal weight for model training

# Initialize storage for OOF predictions
oof_lowers = np.zeros(len(X))
oof_uppers = np.zeros(len(X))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X, y)), total=5, desc="Folds"):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
    # Train lower quantile model
    lower_model = clone(models["lower"])
    lower_model.fit(X_train, y_train)
    lower_pred = lower_model.predict(X_val)
        
    # Train upper quantile model
    upper_model = clone(models["upper"])
    upper_model.fit(X_train, y_train)
    upper_pred = upper_model.predict(X_val)
        
    oof_lowers[val_idx] = lower_pred
    oof_uppers[val_idx] = upper_pred

print("\nModel Performance Evaluation:", oof_lowers, oof_uppers)

# Ensure valid intervals
lower = np.minimum(oof_lowers, oof_uppers)
upper = np.maximum(oof_lowers, oof_uppers)
    
# Calculate MWIS
wis = winkler_score(y, lower, upper)
mwis = np.mean(wis)
model_score = mwis
    
# Calculate coverage
coverage = np.mean((y >= lower) & (y <= upper)) * 100
print(f"Result: MWIS = {mwis:.2f}, Coverage = {coverage:.2f}%")

# Hill Climbing Optimization
print("\nStarting Hill Climbing Optimization...")
current_weights = 0.4
best_score = 100000000000000000.0
    
# Calculate initial combined score
combined_lower = current_weights * oof_lowers
combined_upper = current_weights * oof_uppers
current_score = np.mean(winkler_score(y, combined_lower, combined_upper))
    
print(f"Initial MWIS: {current_score:.4f}")
    
# Optimization loop
for step in tqdm(range(100), desc="Hill Climbing"):
    candidate_weights = current_weights
        
    # Generate candidate weights
    perturbation = np.random.dirichlet([9])[0] - .9
    candidate_weights = candidate_weights + .1 * perturbation
    candidate_weights = np.maximum(candidate_weights, 0)

    # Calculate combined predictions
    combined_lower = candidate_weights * oof_lowers
    combined_upper = candidate_weights * oof_uppers
        
    # Calculate MWIS
    candidate_score = np.mean(winkler_score(y, combined_lower, combined_upper))

    # Update if improvement
    if candidate_score < best_score:
        best_score = candidate_score
        best_weights = candidate_weights
        current_weights = candidate_weights
        print(f"Step {step}: New best MWIS = {best_score:.4f}, Weights = {best_weights}")
# 340027.5577


# Final model training
print("\nTraining Final Models...")
test_preds = {}

# Train lower quantile model
lower_model = models["lower"]
lower_model.fit(X, y)
test_preds["lower"] = lower_model.predict(test_df)

# Train upper quantile model
upper_model = models["upper"]
upper_model.fit(X, y)
test_preds["upper"] = upper_model.predict(test_df)

# Combine test predictions
final_lower = best_weights * test_preds["lower"]
final_upper = best_weights * test_preds["upper"]

# Ensure valid intervals
final_lower, final_upper = np.minimum(final_lower, final_upper), np.maximum(final_lower, final_upper)
final_lower = np.maximum(final_lower, 0)


test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

test_ids = test["id"]
submission = pd.DataFrame({
    "id": test_ids,
    "pi_lower": final_lower,
    "pi_upper": final_upper
})
submission.to_csv("submission.csv", index=False)
print("Submission saved successfully")

submission = pd.read_csv("submission.csv")
submission.head(5)

