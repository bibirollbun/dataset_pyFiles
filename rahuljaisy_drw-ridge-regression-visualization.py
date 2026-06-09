# Suppress warnings for clean output
import warnings; warnings.filterwarnings("ignore")

# System memory management
import gc

# Core data handling libraries
import pandas as pd
import numpy as np

# Feature selection and model components
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Visualization and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Display utility for output formatting
from IPython.display import display



# Load raw training data from Parquet format
df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')

# Replace infinite values with zeros to prevent computation errors
df.replace([np.inf, -np.inf], 0, inplace=True)

# Segment data for training and validation
_tr = df.iloc[:200_000]
_te = df.iloc[500_000:550_000]

# Split features and target labels
X_train = _tr.drop(columns=['label'])
y_train = _tr['label']
X_test  = _te.drop(columns=['label'])
y_test  = _te['label']

# Explicitly delete unused dataframes and trigger garbage collection
del df, _tr, _te
gc.collect()



pipe = Pipeline([
    ('select', SelectKBest(f_regression, k=200)),     # Feature selection: retain top 200 based on correlation
    ('scale', StandardScaler()),                      # Feature scaling: zero-mean, unit variance normalization
    ('ridge', Ridge(alpha=1.0, random_state=42))      # Ridge Regression: L2-regularized linear model
])

# Train the pipeline end to end
pipe.fit(X_train, y_train)



# Predict target values for the test set
y_pred = pipe.predict(X_test)

# Compute evaluation metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
corr, pval = pearsonr(y_test, y_pred)

# Display evaluation results
results = pd.DataFrame([{
    'Model': 'Ridge+SelectK',
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'Pearson Corr': corr,
    'P-value': pval
}])

display(results)



# Select a random sample of 100 points for plotting
idx = np.random.RandomState(42).choice(len(y_test), 100, replace=False)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[idx], marker='o', label='Actual')
plt.plot(y_pred[idx], marker='x', label='Predicted')
plt.title('Actual vs Predicted (100 samples)')
plt.legend()
plt.tight_layout()
plt.show()



# Load the test dataset for final prediction
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')

# Drop target column and replace infinities
test_df.drop(columns=['label'], inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# Make predictions
preds = pipe.predict(test_df)

# Prepare submission file
sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sub['prediction'] = preds

# Save to CSV
sub.to_csv('ridge_selectk_submission.csv', index=False)



# 1. Residual Distribution

residuals = y_test.values - y_pred
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.xlabel("Residual (Actual − Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Predicted vs Actual Scatter

plt.figure(figsize=(6, 6))
plt.scatter(y_test.values, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.title("Predicted vs Actual")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.tight_layout()
plt.show()

# 3. Top-10 Ridge Coefficients (Post-Selection)

# retrieve selected feature names
selected_feats = X_train.columns[pipe.named_steps['select'].get_support()]
# retrieve corresponding coefficients
coefs = pipe.named_steps['ridge'].coef_
# identify top 10 by absolute value
idx_top = np.argsort(np.abs(coefs))[-10:]
plt.figure(figsize=(6, 4))
plt.barh(selected_feats[idx_top], coefs[idx_top])
plt.title("Top-10 Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

