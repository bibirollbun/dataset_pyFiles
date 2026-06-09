# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Memory management
import gc

# Data manipulation
import pandas as pd
import numpy as np

# Modeling
from sklearn.linear_model import LinearRegression

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Visualization
import matplotlib.pyplot as plt

# Display
from IPython.display import display


# Load data
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
train_df.replace([np.inf, -np.inf], 0, inplace=True)

# Subset for training and testing
_train_df = train_df.iloc[:200_000]
_test_df = train_df.iloc[500_000:550_000]

x_train = _train_df.drop(columns=['label'])
y_train = _train_df['label']

x_test = _test_df.drop(columns=['label'])
y_test = _test_df['label']

# Cleanup
del _train_df, _test_df, train_df
gc.collect()


# Define OLS model
OLS = LinearRegression(n_jobs=-1)

# Train
OLS.fit(x_train, y_train)


# Predict
y_pred = OLS.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
corr_coef, p_val = pearsonr(y_test, y_pred)

results = pd.DataFrame({
    'Model': ['Linear OLS'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'Pearson Corr': [corr_coef],
    'P-value': [p_val],
})

display(results)


idx = np.random.RandomState(42).choice(len(y_test), size=100, replace=False)

plt.figure(figsize=(12, 6))
plt.plot(np.array(y_test)[idx], label='Actual', marker='o')
plt.plot(y_pred[idx], label='Predicted (OLS)', marker='x')
plt.title("OLS: Actual vs Predicted (100 Random Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Target")
plt.legend()
plt.tight_layout()
plt.show()


# Load test data
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')

# Drop label column if exists
test_df.drop(columns=['label'], errors='ignore', inplace=True)

# Replace infinite values
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# Align test features with training features
test_df = test_df[x_train.columns]  # Ensures feature consistency

# Make predictions
test_preds = OLS.predict(test_df)

# Load and prepare submission
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = test_preds

# Save submission file
sample_submission.to_csv('sample_submission.csv', index=False)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np

# 1. Standardize features (critical for PCA stability)
_scaler = StandardScaler()
X_tr = _scaler.fit_transform(x_train.values)
X_te = _scaler.transform(x_test.values)

# 2. Automated PCA: retain 95% explained variance → minimal yet informative subspace
_pca = PCA(n_components=0.95, svd_solver='full', random_state=0)
X_tr_pca = _pca.fit_transform(X_tr)
X_te_pca = _pca.transform(X_te)

# 3. Exponential weighting: emphasize most recent observations (decay tuned via heuristic)
_decay = 1e-4
_idx  = np.arange(len(y_train))
_wgts = np.exp(-_decay * (_idx.max() - _idx))  # newer rows receive higher weight

# 4. Fit weighted OLS on reduced subspace
_model = LinearRegression(n_jobs=-1)
_model.fit(X_tr_pca, y_train, sample_weight=_wgts)

# 5. Validation: compute Pearson correlation on held-out test slice
_y_pred = _model.predict(X_te_pca)
_corr, _ = pearsonr(y_test, _y_pred)
print(f'Enhanced PCA-OLS Pearson Corr: {_corr:.6f}')

# 6. Autonomous Submission Logic: if improvement, overwrite predictions
if _corr > corr_coef:
    # scale & project full test set
    X_full_te = _scaler.transform(test_df.values)
    X_full_pca = _pca.transform(X_full_te)
    sample_submission['prediction'] = _model.predict(X_full_pca)
    sample_submission.to_csv('submission_enhanced.csv', index=False)
    print('↳ submission_enhanced.csv generated (better than baseline)')
else:
    print('↳ Baseline remains superior; no submission overwrite.')



