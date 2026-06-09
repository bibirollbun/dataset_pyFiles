# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Memory management
import gc

# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Display utility
from IPython.display import display


# Load training data
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet', engine='pyarrow')
train_df.replace([np.inf, -np.inf], 0, inplace=True)

# Subset data for efficiency
_train_df = train_df.iloc[:200_000]
_test_df = train_df.iloc[500_000:550_000]

x_train = _train_df.drop(columns=['label'])
y_train = _train_df['label']

x_test = _test_df.drop(columns=['label'])
y_test = _test_df['label']

del _train_df, _test_df, train_df
gc.collect()



# Define XGBoost model
XGBR = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method='hist',
    max_bin=64,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

# Train with early stopping
eval_set = [(x_test, y_test)]
XGBR.fit(
    x_train, y_train,
    eval_metric='rmse',
    eval_set=eval_set,
    early_stopping_rounds=10,
    verbose=False
)

# SHAP Interpretability 
import shap
explainer = shap.Explainer(XGBR)
shap_values = explainer(x_test[:100])
shap.plots.beeswarm(shap_values)


# Prediction and metric computation
y_pred = XGBR.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
corr_coef, p_value = pearsonr(y_test, y_pred)

results = pd.DataFrame({
    'Model': ['XGB-Fast'],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'Pearson Corr': [corr_coef],
    'P-value': [p_value],
})

display(results)



# Random sample visualization
idx = np.random.RandomState(42).choice(len(y_test), size=100, replace=False)

plt.figure(figsize=(12, 6))
plt.plot(np.array(y_test)[idx], label='Actual', marker='o')
plt.plot(y_pred[idx], label='Predicted', marker='x')
plt.title("Actual vs Predicted (100 Random Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Target")
plt.legend()
plt.tight_layout()
plt.show()



# Load and clean test data
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')
test_df.drop(columns=['label'], inplace=True)
test_df.replace([np.inf, -np.inf], 0, inplace=True)

# Predict and prepare submission
preds = XGBR.predict(test_df)

sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = preds
sample_submission.to_csv('sample_submission.csv', index=False)


