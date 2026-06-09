# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip', index_col='Id')
df.head()


df['Response'].value_counts()


#finding % missing value

df_nan = (df.isnull().sum().sort_values(ascending=False)/df.shape[0])*100

df_nan


df_nan_filter = df_nan[df_nan>0]

df_nan_filter


#removed columns which is higher thagn 90%
removed_columns = df_nan_filter[df_nan_filter>90].index


df.drop(removed_columns,axis=1,inplace=True)


nan_columns = df_nan_filter[df_nan_filter<=90].index
nan_columns = list(nan_columns)


nan_columns.append('Response')
df_1= df[nan_columns]


df_1.describe()


#Medical_History_1: have min and max has high distance, define frequency number to be used for null. Choose '0' for it

df_1['Medical_History_1'] = df_1['Medical_History_1'].fillna(1)
#df_1['Medical_History_1'].fillna(1, inplace=True)] #2nd method


for col in df_1.columns:
  if col != 'Medical_History_1' and col != 'Response':
    df_1[col] = df_1[col].fillna(df_1[col].mean())


df_1.isnull().sum()


#features importance
import matplotlib.pyplot as plt
import seaborn as sns

corrleation_matrix = df_1.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corrleation_matrix, annot=True)
plt.show()


nan_columns.remove('Response')
nan_columns


data_cleaned = df.drop(nan_columns,axis=1) #removed all missing value



data_info = data_cleaned.select_dtypes(include='object')
data_info['Product_Info_2'].unique()


#using label Encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_cleaned['Product_Info_2'] = le.fit_transform(data_cleaned['Product_Info_2'])


data_corr = data_cleaned.corr()


# Filter features based on correlation with 'Response'
filtered_features = data_corr[
    (data_corr['Response'] > 0.1) | (data_corr['Response'] < -0.1)
]['Response'].index

# Create a new DataFrame with only the selected features
filtered_df = df[filtered_features]


filtered_df.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the filtered DataFrame
correlation_matrix = filtered_df.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Important Features')
plt.show()


filtered_df.to_csv('filtered_data.csv',index=False) #save copy


df=filtered_df


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=UserWarning) # Suppress some common warnings

# Define features (all columns except Response)
features = [col for col in df.columns if col not in ['Response']]
X = df[features]
y = df['Response'] # Target variable (should be integers 1-8)


print(f"Features being used ({len(features)}): {features}")
print(f"Target variable: Response")
print(f"Training data shape: {X.shape}")
# print(f"Test data shape: {X_test.shape}") # Uncomment if using test_df


# --- 2. QWK Metric Function ---
def quadratic_weighted_kappa(y_true, y_pred):
    """Calculates QWK score using sklearn"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# --- 3. Threshold Optimization ---
class OptimizedRounder:
    """
    An optimizer class that finds the best thresholds for converting
    continuous predictions into discrete integer classes (1-8)
    to maximize the Quadratic Weighted Kappa score.
    """
    def __init__(self):
        self.coef_ = None  # Initialize coef_ to None instead of 0
        self.init_thresholds_ = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # Initial guesses

    def _kappa_loss(self, thresholds, predictions, true_values):
        """Objective function to minimize (1 - QWK)"""
        preds_int = self._apply_thresholds(predictions, thresholds)
        return -quadratic_weighted_kappa(true_values, preds_int) # Minimize negative kappa

    def _apply_thresholds(self, predictions, thresholds):
        """Applies thresholds to continuous predictions"""
        # Ensure thresholds are sorted
        sorted_thresholds = np.sort(thresholds)
        preds_int = pd.cut(predictions,
                           bins=[-np.inf] + list(sorted_thresholds) + [np.inf],
                           labels=list(range(1, len(sorted_thresholds) + 2)), # Labels 1 to 8
                           right=False) # Ensure intervals are [min, max)
        return preds_int.astype(int)


    def fit(self, predictions, true_values):
        """Find the optimal thresholds"""
        # Make sure predictions and true_values are numpy arrays
        predictions = np.array(predictions)
        true_values = np.array(true_values)

        loss_partial = partial(self._kappa_loss, predictions=predictions, true_values=true_values)

        # Adjust initial thresholds slightly based on prediction distribution
        # This can sometimes help the optimizer find a better solution
        pred_quantiles = np.quantile(predictions, np.linspace(0, 1, 10))[1:-1] # Roughly percentiles
        initial_guess = np.mean([self.init_thresholds_, pred_quantiles[:7]], axis=0)


        # Use scipy.optimize.minimize to find best thresholds
        # 'Nelder-Mead' is often robust for this type of problem
        result = minimize(loss_partial,
                          initial_guess,
                          method='Nelder-Mead', # or 'L-BFGS-B', 'SLSQP'
                          options={'maxiter': 1000, 'fatol': 1e-7} # Increase maxiter if needed
                         )

        if result.success:
            self.coef_ = np.sort(result.x)
            print(f"Optimized Thresholds: {self.coef_}")
        else:
            # Fallback to initial thresholds if optimization fails
            self.coef_ = np.sort(self.init_thresholds_)
            print(f"Optimization failed. Using initial thresholds: {self.coef_}")
            print(f"Optimization result message: {result.message}")


    def predict(self, predictions):
        """Apply the optimized thresholds to new predictions"""
        if self.coef_ is None:  # Check if coef_ is None instead of 0
            raise ValueError("Optimizer has not been fitted yet.")
        return self._apply_thresholds(predictions, self.coef_)


# --- 4. Cross-Validation and Model Training ---
N_SPLITS = 5 # Number of folds for cross-validation
SEED = 42   # Random seed for reproducibility

# Configure XGBoost parameters
# **NOTE:** These parameters are just examples. You MUST tune them!
# Use libraries like Optuna or Scikit-learn's GridSearchCV/RandomizedSearchCV
xgb_params = {
    'objective': 'reg:squarederror', # Use regression objective
    'eval_metric': 'rmse',          # Evaluate with RMSE during training
    'eta': 0.05,                   # Learning rate
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'gamma': 0.1,
    'lambda': 1,                   # L2 regularization
    'alpha': 0,                    # L1 regularization
    'seed': SEED
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(df)) # To store Out-Of-Fold predictions (continuous)
# test_preds = np.zeros(len(X_test)) # To store test predictions (continuous) # Uncomment if using test_df
fold_scores = []
models = [] # Store trained models if needed

print(f"\nStarting {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")

    # Split data for this fold
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Create XGBoost DMatrix (efficient data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    # dtest = xgb.DMatrix(X_test) # Uncomment if using test_df

    # Train the XGBoost model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,        # Max number of boosting rounds
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,    # Stop if validation score doesn't improve
        verbose_eval=100             # Print evaluation results every 100 rounds
    )

    # --- Predict on validation set (continuous) ---
    val_preds_cont = model.predict(dval, iteration_range=(0, model.best_iteration))
    oof_preds[val_idx] = val_preds_cont # Store OOF predictions

    
    # --- Evaluate fold performance (using thresholds optimized ONLY on this fold's val data) ---
    # This is just for monitoring fold performance, final thresholds use all OOF data
    temp_rounder = OptimizedRounder()
    temp_rounder.fit(val_preds_cont, y_val)
    val_preds_int = temp_rounder.predict(val_preds_cont)
    fold_qwk = quadratic_weighted_kappa(y_val, val_preds_int)
    fold_scores.append(fold_qwk)
    print(f"Fold {fold+1} QWK: {fold_qwk:.4f}")

    models.append(model) # Optional: Store the model

print(f"\n--- Cross-Validation Finished ---")
print(f"Average Fold QWK: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")


# --- 5. Final Threshold Optimization and OOF Score ---
print("\nOptimizing thresholds on combined Out-Of-Fold predictions...")
final_rounder = OptimizedRounder()
final_rounder.fit(oof_preds, y) # Use all OOF predictions and true labels

# Calculate overall OOF QWK score using the final optimized thresholds
oof_preds_int = final_rounder.predict(oof_preds)
overall_oof_qwk = quadratic_weighted_kappa(y, oof_preds_int)
print(f"\nOverall OOF QWK Score (using final thresholds): {overall_oof_qwk:.4f}")
print(f"Final Optimized Thresholds: {final_rounder.coef_}")


print("\n--- Process Complete ---")







