# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------- models ----------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier

from catboost import CatBoostClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
import warnings
import lightgbm as lgb
import os

lgb_params = {
    'verbosity': -1,  # Suppress warnings
    'min_data_in_leaf': 5,  # Explicitly set to match LightGBM's warning
    'min_child_samples': 5,  # Sync with min_data_in_leaf to avoid conflicts
}
model = lgb.LGBMClassifier(**lgb_params)
from lightgbm import LGBMClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


samp_subm_fPath = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
train_fPath = "/kaggle/input/playground-series-s5e3/train.csv"
test_fPath = "/kaggle/input/playground-series-s5e3/test.csv"
samp_subm_df = pd.read_csv(samp_subm_fPath, encoding="ISO-8859-1")
train_df = pd.read_csv(train_fPath, encoding="ISO-8859-1")
test_df = pd.read_csv(test_fPath, encoding="ISO-8859-1")


# Convert 'day' column to categorical
train_df['day'] = pd.Categorical(train_df['day'])
test_df['day'] = pd.Categorical(test_df['day'])


train_df.info()


train_df = train_df.sort_values(by='day')
test_df = test_df.sort_values(by='day')


plt.figure(figsize=(10,6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# Temperature Difference Calculation
train_df['temp_diff'] = train_df['maxtemp'] - train_df['mintemp']
test_df['temp_diff'] = test_df['maxtemp'] - test_df['mintemp']

# Wind Speed Categories
def categorize_wind_speed(speed):
    if speed < 10: 
        return 'Low'
    elif 10 <= speed < 20: 
        return 'Medium'
    else: 
        return 'High'

train_df['windspeed_cat'] = train_df['windspeed'].apply(categorize_wind_speed)
test_df['windspeed_cat'] = test_df['windspeed'].apply(categorize_wind_speed)

train_df = pd.get_dummies(train_df, columns=['windspeed_cat'], drop_first=False)
test_df = pd.get_dummies(test_df, columns=['windspeed_cat'], drop_first=False)

# Cloud/Sunshine Ratio
train_df['cloud_sunshine_ratio'] = train_df['cloud'] / train_df['sunshine'].replace(0, 1e-6)
test_df['cloud_sunshine_ratio'] = test_df['cloud'] / test_df['sunshine'].replace(0, 1e-6)


# Rolling Averages (7-day window)
window_size = 7

# For training data 
train_df['rolling_temp'] = train_df['temparature'].rolling(window=window_size).mean()
train_df['rolling_humidity'] = train_df['humidity'].rolling(window=window_size).mean()
train_df['rolling_cloud'] = train_df['cloud'].rolling(window=window_size).mean()
train_df['rolling_maxtemp'] = train_df['maxtemp'].rolling(window=window_size).mean()
train_df['rolling_mintemp'] = train_df['mintemp'].rolling(window=window_size).mean()
train_df['rolling_temp_diff'] = train_df['temp_diff'].rolling(window=window_size).mean()

# Final Column Drops (different for train/test)
train_drop =  ['id', 'temparature', 'humidity', 'cloud', 'maxtemp', 'mintemp', 'temp_diff']

train_df = train_df.drop(columns=train_drop)
test_id = test_df['id']
test_df = test_df.drop(columns=['id'])
train_df = train_df.rename(columns={
    'rolling_temp': 'temparature',
    'rolling_humidity': 'humidity',
    'rolling_cloud': 'cloud',
    'rolling_maxtemp': 'maxtemp',
    'rolling_mintemp': 'mintemp',
    'rolling_temp_diff': 'temp_diff'
})


# 1. Fill rolling features using median imputation
cols = ['temparature', 'humidity', 'cloud']

# Apply median imputation for missing values
train_df[cols] = train_df[cols].apply(lambda col: col.fillna(col.median()))


# Function to treat temparature outliers using Z-score
def treat_outliers_zscore(series, threshold=3):
    not_null = series.notna()
    
    # Filter out the null values before calculating the z-score
    non_null_series = series[not_null]
    
    # Calculate the Z-scores for the non-null values
    z_scores = np.abs(stats.zscore(non_null_series))
    
    # Create a copy of the original series for treatment
    treated = series.copy()
    
    # Set values greater than the threshold to NaN
    treated.loc[not_null][z_scores > threshold] = np.nan
    
    # Forward fill the NaN values
    return treated.ffill()

# Apply the function to the 'temparature' column
train_df['temparature'] = treat_outliers_zscore(train_df['temparature'], threshold=3)


# Dealing with others outlier
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def treat_outliers(series, method='iqr', threshold=3):
    """Handle outliers in a pandas Series using specified method"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        return series.clip(lower, upper)
    
    elif method == 'log':
        return np.log1p(series)
    
    elif method == 'cap':
        upper = series.quantile(0.99)
        return series.clip(upper=upper)
    
    else:
        return series

# Apply to each column
treatment_plan = {
    'temp_diff': 'iqr',
    'maxtemp': 'iqr',
    'mintemp': 'iqr',
    'dewpoint': 'iqr',
    'winddirection': 'iqr',
    'pressure': 'iqr',
    'sunshine': 'cap',
    'cloud_sunshine_ratio': 'log',
    'humidity': 'iqr',
    'cloud': 'iqr'
}

for col, method in treatment_plan.items():
    if col in train_df.columns:
        train_df[col] = treat_outliers(train_df[col], method)



# ğŸ“Œ Updated model definitions
models = {
    'LightGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'num_leaves': [31, 50, 100],  # Increased range to allow more leaves for flexibility
            'learning_rate': [0.01, 0.05, 0.1],  # Added a mid-value and higher learning rate
            'n_estimators': [50, 100, 200],  # Increased range of estimators to allow more iterations
            'max_depth': [5, 10, 15],  # Expanded depth range to allow deeper trees
            'subsample': [0.8, 0.85, 0.9],  # Adjusted subsample to include a wider range
            'min_data_in_leaf': [5, 10, 20],  # Expanded values to help control the leaf size            
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 150, 200, 250],
            'max_depth': [10, 15, 20, 25],  # Expanded depth range to test larger trees
            'min_samples_split': [2, 5, 10, 15],  # Added more granularity for split control
            'min_samples_leaf': [1, 2, 4],  # Adjusted leaf values for more generalization
            'max_features': ['auto', 'sqrt', 'log2', None],  # Added None to test unbounded feature selection
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='hist', random_state=42),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1],  # Added a higher learning rate for faster convergence
            'n_estimators': [200, 250, 300],  # Increased estimators for a deeper search
            'max_depth': [6, 8, 10],  # Expanded depth range for more complex models
            'subsample': [0.8, 0.85, 0.9],  # Expanded subsample range to test more training data use
            'colsample_bytree': [0.8, 1.0],  # Include 1.0 for full column use
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0, random_state=42),
        'params': {
            'iterations': [100, 500, 1000, 1500],  # Added more iterations for longer training
            'depth': [6, 8, 10, 12],  # Expanded depth for flexibility in learning
            'learning_rate': [0.01, 0.05, 0.1, 0.15],  # Added a higher learning rate for experimentation
            'l2_leaf_reg': [1, 3, 5, 10],  # Expanded the regularization range for better control
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, 20],  # Expanded depth to capture more complexity
            'min_samples_split': [2, 5, 10],  # More granularity for controlling node splitting
        }
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 150, 200],  # Increased number of trees for better model stability
            'max_depth': [10, 15, 20, 25],  # Added larger depths for deeper learning
            'min_samples_split': [5, 10, 15],  # Added more split granularity
            'min_samples_leaf': [1, 2, 4],  # Adjusted leaf size for better generalization
            'max_features': ["sqrt", "log2", None],  # Added None for feature selection flexibility
        }
    }
}



warnings.filterwarnings("ignore", category=UserWarning)

# âœ… Load data - assuming train_df is already loaded
X = train_df.drop(columns=['rainfall'])  # Training features
y = train_df['rainfall']  # Target variable

# âœ… Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_test = test_df.copy()  # Test set (no target available)
# y_test = None  # Set to None since we don't have actual test labels
    
# âœ… Handle missing values for mixed data types
numeric_cols = X_train.select_dtypes(include=['number']).columns
categorical_cols = X_train.select_dtypes(exclude=['number']).columns

# Create and apply imputers
numeric_imputer = SimpleImputer(strategy='median')
if len(numeric_cols) > 0:
    X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = numeric_imputer.transform(X_test[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
if len(categorical_cols) > 0:
    X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

# Ensure y has no missing values
train_mask = y_train.notna()
X_train = X_train[train_mask]
y_train = y_train[train_mask]

# âœ… Scale data for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_models = {}
results = []

for name, config in models.items():
    print(f"ğŸ”� Training {name}...")

    try:
        if not config['params']:
            print(f"âš ï¸� Skipping {name} - empty parameter grid")
            continue

        # Grid Search with ROC-AUC scoring
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5,
            scoring='roc_auc',  
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model

        # Get probability predictions (for ROC-AUC)
        if hasattr(best_model, "predict_proba"):
            y_pred_train_prob = best_model.predict_proba(X_train_scaled)[:, 1]
            y_pred_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_pred_train_prob = best_model.decision_function(X_train_scaled)
            y_pred_test_prob = best_model.decision_function(X_test_scaled)
        else:
            print(f"âš ï¸� {name} does not support probability prediction!")
            continue
        
        # Get hard predictions for accuracy & F1-score
        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)

        # Classification Metrics
        accuracy_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train, average='weighted')
        roc_auc_train = roc_auc_score(y_train, y_pred_train_prob)

        # Only compute test metrics if y_test exists
        if y_test is not None:
            accuracy_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test, average='weighted')
            roc_auc_test = roc_auc_score(y_test, y_pred_test_prob)
        else:
            accuracy_test = f1_test = roc_auc_test = None

        # Store results
        results.append({
            'Model': name,
            'Train Accuracy': accuracy_train,
            'Test Accuracy': accuracy_test,
            'Train F1 Score': f1_train,
            'Test F1 Score': f1_test,
            'Train ROC AUC': roc_auc_train,
            'Test ROC AUC': roc_auc_test,
            'Best Params': str(grid.best_params_)
        })

        print(f"âœ… {name} - Train Acc: {accuracy_train:.4f}, Train F1: {f1_train:.4f}, Train ROC AUC: {roc_auc_train:.4f}")
        if y_test is not None:
            print(f"Test Acc: {accuracy_test:.4f}, Test F1: {f1_test:.4f}, Test ROC AUC: {roc_auc_test:.4f}")
        print(f"Best parameters: {grid.best_params_}\n")

    except Exception as e:
        print(f"â�Œ Failed to train {name}: {str(e)}")

# ğŸ“Œ Final results presentation
results_df = pd.DataFrame(results).sort_values(by='Test ROC AUC', ascending=False)
print("\nğŸ�† Final Model Performance:")
print(results_df[['Model', 'Test Accuracy', 'Test F1 Score', 'Test ROC AUC']].to_markdown(index=False))
print(f"ğŸ“Œ Checking best_models after training {name}: {list(best_models.keys())}")


# Sort results by 'Test ROC AUC' in ascending order
sorted_results_df = results_df.sort_values(by='Test ROC AUC')

# Plotting the Test ROC AUC values
plt.figure(figsize=(10, 6))
plt.barh(sorted_results_df['Model'], sorted_results_df['Test ROC AUC'], color='skyblue')
plt.xlabel('Test ROC AUC')
plt.title('Test ROC AUC for Different Models ')
plt.show()


# 1. Get the exact column order from training data
train_columns = X_train.columns.tolist()

# 2. Select and reorder test columns to match training
X_test_df_aligned = test_df[train_columns].copy()

# 3. Verify no missing columns
missing_cols = set(train_columns) - set(test_df.columns)
if missing_cols:
    raise ValueError(f"Test data missing columns: {missing_cols}")

# 4. Apply preprocessing (using already-fitted transformers)
X_test_df_scaled_cleaned = X_test_df_aligned.copy()

# Numeric imputation
if len(numeric_cols) > 0:
    X_test_df_scaled_cleaned[numeric_cols] = numeric_imputer.transform(
        X_test_df_aligned[numeric_cols]
    )

# Categorical imputation
if len(categorical_cols) > 0:
    X_test_df_scaled_cleaned[categorical_cols] = categorical_imputer.transform(
        X_test_df_aligned[categorical_cols]
    )

# Feature scaling
X_test_df_scaled_cleaned = scaler.transform(X_test_df_scaled_cleaned)

print(f"âœ… Successfully processed test data. Shape: {X_test_df_scaled_cleaned.shape}")


# 1ï¸�âƒ£ Select the best model based on Test ROC AUC
best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]
print(f"\nğŸ�… Using best model: {best_model_name} for rainfall prediction.")

# 2ï¸�âƒ£ Make Predictions
if hasattr(best_model, "predict_proba"):
    predicted_rainfall = best_model.predict_proba(X_test_df_scaled_cleaned)[:, 1]
elif hasattr(best_model, "decision_function"):
    predicted_rainfall = best_model.decision_function(X_test_df_scaled_cleaned)
else:
    predicted_rainfall = best_model.predict(X_test_df_scaled_cleaned)

# 3ï¸�âƒ£ Create Submission DataFrame using `test_id` instead of `test_df.index`
submission = pd.DataFrame({'id': test_id, 'rainfall': predicted_rainfall})

# 4ï¸�âƒ£ Sort by `id` in descending order

# 5ï¸�âƒ£ Ensure the `rainfall` column is formatted with one digit after the decimal point
submission['rainfall'] = submission['rainfall'].round(1)

# Save the submission file as a .csv
samp_subm_fPath = "/kaggle/working/sample_submission.csv"
submission.to_csv(samp_subm_fPath, index=False)

# Optionally, remove any unnecessary files (like .json)
import os
json_file_path = "/kaggle/working/catboost_info/catboost_training.json"
if os.path.exists(json_file_path):
    os.remove(json_file_path)

# Verify the file is saved as .csv
!ls -lh /kaggle/working/

submission.head()

