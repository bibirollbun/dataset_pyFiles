# ===============================
# ğŸ“š Library Imports
# ===============================

# Basic libraries
import os
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Preprocessing
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Machine Learning Models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optimization
import optuna

# Statistical toolss
from scipy import stats

# Model saving & loading
import joblib

# Display input file paths (Kaggle environment)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# ğŸ“‚ Data Loading 
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
predict = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.head()


# Display basic information about the training dataset
train.info()


train.describe().T


def plot_correlation_heatmap(df, figsize=(12, 8)):
    """
    Display the correlation between all numerical features as a heatmap.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Display the correlation heatmap of all features in the training data
corr_matrix = plot_correlation_heatmap(train.drop(columns=['id', 'accident_risk']))


def create_features(df):
    
    df['curavture_accidents'] = df['curvature'] * df['num_reported_accidents']
    
    return df

train = create_features(train)
predict = create_features(predict)


# One-Hot Encoding
def one_hot_encode(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    return df

# categorical variables
encode_columns = ['road_type', 'lighting', 'weather', 'time_of_day']

train = one_hot_encode(train, encode_columns)
predict = one_hot_encode(predict, encode_columns)


# Convert boolean columns

def bool_to_int(df):
    bool_columns = df.select_dtypes(include='bool').columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    return df

train = bool_to_int(train)
predict = bool_to_int(predict)


# Since the residual errors are large, use RobustScaler, which is less sensitive to outliers
def robust_scale(df):
    scaler = RobustScaler()

    numeric_columns = ['curvature','curavture_accidents']

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

train = robust_scale(train)
predict = robust_scale(predict)


def split_data(df,test_size=0.2,random_state=42):
    X = df.drop(columns=['id','accident_risk'])
    y = df['accident_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(train)

predict_X = predict.copy()
predict_X = predict_X.drop(columns=['id'])


# Build an optimized XGBoost regression model
def build_xgboost_model(
    n_estimators=431,
    max_depth=10,
    learning_rate=0.08116012956704159,
    subsample=0.671117833063129,
    colsample_bytree=0.928949134207643,
    gamma=0.02035246955467515,
    min_child_weight=3,
    random_state=42
):
    """Build an optimized XGBoost regression model"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Loss function suitable for regression (RMSE)
        n_estimators=n_estimators,      # Number of boosting trees
        max_depth=max_depth,            # Maximum tree depth
        learning_rate=learning_rate,    # Step size shrinkage used to prevent overfitting
        subsample=subsample,            # Fraction of samples used for each tree
        colsample_bytree=colsample_bytree,  # Fraction of features used per tree
        gamma=gamma,                    # Minimum loss reduction required to make a further partition
        min_child_weight=min_child_weight,  # Minimum sum of instance weight needed in a child
        random_state=random_state       # Random seed for reproducibility
    )
    return model


# Build the model
xgb_model = build_xgboost_model()

# Train the model
xgb_model.fit(X_train, y_train)



# Build an optimized CatBoost model (for regression tasks)
def build_catboost_model(
    iterations=474,
    depth=8,
    learning_rate=0.10105022929242771,
    l2_leaf_reg=3.905251076718455,
    border_count=106,
    random_seed=42,
    verbose=100
):
    """Build an optimized CatBoost regression model"""
    model = cb.CatBoostRegressor(
        iterations=iterations,           # Number of boosting iterations (trees)
        depth=depth,                     # Maximum depth of the trees
        learning_rate=learning_rate,     # Learning rate (controls step size)
        l2_leaf_reg=l2_leaf_reg,         # L2 regularization term (helps prevent overfitting)
        border_count=border_count,       # Number of splits for numerical features
        loss_function='RMSE',            # Loss function suitable for regression
        random_seed=random_seed,         # Random seed for reproducibility
        verbose=verbose,                  # Logging frequency during training
        allow_writing_files=False
    )
    return model


# Build the model
cat_model = build_catboost_model()

# Train the model
cat_model.fit(X_train, y_train)



# Build an optimized LightGBM regression model
def build_lightgbm_model(
    n_estimators=381,
    max_depth=7,
    learning_rate=0.039237697728597476,
    subsample=0.790446416831238,
    colsample_bytree=0.9592823831701073,
    num_leaves=119,
    min_child_samples=29,
    random_state=42
):
    """Build an optimized LightGBM regression model"""
    model = lgb.LGBMRegressor(
        objective='regression',          # For regression tasks
        n_estimators=n_estimators,       # Number of boosting trees
        max_depth=max_depth,             # Maximum depth of each tree
        learning_rate=learning_rate,     # Learning rate
        subsample=subsample,             # Fraction of samples used for each tree
        colsample_bytree=colsample_bytree,  # Fraction of features used per tree
        num_leaves=num_leaves,           # Maximum number of leaves per tree
        min_child_samples=min_child_samples, # Minimum number of samples per leaf
        random_state=random_state,       # Random seed for reproducibility
        verbose=-1                       # Suppress training logs
    )
    return model


# Build the model
lgb_model = build_lightgbm_model()

# Train the model
lgb_model.fit(X_train, y_train)



# Build an optimized Random Forest regression model
def build_random_forest_model(
    n_estimators=195,
    max_depth=11,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features=0.9342067158084462,
    bootstrap=True,
    random_state=42
):
    """Build an optimized Random Forest regression model"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,       # Number of trees in the forest
        max_depth=max_depth,             # Maximum depth of each decision tree
        min_samples_split=min_samples_split, # Minimum number of samples required to split a node
        min_samples_leaf=min_samples_leaf,   # Minimum number of samples required at a leaf node
        max_features=max_features,       # Fraction of features considered for each split
        bootstrap=bootstrap,             # Whether bootstrap samples are used when building trees
        random_state=random_state,       # Random seed for reproducibility
        n_jobs=-1                        # Use all CPU cores for parallel processing
    )
    return model


# Build the model
rf_model = build_random_forest_model()

# Train the model
rf_model.fit(X_train, y_train)



# prediction on Test Data
xgb_pred = xgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


def evalute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results = pd.DataFrame([{
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }])

    return results


xgb_results = evalute_metrics(y_test, xgb_pred).assign(Model="XGBoost")
cat_results = evalute_metrics(y_test, cat_pred).assign(Model="CatBoost")
lgb_results = evalute_metrics(y_test, lgb_pred).assign(Model="LightGBM")
rf_results  = evalute_metrics(y_test, rf_pred).assign(Model="RandomForest")

# çµ�æ�œã‚’çµ�å�ˆã�—ã�¦è¦‹ã‚„ã�™ã��è¡¨ç¤º
results = pd.concat([xgb_results, cat_results, lgb_results, rf_results], ignore_index=True)
results = results[['Model', 'MSE', 'MAE', 'R2']]

display(results.sort_values('MSE'))


import os
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error

# --- Optimization function ---
def optimize_weight(trial):
    w_xgb = trial.suggest_float('xgb_weight', 0.0, 1.0)
    w_lgb = trial.suggest_float('lgb_weight', 0.0, 1.0)
    w_rf  = trial.suggest_float('rf_weight', 0.0, 1.0)
    w_cat = trial.suggest_float('cat_weight', 0.0, 1.0)

    total_weight = w_xgb + w_lgb + w_rf + w_cat
    if total_weight == 0:
        return np.inf  # Invalid case

    # Normalize weights
    w_xgb /= total_weight
    w_lgb /= total_weight
    w_rf  /= total_weight
    w_cat /= total_weight

    # Weighted ensemble prediction
    final_pred = (
        w_xgb * xgb_pred +
        w_lgb * lgb_pred +
        w_rf  * rf_pred +
        w_cat * cat_pred
    )

    mse = mean_squared_error(y_test, final_pred)
    return mse


# --- Run optimization with Optuna ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optimize_weight, n_trials=1000, show_progress_bar=True)

# --- Results ---
best_params = study.best_params
best_w_xgb = best_params['xgb_weight']
best_w_lgb = best_params['lgb_weight']
best_w_rf  = best_params['rf_weight']
best_w_cat = best_params['cat_weight']

# Normalize again
total_weight = best_w_xgb + best_w_lgb + best_w_rf + best_w_cat
best_w_xgb /= total_weight
best_w_lgb /= total_weight
best_w_rf  /= total_weight
best_w_cat /= total_weight

# --- Final ensemble prediction ---
final_pred = (
    best_w_xgb * xgb_pred +
    best_w_lgb * lgb_pred +
    best_w_rf  * rf_pred +
    best_w_cat * cat_pred
)

rmse = np.sqrt(mean_squared_error(y_test, final_pred))

# --- Display results ---
print("\n=== Optimized Weights ===")
print(f"XGBoost: {best_w_xgb:.4f}")
print(f"LightGBM: {best_w_lgb:.4f}")
print(f"RandomForest: {best_w_rf:.4f}")
print(f"CatBoost: {best_w_cat:.4f}")
print(f"\nFinal RMSE: {rmse:.4f}")


ensemble_results = evalute_metrics(y_test, final_pred).assign(Model="Optimized Ensemble")
results = pd.concat(
    [xgb_results, lgb_results, rf_results, cat_results, ensemble_results],
    ignore_index=True
)
results = results[['Model', 'MSE', 'MAE', 'R2']]
display(results)


xgb_pred_new = xgb_model.predict(predict_X)

lgb_pred_new = lgb_model.predict(predict_X)

rf_pred_new = rf_model.predict(predict_X)

cat_pred_new = cat_model.predict(predict_X)

final_pred_new = (best_w_xgb * xgb_pred_new 
                  + best_w_lgb * lgb_pred_new 
                  + best_w_rf * rf_pred_new
                  + best_w_cat * cat_pred_new 
                  )

predict_df = pd.DataFrame(final_pred_new, columns=['accident_risk'])
submission = pd.concat([predict['id'], predict_df], axis=1)

display(submission.head())
print(submission.isnull().sum())


# --- Save to CSV for Kaggle submission ---
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved as 'submission.csv'")

