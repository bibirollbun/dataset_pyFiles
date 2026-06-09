# Standard Libraries
import warnings
from IPython.display import clear_output

# Data Handling & Preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import optuna.visualization as vis

# Machine Learning & Modeling
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Hyperparameter Optimization
import optuna

# Optuna logging setup
optuna.logging.set_verbosity(optuna.logging.INFO)

# Plotly Setup
pio.renderers.default = "iframe_connected"

# Suppress Warnings
warnings.filterwarnings('ignore')


# Define path 
PATH = "/kaggle/input/playground-series-s5e10/"

# Load the data
train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")

train.head()


# Separate features and target
X = train.drop(columns=["accident_risk", "id"])
y = train["accident_risk"]

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Apply Ordinal Encoding to categorical columns in the training data
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# Apply the same encoding to the test data 
X_test_encoded = test.drop(columns=["id"]) 
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# Feature scaling 
scaler = StandardScaler()
X_encoded_scaled = scaler.fit_transform(X_encoded)


# Define the list of categorical features
categorical_features = ['road_type', 'lighting', 'weather', 'road_signs_present', 
                        'public_road', 'time_of_day', 'holiday', 'school_season']

# Set the plot layout
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

# Loop through each categorical feature and plot the average accident risk
for idx, col in enumerate(categorical_features):
    # Calculate average accident risk by category
    data = train.groupby(col)['accident_risk'].mean().sort_values()

    # Plot the data
    data.plot(kind='bar', ax=axes[idx], color=sns.color_palette("viridis", len(data)), edgecolor='black', alpha=0.9)

    # Set title, labels, and grid
    axes[idx].set_title(f'Avg Accident Risk by {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Avg Accident Risk', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

# Adjust layout 
plt.tight_layout()
plt.show()


# Define the list of numerical features
numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Set the plot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

# Loop through each numerical feature
for idx, col in enumerate(numerical_features):
    sns.lineplot(data=train, x=col, y='accident_risk', marker='o', ax=axes[idx], color='#ff0061')

    axes[idx].set_title(f'{col} vs Accident Risk', fontsize=14, fontweight="bold")
    axes[idx].set_xlabel(col, fontsize=12)
    axes[idx].set_ylabel('Accident Risk', fontsize=12)
    axes[idx].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# Set the style
sns.set(style="whitegrid", palette="muted")

# Set the plot size
plt.figure(figsize=(10, 6))

# Plot the distribution of accident risk
sns.histplot(train['accident_risk'], kde=True, bins=20, color='blue', edgecolor='black', alpha=0.6)
plt.title('Accident Risk Distribution', fontsize=16, fontweight="bold")
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()


# Create a new DataFrame 
X_with_target = X_encoded.copy()
X_with_target['accident_risk'] = y

# Compute the correlation matrix 
correlation_matrix = X_with_target.corr()

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(10, 7))
ax = sns.heatmap(correlation_matrix,
                 annot=True,
                 fmt='.2f',
                 linewidths=0.5,
                 cmap="plasma")
plt.title('Correlation Matrix', fontsize=18, fontweight="bold")
plt.show()


# Define the objective function for Optuna 
def objective(trial):
    # Hyperparameter search space
    param = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'enable_categorical': True,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 10)
    }

    # Initialize KFold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []  # Store RMSE scores for each fold

    # KFold cross-validation loop
    for train_idx, val_idx in cv.split(X_encoded_scaled, y):
        X_train_cv, X_val_cv = X_encoded_scaled[train_idx], X_encoded_scaled[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize and train the model
        model = xgb.XGBRegressor(**param)
        model.fit(X_train_cv, y_train_cv, 
                  eval_set=[(X_val_cv, y_val_cv)], 
                  early_stopping_rounds=50, 
                  verbose=False)

        # Predict and calculate RMSE for the fold
        y_pred = model.predict(X_val_cv)
        rmse = np.sqrt(mean_absolute_error(y_val_cv, y_pred))  
        rmse_scores.append(rmse)

    # Return the average RMSE from the KFold validation
    mean_rmse = np.mean(rmse_scores)
    return mean_rmse

# Create an Optuna study for hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)


# SHOW OPTIMIZATION HISTORY
fig = optuna.visualization.plot_optimization_history(study)
fig.show()


# DISPLAY HYPERPARAMETER IMPORTANCES
fig = optuna.visualization.plot_param_importances(study)
fig.show()


# ANALYZE HYPERPARAMETER INTERACTIONS 
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()


# DISPLAY SENSITIVITY OF EACH PARAMETER 
fig = optuna.visualization.plot_slice(study)
fig.show()


# Get best parms
best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params)

# Fit the model on the entire training data
best_model.fit(X_encoded_scaled, y)

# Make predictions on the test set
X_test_scaled = scaler.transform(X_test_encoded) 
y_pred_test = best_model.predict(X_test_scaled)

# Create a submission 
submission = pd.DataFrame({'id': test['id'], 'accident_risk': y_pred_test})
submission.to_csv('/kaggle/working/submission.csv', index=False)

