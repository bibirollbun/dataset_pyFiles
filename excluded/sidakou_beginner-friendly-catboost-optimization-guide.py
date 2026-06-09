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
import catboost as cb
from catboost import CatBoostRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optimization
import optuna

# Statistical tools
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


# Display basic information about the training dataset
print(train.info())
print(train.describe().T)


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


# Build a CatBoost model (for regression tasks)
def build_catboost_model(iterations=100, depth=5, learning_rate=0.1):
    """Function to build a CatBoost regression model"""
    model = cb.CatBoostRegressor(
        iterations=iterations,         # Number of boosting iterations (trees)
        depth=depth,                   # Depth of each decision tree
        learning_rate=learning_rate,   # Learning rate (smaller = more stable, slower convergence)
        loss_function='RMSE',          # Loss function for regression (Root Mean Squared Error)
        random_seed=42,                # Random seed for reproducibility
        verbose=100                    # Logging interval during training
    )
    return model

# Build the model
catboost_model = build_catboost_model()

# Train the model
catboost_model.fit(X_train, y_train)


def explain_model(model):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Visualize individual prediction explanation
    shap.plots.waterfall(shap_values[0])
    
    # Visualize overall feature importance
    shap.plots.beeswarm(shap_values)

# Explain the trained CatBoost model
explain_model(catboost_model)


# Make predictions on the test data
xgb_pred = catboost_model.predict(X_test)


def evalute_metrics(y_true, y_pred):
    results = []

    # Inner function to compute common regression metrics
    def calulate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)   # Mean Squared Error
        mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
        r2 = r2_score(y_true, y_pred)              # RÂ² Score
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }

    # Calculate metrics and store in a list
    results.append(calulate_metrics(y_true, y_pred))
    
    # Convert results to a pandas DataFrame for easy display
    return pd.DataFrame(results)

# Calculate evaluation metrics for the test data
results = evalute_metrics(y_test, xgb_pred)
display(results)


def optimize_catboost(trial):
    """Function for optimizing CatBoost hyperparameters using Optuna"""
    params = {
        # Number of boosting iterations (equivalent to n_estimators in XGBoost)
        'iterations': trial.suggest_int('iterations', 50, 500),

        # Depth of the individual trees (controls model complexity)
        'depth': trial.suggest_int('depth', 3, 10),

        # Learning rate (smaller values make learning slower but more stable)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),

        # L2 regularization coefficient (helps prevent overfitting)
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),

        # Number of splits used to find thresholds for continuous features
        'border_count': trial.suggest_int('border_count', 32, 255),

        # Loss function for regression tasks
        'loss_function': 'RMSE',

        # Random seed for reproducibility
        'random_seed': 37
    }
    
    # Create CatBoost model with current trial parameters (suppress training logs)
    model = CatBoostRegressor(**params, verbose=0)

    # Train model on training set
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# Run hyperparameter optimization
# n_trials defines how many different parameter combinations to test
study = optuna.create_study(direction='minimize')
study.optimize(optimize_catboost, n_trials=30)

# Display the best trial results
print('Best trial:')
trial = study.best_trial
print(f"  MSE: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Re-train CatBoost model using the best parameters found by Optuna
optimized_catboost = CatBoostRegressor(**trial.params, random_seed=42, verbose=100)
optimized_catboost.fit(X_train, y_train)


explain_model(optimized_catboost)


# Predict using the optimized CatBoost model
optimized_pred = optimized_catboost.predict(X_test)

# Evaluate the optimized model performance
optimized_metrics = evalute_metrics(y_test, optimized_pred)

print("before optimized") 
display(results)

# Display the optimized evaluation results
print("after optimized")
display(optimized_metrics)


def simple_residual_plot(y_test, y_pred, figsize=(7, 5)):
    """
    Display a basic residual plot to visualize model errors.

    Parameters
    ----------
    y_test : array-like
        True (actual) target values.
    y_pred : array-like
        Predicted target values from the model.
    figsize : tuple
        Figure size (default: (7, 5)).
    """
    # Calculate residuals (difference between actual and predicted values)
    residuals = y_test - y_pred

    # Create a scatter plot of residuals vs predictions
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)

    # Add a horizontal reference line at y=0
    plt.axhline(y=0, color='r', linestyle='--', lw=2)

    # Set titles and labels
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')

    # Add light grid for readability
    plt.grid(True, alpha=0.3)
    plt.show()

# Visualize residuals for the optimized model
simple_residual_plot(y_test, optimized_pred)


#  Make predictions on the test dataset
predict_y = optimized_catboost.predict(predict_X)

#  Create a DataFrame for predicted results
predict_df = pd.DataFrame(predict_y, columns=['accident_risk'])

#  Combine 'id' column with predictions
submission = pd.concat([predict['id'], predict_df], axis=1)

#  Display the first few rows of the submission file
display(submission.head())

#  Check for any missing values before saving
print(submission.isnull().sum())


# --- Save to CSV for Kaggle submission ---
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file saved as 'submission.csv'")

