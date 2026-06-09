import os


import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
train_df.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_df.head()


# Print the size of the train and test datasets
print(train_df.shape)
print(test_df.shape)


# Check the null values of train_df
train_df.isnull().sum()


# Check the datatypes of the train_df
train_df.dtypes


# Remove the unwanted columns from the dataset
train_df.drop(columns=['id'], inplace=True)


# Encoding the categorical values for the training of dataset
import pickle
from sklearn.preprocessing import LabelEncoder

# Store the encoders 
encoders = {}

for col in train_df.columns:
    if train_df[col].dtype == 'object':
        encoder = LabelEncoder()
        train_df[col] = encoder.fit_transform(train_df[col])
        encoders[col] = encoder

# Save the path of the encoders
with open("lgbm_encoders.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("LightGBM encoders dumped succesfully!")


# Preview the datatypes of dataframe after encoding
train_df.dtypes


import matplotlib.pyplot as plt
import seaborn as sns

# Check the relationship of each features
plt.figure(figsize=(12,8))
sns.heatmap(train_df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title("Correlation Heatmap of Calories", fontsize=20, fontweight='bold', color='red')
plt.xlabel("Features", fontsize=12, fontweight='bold', color='blue')
plt.ylabel("Features", fontsize=12, fontweight='bold', color='green')

# Distribution graph of the Calories
plt.figure(figsize=(12,8))
sns.histplot(train_df['Calories'], kde=True)
plt.title("Distribution graph of the Calories", fontsize=20, fontweight='bold', color='red')
plt.xlabel("Calories", fontsize=20, fontweight='bold', color='blue')
plt.ylabel("Count", fontsize=20, fontweight='bold', color='green')

# Boxplot of the train dataframe for detecting outliers
plt.figure(figsize=(12,8))
sns.boxplot(data=train_df)
plt.title("Boxplot to detect outliers", fontsize=20, fontweight='bold', color='red')
plt.xlabel("Features", fontsize=20, fontweight='bold', color='blue')
plt.ylabel("Count", fontsize=20, fontweight='bold', color='green')

# Final printout/show the graph
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

# Extract the independent and dependent features 
X = train_df.drop(columns=['Calories'])
y = train_df['Calories']

# Train and test the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the path of the scalers
with open("lgbm_scalers", "wb") as f:
    pickle.dump(scaler, f)

print("LightGBM scalers saved successfully!")


import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor

# Objective fuction for optuna
def objective(trial):
    params = {
        # GPU acceleration
        'device_type': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,

        # Suggest LightGBM Hyperparameters
        'random_state': trial.suggest_int('random_state', 300, 1000),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'min_child_samples': trial.suggest_int('min_samples_split', 5, 15),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'colsample_bytree': trial.suggest_float('col_samples_bytree', 0.3, 1.0),
        'boosting_type': trial.suggest_categorical('booster', ['gbdt', 'goss', 'dart', ]) # Boosting type 
    }

    model = LGBMRegressor(**params)

    # Cross val score with R2 Score
    score = cross_val_score(model, X_train_scaled, y_train, cv=7, verbose=-1)

    return np.mean(score)


# Perform optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15, show_progress_bar=True)

# Print best hyperparameters and best score
print("Best hyperparamaters : ", study.best_params)
print("Best Score : ", study.best_value)

# Train the model with best hyperparameters
best_params = study.best_params.copy()

best_params.update({
    'device_type': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
})

# Train the best LightGBM model
best_lgbm_model = LGBMRegressor(**best_params)
best_lgbm_model.fit(X_train_scaled, y_train)


# Predict the model
y_pred = best_lgbm_model.predict(X_test_scaled)
y_pred[0:5]


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Actual vs Predicted Calories
plt.figure(figsize=(12,8))
plt.plot(y_test.values, label="Actual Calories")
plt.plot(y_pred, label="Predicted Calories")
plt.title("Actal vs Predicted Calories Expenditure", fontsize=20, fontweight='bold', color='red')
#plt.xlabel("Actiual Calories", fontsize=12, fontweight='bold', color='blue')
#plt.ylabel("Predicted Calories", fontsize=12, fontweight='bold', color='green')
plt.legend()

# Metrics plot
metrics = {
    'r2': r2_score(y_test, y_pred),
    'mse': mean_squared_error(y_test, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    'mae': mean_absolute_error(y_test, y_pred)
}
plt.figure(figsize=(12,8))
plt.barh(list(metrics.keys()), list(metrics.values()), color=['blue', 'green', 'orange'])
plt.title("Evolution Metrics (LightGBM)")
plt.xlabel("Metrics values", fontsize=12)
plt.ylabel("Metrics Name", fontsize=12)
for i, (key, value) in enumerate(metrics.items()):
    plt.text(value+0.001, i, f"{value:.4f}", va='center', fontsize=11, fontweight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.6)

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(12, 8))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (LightGBM)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (y_test - y_pred)")
plt.grid(True)

# Final layout of the plots
plt.tight_layout()
plt.show()


# Optuna visualizations using Matplotlib
from optuna.visualization.matplotlib import (
      plot_optimization_history,
      plot_param_importances,
      plot_parallel_coordinate,
      plot_edf,
      plot_slice,
      plot_intermediate_values
)

plots = [
    ('Optimization History', plot_optimization_history),
    ('Parameter Importances', plot_param_importances),
    ('Parallel Coordinates', plot_parallel_coordinate),
    ("Empirical Distribution Function", plot_edf),
    ('Paramater Slice plot', plot_slice),
    ('Intermmediate values learning Curve', plot_intermediate_values)
]

# Display all plots properly resized
for name, plot_func in plots:
    print(f"\n{name}")
    plt.figure(figsize=(20,12))
    
    # Call the plotting function directly on this figure
    ax = plot_func(study)
    
    # Add title (use suptitle if it's a figure)
    if hasattr(ax, "set_title"):
        ax.set_title(name, fontsize=14)
    else:
        plt.title(name, fontsize=14)
    
    plt.tight_layout()
    plt.show()


import joblib
from sklearn.pipeline import make_pipeline

# Create the pipeline
pipe = make_pipeline(scaler, best_lgbm_model)
pipe.fit(X_train, y_train)

# Save the path of the trained model
joblib.dump(pipe, "lightgbm_model.pkl")

print("LightGBM Model saved successfully!")

