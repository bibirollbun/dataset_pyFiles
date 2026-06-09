# Import libraries
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    RANSACRegressor,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
pd.set_option("display.float_format", "{:.4f}".format)

# Static variables
RANDOM_STATE = 42
N_ITER_GRID_SEARCH = 400  # low value for quicker testing in local environment
CV_GRID_SEARCH = 3

# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col="id")
kaggle_test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv", index_col="id")


# Shape of datasets
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {kaggle_test_df.shape}")

# Display first rows of training set
train_df.head()


# Basic information about the dataset
train_df.info()


# Statistical summary
train_df.describe(include="all")


# Check for missing values
train_df.isnull().sum()


# Target variable distribution
print("\nTarget Variable (accident_risk) Statistics:")
print(train_df["accident_risk"].describe())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train_df["accident_risk"], bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Accident Risk")
plt.ylabel("Frequency")
plt.title("Distribution of Accident Risk")

plt.subplot(1, 2, 2)
plt.violinplot(train_df["accident_risk"])
plt.ylabel("Accident Risk")
plt.title("Accident Risk - Violin Plot")

plt.tight_layout()
plt.show()


# Separate features and target
X = train_df.drop("accident_risk", axis=1)
y = train_df["accident_risk"]

# Identify feature types
numerical_features = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
categorical_features = ["road_type", "lighting", "weather", "time_of_day"]
boolean_features = ["road_signs_present", "public_road", "holiday", "school_season"]

print(f"Total features: {len(X.columns)}")
print(f"Numerical: {len(numerical_features)}")
print(f"Categorical: {len(categorical_features)}")
print(f"Boolean: {len(boolean_features)}")
print(f"\nTarget variable range: [{y.min():.2f}, {y.max():.2f}]")


# Split data: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

print("Data split:")
print(f"  Train:      {X_train.shape[0]:>6} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation: {X_val.shape[0]:>6} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:       {X_test.shape[0]:>6} samples ({X_test.shape[0]/len(X)*100:.1f}%)")


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ("bool", "passthrough", boolean_features),
    ],
    remainder="drop",
)

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)
X_kaggle_processed = preprocessor.transform(kaggle_test_df)

print(f"Original features: {X_train.shape[1]}")
print(f"After preprocessing: {X_train_processed.shape[1]}")


# Define light and fast models to evaluate
models = {
    # Linear Models
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
    "BayesianRidge": BayesianRidge(),
    "HuberRegressor": HuberRegressor(),
    "RANSAC": RANSACRegressor(random_state=RANDOM_STATE),
    
    # Tree - based models
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    # "RandomForest": RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
    
    # Gradient Boosting Models
    "XGBoost": XGBRegressor(n_estimators=50, random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=50, random_state=RANDOM_STATE, verbose=-1),
    
    # Ensemble Methods

    "Bagging": BaggingRegressor(n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1),
    
    # Other Fast Models
    "PLSRegression": PLSRegression(n_components=2),
}

results = []
trained_models = {}

# Minimalist tqdm without emojis
for name, model in tqdm(models.items(), desc="Training models"):
    start_time = time.time()

    # Train model
    model.fit(X_train_processed, y_train)

    # Store the trained model
    trained_models[name] = model

    # Predict on validation set
    y_pred = model.predict(X_val_processed)

    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    training_time = time.time() - start_time

    # Store results
    results.append({"Model": name, "R-Squared": r2, "RMSE": rmse, "MAE": mae, "Time (s)": training_time})


# Create results dataframe
results_df = pd.DataFrame(results).sort_values("R-Squared", ascending=False).reset_index(drop=True)


display(results_df)


# Visualize results with value labels
plt.figure(figsize=(12, 8))

# Plot 1: R-squared scores
plt.subplot(2, 1, 1)
ax1 = sns.barplot(data=results_df, x="R-Squared", y="Model", hue="Model", palette="viridis",)
plt.title("Model Comparison - RÂ² Scores")
plt.xlim(0, 1)

# Add value labels on bars
for i, v in enumerate(results_df["R-Squared"]):
    ax1.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="normal")

# Plot 2: RMSE values
plt.subplot(2, 1, 2)
ax2 = sns.barplot(data=results_df, x="RMSE", y="Model", hue="Model", palette="viridis")
plt.title("Model Comparison - RMSE (Lower is Better)")

# Add value labels on bars
for i, v in enumerate(results_df["RMSE"]):
    ax2.text(v + 0.001, i, f"{v:.4f}", va="center", fontweight="normal")

plt.tight_layout()
plt.show()


# XGBoost parameter grid
xgb_param_grid = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1, 2],
    "reg_lambda": [1, 1.5, 2, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.5, 1],
    "min_child_weight": [1, 3, 5, 7],
}
print("XGBoost Grid Search...")
print("=" * 45)

# Get original XGBoost performance
original_xgb_r2 = results_df[results_df["Model"] == "XGBoost"]["R-Squared"].iloc[0]

# Perform Grid Search
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    xgb_param_grid,
    n_iter=N_ITER_GRID_SEARCH,
    cv=CV_GRID_SEARCH,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
)

xgb_search.fit(X_train_processed, y_train)

# Store tuned model
trained_models["XGBoost_Tuned"] = xgb_search.best_estimator_

# Evaluate tuned model
y_val_pred_xgb = trained_models["XGBoost_Tuned"].predict(X_val_processed)
tuned_xgb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
tuned_xgb_r2 = r2_score(y_val, y_val_pred_xgb)

# Get original RMSE for comparison
original_xgb_rmse = results_df[results_df["Model"] == "XGBoost"]["RMSE"].iloc[0]

print(f"\nXGBoost Tuning Complete!")
print(f"Best parameters: {xgb_search.best_params_}")
print(f"Original RMSE: {original_xgb_rmse:.4f}")
print(f"Tuned RMSE: {tuned_xgb_rmse:.4f}")
print(f"Improvement: {original_xgb_rmse - tuned_xgb_rmse:+.4f}")
print(f"RÂ²: {tuned_xgb_r2:.4f}")


# LightGBM grid
lgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "num_leaves": [31, 40, 50, 63],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2],
    "min_child_samples": [10, 20, 30],
    "min_child_weight": [0.01, 0.1, 1],
    "min_split_gain": [0, 0.1],
}

print("LightGBM Grid Search...")
print("=" * 50)

# Get original model performance
original_lgb_r2 = results_df[results_df["Model"] == "LightGBM"]["R-Squared"].iloc[0]
original_lgb_rmse = results_df[results_df["Model"] == "LightGBM"]["RMSE"].iloc[0]
lgb_search = RandomizedSearchCV(
    LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
    lgb_param_grid,
    n_iter=N_ITER_GRID_SEARCH,
    cv=CV_GRID_SEARCH,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE,
)

lgb_search.fit(X_train_processed, y_train)

# Store tuned model
trained_models["LightGBM_Tuned"] = lgb_search.best_estimator_

# Evaluate tuned model
y_val_pred_lgb = trained_models["LightGBM_Tuned"].predict(X_val_processed)
tuned_lgb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
tuned_lgb_r2 = r2_score(y_val, y_val_pred_lgb)

print(f"\nLightGBM Tuning Complete!")
print(f"Best parameters: {lgb_search.best_params_}")
print(f"Original RMSE: {original_lgb_rmse:.4f}")
print(f"Tuned RMSE: {tuned_lgb_rmse:.4f}")
print(f"Improvement: {original_lgb_rmse - tuned_lgb_rmse:+.4f}")
print(f"RÂ²: {tuned_lgb_r2:.4f}")


# Create ensemble of tuned models
ensemble_models = [("xgb_tuned", trained_models["XGBoost_Tuned"]), ("lgb_tuned", trained_models["LightGBM_Tuned"])]

# Voting Regressor with equal weights
ensemble = VotingRegressor(estimators=ensemble_models, weights=None, n_jobs=-1)  # Equal weights

# Train ensemble on the same training data
print("Training ensemble model...")
ensemble.fit(X_train_processed, y_train)

# Store the ensemble model for later evaluation
trained_models["Ensemble_XGB_LGB"] = ensemble

print("Ensemble model created successfully!")


print("Ensemble Validation Performance")
print("=" * 40)

# Evaluate only ensemble on validation set
y_val_pred_ensemble = trained_models["Ensemble_XGB_LGB"].predict(X_val_processed)

# Calculate metrics
val_r2 = r2_score(y_val, y_val_pred_ensemble)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_ensemble))


print(f"Ensemble XGB+LGB Validation Results:")
print(f"RMSE: {val_rmse:.4f}")
print(f"RÂ²: {val_r2:.4f}")


# Models to evaluate
models_to_evaluate = {
    "XGBoost": trained_models["XGBoost"],
    "LightGBM": trained_models["LightGBM"],
    "XGBoost_Tuned": trained_models["XGBoost_Tuned"],
    "LightGBM_Tuned": trained_models["LightGBM_Tuned"],
    "Ensemble_XGB_LGB": trained_models["Ensemble_XGB_LGB"],
}

test_results = []

print("Evaluating models on test set...")
for name, model in models_to_evaluate.items():
    # Predict on test set
    y_test_pred = model.predict(X_test_processed)

    # Calculate metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    test_results.append({"Model": name, "Test_R2": test_r2, "Test_RMSE": test_rmse, "Test_MAE": test_mae})

# Create results dataframe
test_results_df = pd.DataFrame(test_results).sort_values("Test_RMSE")

print("\nFinal Test Set Performance (Sorted by RMSE):")
print("=" * 60)
display(test_results_df)


# Visualize results
plt.figure(figsize=(12, 4))

# Plot 1: RMSE comparison
plt.subplot(1, 2, 1)
ax1 = sns.barplot(data=test_results_df, x='Test_RMSE', y='Model', hue='Model', palette='viridis')
plt.title('Test Set RMSE\n(Lower is Better)')
plt.xlabel('RMSE')
# Add value labels
for i, v in enumerate(test_results_df['Test_RMSE']):
    ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='normal')

# Plot 2: RÂ² comparison  
plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=test_results_df, x='Test_R2', y='Model', hue='Model', palette='viridis')
plt.title('Test Set RÂ²\n(Higher is Better)')
plt.xlabel('RÂ²')
# Add value labels
for i, v in enumerate(test_results_df['Test_R2']):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='normal')

plt.tight_layout()
plt.show()


# Summary
best_model = test_results_df.iloc[0]
print(f"\nğŸ�† Best Model: {best_model['Model']}")
print(f"ğŸ“Š Test RMSE: {best_model['Test_RMSE']:.4f}")
print(f"ğŸ“ˆ Test RÂ²: {best_model['Test_R2']:.4f}")


# Get the best model from test results
best_model_name = test_results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"Using best model: {best_model_name}")
print(f"Test RMSE: {test_results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"Test RÂ²: {test_results_df.iloc[0]['Test_R2']:.4f}")

# Make predictions on Kaggle test set
kaggle_predictions = best_model.predict(X_kaggle_processed)

# Create submission dataframe
submission_df = pd.DataFrame({"id": kaggle_test_df.index, "accident_risk": kaggle_predictions})

# Ensure predictions are in valid range [0, 1]
submission_df["accident_risk"] = submission_df["accident_risk"].clip(0, 1)

# Save to CSV

submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)

print(f"\nâœ… Submission file created: {submission_file}")
print(f"ğŸ“Š Predictions range: [{submission_df['accident_risk'].min():.4f}, {submission_df['accident_risk'].max():.4f}]")
print(f"ğŸ“� File saved with {len(submission_df)} predictions")

# Show sample of submission
print("\nSample of submission file:")
display(submission_df.head(10))

