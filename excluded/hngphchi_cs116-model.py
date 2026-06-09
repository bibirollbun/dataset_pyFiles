import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def create_age_groups(age):
    """Create age groups based on real data analysis"""
    if age <= 24:
        return 0  # 18-24 (Young Adult)
    elif age <= 34:
        return 1  # 25-34 (Early Adult)
    elif age <= 44:
        return 2  # 35-44 (Middle Adult)
    elif age <= 54:
        return 3  # 45-54 (Late Adult)
    elif age <= 64:
        return 4  # 55-64 (Pre-Senior)
    else:
        return 5  # 65-80 (Senior)

def optimized_feature_engineering(df):
    """
    Feature engineering focusing on top performing features only
    """
    print("Starting optimized feature engineering...")
    
    # Convert Sex to numeric
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # Create age groups
    df['Age_Group'] = df['Age'].apply(create_age_groups)
    
    # === Core Features (Top performers from importance analysis) ===
    
    # BMI calculation
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Heart Rate Percentage
    df['Heart_Rate_pct'] = df['Heart_Rate'] / (220 - df['Age'])
    
    # Duration normalization by age group
    duration_means = {0: 15.405, 1: 15.192, 2: 15.741, 3: 14.986, 4: 15.440, 5: 16.265}
    df['Duration_norm'] = df['Duration'] / df['Age_Group'].map(duration_means)
    
    # Age-Weighted Intensity (normalized) - Top 5 feature
    df['AWI'] = (df['Duration'] * df['Heart_Rate_pct']) / df['Age']
    awi_means = {0: 0.359, 1: 0.273, 2: 0.222, 3: 0.177, 4: 0.162, 5: 0.156}
    df['AWI_norm'] = df['AWI'] / df['Age_Group'].map(awi_means)
    
    # Thermal Stress Index - Top performer
    df['TSI'] = 5 * ((df['Body_Temp'] - 36.5) / (41.5 - 36.5)) + \
                5 * ((df['Heart_Rate'] - 60) / ((220 - df['Age']) - 60))
    
    # Rate of Perceived Exertion
    df['RPE'] = df['Heart_Rate_pct'] + 0.1 * (df['Body_Temp'] - 37)
    
    # Cardiac Load Index with normalization
    df['CLI'] = (df['Heart_Rate'] * df['Duration']) / df['Weight']
    cli_means = {0: 21.712, 1: 21.165, 2: 21.717, 3: 20.356, 4: 21.040, 5: 22.308}
    df['CLI_norm'] = df['CLI'] / df['Age_Group'].map(cli_means)
    
    # Work Load Index
    df['WLI'] = df['Heart_Rate'] * df['Duration'] * df['Weight']
    wli_means = {0: 112944, 1: 112456, 2: 118311, 3: 112966, 4: 117502, 5: 126107}
    df['WLI_norm'] = df['WLI'] / df['Age_Group'].map(wli_means)
    
    # VO2 Proxy with normalization
    df['VO2_Proxy'] = np.where(
        df['Sex'] == 0,  # female
        (0.85 * df['Duration']) / (df['Heart_Rate_pct'] * df['Age']),
        (1.00 * df['Duration']) / (df['Heart_Rate_pct'] * df['Age'])
    )
    vo2_means = {0: 1.317, 1: 0.930, 2: 0.673, 3: 0.488, 4: 0.387, 5: 0.314}
    df['VO2_norm'] = df['VO2_Proxy'] / df['Age_Group'].map(vo2_means)
    
    # === Top Ratio Features ===
    df['Temp_Duration_ratio'] = df['Body_Temp'] / df['Duration']  # Top 3 feature
    df['Thermal_Efficiency'] = df['Duration'] / (df['Body_Temp'] - 36.5)  # Top 4 feature
    df['Duration_HR_ratio'] = df['Duration'] / df['Heart_Rate']
    
    # Heat Production
    df['Heat_Production'] = (df['Body_Temp'] - 36.5) * df['Weight'] * df['Duration']
    
    # Cardiovascular Fitness
    df['CV_Fitness'] = (220 - df['Age'] - df['Heart_Rate']) / (220 - df['Age'])
    
    # Fatigue Index with normalization
    df['FI'] = (df['Heart_Rate_pct'] ** 2) / df['Duration']
    fi_means = {0: 0.024, 1: 0.028, 2: 0.027, 3: 0.032, 4: 0.036, 5: 0.040}
    df['FI_norm'] = df['FI'] / df['Age_Group'].map(fi_means)
    
    # Heart Rate Percentage normalization
    hr_pct_means = {0: 0.481, 1: 0.500, 2: 0.531, 3: 0.557, 4: 0.596, 5: 0.649}
    df['HR_pct_norm'] = df['Heart_Rate_pct'] / df['Age_Group'].map(hr_pct_means)
    
    # Gender-adjusted Heart Rate
    df['HR_sex_adj'] = df['Heart_Rate'] * (1 + 0.03 * df['Sex'])
    
    # === Top Interaction Features ===
    df['BMI_HR_Duration'] = df['BMI'] * df['Heart_Rate'] * df['Duration']  # #1 feature!
    df['Weight_Duration_HR'] = df['Weight'] * df['Duration'] * df['Heart_Rate_pct']
    
    # === Polynomial Features (Top performers only) ===
    df['Duration_sq'] = df['Duration'] ** 2  # #2 feature!
    df['Heart_Rate_sq'] = df['Heart_Rate'] ** 2
    
    # === Key Gender and Age Interactions ===
    df['Heart_Rate_pct_sex'] = df['Heart_Rate_pct'] * df['Sex']
    df['TSI_sex'] = df['TSI'] * df['Sex']
    df['Duration_sex'] = df['Duration'] * df['Sex']
    df['CLI_sex'] = df['CLI'] * df['Sex']
    
    # Age group interactions
    df['CLI_age_group'] = df['CLI'] * df['Age_Group']
    df['Duration_age_group'] = df['Duration'] * df['Age_Group']
    df['Heart_Rate_pct_age_group'] = df['Heart_Rate_pct'] * df['Age_Group']
    
    # === Additional Important Features ===
    df['Metabolic_Efficiency'] = df['Heart_Rate_pct'] * df['Weight'] / df['Duration']
    
    # BMR calculation and interactions
    df['BMR'] = np.where(
        df['Sex'] == 0,  # female
        10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] - 161,
        10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5
    )
    df['BMR_Duration'] = df['BMR'] * df['Duration']
    df['BMR_HR_pct'] = df['BMR'] * df['Heart_Rate_pct']
    df['BMR_age_group'] = df['BMR'] * df['Age_Group']
    
    # Additional ratios
    df['BMR_HR_ratio'] = df['BMR'] / df['Heart_Rate']
    
    print(f"Feature engineering completed. New shape: {df.shape}")
    return df

def prepare_data():
    """Prepare training and test data with optimized features"""
    # Combine for consistent feature engineering
    df_merged = pd.concat([train.drop(columns='Calories'), test], axis=0).reset_index(drop=True)
    
    # Apply feature engineering
    df_processed = optimized_feature_engineering(df_merged)
    
    # Split back to train/test
    X_train = df_processed.iloc[:len(train)].drop(columns=['id'])
    X_test = df_processed.iloc[len(train):].drop(columns=['id'])
    y_train = train['Calories'].values
    
    # Select only top performing features (importance > 0.001)
    top_features = [
        'BMI_HR_Duration', 'Duration_sq', 'Temp_Duration_ratio', 'Thermal_Efficiency',
        'AWI_norm', 'Duration', 'Duration_norm', 'TSI', 'Duration_HR_ratio',
        'Weight_Duration_HR', 'Heart_Rate', 'Heart_Rate_sq', 'CLI_norm', 'WLI',
        'RPE', 'CLI', 'Heat_Production', 'WLI_norm', 'VO2_norm', 'HR_sex_adj',
        'Body_Temp', 'Heart_Rate_pct', 'CV_Fitness', 'FI_norm', 'HR_pct_norm',
        'CLI_age_group', 'Heart_Rate_pct_sex', 'TSI_sex', 'AWI', 'Duration_age_group',
        'Metabolic_Efficiency', 'Duration_sex', 'Heart_Rate_pct_age_group',
        'BMR_Duration', 'BMR_HR_pct', 'FI', 'VO2_Proxy', 'Age', 'CLI_sex',
        'BMR_HR_ratio', 'BMR_age_group'
    ]
    
    # Filter to only include features that exist and are important
    available_features = [f for f in top_features if f in X_train.columns]
    X_train_selected = X_train[available_features]
    X_test_selected = X_test[available_features]
    
    print(f"Selected {len(available_features)} top features from {len(X_train.columns)} total features")
    
    return X_train_selected, X_test_selected, y_train

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    y_pred = np.clip(y_pred, 0.001, None)
    y_true = np.clip(y_true, 0.001, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Prepare data
X_train, X_test, y_train = prepare_data()

print(f"\nFinal dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")

print(f"\nSelected features: {list(X_train.columns)}")

# Optimized Random Forest parameters from your analysis
optimized_params = {
    'n_estimators': 250,
    'max_depth': 19,
    'min_samples_split': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nUsing optimized Random Forest parameters:")
for param, value in optimized_params.items():
    print(f"  {param}: {value}")

# Train model
print("\nTraining Random Forest with optimized parameters and selected features...")
model = RandomForestRegressor(**optimized_params)
model.fit(X_train, y_train)

# Model evaluation
print("\nEvaluating model performance...")
train_pred = model.predict(X_train)
train_rmsle = rmsle(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)

print(f"\nTraining Performance:")
print(f"Training RMSLE: {train_rmsle:.6f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Training R²: {train_r2:.4f}")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Train fold model
    fold_model = RandomForestRegressor(**optimized_params)
    fold_model.fit(X_train_fold, y_train_fold)
    
    # Predict and evaluate
    y_pred_fold = fold_model.predict(X_val_fold)
    fold_rmsle = rmsle(y_val_fold, y_pred_fold)
    cv_rmsle_scores.append(fold_rmsle)
    
    print(f"Fold {fold + 1}: RMSLE = {fold_rmsle:.6f}")

cv_mean_rmsle = np.mean(cv_rmsle_scores)
cv_std_rmsle = np.std(cv_rmsle_scores)

print(f"\nCross-validation Results:")
print(f"Mean RMSLE: {cv_mean_rmsle:.6f}")
print(f"Std RMSLE: {cv_std_rmsle:.6f}")
print(f"95% Confidence Interval: {cv_mean_rmsle:.6f} ± {1.96 * cv_std_rmsle:.6f}")

# Feature importance analysis
print("\nAnalyzing feature importance...")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop Features Importance (Selected Features Only):")
print(feature_importance.to_string(index=False))

# Generate predictions
print("\nGenerating test predictions...")
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0.001, None)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})

submission.to_csv('submission_optimized_rf_selected_features.csv', index=False)
print(f"Submission saved to 'submission_optimized_rf_selected_features.csv'")

# Final statistics
print(f"\nPrediction Statistics:")
print(f"Test predictions - Min: {test_pred.min():.2f}, Max: {test_pred.max():.2f}")
print(f"Test predictions - Mean: {test_pred.mean():.2f}, Std: {test_pred.std():.2f}")
print(f"Training targets - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
print(f"Training targets - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")

# Feature importance visualization
plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances - Optimized Random Forest (Selected Features)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print(f"\nOptimized model summary:")
print(f"- Total features: {len(X_train.columns)} (reduced from original)")
print(f"- Cross-validation RMSLE: {cv_mean_rmsle:.6f} ± {cv_std_rmsle:.6f}")
print(f"- Model trained with optimized hyperparameters")
print(f"- Feature selection based on importance > 0.001")
print("\nModel training completed successfully!")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def create_age_groups(age):
    """Create age groups based on real data analysis"""
    if age <= 24:
        return 0  # 18-24 (Young Adult)
    elif age <= 34:
        return 1  # 25-34 (Early Adult)
    elif age <= 44:
        return 2  # 35-44 (Middle Adult)
    elif age <= 54:
        return 3  # 45-54 (Late Adult)
    elif age <= 64:
        return 4  # 55-64 (Pre-Senior)
    else:
        return 5  # 65-80 (Senior)

def optimized_feature_engineering(df):
    """
    Feature engineering focusing on top performing features only
    """
    print("Starting optimized feature engineering...")
    
    # Convert Sex to numeric
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # Create age groups
    df['Age_Group'] = df['Age'].apply(create_age_groups)
    
    # === Core Features (Top performers from importance analysis) ===
    
    # BMI calculation
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Heart Rate Percentage
    df['Heart_Rate_pct'] = df['Heart_Rate'] / (220 - df['Age'])
    
    # Duration normalization by age group
    duration_means = {0: 15.405, 1: 15.192, 2: 15.741, 3: 14.986, 4: 15.440, 5: 16.265}
    df['Duration_norm'] = df['Duration'] / df['Age_Group'].map(duration_means)
    
    # Age-Weighted Intensity (normalized) - Top 5 feature
    df['AWI'] = (df['Duration'] * df['Heart_Rate_pct']) / df['Age']
    awi_means = {0: 0.359, 1: 0.273, 2: 0.222, 3: 0.177, 4: 0.162, 5: 0.156}
    df['AWI_norm'] = df['AWI'] / df['Age_Group'].map(awi_means)
    
    # Thermal Stress Index - Top performer
    df['TSI'] = 5 * ((df['Body_Temp'] - 36.5) / (41.5 - 36.5)) + \
                5 * ((df['Heart_Rate'] - 60) / ((220 - df['Age']) - 60))
    
    # Rate of Perceived Exertion
    df['RPE'] = df['Heart_Rate_pct'] + 0.1 * (df['Body_Temp'] - 37)
    
    # Cardiac Load Index with normalization
    df['CLI'] = (df['Heart_Rate'] * df['Duration']) / df['Weight']
    cli_means = {0: 21.712, 1: 21.165, 2: 21.717, 3: 20.356, 4: 21.040, 5: 22.308}
    df['CLI_norm'] = df['CLI'] / df['Age_Group'].map(cli_means)
    
    # Work Load Index
    df['WLI'] = df['Heart_Rate'] * df['Duration'] * df['Weight']
    wli_means = {0: 112944, 1: 112456, 2: 118311, 3: 112966, 4: 117502, 5: 126107}
    df['WLI_norm'] = df['WLI'] / df['Age_Group'].map(wli_means)
    
    # VO2 Proxy with normalization
    df['VO2_Proxy'] = np.where(
        df['Sex'] == 0,  # female
        (0.85 * df['Duration']) / (df['Heart_Rate_pct'] * df['Age']),
        (1.00 * df['Duration']) / (df['Heart_Rate_pct'] * df['Age'])
    )
    vo2_means = {0: 1.317, 1: 0.930, 2: 0.673, 3: 0.488, 4: 0.387, 5: 0.314}
    df['VO2_norm'] = df['VO2_Proxy'] / df['Age_Group'].map(vo2_means)
    
    # === Top Ratio Features ===
    df['Temp_Duration_ratio'] = df['Body_Temp'] / df['Duration']  # Top 3 feature
    df['Thermal_Efficiency'] = df['Duration'] / (df['Body_Temp'] - 36.5)  # Top 4 feature
    df['Duration_HR_ratio'] = df['Duration'] / df['Heart_Rate']
    
    # Heat Production
    df['Heat_Production'] = (df['Body_Temp'] - 36.5) * df['Weight'] * df['Duration']
    
    # Cardiovascular Fitness
    df['CV_Fitness'] = (220 - df['Age'] - df['Heart_Rate']) / (220 - df['Age'])
    
    # Fatigue Index with normalization
    df['FI'] = (df['Heart_Rate_pct'] ** 2) / df['Duration']
    fi_means = {0: 0.024, 1: 0.028, 2: 0.027, 3: 0.032, 4: 0.036, 5: 0.040}
    df['FI_norm'] = df['FI'] / df['Age_Group'].map(fi_means)
    
    # Heart Rate Percentage normalization
    hr_pct_means = {0: 0.481, 1: 0.500, 2: 0.531, 3: 0.557, 4: 0.596, 5: 0.649}
    df['HR_pct_norm'] = df['Heart_Rate_pct'] / df['Age_Group'].map(hr_pct_means)
    
    # Gender-adjusted Heart Rate
    df['HR_sex_adj'] = df['Heart_Rate'] * (1 + 0.03 * df['Sex'])
    
    # === Top Interaction Features ===
    df['BMI_HR_Duration'] = df['BMI'] * df['Heart_Rate'] * df['Duration']  # #1 feature!
    df['Weight_Duration_HR'] = df['Weight'] * df['Duration'] * df['Heart_Rate_pct']
    
    # === Polynomial Features (Top performers only) ===
    df['Duration_sq'] = df['Duration'] ** 2  # #2 feature!
    df['Heart_Rate_sq'] = df['Heart_Rate'] ** 2
    
    # === Key Gender and Age Interactions ===
    df['Heart_Rate_pct_sex'] = df['Heart_Rate_pct'] * df['Sex']
    df['TSI_sex'] = df['TSI'] * df['Sex']
    df['Duration_sex'] = df['Duration'] * df['Sex']
    df['CLI_sex'] = df['CLI'] * df['Sex']
    
    # Age group interactions
    df['CLI_age_group'] = df['CLI'] * df['Age_Group']
    df['Duration_age_group'] = df['Duration'] * df['Age_Group']
    df['Heart_Rate_pct_age_group'] = df['Heart_Rate_pct'] * df['Age_Group']
    
    # === Additional Important Features ===
    df['Metabolic_Efficiency'] = df['Heart_Rate_pct'] * df['Weight'] / df['Duration']
    
    # BMR calculation and interactions
    df['BMR'] = np.where(
        df['Sex'] == 0,  # female
        10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] - 161,
        10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5
    )
    df['BMR_Duration'] = df['BMR'] * df['Duration']
    df['BMR_HR_pct'] = df['BMR'] * df['Heart_Rate_pct']
    df['BMR_age_group'] = df['BMR'] * df['Age_Group']
    
    # Additional ratios
    df['BMR_HR_ratio'] = df['BMR'] / df['Heart_Rate']
    
    print(f"Feature engineering completed. New shape: {df.shape}")
    return df

def prepare_data():
    """Prepare training and test data with optimized features"""
    # Combine for consistent feature engineering
    df_merged = pd.concat([train.drop(columns='Calories'), test], axis=0).reset_index(drop=True)
    
    # Apply feature engineering
    df_processed = optimized_feature_engineering(df_merged)
    
    # Split back to train/test
    X_train = df_processed.iloc[:len(train)].drop(columns=['id'])
    X_test = df_processed.iloc[len(train):].drop(columns=['id'])
    y_train = train['Calories'].values
    
    # Select only top performing features (importance > 0.001)
    top_features = [
        'BMI_HR_Duration', 'Temp_Duration_ratio', 'Thermal_Efficiency',
        'AWI_norm', 'Duration_norm', 'TSI', 'Duration_HR_ratio',
        'Weight_Duration_HR', 'CLI_norm',
        'RPE', 'Heat_Production', 'WLI_norm', 'VO2_norm', 'HR_sex_adj', 
        'CV_Fitness', 'FI_norm', 'HR_pct_norm',
        'CLI_age_group', 'Heart_Rate_pct_sex', 'TSI_sex', 'AWI', 'Duration_age_group',
        'Metabolic_Efficiency', 'Duration_sex', 'Heart_Rate_pct_age_group',
        'BMR_Duration', 'BMR_HR_pct', 'CLI_sex',
        'BMR_HR_ratio', 'BMR_age_group'
    ]
    
    # Filter to only include features that exist and are important
    available_features = [f for f in top_features if f in X_train.columns]
    X_train_selected = X_train[available_features]
    X_test_selected = X_test[available_features]
    
    print(f"Selected {len(available_features)} top features from {len(X_train.columns)} total features")
    
    return X_train_selected, X_test_selected, y_train

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    y_pred = np.clip(y_pred, 0.001, None)
    y_true = np.clip(y_true, 0.001, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Prepare data
X_train, X_test, y_train = prepare_data()

print(f"\nFinal dataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")

print(f"\nSelected features: {list(X_train.columns)}")

# Optimized Random Forest parameters from your analysis
optimized_params = {
    'n_estimators': 250,
    'max_depth': 19,
    'min_samples_split': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\nUsing optimized Random Forest parameters:")
for param, value in optimized_params.items():
    print(f"  {param}: {value}")

# Train model
print("\nTraining Random Forest with optimized parameters and selected features...")
model = RandomForestRegressor(**optimized_params)
model.fit(X_train, y_train)

# Model evaluation
print("\nEvaluating model performance...")
train_pred = model.predict(X_train)
train_rmsle = rmsle(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)

print(f"\nTraining Performance:")
print(f"Training RMSLE: {train_rmsle:.6f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Training R²: {train_r2:.4f}")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Train fold model
    fold_model = RandomForestRegressor(**optimized_params)
    fold_model.fit(X_train_fold, y_train_fold)
    
    # Predict and evaluate
    y_pred_fold = fold_model.predict(X_val_fold)
    fold_rmsle = rmsle(y_val_fold, y_pred_fold)
    cv_rmsle_scores.append(fold_rmsle)
    
    print(f"Fold {fold + 1}: RMSLE = {fold_rmsle:.6f}")

cv_mean_rmsle = np.mean(cv_rmsle_scores)
cv_std_rmsle = np.std(cv_rmsle_scores)

print(f"\nCross-validation Results:")
print(f"Mean RMSLE: {cv_mean_rmsle:.6f}")
print(f"Std RMSLE: {cv_std_rmsle:.6f}")
print(f"95% Confidence Interval: {cv_mean_rmsle:.6f} ± {1.96 * cv_std_rmsle:.6f}")

# Feature importance analysis
print("\nAnalyzing feature importance...")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop Features Importance (Selected Features Only):")
print(feature_importance.to_string(index=False))

# Generate predictions
print("\nGenerating test predictions...")
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 0.001, None)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})

submission.to_csv('submission_optimized_rf_selected_features_2.csv', index=False)
print(f"Submission saved to 'submission_optimized_rf_selected_features.csv'")

# Final statistics
print(f"\nPrediction Statistics:")
print(f"Test predictions - Min: {test_pred.min():.2f}, Max: {test_pred.max():.2f}")
print(f"Test predictions - Mean: {test_pred.mean():.2f}, Std: {test_pred.std():.2f}")
print(f"Training targets - Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
print(f"Training targets - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")

# Feature importance visualization
plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances - Optimized Random Forest (Selected Features)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print(f"\nOptimized model summary:")
print(f"- Total features: {len(X_train.columns)} (reduced from original)")
print(f"- Cross-validation RMSLE: {cv_mean_rmsle:.6f} ± {cv_std_rmsle:.6f}")
print(f"- Model trained with optimized hyperparameters")
print(f"- Feature selection based on importance > 0.001")
print("\nModel training completed successfully!")

