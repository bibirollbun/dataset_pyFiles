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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


# Set random seed for reproducibility
np.random.seed(42)

print("====================================")
print("Advanced Rainfall Prediction Model")
print("====================================")

# Load the data
print("\nLoading and preprocessing data...")
base_dir = '/kaggle/input/playground-series-s5e3/'
train_data = pd.read_csv(base_dir + 'train.csv')
test_data = pd.read_csv(base_dir + 'test.csv')


# Check for missing values
print("Checking for missing values:")
print("  Train data missing values:", train_data.isna().sum().sum())
print("  Test data missing values:", test_data.isna().sum().sum())
print("  Missing values by column in test data:")
missing_cols = test_data.isna().sum()[test_data.isna().sum() > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("  No missing values in specific columns")


# Separate features and target
X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']
test_features = test_data.drop(['id'], axis=1)

# Print class distribution
print("\nClass distribution in training data:")
print(y.value_counts(normalize=True) * 100)

# ================================================================
# Advanced Feature Engineering
# ================================================================
print("\nPerforming advanced feature engineering...")

# 1. Create cyclical features for day (to capture seasonality)
def create_cyclical_features(df, col, period):
    """Create sin and cos features to capture cyclical nature of time variables"""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/period)
    return df

# Apply to both train and test
X = create_cyclical_features(X, 'day', 365)
test_features = create_cyclical_features(test_features, 'day', 365)


# 2. Create seasonal indicators (quarters, meteorological seasons)
def add_season_features(df, day_col):
    """Add season indicators based on day of year"""
    # Simple season (4 seasons)
    day = df[day_col]
    
    # Meteorological seasons in Northern Hemisphere
    conditions = [
        (day >= 1) & (day <= 59),     # Winter: Jan 1 - Feb 28
        (day >= 60) & (day <= 151),   # Spring: Mar 1 - May 31
        (day >= 152) & (day <= 243),  # Summer: Jun 1 - Aug 31
        (day >= 244) & (day <= 334),  # Fall: Sep 1 - Nov 30
        (day >= 335) & (day <= 365)   # Winter: Dec 1 - Dec 31
    ]
    seasons = [0, 1, 2, 3, 0]  # 0:Winter, 1:Spring, 2:Summer, 3:Fall
    df['season'] = np.select(conditions, seasons, default=0)
    
    # Create dummy variables for seasons
    for season in range(4):
        df[f'season_{season}'] = (df['season'] == season).astype(int)
    
    # Quarter of the year (1, 2, 3, 4)
    df['quarter'] = pd.cut(df[day_col], bins=[0, 90, 181, 273, 365], labels=[1, 2, 3, 4]).astype(int)
    
    # Create dummy variables for quarters
    for quarter in range(1, 5):
        df[f'quarter_{quarter}'] = (df['quarter'] == quarter).astype(int)
    
    return df

X = add_season_features(X, 'day')
test_features = add_season_features(test_features, 'day')


# 3. Create more weather-related features and interactions
def create_weather_features(df):
    """Create advanced weather-related features"""
    # Temperature-related
    if 'temparature' in df.columns and 'maxtemp' in df.columns and 'mintemp' in df.columns:
        # Temperature range
        df['temp_range'] = df['maxtemp'] - df['mintemp']
        
        # Temperature relative to min/max
        df['temp_relative_min'] = df['temparature'] - df['mintemp']
        df['temp_relative_max'] = df['maxtemp'] - df['temparature']
        
        # Temperature variability
        df['temp_variability'] = df['temp_range'] / df['temparature']
    
    # Humidity and temperature interaction
    if 'humidity' in df.columns:
        # Different variants of humidity interactions
        if 'temparature' in df.columns:
            df['humidity_temp'] = df['humidity'] * df['temparature']
            df['humidity_temp_ratio'] = df['humidity'] / df['temparature']
        
        # Humidity squared (to capture nonlinear effects)
        df['humidity_squared'] = df['humidity'] ** 2
    
    # Cloud and sunshine interactions
    if 'cloud' in df.columns and 'sunshine' in df.columns:
        df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 0.1)  # Adding 0.1 to avoid division by zero
        df['cloud_sunshine_product'] = df['cloud'] * df['sunshine']
    
    # Wind features
    if 'windspeed' in df.columns:
        df['windspeed_squared'] = df['windspeed'] ** 2
        
        if 'winddirection' in df.columns:
            # Create wind component features (x and y direction)
            df['wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
            df['wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))
    
    # Pressure interactions
    if 'pressure' in df.columns:
        if 'temparature' in df.columns:
            df['pressure_temp'] = df['pressure'] * df['temparature']
        if 'humidity' in df.columns:
            df['pressure_humidity'] = df['pressure'] * df['humidity']
    
    return df

X = create_weather_features(X)
test_features = create_weather_features(test_features)


# 4. Create polynomial features for important numerical variables
# Identify important features to use for polynomial expansion (to avoid explosion of features)
important_weather_features = ['pressure', 'humidity', 'cloud', 'windspeed']
X_poly_subset = X[important_weather_features]
test_poly_subset = test_features[important_weather_features]

# Create polynomial features (degree 2) for selected features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_poly_subset)
test_poly = poly.transform(test_poly_subset)

# Convert to DataFrame with appropriate column names
poly_feature_names = poly.get_feature_names_out(important_weather_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
test_poly_df = pd.DataFrame(test_poly, columns=poly_feature_names)

# Add polynomial features to original features
X = pd.concat([X.reset_index(drop=True), X_poly_df.iloc[:, len(important_weather_features):]], axis=1)
test_features = pd.concat([test_features.reset_index(drop=True), test_poly_df.iloc[:, len(important_weather_features):]], axis=1)

print(f"After feature engineering: {X.shape[1]} features created")

# ================================================================
# Feature Selection
# ================================================================
print("\nPerforming feature selection...")

# 1. Check correlation to drop highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]  # Increased threshold to 0.9
print(f"Dropping {len(to_drop)} highly correlated features")

# Drop highly correlated features
X = X.drop(to_drop, axis=1)
test_features = test_features.drop(to_drop, axis=1)


# 2. Calculate mutual information for feature importance
def select_features_by_importance(X, y, threshold=0.01):
    """Select features based on mutual information with target"""
    # Ensure no missing values for mutual_info_classif
    X_temp = SimpleImputer(strategy='median').fit_transform(X)
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X_temp, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    
    # Sort features by importance
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # Select features with importance above threshold
    selected_features = mi_scores[mi_scores > threshold].index.tolist()
    
    print(f"Top 10 features by mutual information:")
    for idx, (feature, score) in enumerate(mi_scores.head(10).items()):
        print(f"  {idx+1}. {feature}: {score:.4f}")
    
    return selected_features

important_features = select_features_by_importance(X, y, threshold=0.005)
print(f"Selected {len(important_features)} features based on importance")

# Keep only important features
X = X[important_features]
test_features = test_features[important_features]
print(f"Final feature count: {X.shape[1]}")


# ================================================================
# Train-Test Split with Stratification
# ================================================================
print("\nSplitting data for training and validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================================
# Model Training and Evaluation
# ================================================================
# Create a list of models
models = {
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM': SVC(
        probability=True, 
        class_weight='balanced',
        random_state=42
    )
}


# Train and evaluate models
print("\nTraining and evaluating models...")
best_model = None
best_score = 0

results = {}

# Create preprocessor
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model with cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Fit on the whole training set
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate PR AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Store results
    results[name] = {
        'cv_score': cv_scores.mean(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'model': pipeline
    }
    
    # Print results
    print(f"Validation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}_v2.png')
    plt.close()
    
    # Check if this is the best model
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = name



# ================================================================
# Create Ensemble Model
# ================================================================
print("\nCreating ensemble model...")

# Get the best 3 models
sorted_models = sorted(results.items(), key=lambda x: x[1]['pr_auc'], reverse=True)
top_models = [model_name for model_name, _ in sorted_models[:3]]

print(f"Top 3 models: {top_models}")

# Create a voting ensemble of the top 3 models
voting_ensemble = VotingClassifier(
    estimators=[
        (name.lower().replace(' ', '_'), results[name]['model']) 
        for name in top_models
    ],
    voting='soft'  # Use probability predictions for voting
)

# Train the voting ensemble
voting_ensemble.fit(X_train, y_train)

# Make predictions with the ensemble
y_pred_ensemble = voting_ensemble.predict(X_val)
y_pred_proba_ensemble = voting_ensemble.predict_proba(X_val)[:, 1]

# Calculate metrics for the ensemble
accuracy_ensemble = accuracy_score(y_val, y_pred_ensemble)
precision_ensemble = precision_score(y_val, y_pred_ensemble)
recall_ensemble = recall_score(y_val, y_pred_ensemble)
f1_ensemble = f1_score(y_val, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_val, y_pred_proba_ensemble)

# Calculate PR AUC for the ensemble
precision_curve_ensemble, recall_curve_ensemble, _ = precision_recall_curve(y_val, y_pred_proba_ensemble)
pr_auc_ensemble = auc(recall_curve_ensemble, precision_curve_ensemble)

# Print ensemble results
print("\nEnsemble Model Results:")
print(f"  Accuracy:  {accuracy_ensemble:.4f}")
print(f"  Precision: {precision_ensemble:.4f}")
print(f"  Recall:    {recall_ensemble:.4f}")
print(f"  F1 Score:  {f1_ensemble:.4f}")
print(f"  ROC AUC:   {roc_auc_ensemble:.4f}")
print(f"  PR AUC:    {pr_auc_ensemble:.4f}")

# Check if ensemble is better than the best individual model
if pr_auc_ensemble > best_score:
    best_score = pr_auc_ensemble
    best_model = "Ensemble"
    print("Ensemble model is the best model!")
    # Add ensemble to results
    results["Ensemble"] = {
        'accuracy': accuracy_ensemble,
        'precision': precision_ensemble,
        'recall': recall_ensemble,
        'f1_score': f1_ensemble,
        'roc_auc': roc_auc_ensemble,
        'pr_auc': pr_auc_ensemble,
        'model': voting_ensemble
    }
else:
    print(f"Best individual model ({best_model}) outperforms the ensemble.")


# Print summary of all models
print("\n===== Model Performance Summary =====")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  PR AUC:    {metrics['pr_auc']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

print(f"\nBest model: {best_model} with PR-AUC score {best_score:.4f}")

# ================================================================
# Fine-tune the Best Model (if it's Random Forest)
# ================================================================
if best_model == "Random Forest":
    print(f"\nFine-tuning the best model: {best_model}")
    
    # Define hyperparameter grid for Random Forest
    param_grid = {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [8, 10, 12],
        'model__min_samples_split': [5, 10, 15],
        'model__min_samples_leaf': [2, 4, 6]
    }
    
    # Create a pipeline for grid search
    best_pipeline = results[best_model]['model']
    
    # Use grid search with cross-validation
    grid_search = GridSearchCV(
        best_pipeline, 
        param_grid, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit grid search to data
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    # Evaluate best model from grid search
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_val)
    y_pred_proba = best_pipeline.predict_proba(X_val)[:, 1]
    
    # Calculate metrics for tuned model
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    # Calculate PR AUC for tuned model
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\nTuned Model Performance:")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Update best model if tuned model is better
    if pr_auc > best_score:
        best_score = pr_auc
        results[best_model]['model'] = best_pipeline
        print(f"Tuned model improved PR-AUC from {best_score:.4f} to {pr_auc:.4f}")
    else:
        print("Tuning did not improve model performance")


# ================================================================
# Generate Predictions on Test Set
# ================================================================
print("\nGenerating predictions for test set...")

# Use the best model to predict on test data
if best_model == "Ensemble":
    final_model = voting_ensemble
else:
    final_model = results[best_model]['model']

# Generate predictions
test_predictions = final_model.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_predictions
})

submission.to_csv('submission_v2.csv', index=False)
print("Submission file created successfully: submission_v2.csv")

# Save feature importances if possible
if best_model in ["Random Forest", "Gradient Boosting", "AdaBoost"]:
    try:
        # Get the feature importances
        if best_model == "Ensemble":
            # For ensemble, use the first model's feature importances if available
            for estimator_name, estimator in voting_ensemble.named_estimators_.items():
                if hasattr(estimator.named_steps['model'], 'feature_importances_'):
                    model = estimator.named_steps['model']
                    break
        else:
            model = results[best_model]['model'].named_steps['model']
        
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Create dataframe of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Save to CSV
        feature_importance_df.to_csv('feature_importances_v2.csv', index=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model}')
        plt.tight_layout()
        plt.savefig('feature_importances_v2.png')
        plt.close()
        
        print("Feature importances saved to feature_importances_v2.csv and feature_importances_v2.png")
    except:
        print("Could not extract feature importances")

print("\nModel training and prediction completed successfully!") 




