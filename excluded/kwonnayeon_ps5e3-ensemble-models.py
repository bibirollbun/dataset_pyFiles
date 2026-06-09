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


# Basic data handling
import pandas as pd
import numpy as np

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Data preprocessing
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# Model evaluation & validation
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.base import clone

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Boosting frameworks
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Optimization
import optuna

# Deep learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Load competition data
# - Training dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').set_index('id')

# - Testing dataset
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv').set_index('id')

# Load original data (Hong Kong rainfall dataset)
original = pd.read_csv("/kaggle/input/hongkongrainfall/hongkong.csv", encoding='latin-1')

# Load submission template
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# Print basic dataset information
print("\nData shape (rows, columns):")
print(original.shape)

# Display column names
print("\nColumn information:")
print(original.columns)

# Preview first 5 rows
print("\nFirst 5 rows of data:")
print(original.head())

# Summary statistics for numerical columns
print("\nStatistical information:")
print(original.describe())

# Check data types for each column
print("\nData type information:")
print(original.dtypes)

# Check for missing values in each column
print("\nMissing value check:")
print(original.isnull().sum())


# Explore rainfall data
print("Rainfall data type:", original['rainfall'].dtype)
print("Unique rainfall values:", original['rainfall'].unique())
print("Rainfall value distribution:\n", original['rainfall'].value_counts())


# Explore competition training data
print("\nData shape (rows, columns):")
print(train_data.shape)

# Display column names
print("\nColumn information:")
print(train_data.columns)

# Preview first 5 rows
print("\nFirst 5 rows of data:")
print(train_data.head())

# Summary statistics for numerical columns
print("\nStatistical information:")
print(train_data.describe())

# Check data types for each column
print("\nData type information:")
print(train_data.dtypes)

# Check for missing values in each column
print("\nMissing value check:")
print(train_data.isnull().sum())


# Set graph size
plt.figure(figsize=(15, 10))

# Select numerical columns
numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns

# Create histograms
for i, column in enumerate(numeric_columns):
    if column != 'id' and column != 'rainfall':  # Exclude target variable and ID
        plt.subplot(3, 4, i + 1)
        sns.histplot(train_data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.tight_layout()

plt.show()


# Visualize target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='rainfall', data=train_data)
plt.title('Target Variable Distribution (rainfall)')
plt.ylabel('Count')

rain_counts = [train_data['rainfall'].value_counts()[0], train_data['rainfall'].value_counts()[1]]

for i, count in enumerate(rain_counts):
    plt.text(i, count + 10, f'{count} ({count/len(train_data):.1%})', 
             ha='center', va='bottom')
plt.show()


# Check correlation between numerical variables
plt.figure(figsize=(12, 10))
corr = train_data.select_dtypes(include=['int64', 'float64']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Print correlation with target variable
target_corr = corr['rainfall'].sort_values(ascending=False)
print("Features correlation with target (rainfall):")
print(target_corr)


# Outlier detection
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns):
    if column != 'id' and column != 'rainfall':  # Exclude target variable and ID
        plt.subplot(3, 4, i + 1)
        sns.boxplot(x='rainfall', y=column, data=train_data)
        plt.title(f'Boxplot of {column} by rainfall')
        plt.tight_layout()
plt.show()


# Temporal Pattern Analysis using the 'day' variable
if 'day' in train_data.columns:
    # Check the data type of 'day'
    print(f"Data type of 'day': {train_data['day'].dtype}")
    
    # If 'day' is not already in datetime format, convert it if possible
    # (Skip this if 'day' is just a number without actual date meaning)
    
    # Analyze rainfall probability by day
    plt.figure(figsize=(12, 6))
    daily_rain_prob = train_data.groupby('day')['rainfall'].mean()
    daily_rain_prob.plot(kind='line')
    plt.title('Daily Rain Probability')
    plt.xlabel('Day')
    plt.ylabel('Probability of Rain')
    plt.grid(True)
    plt.show()
    
    # Count of rainy days vs non-rainy days by day
    plt.figure(figsize=(12, 6))
    rain_counts_by_day = pd.crosstab(train_data['day'], train_data['rainfall'])
    rain_counts_by_day.plot(kind='bar', stacked=True)
    plt.title('Rain vs No Rain Counts by Day')
    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.legend(['No Rain', 'Rain'])
    plt.show()


# Source: Partially adapted from Kaggle notebook 'Rainfall Prediction +0.90 with Catboost' by Berker ERYILMAZ
# Original notebook features were modified and extended for this implementation

def add_features(df):
    """
    Comprehensive feature engineering for weather data
    """
    df = df.copy()  # Create a copy to avoid modifying the original
    
    # Handle string format data
    for col in ['rainfall', 'sunshine', 'radiation', 'evaporation']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Circular encoding for wind direction
    wind_rad = np.radians(df['winddirection'])
    df['winddirection_sin'] = np.sin(wind_rad)
    df['winddirection_cos'] = np.cos(wind_rad)
    
    # 2. Temperature features to handle multicollinearity
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    
    # 3. Create interaction features
    df["dew_humidity"] = df["dewpoint"] * df["humidity"]
    df["cloud_windspeed"] = df["cloud"] * df["windspeed"]
    df["cloud_to_humidity"] = df["cloud"] / df["humidity"]
    df["temp_to_sunshine"] = df["sunshine"] / df["temparature"]
    df['wind_temp_interaction'] = df['windspeed'] * df['temparature']
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1)
    df["dew_humidity/sun"] = df["dewpoint"] * df["humidity"] / (df['sunshine'] + 1)
    df["dew_humidity_+"] = df["dewpoint"] * df["humidity"]
    df['humidity_sunshine_*'] = df["humidity"] * df['sunshine']
    df["cloud_humidity/pressure"] = (df["cloud"] * df["humidity"]) / df["pressure"]
    
    # 4. Cloud, Sunshine, Humidity interactions (from first code snippet)
    df['cloud_humidity'] = df['cloud'] * df['humidity']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['humidity_sunshine'] = df['humidity'] * df['sunshine']
    
    # 5. Extract temporal features
    df['month'] = ((df['day'] - 1) // 30 + 1).clip(upper=12)
    df['season'] = df['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                     else 2 if 6 <= x <= 8  # Summer
                                     else 3 if 9 <= x <= 11  # Autumn
                                     else 0)  # Winter
    
    # 6. Seasonal features
    df['season_cloud_trend'] = df['cloud'] * df['season']
    df['season_cloud_deviation'] = df['cloud'] - df.groupby('season')['cloud'].transform('mean')
    df['season_temperature'] = df['temparature'] * df['season']
    
    # 7. Drop unnecessary columns
    # Keep columns needed for additional features or by models
    df = df.drop(columns=["month", "maxtemp", "winddirection", "humidity", 
                         "temparature", "pressure", "day", "season"])
    
    return df


# Apply to train and test datasets
train_data = add_features(train_data)
test_data = add_features(test_data)

# Check the result
print(train_data.head())
print(train_data.shape)


# Select features and target variable
X = train_data.drop(['rainfall'], axis=1)
y = train_data['rainfall']
X_test = test_data.copy()


# Print columns with NaN values
print(X_test.isna().sum())


# Fill NaN values in wind direction columns with their respective modes
for col in ['winddirection_sin', 'winddirection_cos']:
    X_test[col].fillna(X[col].mode()[0], inplace=True)


# Alternative: Using circular mean for directional data
# Wind direction is circular data (0-360 degrees) requiring special handling for mean calculation
# import numpy as np
# angles = np.radians(x['winddirection'].dropna())
# mean_sin = np.mean(np.sin(angles))
# mean_cos = np.mean(np.cos(angles))
# mean_angle = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
# x_test['winddirection'].fillna(mean_angle, inplace=True)

# Source: Kaggle notebook 'PS5E3 | Rainfall Prediction | Classification' by Minato Namikaze


models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_estimators=100),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=100, verbose=0)
}

def evaluate_all_models(X, y, X_test=None, test_index=None, folds=5):
    # Train models using StratifiedKFold CV
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    auc_scores = {}
    roc_curves = {}
    test_preds = {}
    
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        
        if X_test is not None:
            test_fold_preds = np.zeros((folds, len(X_test)))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if isinstance(model, lgb.LGBMClassifier):
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50, verbose=0)])
            elif isinstance(model, xgb.XGBClassifier):
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=0)
            elif isinstance(model, CatBoostClassifier):
                model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=0)
            else:
                model.fit(X_train, y_train)
            
            # Save out-of-fold predictions
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            
            # Make test predictions if test data provided
            if X_test is not None:
                test_fold_preds[fold] = model.predict_proba(X_test)[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y, oof_preds)
        auc_scores[name] = auc_score
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, oof_preds)
        roc_curves[name] = (fpr, tpr, auc_score)
        
        # Average test predictions across folds
        if X_test is not None:
            test_preds[name] = test_fold_preds.mean(axis=0)
        
        print(f"{name}: AUC = {auc_score:.4f}")
    
    return auc_scores, test_preds, roc_curves


# Model evaluation
auc_scores, test_predictions, roc_curves = evaluate_all_models(
    X, y, 
    X_test=X_test,
    test_index=test_data.index,
    folds=5
)


plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()


# Plot AUC scores
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=list(auc_scores.keys()), y=list(auc_scores.values()))

# Annotate the bars with AUC scores
for i, score in enumerate(auc_scores.values()):
    ax.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45)
plt.ylabel("AUC Score")
plt.xlabel("Models")
plt.title("Model AUC Score Comparison")
plt.ylim(0.5, 1)  
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()

# Source: Kaggle notebook 'Rainfall Prediction +0.90 with Catboost' by Berker ERYILMAZ


# Get the best model name (already calculated)
best_model_name = max(auc_scores.items(), key=lambda x: x[1])[0]

# Get the actual model object
best_model = models[best_model_name]

# Check for feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
elif hasattr(best_model, 'coef_'):
    # For logistic regression or linear models
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'
else:
    print(f"Model {best_model_name} doesn't provide feature importances or coefficients")
    feature_importance = None

# Only visualize if feature importance is available
if feature_importance is not None:
    # Get feature names (verify these are the actual features used in training)
    feature_names = X.columns  # X is the feature DataFrame used for model training
    
    # Create feature importance DataFrame
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance in descending order
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    # Display only top 15 features (optional)
    feature_df = feature_df.head(15)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title(f"{importance_type} for {best_model_name}")
    plt.tight_layout()
    plt.show()


# Select the best model based on AUC
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]

# Create StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': X.columns,  # Use X.columns instead of train_data.drop
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# List of top N features to try 
top_feature_counts = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Variables to track the best AUC and corresponding top features
best_auc_top = 0
best_top_n = 0
best_oof_preds_top = None

# Loop over different top feature counts
for top_n in top_feature_counts:
    print(f"\nTesting with top {top_n} features...")
    
    # Select the top N features
    top_features = feature_df.head(top_n)['Feature'].tolist()
    
    # Prepare the data with the selected top N features
    X_top = X[top_features]  # Simplified indexing
    X_test_top = X_test[top_features]  # Simplified indexing

    # Initialize out-of-fold predictions
    oof_preds_top = np.zeros(len(y))
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_top, y)):
        X_train, X_val = X_top.iloc[train_idx], X_top.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone the model to avoid data leakage between folds
        model_clone = clone(best_model)
        
        # Train the model - handle different model types
        if isinstance(model_clone, lgb.LGBMClassifier):
            model_clone.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=0)])
        elif isinstance(model_clone, xgb.XGBClassifier):
            model_clone.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=0)
        elif isinstance(model_clone, CatBoostClassifier):
            model_clone.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=0)
        else:
            model_clone.fit(X_train, y_train)
            
        # Predict on validation fold
        oof_preds_top[val_idx] = model_clone.predict_proba(X_val)[:, 1]

    # Calculate and print AUC score for top N features model
    auc_score_top = roc_auc_score(y, oof_preds_top)
    print(f"AUC for top {top_n} features: {auc_score_top:.4f}")
    
    # Track the best AUC and corresponding features
    if auc_score_top > best_auc_top:
        best_auc_top = auc_score_top
        best_top_n = top_n
        best_oof_preds_top = oof_preds_top.copy()

print(f"\nBest performance achieved with top {best_top_n} features. AUC: {best_auc_top:.4f}")

# Get the best features
best_features = feature_df.head(best_top_n)

# Visualize the best features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=best_features)
plt.title(f"Top {best_top_n} Features ({importance_type}) for {best_model_name}")
plt.tight_layout()
plt.show()

print("=" * 50)
print(f"Best Model: {best_model_name}")
print(f"Best AUC: {best_auc_top:.4f} using Top {best_top_n} Features")
print("=" * 50)


def create_ensemble_prediction(auc_scores, test_predictions, test_index):
    """
    Create an ensemble prediction using the top 3 models based on AUC scores
    
    Parameters:
    -----------
    auc_scores : dict
        Dictionary of model names and their AUC scores
    test_predictions : dict
        Dictionary of model names and their test predictions
    test_index : array-like
        Index values for test data
        
    Returns:
    --------
    submission : pandas DataFrame
        Submission dataframe with ensemble predictions
    """
    # Get the top 3 models based on AUC scores
    top_models = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_model_names = [model[0] for model in top_models]
    
    print("Creating ensemble with top 3 models:")
    for name, auc in top_models:
        print(f"- {name}: AUC = {auc:.4f}")
    
    # Calculate weights based on AUC scores
    total_auc = sum(auc for _, auc in top_models)
    weights = [auc/total_auc for _, auc in top_models]
    
    # Create weighted ensemble predictions
    ensemble_preds = np.zeros(len(test_predictions[top_model_names[0]]))
    for i, (name, _) in enumerate(top_models):
        ensemble_preds += weights[i] * test_predictions[name]
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_index,
        'rainfall': ensemble_preds
    })
    
    # Save the submission
    submission.to_csv('ensemble_submission.csv', index=False)
    print("Ensemble submission saved as 'ensemble_submission.csv'")
    
    return submission


# Create ensemble prediction
ensemble_submission = create_ensemble_prediction(
    auc_scores, 
    test_predictions, 
    test_data.index
)

# Save as submission.csv (overwrite the ensemble_submission.csv)
ensemble_submission.to_csv('submission.csv', index=False)
print("\nEnsemble prediction saved as 'submission.csv'")

