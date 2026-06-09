# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor

# Feature selection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Styling
plt.style.use('ggplot')
sns.set_palette("Set2")
%matplotlib inline


# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Display dataset information
print("Training Data Overview:")
print("=" * 50)
train.info()


# Display first few rows
print("\nFirst 5 rows of training data:")
display(train.head())


# Identify target variable
target_col = [col for col in train.columns if col not in test.columns][0]
print(f"Target variable: {target_col}")


# Check for missing values
print("\nMissing values summary:")
print("Train data:")
print(train.isnull().sum().sum())
print("Test data:")
print(test.isnull().sum().sum())


print("Performing feature engineering...")

# Separate features and target
X_train = train.drop(columns=[target_col])
y_train = train[target_col]
X_test = test.copy()

# Store ID column if exists
id_col = None
for col in ['id', 'ID', 'Id']:
    if col in X_train.columns:
        id_col = col
        train_ids = X_train[id_col]
        test_ids = X_test[id_col]
        X_train = X_train.drop(columns=[id_col])
        X_test = X_test.drop(columns=[id_col])
        break

# Create demographic ratio features
feature_cols = X_train.columns.tolist()

# Age-based ratios
age_cols = [col for col in feature_cols if 'AGE' in col]
if age_cols and 'AGE_25_PLUS_PCT' in feature_cols:
    # Youth ratio (under 25)
    youth_cols = [col for col in age_cols if any(x in col for x in ['U18', '18_24'])]
    if youth_cols:
        X_train['youth_ratio'] = X_train[youth_cols].sum(axis=1)
        X_test['youth_ratio'] = X_test[youth_cols].sum(axis=1)
    
    # Senior ratio (65+)
    senior_cols = [col for col in age_cols if any(x in col for x in ['65_69', '70_79', '80_PLUS'])]
    if senior_cols:
        X_train['senior_ratio'] = X_train[senior_cols].sum(axis=1)
        X_test['senior_ratio'] = X_test[senior_cols].sum(axis=1)

# Veteran ratios
if 'VETERAN_POP_PCT' in feature_cols:
    X_train['veteran_ratio'] = X_train['VETERAN_POP_PCT']
    X_test['veteran_ratio'] = X_test['VETERAN_POP_PCT']

# Disability ratios
if 'DISABILITY_POP_PCT' in feature_cols:
    X_train['disability_ratio'] = X_train['DISABILITY_POP_PCT']
    X_test['disability_ratio'] = X_test['DISABILITY_POP_PCT']

# Family household ratios
if 'FAMILY_HH_CHILD_LT18_PCT' in feature_cols and 'FAMILY_HH_TOTAL' in feature_cols:
    X_train['family_with_children_ratio'] = X_train['FAMILY_HH_CHILD_LT18_PCT'] / X_train['FAMILY_HH_TOTAL'].replace(0, 1)
    X_test['family_with_children_ratio'] = X_test['FAMILY_HH_CHILD_LT18_PCT'] / X_test['FAMILY_HH_TOTAL'].replace(0, 1)

# Race/ethnicity diversity index
race_cols = [col for col in feature_cols if 'RACE' in col and 'PCT' in col]
if race_cols:
    # Calculate diversity using Shannon entropy
    race_data = X_train[race_cols].div(X_train[race_cols].sum(axis=1).replace(0, 1), axis=0)
    race_data_test = X_test[race_cols].div(X_test[race_cols].sum(axis=1).replace(0, 1), axis=0)
    
    # Replace negative values and infinities with 0
    race_data = race_data.replace([np.inf, -np.inf], 0).fillna(0)
    race_data_test = race_data_test.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate entropy
    entropy = -((race_data * np.log(race_data + 1e-10)).sum(axis=1))
    entropy_test = -((race_data_test * np.log(race_data_test + 1e-10)).sum(axis=1))
    
    X_train['diversity_index'] = entropy
    X_test['diversity_index'] = entropy_test

# Handle any infinities or NaN values created
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Features after engineering: {X_train.shape[1]}")

# Visualize new features
new_features = [col for col in X_train.columns if col not in feature_cols]
if new_features:
    fig, axes = plt.subplots(1, len(new_features), figsize=(5*len(new_features), 4))
    if len(new_features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(new_features):
        axes[i].scatter(X_train[feature], y_train, alpha=0.6)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target_col)
        axes[i].set_title(f'{feature} vs {target_col}')
    
    plt.tight_layout()
    plt.show()


print("\nPerforming feature selection...")

# Remove constant features
constant_features = []
for col in X_train.columns:
    if X_train[col].nunique() == 1:
        constant_features.append(col)

if constant_features:
    print(f"Removing {len(constant_features)} constant features")
    X_train = X_train.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

# Select top features using mutual information
selector = SelectKBest(score_func=mutual_info_regression, k=min(30, X_train.shape[1]))
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    'feature': X_train.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("\nTop features by mutual information:")
display(feature_scores.head(15))

# Visualize feature importance
plt.figure(figsize=(10, 8))
top_n = min(15, len(feature_scores))
sns.barplot(x='score', y='feature', data=feature_scores.head(top_n))
plt.title(f'Top {top_n} Features by Mutual Information')
plt.xlabel('Mutual Information Score')
plt.tight_layout()
plt.show()

# Keep top features
top_features = feature_scores.nlargest(min(30, len(feature_scores)), 'score')['feature'].tolist()
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

print(f"Selected {len(top_features)} features for modeling")


print("\nTraining models...")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, random_state=42, verbose=False),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'Bagging': BaggingRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'BayesianRidge': BayesianRidge(),
    'HuberRegressor': HuberRegressor(max_iter=5000),
    'SGDRegressor': SGDRegressor(max_iter=5000, tol=1e-3, random_state=42),
    'PassiveAggressive': PassiveAggressiveRegressor(max_iter=5000, random_state=42)
}

# Cross-validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    
    cv_results.append({
        'model': name,
        'mean_rmse': rmse_scores.mean(),
        'std_rmse': rmse_scores.std(),
        'scores': rmse_scores
    })
    
    print(f"{name}: RMSE = {rmse_scores.mean():.6f} (+/- {rmse_scores.std():.6f})")

# Create results dataframe
results_df = pd.DataFrame(cv_results).sort_values('mean_rmse')

# Visualize model performance
plt.figure(figsize=(10, 6))
sns.barplot(x='mean_rmse', y='model', data=results_df, palette='viridis')
plt.title('Model Performance Comparison (Lower RMSE is Better)')
plt.xlabel('RMSE')
plt.tight_layout()
plt.show()

# Select best model
best_model_name = results_df.iloc[0]['model']
print(f"\nBest model: {best_model_name}")


print("\nTraining final models for ensemble...")

# Train top 3 models for ensemble
top_models = results_df.head(3)['model'].tolist()
predictions = []
model_performances = []

for name in top_models:
    print(f"Training {name}...")
    model = models[name]
    model.fit(X_train_scaled, y_train)
    
    # Training performance
    train_pred = model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    pred = model.predict(X_test_scaled)
    predictions.append(pred)
    model_performances.append({
        'model': name,
        'train_rmse': train_rmse,
        'train_r2': train_r2
    })

# Ensemble predictions (weighted average based on CV scores)
weights = []
for name in top_models:
    model_result = results_df[results_df['model'] == name].iloc[0]
    # Lower RMSE gets higher weight
    weight = 1 / model_result['mean_rmse']
    weights.append(weight)

weights = np.array(weights) / np.sum(weights)
final_predictions = np.average(predictions, weights=weights, axis=0)

print("\nEnsemble weights:")
for i, name in enumerate(top_models):
    print(f"{name}: {weights[i]:.4f}")


# Create Submission
print("\nCreating submission file...")

# Ensure predictions match submission format
if id_col:
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: final_predictions
    })
else:
    submission = pd.DataFrame({
        'index': range(len(final_predictions)),
        target_col: final_predictions
    })

# Match sample submission format
submission.columns = sample_sub.columns
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created!")
print(f"Submission shape: {submission.shape}")
print("\nFirst few predictions:")
display(submission.head(10))


# Feature Importance from best tree-based model
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']:
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': top_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 most important features:")
        display(feature_importance.head(20))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()


# Prediction distribution visualization
plt.figure(figsize=(12, 5))

# Training data distribution
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=20, alpha=0.7, label='Actual (Train)', color='blue')
plt.xlabel('Homeless Rate')
plt.ylabel('Frequency')
plt.title('Training Data Distribution')
plt.legend()

# Test predictions distribution
plt.subplot(1, 2, 2)
plt.hist(final_predictions, bins=20, alpha=0.7, label='Predicted (Test)', color='green')
plt.xlabel('Homeless Rate')
plt.ylabel('Frequency')
plt.title('Test Predictions Distribution')
plt.legend()

plt.tight_layout()
plt.show()


# Model performance summary
performance_df = pd.DataFrame(model_performances)
print("\nModel Performance on Training Data:")
display(performance_df)

print("\nProcess completed successfully!")

