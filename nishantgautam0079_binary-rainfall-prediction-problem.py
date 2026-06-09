!pip install pandas numpy scikit-learn xgboost lightgbm catboost


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Handle missing values in training and test sets
print(f"Missing values in training set: {train_df.isnull().sum().sum()}")
print(f"Missing values in test set: {test_df.isnull().sum().sum()}")
imputer = SimpleImputer(strategy='median')
if train_df.isnull().sum().sum() > 0:
    train_imputed = imputer.fit_transform(train_df.select_dtypes(include=[np.number]))
    train_df = pd.DataFrame(train_imputed, columns=train_df.select_dtypes(include=[np.number]).columns)
if test_df.isnull().sum().sum() > 0:
    test_imputed = imputer.fit_transform(test_df.select_dtypes(include=[np.number]))
    test_df = pd.DataFrame(test_imputed, columns=test_df.select_dtypes(include=[np.number]).columns)

# Validate and separate features and target
if 'id' in train_df.columns and 'rainfall' in train_df.columns:
    X = train_df.drop(['id', 'rainfall'], axis=1)
    y = train_df['rainfall']
else:
    raise ValueError("Required columns 'id' or 'rainfall' not found in train_df")
if 'id' in test_df.columns:
    X_test = test_df.drop('id', axis=1)
else:
    raise ValueError("Required column 'id' not found in test_df")

# Check for highly correlated features
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(f"Highly correlated features: {high_corr_features}")
if high_corr_features:
    X = X.drop(high_corr_features, axis=1)
    X_test = X_test.drop(high_corr_features, axis=1)
    print(f"Removed features: {high_corr_features}")

# Create new features with column validation
def create_features(df):
    df = df.copy()
    if 'maxtemp' in df.columns and 'mintemp' in df.columns:
        df['temp_range'] = df['maxtemp'] - df['mintemp']
    else:
        print("Warning: 'maxtemp' or 'mintemp' not found in dataset")
    if 'temperature' in df.columns and 'dewpoint' in df.columns:
        df['dew_depression'] = df['temperature'] - df['dewpoint']
    else:
        print("Warning: 'temperature' or 'dewpoint' not found in dataset")
    if 'day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    else:
        print("Warning: 'day' not found in dataset")
    return df

# Apply feature engineering
X = create_features(X)
X_test = create_features(X_test)

# Handle categorical features (if any)
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
X = pd.DataFrame(X_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Check class distribution
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Rainfall percentage: {y.mean():.2%}")

# Handle class imbalance using class weights
class_weights = {0: len(y) / (2 * (len(y) - y.sum())),
                 1: len(y) / (2 * y.sum())}
print(f"Class weights: {class_weights}")

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models with optimized parameters
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight='balanced'
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss',
        scale_pos_weight=(len(y) - y.sum()) / y.sum()
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42,
        class_weight='balanced'
    ),
    'CatBoost': CatBoostClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=0,
        auto_class_weights='Balanced'
    ),
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    results[name] = auc_score
    print(f"{name} ROC AUC: {auc_score:.4f}")

# Perform hyperparameter tuning on the best model
best_model_name = max(results, key=results.get)
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    base_model = XGBClassifier(random_state=42, eval_metric='logloss',
                              scale_pos_weight=(len(y) - y.sum()) / y.sum())
elif best_model_name == 'LightGBM':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    base_model = LGBMClassifier(class_weight='balanced', random_state=42)
else:  # CatBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    base_model = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced')

grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters for {best_model_name}: {grid_search.best_params_}")

# Cross-validation with best model
print(f"\nPerforming cross-validation with {best_model_name}...")
cv_scores = cross_val_score(best_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='roc_auc', n_jobs=1)
print(f"{best_model_name} Cross-Validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train ensemble model
print("\nTraining ensemble model...")
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft',
    weights=[results[name] for name in models.keys()]
)
ensemble.fit(X_train, y_train)
y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
print(f"Ensemble ROC AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")

# Train final model (ensemble or best tuned model based on validation performance)
final_model = ensemble if roc_auc_score(y_val, y_pred_proba) > results[best_model_name] else best_model
print(f"\nTraining final model ({'ensemble' if final_model == ensemble else best_model_name}) on all data...")
final_model.fit(X, y)

# Make predictions on test set
test_preds = final_model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")
print(f"Sample of predictions:\n{submission.head()}")

# Feature importance analysis (if applicable)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
else:
    print("\nFeature importance not available for ensemble model")

# Additional improvement suggestions
print("\nTo improve your score further, consider:")
print("1. Adding the original dataset mentioned in the competition description")
print("2. Exploring more advanced feature engineering techniques")
print("3. Trying stacking with a meta-learner")
print("4. Incorporating external weather data if allowed by the competition")







