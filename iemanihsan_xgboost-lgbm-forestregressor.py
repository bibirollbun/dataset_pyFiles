import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from joblib import parallel_backend, dump
import warnings
warnings.filterwarnings("ignore")
import time


# Start timer
start_time = time.time()


# RMSLE calculation with epsilon to prevent log(0)
def rmsle(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None) + epsilon
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Enhanced feature engineering
def create_features(df):
    df = df.copy()
    
    # Basic features
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # Harris-Benedict BMR calculation
    df['BMR'] = df.apply(lambda x: 
        (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] + 5) if x['Sex'] == 'male' 
        else (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] - 161), axis=1)
    
    # Heart rate related features
    df['Max_HR'] = 208 - 0.7 * df['Age']
    df['HR_Reserve'] = df['Heart_Rate'] / df['Max_HR']
    df['HR_Zone'] = pd.cut(df['HR_Reserve'], 
                          bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                          labels=[1, 2, 3, 4, 5]).astype('float')
    
    # Performance related features
    df['MET'] = (df['Heart_Rate'] * df['Duration']) / (df['BMR'] / 24 / 60)
    df['Exercise_Intensity'] = df['Heart_Rate'] * df['Duration'] / 60
    df['Temp_HR_Ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['BMR_HR_Interaction'] = df['BMR'] * df['HR_Reserve']
    
    # Non-linear transformations
    df['HR_Squared'] = df['Heart_Rate'] ** 2
    df['Duration_Sqrt'] = np.sqrt(df['Duration'])
    df['Duration_Log'] = np.log1p(df['Duration'])
    df['Thermal_Stress'] = (df['Body_Temp'] - 36.5) * df['HR_Reserve']
    
    # Interaction terms
    df['Weight_Duration'] = df['Weight'] * df['Duration_Sqrt']
    df['BMI_HR'] = df['BMI'] * df['Heart_Rate']
    
    return df


# Load and preprocess data
print("Loading and preprocessing data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').dropna()
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Apply feature engineering
train_processed = create_features(train)
test_processed = create_features(test)

# Log transform target variable
y = np.log1p(train_processed['Calories'])
X = train_processed.drop(['Calories', 'Sex', 'id'], axis=1)
X_test = test_processed.drop(['Sex', 'id'], axis=1)


# Create preprocessing pipelines

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson'))
])

preprocessor = ColumnTransformer(
    [('num', numerical_transformer, numerical_features)],
    remainder='drop',
    n_jobs=-1
)


# Base models with optimized hyperparameters
def get_xgb():
    params = {
        'objective': 'reg:squaredlogerror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'n_estimators': 200,
        'tree_method': 'hist',
        'random_state': 42
    }
    try:
        import cupy
        params['device'] = 'cuda'
        print("Using CUDA for XGBoost")
    except ImportError:
        print("Using CPU for XGBoost")
    return xgb.XGBRegressor(**params)

def get_lgbm():
    return lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=32,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )

def get_rf():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )


# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop 10 most important features:")
        for i in range(min(10, len(feature_names))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


# Stacking ensemble with Ridge as meta-learner
base_models = [
    ('xgb', get_xgb()),
    ('lgbm', get_lgbm()),
    ('rf', get_rf())
]

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    n_jobs=-1,
    passthrough=True  # Include original features in final model
)


# Full pipeline
ensemble_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stacked_model)
])


# Perform cross-validation with parallel processing
print("Validating ensemble model with cross-validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

with parallel_backend('threading', n_jobs=-1):
    cv_scores = cross_val_score(
        ensemble_pipeline, 
        X, 
        y, 
        cv=cv, 
        scoring=rmsle_scorer, 
        verbose=1
    )

print(f"Cross-validation RMSLE scores: {np.abs(cv_scores)}")
print(f"Mean RMSLE: {np.abs(cv_scores).mean():.5f}")
print(f"Std RMSLE: {np.abs(cv_scores).std():.5f}")


# Train final model with all data
print("Training final ensemble model...")
ensemble_pipeline.fit(X, y)


# Print feature importance for RF (one of the base models)
rf_model = ensemble_pipeline.named_steps['model'].estimators_[2][1]
preprocessed_X = ensemble_pipeline.named_steps['preprocessor'].transform(X)
if isinstance(preprocessed_X, np.ndarray):
    feature_names = [f"feature_{i}" for i in range(preprocessed_X.shape[1])]
else:
    feature_names = numerical_features
analyze_feature_importance(rf_model, feature_names)


# Make predictions
test_predictions = np.expm1(ensemble_pipeline.predict(X_test))
test_predictions = np.clip(test_predictions, 0, None)

# Create submission
submission = pd.DataFrame({'id': test['id'], 'Calories': test_predictions})
submission.to_csv('ensemble_submission.csv', index=False)
print("Ensemble submission file created!")

# Save model
dump(ensemble_pipeline, 'ensemble_model.pkl')
print("Ensemble model saved to disk.")


# Calculate and print elapsed time
elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

