import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.linear_model import Ridge
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
from sklearn.utils.validation import check_X_y, check_array

warnings.filterwarnings("ignore")


# GPU-safe RMSLE
def rmsle(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None) + epsilon
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['BMI'] = X['Weight'] / (X['Height'] / 100) ** 2
        X['Sex_encoded'] = X['Sex'].map({'male': 1, 'female': 0})
        X['BMR'] = X.apply(lambda x: (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] + 5)
                         if x['Sex'] == 'male' else (10 * x['Weight'] + 6.25 * x['Height'] - 5 * x['Age'] - 161), axis=1)
        X['Max_HR'] = 208 - 0.7 * X['Age']
        X['HR_Reserve'] = X['Heart_Rate'] / X['Max_HR']
        X['HR_Zone'] = pd.cut(X['HR_Reserve'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], labels=[1, 2, 3, 4, 5]).astype(float)
        X['MET'] = (X['Heart_Rate'] * X['Duration']) / (X['BMR'] / 24 / 60)
        X['Thermal_Load'] = (X['Body_Temp'] - 36.5) * X['Duration']
        X['Work_Volume'] = X['Duration'] * X['Heart_Rate'] / X['Weight']
        X['Caloric_Rate'] = X['BMR'] * X['Duration'] / 1440
        X['Temp_HR_Ratio'] = X['Body_Temp'] / X['Heart_Rate']
        X['Age_HR_Interaction'] = X['Age'] * X['Heart_Rate'] / 100
        X['Weight_Temp_Interaction'] = X['Weight'] * (X['Body_Temp'] - 36)
        X['BMR_HR_Interaction'] = X['BMR'] * X['HR_Reserve']
        X['HR_Squared'] = X['Heart_Rate'] ** 2
        X['Duration_Sqrt'] = np.sqrt(X['Duration'])
        return X


# ------------------------ Custom Ensemble ------------------------ #
# Simplified and memory-efficient ensemble
class SimpleStackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models if base_models is not None else {}
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.base_models_ = None
        self.meta_model_ = None
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        # Clone and fit base models
        self.base_models_ = {name: clone(model) for name, model in self.base_models.items()}
        
        # Train with simple cross-validation to generate meta features
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                model_copy = clone(model)
                model_copy.fit(X_train, y_train)
                
                # Handle potential multi-output predictions and ensure they're flattened
                preds = model_copy.predict(X_val)
                if len(preds.shape) > 1 and preds.shape[1] > 1:
                    preds = preds[:, 0]  # Take only first column if multi-output
                
                meta_features[val_idx, i] = preds
                
        # Train base models on full data
        for name, model in self.base_models_.items():
            model.fit(X, y)
            
        # Train meta model
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(meta_features, y)
        
        return self
        
    def predict(self, X):
        check_array(X)
        meta_features = np.zeros((X.shape[0], len(self.base_models_)))
        
        for i, (_, model) in enumerate(self.base_models_.items()):
            # Handle potential multi-output predictions
            preds = model.predict(X)
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                preds = preds[:, 0]  # Take only first column if multi-output
                
            meta_features[:, i] = preds
            
        return self.meta_model_.predict(meta_features)
        
    def set_params(self, **params):
        for param, value in params.items():
            if '__' not in param:
                super().set_params(**{param: value})
            else:
                components = param.split('__', 1)
                if components[0] == 'meta_model':
                    self.meta_model.set_params(**{components[1]: value})
                elif components[0] == 'base_models':
                    model_param = components[1].split('__', 1)
                    model_name, param_name = model_param[0], model_param[1]
                    self.base_models[model_name].set_params(**{param_name: value})
        return self


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Clean and prepare data
train.dropna(inplace=True)
y = np.log1p(train['Calories'])
train.drop(columns=['Calories'], inplace=True)


# Feature selection based on feature engineering output
def get_features(df):
    fe = FeatureEngineer()
    transformed = fe.transform(df.head())  # Just to get column names
    categorical_features = ['HR_Zone']
    numerical_features = [col for col in transformed.columns 
                          if col not in categorical_features + ['Sex', 'id']]
    return numerical_features, categorical_features

# Get feature lists without leaking data
numerical_features, categorical_features = get_features(train)


# Create preprocessor
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# Create base models with CPU-only configuration for stability
xgb_model = xgb.XGBRegressor(
    objective='reg:squaredlogerror', 
    eval_metric='rmse',
    tree_method='hist',  # Use hist instead of GPU
    random_state=42,
    n_jobs=1,  # Limit to 1 job to reduce memory usage
    base_score=0.5,  # Default value to ensure proper initialization
    booster='gbtree'  # Explicitly set booster type
)

cat_model = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='RMSE', 
    task_type='CPU',  # Use CPU instead of GPU
    random_seed=42, 
    verbose=0,
    thread_count=1,  # Limit threads to reduce memory usage
    bootstrap_type='Bayesian'  # Use a more stable bootstrapping method
)


# Create a simpler pipeline first to debug
# Use a single model instead of ensemble to verify the pipeline works
simple_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])


# Test the simple pipeline first
print("Testing simple pipeline...")
cv = KFold(n_splits=2, shuffle=True, random_state=42)
simple_scores = []

for train_idx, val_idx in cv.split(train):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    simple_pipeline.fit(X_train, y_train)
    val_preds = simple_pipeline.predict(X_val)
    score = rmsle(y_val, val_preds)
    simple_scores.append(score)
    print(f"Validation RMSLE: {score:.5f}")

print(f"Average RMSLE: {np.mean(simple_scores):.5f}")


# Now create the ensemble with simplified structure
# We'll use just 2 models to reduce complexity
base_models = {'ridge': Ridge(alpha=1.0), 'xgb': xgb_model}
meta_model = Ridge(alpha=1.0)
ensemble = SimpleStackingEnsemble(base_models=base_models, meta_model=meta_model)

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('ensemble', ensemble)
])

# Simplified parameter grid for reduced model complexity
xgb_params = {
    'ensemble__base_models__xgb__n_estimators': [50, 100],
    'ensemble__base_models__xgb__learning_rate': [0.1],
    'ensemble__base_models__xgb__max_depth': [3],
}

ridge_params = {
    'ensemble__base_models__ridge__alpha': [0.1, 1.0, 10.0],
}

param_distributions = {**xgb_params, **ridge_params}


# Minimal search space and iterations
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=2,  # Extremely reduced for debugging
    scoring=rmsle_scorer,
    cv=KFold(n_splits=2, shuffle=True, random_state=42),  # Reduced for debugging
    verbose=2,
    random_state=42,
    n_jobs=1,  # Use single job to avoid memory issues
    error_score='raise'  # Raise error immediately so we can see what's happening
)



# Fit the model with try-except to catch and debug any errors
try:
    search.fit(train, np.array(y))
except Exception as e:
    print(f"Error during search: {e}")
    print("\nAttempting direct pipeline fit...")
    
    # Try a direct fit with the pipeline
    try:
        pipeline.fit(train, np.array(y))
        print("Direct pipeline fit successful")
        
        # Make predictions
        preds = pipeline.predict(test)
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        
        # Create submission
        submission = pd.DataFrame({'id': test['id'], 'Calories': preds})
        submission.to_csv('submission.csv', index=False)
        print("Submission file created")
        joblib.dump(pipeline, 'final_model.pkl')
        
        # Exit script
        import sys
        sys.exit(0)
    except Exception as e2:
        print(f"Error during direct fit: {e2}")
        
        # Try an even simpler approach
        print("\nTrying final fallback approach with just Ridge regression...")
        final_pipeline = Pipeline([
            ('feature_engineering', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))
        ])
        
        try:
            final_pipeline.fit(train, np.array(y))
            print("Final fallback pipeline fit successful")
            
            # Make predictions
            preds = final_pipeline.predict(test)
            preds = np.expm1(preds)
            preds = np.clip(preds, 0, None)
            
            # Create submission
            submission = pd.DataFrame({'id': test['id'], 'Calories': preds})
            submission.to_csv('submission.csv', index=False)
            print("Submission file created")
            joblib.dump(final_pipeline, 'final_model.pkl')
        except Exception as e3:
            print(f"Error during fallback approach: {e3}")
            print("Unable to create a working model")


print(f"Best Score: {-search.best_score_:.5f}")
print(f"Best Params: {search.best_params_}")

# Make predictions
preds = search.best_estimator_.predict(test)
preds = np.expm1(preds)
preds = np.clip(preds, 0, None)


# Create submission
submission = pd.DataFrame({'id': test['id'], 'Calories': preds})
submission.to_csv('submission.csv', index=False)

print("Submission file created and model saved.")
joblib.dump(search.best_estimator_, 'best_model.pkl')

