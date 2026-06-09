import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

class BackpackPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        df = df.copy()
        
        # Price ranges based on brand and material
        df['brand_material'] = df['Brand'] + '_' + df['Material']
        
        # Size-related features
        df['capacity_per_compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
        
        # Interaction features
        df['is_premium'] = ((df['Brand'] == 'Adidas') & 
                          (df['Material'] == 'Leather') & 
                          (df['Weight Capacity (kg)'] > 20)).astype(int)
        
        # Binary feature combinations
        df['waterproof_laptop'] = ((df['Waterproof'] == 'true') & 
                                 (df['Laptop Compartment'] == 'true')).astype(int)
        
        return df
    
    def fit_transform(self, df):
        df = self.create_features(df)
        
        # Handle missing values
        for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
            df[col].fillna(df[col].mode()[0], inplace=True)
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Convert boolean columns
        df['Laptop Compartment'] = df['Laptop Compartment'].map({'true': 1, 'false': 0})
        df['Waterproof'] = df['Waterproof'].map({'true': 1, 'false': 0})
        
        # Handle numerical missing values
        df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median(), inplace=True)
        
        return df
    
    def transform(self, df):
        df = self.create_features(df)
        
        for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = self.label_encoders[col].transform(df[col])
        
        df['Laptop Compartment'] = df['Laptop Compartment'].map({'true': 1, 'false': 0})
        df['Waterproof'] = df['Waterproof'].map({'true': 1, 'false': 0})
        df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median(), inplace=True)
        
        return df

def objective(trial, X, y, cv):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(params, train_data, valid_sets=[val_data], 
                         callbacks=[lgb.early_stopping(50)]
            )
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
    
    return np.mean(scores)

# Preprocess data
print("Preprocessing data...")
preprocessor = BackpackPreprocessor()
train_processed = preprocessor.fit_transform(train)
test_processed = preprocessor.transform(test)

# Prepare features
feature_cols = [col for col in train_processed.columns 
               if col not in ['id', 'Price', 'brand_material']]
X = train_processed[feature_cols].values
y = train_processed['Price'].values

# Optimize hyperparameters
print("Optimizing hyperparameters...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y, cv), n_trials=50)

# Get best parameters
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
print("\nBest parameters:", best_params)

# Train final model with CV
print("\nTraining final model with CV...")
cv_scores = []
test_predictions = np.zeros(len(test_processed))

for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
    print(f"Training fold {fold}...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(best_params, train_data, valid_sets=[val_data], 
                     callbacks=[lgb.early_stopping(50)])
    
    # Validate
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    cv_scores.append(rmse)
    
    # Predict test
    test_predictions += model.predict(test_processed[feature_cols].values) / 5

print(f"\nCV RMSE scores: {cv_scores}")
print(f"Mean CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Price': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))




