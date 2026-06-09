import pandas as pd
import numpy as np

train_data = pd.read_csv("/kaggle/input/buying-wine/train.csv")
train_data.head()


print("\nMissing values in training data:")
print(train_data.isnull().sum())


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report

def engineer_health_features(df):
    """
    Create health-specific features for smoking prediction
    """
    df_engineered = df.copy()
    
    # 1. Body Composition
    df_engineered['bmi'] = df_engineered['weight(kg)'] / ((df_engineered['height(cm)']/100) ** 2)
    df_engineered['waist_to_height'] = df_engineered['waist(cm)'] / df_engineered['height(cm)']
    
    # 2. Blood Pressure Analysis
    df_engineered['pulse_pressure'] = df_engineered['systolic'] - df_engineered['relaxation']
    df_engineered['mean_arterial_pressure'] = (
        df_engineered['relaxation'] + 
        (df_engineered['pulse_pressure'] / 3)
    )
    
    # 3. Lipid Ratios
    df_engineered['cholesterol_hdl_ratio'] = df_engineered['Cholesterol'] / df_engineered['HDL']
    df_engineered['triglyceride_hdl_ratio'] = df_engineered['triglyceride'] / df_engineered['HDL']
    df_engineered['ldl_hdl_ratio'] = df_engineered['LDL'] / df_engineered['HDL']
    
    # 4. Liver Function
    df_engineered['ast_alt_ratio'] = df_engineered['AST'] / df_engineered['ALT']
    df_engineered['liver_stress_index'] = (
        df_engineered['AST'] + 
        df_engineered['ALT'] + 
        df_engineered['Gtp']
    ) / 3
    
    # 5. Age-Related Risk Categories
    df_engineered['age_group'] = pd.cut(
        df_engineered['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=['young_adult', 'adult', 'middle_age', 'senior', 'elderly']
    )
    
    # 6. BMI Categories
    df_engineered['bmi_category'] = pd.cut(
        df_engineered['bmi'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['underweight', 'normal', 'overweight', 'obese']
    )
    
    # 7. Blood Pressure Categories
    df_engineered['bp_category'] = pd.cut(
        df_engineered['systolic'],
        bins=[0, 120, 130, 140, 300],
        labels=['normal', 'elevated', 'high_stage1', 'high_stage2']
    )
    
    # 8. Metabolic Health Score
    df_engineered['metabolic_score'] = (
        (df_engineered['bmi'] < 25).astype(int) +
        (df_engineered['systolic'] < 130).astype(int) +
        (df_engineered['fasting blood sugar'] < 100).astype(int) +
        (df_engineered['triglyceride'] < 150).astype(int) +
        (df_engineered['HDL'] > 40).astype(int)
    )
    
    # 9. Sensory Health
    df_engineered['vision_asymmetry'] = abs(
        df_engineered['eyesight(left)'] - df_engineered['eyesight(right)']
    )
    df_engineered['hearing_asymmetry'] = abs(
        df_engineered['hearing(left)'] - df_engineered['hearing(right)']
    )
    
    # 10. Cardiovascular Risk Score
    df_engineered['cardio_risk_score'] = (
        (df_engineered['systolic'] > 130).astype(int) +
        (df_engineered['cholesterol_hdl_ratio'] > 5).astype(int) +
        (df_engineered['ldl_hdl_ratio'] > 3).astype(int) +
        (df_engineered['triglyceride'] > 150).astype(int) +
        (df_engineered['waist_to_height'] > 0.5).astype(int)
    )
    
    # Convert categorical features to dummy variables
    categorical_columns = ['age_group', 'bmi_category', 'bp_category']
    dummy_features = pd.get_dummies(df_engineered[categorical_columns], prefix=categorical_columns)
    df_engineered = pd.concat([df_engineered, dummy_features], axis=1)
    df_engineered.drop(categorical_columns, axis=1, inplace=True)
    
    # 11. Interaction Features
    df_engineered['age_bmi'] = df_engineered['age'] * df_engineered['bmi']
    df_engineered['age_bp'] = df_engineered['age'] * df_engineered['systolic']
    df_engineered['liver_bmi'] = df_engineered['liver_stress_index'] * df_engineered['bmi']
    df_engineered['liver_lipids'] = df_engineered['liver_stress_index'] * df_engineered['cholesterol_hdl_ratio']
    
    # Handle potential infinities and NaN values
    df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)
    df_engineered = df_engineered.fillna(df_engineered.median())
    
    return df_engineered

def prepare_data(train_path, test_path=None):
    """
    Prepare data for smoking prediction model
    """
    # Read training data
    train_df = pd.read_csv(train_path)
    
    # Engineer features for training data
    train_df = engineer_health_features(train_df)
    
    # Split features and target
    features_to_use = [col for col in train_df.columns if col not in ['id', 'smoking']]
    X = train_df[features_to_use]
    y = train_df['smoking'] if 'smoking' in train_df.columns else None
    
    # Handle test data if provided
    test_df = None
    test_ids = None
    if test_path:
        test_df = pd.read_csv(test_path)
        test_ids = test_df['id'].copy()
        
        # Engineer features for test data
        test_df = engineer_health_features(test_df)
        test_df = test_df[features_to_use]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    if test_df is not None:
        test_scaled = scaler.transform(test_df)
        return X_scaled, y.values if y is not None else None, test_scaled, test_ids
    
    return X_scaled, y.values if y is not None else None

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import optuna
import shap
from scipy.stats import spearmanr

def remove_highly_correlated_features(df, threshold=0.85):
    """
    Remove highly correlated features using Spearman correlation
    """
    # Calculate correlation matrix
    corr_matrix = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).corr(method='spearman')
    
    # Create upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    
    return df.drop(to_drop, axis=1)

def objective(trial, X, y):
    """
    Optuna objective function for hyperparameter optimization
    """
    param = {
        'objective':'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
    }
    
    # Implement stratified k-fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store scores
    scores = []
    
    # Perform k-fold cross-validation
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(param, dtrain, num_boost_round=1000,
                         evals=[(dval, 'val')],
                         early_stopping_rounds=50,
                         verbose_eval=False)
        
        # Get best score
        scores.append(model.best_score)
    
    # Return mean of scores
    return np.mean(scores)

def train_improved_model(X, y, n_trials=50):
    """
    Train improved XGBoost model with hyperparameter optimization and cross-validation
    """
    print("Starting hyperparameter optimization...")
    
    # Create study object
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective':'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42
    })
    
    print("\nBest parameters:", best_params)
    print("Best score:", study.best_value)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    dtrain = xgb.DMatrix(X, label=y)
    final_model = xgb.train(best_params, dtrain, num_boost_round=1000)
    
    # Calculate SHAP values
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    # Get feature importance based on SHAP
    feature_importance = pd.DataFrame({
        'feature': dtrain.feature_names,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features (SHAP):")
    print(feature_importance.head(10))
    
    return final_model, feature_importance, shap_values

def prepare_improved_data(train_path, test_path=None):
    """
    Improved data preparation with correlation removal
    """
    # Original data preparation
    X, y, X_test, test_ids = prepare_data(train_path, test_path)
    
    # Convert to DataFrame for feature selection
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Remove highly correlated features
    X_df = remove_highly_correlated_features(X_df)
    
    # Convert back to numpy arrays
    X = X_df.values
    
    if test_path:
        # Apply same feature selection to test data
        X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        X_test = X_test_df[X_df.columns].values
        return X, y, X_test, test_ids
    
    return X, y

def predict_with_confidence(model, X_test, test_ids, threshold=0.5):
    """
    Generate predictions with confidence scores
    """
    dtest = xgb.DMatrix(X_test)
    
    # Get probability predictions
    probabilities = model.predict(dtest, output_margin=False)
    
    # Create submission DataFrame with confidence scores
    submission_df = pd.DataFrame({
        'id': test_ids,
        'smoking': probabilities,
    })
    
    return submission_df

def main_improved():
    try:
        # Prepare data with correlation removal
        print("Loading and preparing data...")
        train_path = "/kaggle/input/buying-wine/train.csv"
        test_path = "/kaggle/input/buying-wine/test.csv"
        
        X, y, X_test, test_ids = prepare_improved_data(train_path, test_path)
        
        # Train improved model
        model, feature_importance, shap_values = train_improved_model(X, y)
        
        # Generate predictions with confidence scores
        predictions_df = predict_with_confidence(model, X_test, test_ids)
        
        # Save predictions
        predictions_df.to_csv("/kaggle/working/submission.csv", index=False)
        
        return model, predictions_df, feature_importance, shap_values
        
    except Exception as e:
        print(f"\nAn error occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise e

if __name__ == "__main__":
    model, predictions, importance, shap_vals = main_improved()

