# ============================================================
# Road Accident Risk Prediction (Playground Series S5E10)
# Full EDA + Visualization + Ensemble Modeling
# Author: Naimul Hasan Shadesh
# ============================================================

# =====================
# 1. Import Libraries
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set(style="whitegrid", palette="coolwarm")
plt.rcParams['figure.figsize'] = (10, 6)


# =====================
# 2. Load Dataset
# =====================
def load_data():
    """Load training and test datasets"""
    train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
    
    target = 'accident_risk'
    
    print("âœ… Train shape:", train.shape)
    print("âœ… Test shape:", test.shape)
    
    return train, test, sample_submission, target


# =====================
# 3. Data Exploration
# =====================
def explore_data(train, target):
    """Perform exploratory data analysis"""
    print("\nðŸ”¹ Training Data Overview:\n")
    display(train.head())
    
    print("\nðŸ”¹ Dataset Info:\n")
    display(train.info())
    
    print("\nðŸ”¹ Missing Values:\n")
    missing_vals = train.isnull().sum().sort_values(ascending=False).head(10)
    display(missing_vals)
    
    # Data type split
    numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)
    categorical_features = train.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\nðŸ“Š Numeric Features: {len(numerical_features)}")
    print(f"ðŸ“Š Categorical Features: {len(categorical_features)}")
    
    return numerical_features, categorical_features


# =====================
# 4. Visualization Functions
# =====================
def plot_target_distribution(train, target):
    """Plot distribution of target variable"""
    plt.figure(figsize=(8, 5))
    sns.histplot(train[target], bins=30, kde=True, color='royalblue')
    plt.title("Distribution of Target: accident_risk", fontsize=14)
    plt.xlabel("Accident Risk (0â€“1)")
    plt.ylabel("Frequency")
    plt.show()


def plot_correlation_analysis(train, numerical_features, target):
    """Plot correlation heatmap and top correlated features"""
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = train[numerical_features + [target]].corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, annot=False)
    plt.title("Correlation Heatmap with Target", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Top correlated features
    corr_target = corr[target].drop(target).sort_values(ascending=False)
    top_corr_features = corr_target.head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr_features.values, y=top_corr_features.index, palette='viridis')
    plt.title("Top 10 Features Correlated with Target")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.show()
    
    return corr_target


def plot_feature_distributions(train, numerical_features):
    """Plot distributions of sample numerical features"""
    sample_cols = np.random.choice(numerical_features, 
                                 size=min(5, len(numerical_features)), 
                                 replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(sample_cols):
        sns.histplot(train[col], bins=30, kde=True, color='teal', ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    
    # Remove empty subplots
    for i in range(len(sample_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return sample_cols


def plot_feature_target_relationships(train, sample_cols, target):
    """Plot relationships between features and target"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(sample_cols):
        sns.scatterplot(x=train[col], y=train[target], alpha=0.3, color='purple', ax=axes[i])
        axes[i].set_title(f"{col} vs accident_risk")
    
    # Remove empty subplots
    for i in range(len(sample_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


# =====================
# 5. Data Preprocessing
# =====================
def preprocess_data(train, test, numerical_features, categorical_features):
    """Handle missing values and encode categorical variables"""
    # Handle missing values for numerical features
    for col in numerical_features:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(train[col].median(), inplace=True)
    
    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    return train, test


def engineer_features(train, test, numerical_features):
    """Create new features through feature engineering"""
    # Create squared features
    for col in numerical_features:
        train[f"{col}_sq"] = train[col] ** 2
        test[f"{col}_sq"] = test[col] ** 2
    
    # Create ratio feature if possible
    if len(numerical_features) >= 2:
        train['feature_ratio'] = train[numerical_features[0]] / (train[numerical_features[1]] + 1e-6)
        test['feature_ratio'] = test[numerical_features[0]] / (test[numerical_features[1]] + 1e-6)
    
    return train, test


def scale_features(train, test, numerical_features):
    """Scale numerical features"""
    scaler = StandardScaler()
    scaled_cols = numerical_features + [f"{c}_sq" for c in numerical_features]
    
    # Add ratio feature if it exists
    if 'feature_ratio' in train.columns:
        scaled_cols.append('feature_ratio')
    
    train[scaled_cols] = scaler.fit_transform(train[scaled_cols])
    test[scaled_cols] = scaler.transform(test[scaled_cols])
    
    return train, test, scaler


# =====================
# 6. Ensemble Modeling
# =====================
def create_ensemble_models():
    """Define model parameters for ensemble"""
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'random_state': 42
    }
    
    return lgb_params


def train_ensemble(X, y, X_test, n_splits=5):
    """Train ensemble model with cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    lgb_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))
    cat_preds = np.zeros(len(X_test))
    oof_rmse = []
    
    lgb_params = create_ensemble_models()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nðŸ”¸ Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val)
        
        model_lgb = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
        )
        
        # XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.3,
            random_state=42,
            tree_method='hist'
        )
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=200,
            verbose=False
        )
        
        # CatBoost
        model_cat = cb.CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            random_seed=42,
            loss_function='RMSE',
            verbose=False
        )
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), 
                     early_stopping_rounds=200, verbose=False)
        
        # Ensemble validation predictions
        val_pred = (
            0.4 * model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration) +
            0.35 * model_xgb.predict(X_val) +
            0.25 * model_cat.predict(X_val)
        )
        
        fold_rmse = mean_squared_error(y_val, val_pred, squared=False)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f}")
        oof_rmse.append(fold_rmse)
        
        # Test predictions
        lgb_preds += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / kf.n_splits
        xgb_preds += model_xgb.predict(X_test) / kf.n_splits
        cat_preds += model_cat.predict(X_test) / kf.n_splits
        
        # Store first fold model for feature importance
        if fold == 0:
            first_model_lgb = model_lgb
    
    return lgb_preds, xgb_preds, cat_preds, oof_rmse, first_model_lgb


# =====================
# 7. Visualization of Results
# =====================
def plot_feature_importance(model_lgb, X):
    """Plot feature importance from LightGBM model"""
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model_lgb.feature_importance()
    }).sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(y='feature', x='importance', data=importance, palette='rocket')
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(predictions):
    """Plot distribution of final predictions"""
    plt.figure(figsize=(8, 5))
    sns.histplot(predictions, bins=30, kde=True, color='green')
    plt.title("Distribution of Predicted Accident Risk (0â€“1)")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Frequency")
    plt.show()


# =====================
# 8. Main Execution
# =====================
def main():
    """Main execution function"""
    # Load data
    train, test, sample_submission, target = load_data()
    
    # Explore data
    numerical_features, categorical_features = explore_data(train, target)
    
    # Visualizations
    plot_target_distribution(train, target)
    corr_target = plot_correlation_analysis(train, numerical_features, target)
    sample_cols = plot_feature_distributions(train, numerical_features)
    plot_feature_target_relationships(train, sample_cols, target)
    
    # Preprocessing
    train, test = preprocess_data(train, test, numerical_features, categorical_features)
    train, test = engineer_features(train, test, numerical_features)
    train, test, scaler = scale_features(train, test, numerical_features)
    
    # Prepare data for modeling
    X = train.drop(columns=[target, 'id'] if 'id' in train.columns else [target])
    y = train[target]
    X_test = test.drop(columns=['id'] if 'id' in test.columns else [])
    
    # Train ensemble model
    lgb_preds, xgb_preds, cat_preds, oof_rmse, model_lgb = train_ensemble(X, y, X_test)
    
    # Final blending
    final_preds = (0.4 * lgb_preds + 0.35 * xgb_preds + 0.25 * cat_preds)
    cv_rmse = np.mean(oof_rmse)
    print(f"\nâœ… Average CV RMSE: {cv_rmse:.6f}")
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test['id'],
        'accident_risk': np.clip(final_preds, 0, 1)
    })
    submission.to_csv('submission.csv', index=False)
    print("\nâœ… submission.csv created successfully!")
    
    # Final visualizations
    plot_feature_importance(model_lgb, X)
    plot_prediction_distribution(final_preds)
    
    return submission, cv_rmse


# Execute main function
if __name__ == "__main__":
    submission, cv_score = main()
    print(f"\nðŸŽ¯ Final Model Performance: RMSE = {cv_score:.6f}")

