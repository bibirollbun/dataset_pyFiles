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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("viridis")

# Load the dataset
def load_data(train_path, test_path):
    """
    Load the dataset from CSV files
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Train dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")
        
        # Display column names to help identify the target column
        print(f"Train columns: {list(train_df.columns)}")
        print(f"Test columns: {list(test_df.columns)}")
        
        return train_df, test_df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None, None

# Exploratory Data Analysis
def perform_eda(df, target_col=None):
    """
    Perform exploratory data analysis
    """
    print("=== Exploratory Data Analysis ===")
    
    # Display basic information
    print("\n1. Dataset Info:")
    print(df.info())
    
    # Display first few rows
    print("\n2. First 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\n3. Missing values:")
    print(df.isnull().sum())
    
    # Statistical summary
    print("\n4. Statistical summary:")
    print(df.describe())
    
    # Check target variable distribution if target column is specified and exists
    if target_col and target_col in df.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[target_col], kde=True)
        plt.title(f'Distribution of {target_col}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[target_col])
        plt.title(f'Boxplot of {target_col}')
        
        plt.tight_layout()
        plt.show()
    elif target_col:
        print(f"\nTarget column '{target_col}' not found in the dataset.")
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:  # Only create correlation matrix if we have multiple numeric columns
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
    else:
        print("Not enough numeric columns to create a correlation matrix.")
    
    # Feature distributions
    numeric_features = numeric_df.columns
    if target_col and target_col in numeric_features:
        numeric_features = numeric_features.drop(target_col)
    
    n_features = len(numeric_features)
    if n_features > 0:
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(16, 4 * n_rows))
        for i, col in enumerate(numeric_features):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
        plt.show()
    else:
        print("No numeric features to display.")

# Feature Engineering
def engineer_features(df, target_col=None):
    """
    Perform feature engineering
    """
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Create interaction features based on domain knowledge
    if 'Energy' in df_processed.columns and 'Danceability' in df_processed.columns:
        df_processed['Energy_Danceability'] = df_processed['Energy'] * df_processed['Danceability']
    
    if 'Loudness' in df_processed.columns and 'Energy' in df_processed.columns:
        df_processed['Loudness_Energy'] = df_processed['Loudness'] * df_processed['Energy']
    
    if 'Acousticness' in df_processed.columns and 'Energy' in df_processed.columns:
        df_processed['Acousticness_Energy'] = df_processed['Acousticness'] * df_processed['Energy']
    
    # Create polynomial features for important variables
    if 'Loudness' in df_processed.columns:
        df_processed['Loudness_squared'] = df_processed['Loudness'] ** 2
    
    if 'Energy' in df_processed.columns:
        df_processed['Energy_squared'] = df_processed['Energy'] ** 2
    
    # Create ratio features
    if 'Valence' in df_processed.columns and 'Energy' in df_processed.columns:
        df_processed['Valence_Energy_Ratio'] = df_processed['Valence'] / (df_processed['Energy'] + 1e-6)
    
    if 'Danceability' in df_processed.columns and 'Tempo' in df_processed.columns:
        df_processed['Danceability_Tempo_Ratio'] = df_processed['Danceability'] / (df_processed['Tempo'] + 1e-6)
    
    # Create categorical features if needed
    if 'Key' in df_processed.columns:
        df_processed['Key_category'] = pd.cut(df_processed['Key'], bins=5, labels=False)
    
    # Create bins for continuous variables
    if 'Duration_ms' in df_processed.columns:
        df_processed['Duration_bin'] = pd.qcut(df_processed['Duration_ms'], q=5, labels=False, duplicates='drop')
    
    print(f"Original features: {len(df.columns)}")
    print(f"After feature engineering: {len(df_processed.columns)}")
    
    return df_processed

# Model Training and Evaluation
class BPMPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        
    def prepare_data(self, df, target_col=None, test_size=0.2, random_state=42):
        """
        Prepare data for training
        """
        # Check if target column exists
        if target_col and target_col not in df.columns:
            print(f"Target column '{target_col}' not found in the dataset.")
            print(f"Available columns: {list(df.columns)}")
            return None, None, None, None
        
        # Separate features and target
        drop_cols = []
        if target_col:
            drop_cols.append(target_col)
        if 'id' in df.columns:
            drop_cols.append('id')
            
        X = df.drop(columns=drop_cols, errors='ignore')
        
        # Remove non-numeric columns for simplicity
        X = X.select_dtypes(include=[np.number])
        self.feature_names = X.columns.tolist()
        
        # Get target variable if specified
        y = df[target_col] if target_col else None
        
        if target_col:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            # For test data without target
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, None, None, None
    
    def train_models(self, X_train, y_train):
        """
        Train multiple advanced models
        """
        # Define models with initial parameters
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=300, 
                max_depth=20, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'CatBoost': CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=0
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        """
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"\n{name} Performance:")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
        
        return results
    
    def cross_validate_models(self, X, y, cv=5):
        """
        Perform cross-validation for all models
        """
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            cv_results[name] = {
                'MAE_mean': -cv_scores.mean(),
                'MAE_std': cv_scores.std()
            }
            print(f"{name} CV MAE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return cv_results
    
    def create_ensemble(self, X_train, y_train):
        """
        Create a stacking ensemble model
        """
        # Define base models
        base_models = [
            ('xgb', XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )),
            ('lgbm', LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=0
            ))
        ]
        
        # Define meta model
        meta_model = Ridge(alpha=1.0, random_state=42)
        
        # Create stacking ensemble
        ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.models['Ensemble'] = ensemble
        self.best_model = ensemble
        
        return ensemble
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """
        Perform hyperparameter tuning for a specific model
        """
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = XGBRegressor(random_state=42)
        
        elif model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 8, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = LGBMRegressor(random_state=42)
        
        else:
            print("Model not supported for hyperparameter tuning")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best score: {-grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self, model_name='XGBoost'):
        """
        Analyze feature importance
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importances - {model_name}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print top 10 features
            print(f"\nTop 10 Most Important Features for {model_name}:")
            for i in range(min(10, len(importances))):
                print(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        else:
            print("Model doesn't support feature importance analysis")

# Main execution function
def main():
    """
    Main function to run the complete BPM prediction pipeline
    """
    # Load your dataset
    train_path = "/kaggle/input/playground-series-s5e9/train.csv"
    test_path = "/kaggle/input/playground-series-s5e9/test.csv"
    
    train_df, test_df = load_data(train_path, test_path)
    
    if train_df is None:
        return
    
    # Identify the target column
    # Look for common target column names
    possible_targets = ['BPM', 'bpm', 'target', 'TARGET', 'y']
    target_col = None
    
    for col in possible_targets:
        if col in train_df.columns:
            target_col = col
            break
    
    if target_col is None:
        # If no common target name found, try to identify it
        print("Could not identify target column. Please specify the target column name.")
        # Let's assume the last column is the target for now
        target_col = train_df.columns[-1]
        print(f"Assuming target column is: {target_col}")
    
    print(f"Using '{target_col}' as the target variable.")
    
    # Perform EDA
    perform_eda(train_df, target_col=target_col)
    
    # Feature engineering
    train_processed = engineer_features(train_df, target_col=target_col)
    
    # Initialize predictor
    predictor = BPMPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        train_processed, target_col=target_col
    )
    
    # Check if data preparation was successful
    if X_train is None:
        print("Data preparation failed. Exiting.")
        return
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Cross-validate models
    cv_results = predictor.cross_validate_models(X_train, y_train, cv=5)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Create ensemble model
    ensemble = predictor.create_ensemble(X_train, y_train)
    ensemble_results = predictor.evaluate_models(X_test, y_test)
    
    # Hyperparameter tuning for the best model
    best_model_tuned = predictor.hyperparameter_tuning(X_train, y_train, 'XGBoost')
    
    # Feature importance analysis
    predictor.feature_importance_analysis('XGBoost')
    
    # Compare model performance
    comparison_df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index')
    comparison_df = comparison_df.sort_values('RMSE')
    
    plt.figure(figsize=(12, 6))
    comparison_df['RMSE'].plot(kind='bar')
    plt.title('Model Comparison by RMSE')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Prepare test data for submission
    test_processed = engineer_features(test_df)
    test_processed = test_processed.select_dtypes(include=[np.number])
    test_scaled = predictor.scaler.transform(test_processed)
    
    # Make predictions with the best model
    best_predictions = predictor.best_model.predict(test_scaled)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'BPM': best_predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")
    
    # Save the best model
    import joblib
    joblib.dump(predictor.best_model, 'best_bpm_predictor.pkl')
    joblib.dump(predictor.scaler, 'scaler.pkl')
    
    print("Training completed! Best model saved as 'best_bpm_predictor.pkl'")

if __name__ == "__main__":
    main()

