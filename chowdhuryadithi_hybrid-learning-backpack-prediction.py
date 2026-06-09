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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Machine Learning models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
    extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
    
    print("Initial shapes:")
    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print(f"Extra: {extra.shape}")
    
    return train, test, extra


def perform_eda(df):
    plt.figure(figsize=(15, 5))
    
    # Price distribution
    plt.subplot(131)
    sns.boxplot(y=df['Price'])
    plt.title('Price Distribution (Box Plot)')
    
    plt.subplot(132)
    sns.histplot(df['Price'], kde=True)
    plt.title('Price Distribution (Histogram)')
    
    plt.subplot(133)
    stats.probplot(df['Price'], dist="norm", plot=plt)
    plt.title('Price Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Missing values analysis
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum() / len(df) * 100
    missing.plot(kind='bar')
    plt.title('Missing Values Percentage')
    plt.tight_layout()
    plt.show()



def preprocess_data(train_df, test_df, extra_df):
    # Combine training data
    train_df = pd.concat([train_df, extra_df], ignore_index=True)
    train_df = train_df.drop_duplicates()
    
    # Handle missing values
    for df in [train_df, test_df]:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna('missing', inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Price']
    
    for col in categorical_cols:
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'Price']]
    
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
    return train_df, test_df


def apply_pca(X_train, X_test):
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    plt.show()
    
    return X_train_pca, X_test_pca, pca


def train_ml_models(X_train, X_test, y_train, y_test):
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            verbose=False,
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        r2 = r2_score(y_test, test_pred)
        
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'r2_score': r2
        }
        trained_models[name] = model
        
        print(f"{name} Results:")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
    
    return results, trained_models


def create_deep_learning_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse']
    )
    
    return model


def plot_results(results):
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE comparison
    models = list(results.keys())
    train_rmse = [results[m]['train_rmse'] for m in models]
    test_rmse = [results[m]['test_rmse'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    print("Step 1: Loading data...")
    train, test, extra = load_data()
    
    # Perform EDA
    print("\nStep 2: Performing EDA...")
    perform_eda(train)
    
    # Preprocess data
    print("\nStep 3: Preprocessing data...")
    train_processed, test_processed = preprocess_data(train.copy(), test.copy(), extra.copy())
    
    # Prepare features and target
    X = train_processed.drop(['Price', 'id'], axis=1)
    y = train_processed['Price']
    test_data_for_prediction = test_processed.drop('id', axis=1) 
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply PCA
    print("\nStep 4: Applying PCA...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)
    
    # Train ML models
    print("\nStep 5: Training Machine Learning models...")
    ml_results, trained_models = train_ml_models(X_train_pca, X_test_pca, y_train, y_test)
    
    
    # Train Deep Learning model
    print("\nStep 6: Training Deep Learning model...")
    dl_model = create_deep_learning_model(X_train_pca.shape[1])
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5)
    ]
    
    history = dl_model.fit(
        X_train_pca, y_train,
        validation_data=(X_test_pca, y_test),
        epochs=2,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot DL training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Deep Learning Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Evaluate DL model
    dl_train_pred = dl_model.predict(X_train_pca)
    dl_test_pred = dl_model.predict(X_test_pca)
    
    dl_train_rmse = np.sqrt(mean_squared_error(y_train, dl_train_pred))
    dl_test_rmse = np.sqrt(mean_squared_error(y_test, dl_test_pred))
    dl_r2 = r2_score(y_test, dl_test_pred)
    
    ml_results['Deep Learning'] = {
        'train_rmse': dl_train_rmse,
        'test_rmse': dl_test_rmse,
        'r2_score': dl_r2
    }
    
    # Plot final results
    print("\nStep 7: Plotting final results...")
    plot_results(ml_results)
    
    # Create ensemble predictions
    print("\nStep 8: Creating ensemble predictions...")
    test_pca = pca.transform(test_data_for_prediction) 
    final_predictions = np.zeros(len(test_processed)) 
    
    # Weighted average of all models
    weights = {
        'XGBoost': 0.3,
        'LightGBM': 0.25,
        'CatBoost': 0.25,
        'Deep Learning': 0.2
    }
    
    for name, model in trained_models.items():
        final_predictions += weights[name] * model.predict(test_pca)
    
    final_predictions += weights['Deep Learning'] * dl_model.predict(test_pca).flatten()
    
     # Create submission file
    submission = pd.DataFrame({
        'id': test_processed['id'],
        'Price': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created successfully!")

if __name__ == "__main__":
    main()

