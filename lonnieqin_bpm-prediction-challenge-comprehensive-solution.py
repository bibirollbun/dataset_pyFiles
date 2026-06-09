import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CFG:
    n_trials = 50
    is_gpu = True
# Setup KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')  # Fixed: should be test.csv
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

_ = train.pop("id")
_ = test.pop('id')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Basic info about the datasets
print("\n=== Train Dataset Info ===")
print(train.info())
print("\n=== Train Dataset Description ===")
print(train.describe())

# Check for missing values
print(f"\nMissing values in train: {train.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")

# Target distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train['BeatsPerMinute'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribution of BeatsPerMinute')
plt.xlabel('BPM')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(train['BeatsPerMinute'])
plt.title('BoxPlot of BeatsPerMinute')
plt.ylabel('BPM')

plt.tight_layout()
plt.show()

print(f"\nTarget statistics:")
print(f"Mean BPM: {train['BeatsPerMinute'].mean():.2f}")
print(f"Std BPM: {train['BeatsPerMinute'].std():.2f}")
print(f"Min BPM: {train['BeatsPerMinute'].min():.2f}")
print(f"Max BPM: {train['BeatsPerMinute'].max():.2f}")


# Get feature columns (excluding ID and target)
feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
print(f"Number of features: {len(feature_cols)}")
print(f"Features: {feature_cols[:10]}...")  # Show first 10 features

# Correlation analysis
plt.figure(figsize=(15, 12))
# Select top features with highest correlation to target
correlations = train[feature_cols].corrwith(train['BeatsPerMinute']).abs().sort_values(ascending=False)
top_features = correlations.head(20).index.tolist()

# Create correlation heatmap for top features
corr_matrix = train[top_features + ['BeatsPerMinute']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix - Top 20 Features vs Target')
plt.tight_layout()
plt.show()

print("Top 10 features by correlation with target:")
for i, (feat, corr) in enumerate(correlations.head(10).items()):
    print(f"{i+1}. {feat}: {corr:.4f}")


class DataPreprocessor:
    def __init__(self, select_best_features=False):
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.select_best_features = select_best_features
        
    def fit(self, X, y=None):
        # Initialize scaler
        self.scaler = RobustScaler()
        # Feature selection based on correlation and statistical tests
        if y is not None and self.select_best_features:
            # Select features with high correlation or statistical significance
            correlations = pd.DataFrame(X).corrwith(pd.Series(y)).abs()
            
            # Select top features by correlation
            top_corr_features = correlations.nlargest(50).index.tolist()
            
            # Statistical feature selection
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(100, X.shape[1]))
            self.feature_selector.fit(X, y)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            stat_selected_features = X.columns[selected_mask].tolist()
            
            # Combine both methods
            self.selected_features = list(set(top_corr_features + stat_selected_features))
            print(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
        else:
            self.selected_features = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X[self.selected_features])
        return self
    
    def transform(self, X):
            
        # Select features
        X_selected = X[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        X_scaled = pd.DataFrame(X_scaled, columns=self.selected_features, index=X.index)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Separate features and target
X = train.drop(['BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']

# Fit preprocessor and transform data
X_processed = preprocessor.fit_transform(X, y)
X_test_processed = preprocessor.transform(test)

print(f"Processed training data shape: {X_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")


def create_additional_features(df):
    """Create additional engineered features"""
    df_new = df.copy()
    df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)
    df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
    df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
    df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']
    # Statistical features
    df_new['feature_mean'] = df_new.mean(axis=1)
    df_new['feature_std'] = df_new.std(axis=1)
    df_new['feature_max'] = df_new.max(axis=1)
    df_new['feature_min'] = df_new.min(axis=1)
    df_new['feature_median'] = df_new.median(axis=1)
    df_new['feature_range'] = df_new['feature_max'] - df_new['feature_min']
    df_new['feature_skew'] = df_new.skew(axis=1)
    
    # Interaction features (top correlated features)
    feature_cols = [col for col in df_new.columns if col.startswith('feature_') == False]
    if len(feature_cols) >= 2:
        # Create interactions between top features
        for i in range(min(5, len(feature_cols))):
            for j in range(i+1, min(5, len(feature_cols))):
                df_new[f'interaction_{i}_{j}'] = df_new.iloc[:, i] * df_new.iloc[:, j]
    
    # Polynomial features for top features
    for i in range(min(3, len(feature_cols))):
        df_new[f'poly2_{i}'] = df_new.iloc[:, i] ** 2
        df_new[f'poly3_{i}'] = df_new.iloc[:, i] ** 3
    
    return df_new


# Create engineered features
X_engineered = create_additional_features(X_processed)
X_test_engineered = create_additional_features(X_test_processed)

print(f"Features after engineering - Train: {X_engineered.shape}")
print(f"Features after engineering - Test: {X_test_engineered.shape}")

# Final feature selection (remove ID for modeling)
feature_columns = [col for col in X_engineered.columns if col != 'ID']
X_final = X_engineered[feature_columns]
X_test_final = X_test_engineered[feature_columns]


# LightGBM hyperparameter tuning using first fold
def objective_lgb(trial):
    # Split first fold for hyperparameter tuning
    train_idx, val_idx = next(kfold.split(X_final, y))
    X_train_fold, X_val_fold = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
        'verbosity': -1,
        'random_state': 42,
        'device': 'gpu' if CFG.is_gpu else 'cpu'
    }
    
    # Train model
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    
    return rmse

# Hyperparameter tuning for LightGBM
print("Optimizing LightGBM hyperparameters...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=CFG.n_trials, show_progress_bar=True)

print("Best LightGBM parameters:", study_lgb.best_params)
best_lgb_params = study_lgb.best_params
best_lgb_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42
})

# Train final LightGBM model with cross-validation
lgb_scores = []
lgb_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_final, y)):
    X_train_fold, X_val_fold = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    model = lgb.train(
        best_lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    lgb_scores.append(rmse)
    lgb_models.append(model)
    
    print(f"LightGBM Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nLightGBM CV Score: {np.mean(lgb_scores):.4f} (+/- {np.std(lgb_scores)*2:.4f})")


# CatBoost hyperparameter tuning
def objective_catboost(trial):
    # Split first fold for hyperparameter tuning
    train_idx, val_idx = next(kfold.split(X_final, y))
    X_train_fold, X_val_fold = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Suggest hyperparameters
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 42,
        'verbose': False,
        'task_type': 'GPU' if CFG.is_gpu else 'CPU'
    }
    
    # Train model
    model = cb.CatBoostRegressor(**params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict and calculate RMSE
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    
    return rmse

# Hyperparameter tuning for CatBoost
print("Optimizing CatBoost hyperparameters...")
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=CFG.n_trials, show_progress_bar=True)

print("Best CatBoost parameters:", study_catboost.best_params)
best_catboost_params = study_catboost.best_params
best_catboost_params.update({
    'random_seed': 42,
    'verbose': False
})

# Train final CatBoost model with cross-validation
catboost_scores = []
catboost_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_final, y)):
    X_train_fold, X_val_fold = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = cb.CatBoostRegressor(**best_catboost_params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    catboost_scores.append(rmse)
    catboost_models.append(model)
    
    print(f"CatBoost Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nCatBoost CV Score: {np.mean(catboost_scores):.4f} (+/- {np.std(catboost_scores)*2:.4f})")


# Prepare data for neural network
scaler_nn = StandardScaler()
X_scaled = scaler_nn.fit_transform(X_final)
y_scaled = (y - y.mean()) / y.std()  # Scale target for better training

# DNN hyperparameter tuning
def objective_dnn(trial):
    # Split first fold for hyperparameter tuning
    train_idx, val_idx = next(kfold.split(X_scaled, y_scaled))
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y_scaled.iloc[train_idx], y_scaled.iloc[val_idx]
    
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    n_units = trial.suggest_int('n_units', 64, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048])
    
    # Build model
    model = keras.Sequential()
    model.add(layers.Dense(n_units, activation='relu', input_dim=X_scaled.shape[1]))
    model.add(layers.Dropout(dropout_rate))
    
    for _ in range(n_layers - 1):
        model.add(layers.Dense(n_units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    
    # Train model
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=30,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2
    )
    
    # Predict and calculate RMSE
    y_pred_scaled = model.predict(X_val_fold, verbose=0)
    y_pred = y_pred_scaled.flatten() * y.std() + y.mean()  # Unscale predictions
    y_val_unscaled = y_val_fold * y.std() + y.mean()  # Unscale true values
    
    rmse = np.sqrt(mean_squared_error(y_val_unscaled, y_pred))
    
    return rmse

# Hyperparameter tuning for DNN
print("Optimizing DNN hyperparameters...")
study_dnn = optuna.create_study(direction='minimize')
study_dnn.optimize(
    objective_dnn, 
    #n_trials=CFG.n_trials, 
    n_trials=10, 
    show_progress_bar=True
)

print("Best DNN parameters:", study_dnn.best_params)
best_dnn_params = study_dnn.best_params

# Train final DNN model with cross-validation
dnn_scores = []
dnn_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y_scaled)):
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y_scaled.iloc[train_idx], y_scaled.iloc[val_idx]
    
    # Build model with best parameters
    model = keras.Sequential()
    model.add(layers.Dense(best_dnn_params['n_units'], activation='relu', input_dim=X_scaled.shape[1]))
    model.add(layers.Dropout(best_dnn_params['dropout_rate']))
    
    for _ in range(best_dnn_params['n_layers'] - 1):
        model.add(layers.Dense(best_dnn_params['n_units'], activation='relu'))
        model.add(layers.Dropout(best_dnn_params['dropout_rate']))
    
    model.add(layers.Dense(1))
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=best_dnn_params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mae')
    
    # Train model
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=30,
        batch_size=best_dnn_params['batch_size'],
        callbacks=[early_stopping],
        verbose=2
    )
    
    # Predict and calculate RMSE
    y_pred_scaled = model.predict(X_val_fold, verbose=0)
    y_pred = y_pred_scaled.flatten() * y.std() + y.mean()
    y_val_unscaled = y_val_fold * y.std() + y.mean()
    
    rmse = np.sqrt(mean_squared_error(y_val_unscaled, y_pred))
    dnn_scores.append(rmse)
    dnn_models.append(model)
    
    print(f"DNN Fold {fold+1} RMSE: {rmse:.4f}")

print(f"\nDNN CV Score: {np.mean(dnn_scores):.4f} (+/- {np.std(dnn_scores)*2:.4f})")



# Compare all models
model_results = {
    'LightGBM': (np.mean(lgb_scores), np.std(lgb_scores)),
    'CatBoost': (np.mean(catboost_scores), np.std(catboost_scores)),
    'DNN': (np.mean(dnn_scores), np.std(dnn_scores))
}

print("\n=== Model Comparison ===")
for model_name, (mean_score, std_score) in model_results.items():
    print(f"{model_name}: {mean_score:.4f} (+/- {std_score*2:.4f})")

# Find best model
best_model_name = min(model_results.keys(), key=lambda x: model_results[x][0])
print(f"\nBest Model: {best_model_name} with RMSE: {model_results[best_model_name][0]:.4f}")

# Feature importance for tree-based models
plt.figure(figsize=(15, 8))

# LightGBM feature importance
plt.subplot(1, 2, 1)
feature_importance_lgb = np.mean([model.feature_importance() for model in lgb_models], axis=0)
top_features_idx = np.argsort(feature_importance_lgb)[-20:]
plt.barh(range(len(top_features_idx)), feature_importance_lgb[top_features_idx])
plt.yticks(range(len(top_features_idx)), [feature_columns[i] for i in top_features_idx])
plt.title('LightGBM - Top 20 Feature Importances')
plt.xlabel('Importance')

# CatBoost feature importance
plt.subplot(1, 2, 2)
feature_importance_cb = np.mean([model.feature_importances_ for model in catboost_models], axis=0)
top_features_idx = np.argsort(feature_importance_cb)[-20:]
plt.barh(range(len(top_features_idx)), feature_importance_cb[top_features_idx])
plt.yticks(range(len(top_features_idx)), [feature_columns[i] for i in top_features_idx])
plt.title('CatBoost - Top 20 Feature Importances')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()



# Scale test data for DNN
X_test_scaled = scaler_nn.transform(X_test_final)

# Generate predictions from all models
lgb_predictions = np.mean([model.predict(X_test_final) for model in lgb_models], axis=0)
catboost_predictions = np.mean([model.predict(X_test_final) for model in catboost_models], axis=0)
dnn_predictions = np.mean([model.predict(X_test_scaled, batch_size=512, verbose=0).flatten() * y.std() + y.mean() 
                          for model in dnn_models], axis=0)

# Create ensemble prediction (weighted average based on CV performance)
weights = {
    'LightGBM': 1 / model_results['LightGBM'][0],
    'CatBoost': 1 / model_results['CatBoost'][0], 
    'DNN': 1 / model_results['DNN'][0]
}

# Normalize weights
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print(f"Ensemble weights: {weights}")

# Weighted ensemble
ensemble_predictions = (
    weights['LightGBM'] * lgb_predictions +
    weights['CatBoost'] * catboost_predictions +
    weights['DNN'] * dnn_predictions
)

# Create submission file
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'BeatsPerMinute': ensemble_predictions
})

# Ensure predictions are within reasonable bounds (optional)
submission['BeatsPerMinute'] = submission['BeatsPerMinute'].clip(
    lower=train['BeatsPerMinute'].quantile(0.01),
    upper=train['BeatsPerMinute'].quantile(0.99)
)

print(f"\nSubmission statistics:")
print(f"Mean prediction: {submission['BeatsPerMinute'].mean():.2f}")
print(f"Std prediction: {submission['BeatsPerMinute'].std():.2f}")
print(f"Min prediction: {submission['BeatsPerMinute'].min():.2f}")
print(f"Max prediction: {submission['BeatsPerMinute'].max():.2f}")

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")

# Display first few predictions
print("\nFirst 10 predictions:")
print(submission.head(10))


# Plot prediction distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist([train['BeatsPerMinute'], lgb_predictions, catboost_predictions, dnn_predictions], 
         bins=30, alpha=0.6, label=['Train Target', 'LightGBM', 'CatBoost', 'DNN'])
plt.legend()
plt.title('Prediction Distributions')
plt.xlabel('BPM')

plt.subplot(1, 3, 2)
plt.hist([train['BeatsPerMinute'], ensemble_predictions], 
         bins=30, alpha=0.7, label=['Train Target', 'Ensemble'])
plt.legend()
plt.title('Ensemble vs Target Distribution')
plt.xlabel('BPM')

plt.subplot(1, 3, 3)
plt.boxplot([train['BeatsPerMinute'], ensemble_predictions], 
            labels=['Train Target', 'Ensemble Pred'])
plt.title('Box Plot Comparison')
plt.ylabel('BPM')

plt.tight_layout()
plt.show()

print("Analysis completed! The ensemble model combines the strengths of all three approaches:")
print("- LightGBM: Fast and efficient gradient boosting")
print("- CatBoost: Robust handling of features with built-in regularization") 
print("- DNN: Ability to capture complex non-linear relationships")
print("\nThe weighted ensemble should provide better generalization than any single model.")

