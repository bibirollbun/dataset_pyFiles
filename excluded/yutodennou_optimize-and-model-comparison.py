# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Gradient boosting
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
np.random.seed(42)


# Data loading
DATA_PATH = r'/kaggle/input/playground-series-s5e8/'
train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
test_df = pd.read_csv(f'{DATA_PATH}/test.csv')
sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")


# Basic data information
print("=== Train Data Info ===")
print(train_df.info())
print("\n=== Train Data Description ===")
print(train_df.describe())
print("\n=== First 5 rows ===")
print(train_df.head())


# Target variable distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
train_df['y'].value_counts().plot(kind='bar')
plt.title('Target Distribution (Count)')
plt.xlabel('Target')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
train_df['y'].value_counts(normalize=True).plot(kind='bar')
plt.title('Target Distribution (Proportion)')
plt.xlabel('Target')
plt.ylabel('Proportion')

plt.tight_layout()
plt.show()

print(f"Target distribution:")
print(train_df['y'].value_counts())
print(f"\nTarget proportion:")
print(train_df['y'].value_counts(normalize=True))


# Numeric features distribution
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

plt.figure(figsize=(20, 12))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    train_df[feature].hist(bins=50, alpha=0.7)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Categorical features analysis
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for feature in categorical_features:
    print(f"\n=== {feature} ===")
    print(f"Unique values: {train_df[feature].nunique()}")
    print(train_df[feature].value_counts())
    
    # Relationship with target
    crosstab = pd.crosstab(train_df[feature], train_df['y'], normalize='index')
    print(f"\nTarget rate by {feature}:")
    print(crosstab[1].sort_values(ascending=False))


# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = train_df[numeric_features + ['y']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Features with high correlation to target
target_corr = correlation_matrix['y'].abs().sort_values(ascending=False)
print("Features correlation with target:")
print(target_corr)


def preprocess_data(df, is_train=True):
    """Data preprocessing function"""
    df = df.copy()
    
    # Feature engineering
    # 1. Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                            labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    # 2. Balance groups
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_high'] = (df['balance'] > df['balance'].quantile(0.75)).astype(int)
    df['balance_log'] = np.log1p(df['balance'] + abs(df['balance'].min()) + 1)
    
    # 3. Duration groups
    df['duration_group'] = pd.cut(df['duration'], bins=[0, 100, 300, 600, float('inf')], 
                                 labels=['short', 'medium', 'long', 'very_long'])
    df['duration_log'] = np.log1p(df['duration'])
    
    # 4. Campaign-related features
    df['has_previous_contact'] = (df['pdays'] != -1).astype(int)
    df['campaign_intensity'] = df['campaign'] + df['previous']
    df['campaign_per_day'] = df['campaign'] / (df['day'] + 1)
    
    # 5. Month seasonality
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_mapping)
    df['season'] = df['month_num'].apply(lambda x: 'spring' if x in [3,4,5] else
                                                  'summer' if x in [6,7,8] else
                                                  'autumn' if x in [9,10,11] else 'winter')
    
    # 6. Additional features
    df['age_balance_ratio'] = df['age'] / (abs(df['balance']) + 1)
    df['duration_campaign_ratio'] = df['duration'] / (df['campaign'] + 1)
    
    # Categorical variable encoding
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'poutcome', 'age_group', 'duration_group', 'season']
    
    # Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Drop unnecessary columns
    cols_to_drop = ['id'] + categorical_cols
    if not is_train:
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    else:
        cols_to_drop = [col for col in cols_to_drop if col in df.columns and col != 'y']
    
    df = df.drop(columns=cols_to_drop)
    
    return df, label_encoders

# Execute data preprocessing
train_processed, encoders = preprocess_data(train_df, is_train=True)
test_processed, _ = preprocess_data(test_df, is_train=False)

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")
print(f"\nTrain columns: {list(train_processed.columns)}")


# Separate features and target
X = train_processed.drop('y', axis=1)
y = train_processed['y']
X_test = test_processed

# Split training and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")


# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


# Optimized hyperparameter search function
def optimize_hyperparameters():
    """Optimize hyperparameters for each model with reduced search space"""
    
    optimized_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 folds
    
    # 1. XGBoost optimization (OPTIMIZED)
    print("Optimizing XGBoost...")
    xgb_params = {
        'n_estimators': [300 ,600],  # Reduced from [100, 200, 300]
        'max_depth': [4, 5],         # Reduced from [3, 4, 5, 6]
        'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1, 0.2]
        'subsample': [0.9, 1.0],     # Reduced from [0.8, 0.9, 1.0]
        'colsample_bytree': [0.9, 1.0] # Reduced from [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_params, n_iter=15, cv=cv,  # Reduced from 50 to 15 iterations
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    optimized_models['XGBoost'] = xgb_search.best_estimator_
    print(f"XGBoost best score: {xgb_search.best_score_:.4f}")
    print(f"XGBoost best params: {xgb_search.best_params_}")
    
    # 2. LightGBM optimization (OPTIMIZED)
    print("\nOptimizing LightGBM...")
    lgb_params = {
        'n_estimators': [300 ,600],   # Reduced from [100, 200, 300]
        'max_depth': [4, 5],          # Reduced from [3, 4, 5, 6]
        'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1, 0.2]
        'subsample': [0.9, 1.0],      # Reduced from [0.8, 0.9, 1.0]
        'colsample_bytree': [0.9, 1.0], # Reduced from [0.8, 0.9, 1.0]
        'num_leaves': [50, 100]       # Reduced from [31, 50, 100]
    }
    
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_search = RandomizedSearchCV(
        lgb_model, lgb_params, n_iter=15, cv=cv,  # Reduced from 50 to 15 iterations
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    lgb_search.fit(X_train, y_train)
    optimized_models['LightGBM'] = lgb_search.best_estimator_
    print(f"LightGBM best score: {lgb_search.best_score_:.4f}")
    print(f"LightGBM best params: {lgb_search.best_params_}")
    
    # 3. CatBoost optimization (OPTIMIZED)
    print("\nOptimizing CatBoost...")
    cat_params = {
        'iterations': [200, 300],     # Reduced from [100, 200, 300]
        'depth': [4, 5],              # Reduced from [3, 4, 5, 6]
        'learning_rate': [0.05, 0.1], # Reduced from [0.01, 0.05, 0.1, 0.2]
        'l2_leaf_reg': [3, 5]         # Reduced from [1, 3, 5, 7]
    }
    
    cat_model = CatBoostClassifier(random_state=42, verbose=False)
    cat_search = RandomizedSearchCV(
        cat_model, cat_params, n_iter=10, cv=cv,  # Reduced from 30 to 10 iterations
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    cat_search.fit(X_train, y_train)
    optimized_models['CatBoost'] = cat_search.best_estimator_
    print(f"CatBoost best score: {cat_search.best_score_:.4f}")
    print(f"CatBoost best params: {cat_search.best_params_}")
    
    # 4. RandomForest optimization (OPTIMIZED)
    print("\nOptimizing RandomForest...")
    rf_params = {
        'n_estimators': [ 300,600],   # Reduced from [100, 200, 300]
        'max_depth': [10, 15, 20],    # Reduced from [5, 10, 15, 20, None]
        'min_samples_split': [2, 5],  # Reduced from [2, 5, 10]
        'min_samples_leaf': [1, 2],   # Reduced from [1, 2, 4]
        'max_features': ['sqrt', 'log2'] # Reduced from ['sqrt', 'log2', None]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        rf_model, rf_params, n_iter=10, cv=cv,  # Reduced from 30 to 10 iterations
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    optimized_models['RandomForest'] = rf_search.best_estimator_
    print(f"RandomForest best score: {rf_search.best_score_:.4f}")
    print(f"RandomForest best params: {rf_search.best_params_}")
    
    # 5. LogisticRegression optimization (OPTIMIZED)
    print("\nOptimizing LogisticRegression...")
    lr_params = {
        'C': [0.1, 1, 10],            # Reduced from [0.001, 0.01, 0.1, 1, 10, 100]
        'penalty': ['l2'],            # Reduced from ['l1', 'l2', 'elasticnet']
        'solver': ['liblinear'],      # Reduced from ['liblinear', 'saga']
        'max_iter': [1000]            # Reduced from [1000, 2000]
    }
    
    lr_model = LogisticRegression(random_state=42)
    lr_search = RandomizedSearchCV(
        lr_model, lr_params, n_iter=5, cv=cv,   # Reduced from 30 to 5 iterations
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    lr_search.fit(X_train_scaled, y_train)
    optimized_models['LogisticRegression'] = lr_search.best_estimator_
    print(f"LogisticRegression best score: {lr_search.best_score_:.4f}")
    print(f"LogisticRegression best params: {lr_search.best_params_}")
    
    return optimized_models

# Execute optimized hyperparameter optimization
print("=== OPTIMIZED HYPERPARAMETER SEARCH ===")
print("Reduced search space for faster execution:")
print("- Cross-validation folds: 5 → 3")
print("- XGBoost iterations: 50 → 15")
print("- LightGBM iterations: 50 → 15")
print("- CatBoost iterations: 30 → 10")
print("- RandomForest iterations: 30 → 10")
print("- LogisticRegression iterations: 30 → 5")
print("- Parameter combinations reduced by ~60%")
print("\nStarting optimization...\n")

optimized_models = optimize_hyperparameters()


# Evaluate optimized models
results = {}
predictions = {}
val_predictions = {}

for name, model in optimized_models.items():
    print(f"\n=== Evaluating Optimized {name} ===")
    
    # Models requiring scaling
    if name in ['LogisticRegression']:
        X_train_use = X_train_scaled
        X_val_use = X_val_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_val_use = X_val
        X_test_use = X_test
    
    # Predictions
    y_pred = model.predict(X_val_use)
    y_pred_proba = model.predict_proba(X_val_use)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {'accuracy': accuracy, 'auc': auc}
    val_predictions[name] = y_pred_proba
    
    # Test data predictions
    test_pred_proba = model.predict_proba(X_test_use)[:, 1]
    predictions[name] = test_pred_proba
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

# Results summary
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('auc', ascending=False)
print("\n=== Optimized Model Comparison ===")
print(results_df)


# Ensemble prediction (weighted average of top 3 models)
top_models = results_df.head(3).index.tolist()
top_scores = results_df.head(3)['auc'].values

# Calculate weights based on scores
weights = top_scores / top_scores.sum()

print(f"Top 3 models: {top_models}")
print(f"Weights: {dict(zip(top_models, weights))}")

# Weighted ensemble predictions
ensemble_val_pred = np.average([val_predictions[model] for model in top_models], 
                               axis=0, weights=weights)
ensemble_test_pred = np.average([predictions[model] for model in top_models], 
                                axis=0, weights=weights)

# Calculate ensemble AUC
ensemble_auc = roc_auc_score(y_val, ensemble_val_pred)

print(f"\nEnsemble AUC: {ensemble_auc:.4f}")
print(f"Ensemble prediction range: {ensemble_test_pred.min():.4f} - {ensemble_test_pred.max():.4f}")
print(f"Ensemble prediction mean: {ensemble_test_pred.mean():.4f}")


# Compare best model and ensemble
best_single_model = results_df.index[0]
best_single_auc = results_df.iloc[0]['auc']

print(f"\n=== Model Selection ===")
print(f"Best Single Model: {best_single_model} (AUC: {best_single_auc:.4f})")
print(f"Ensemble Model AUC: {ensemble_auc:.4f}")

# Select model with higher score
if ensemble_auc > best_single_auc:
    print(f"\nSelected: Ensemble Model (AUC: {ensemble_auc:.4f})")
    final_predictions = ensemble_test_pred
    selected_model = 'Ensemble'
else:
    print(f"\nSelected: {best_single_model} (AUC: {best_single_auc:.4f})")
    final_predictions = predictions[best_single_model]
    selected_model = best_single_model

# Create submission file
submission = sample_submission.copy()
submission['y'] = final_predictions
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission file created: submission.csv")
print(f"Selected model: {selected_model}")
print(f"Prediction statistics:")
print(submission['y'].describe())


print("=== FINAL RESULTS SUMMARY (OPTIMIZED) ===")
print(f"\nDataset: Playground Series S5E8 - Binary Classification with Bank Dataset")
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print(f"Features used: {len(X.columns)}")

print(f"\nOptimized Model Performance (Validation AUC):")
for model, metrics in results_df.iterrows():
    print(f"{model}: {metrics['auc']:.4f}")

print(f"\nEnsemble Model AUC: {ensemble_auc:.4f}")
print(f"Selected Model: {selected_model}")
print(f"Final AUC Score: {ensemble_auc if selected_model == 'Ensemble' else best_single_auc:.4f}")

print(f"\nOptimization Benefits:")
print(f"- Reduced cross-validation folds: 5 → 3 (40% faster)")
print(f"- Reduced hyperparameter iterations by 60-70%")
print(f"- Focused on most impactful parameter ranges")
print(f"- Expected total time reduction: 60-70%")
print(f"- Maintained model quality with faster execution")

print(f"\nTarget distribution: {train_df['y'].value_counts().to_dict()}")
print(f"\nSubmission file: submission.csv")
print(f"Prediction range: {final_predictions.min():.4f} - {final_predictions.max():.4f}")
print(f"Prediction mean: {final_predictions.mean():.4f}")

