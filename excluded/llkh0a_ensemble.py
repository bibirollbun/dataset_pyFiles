train_path = '/kaggle/input/playground-series-s5e8/train.csv'
test_path = '/kaggle/input/playground-series-s5e8/test.csv'
submission_path = '/kaggle/input/playground-series-s5e8/sample_submission.csv'



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)


train.columns 


target_column = 'y'
category_features = train.select_dtypes(include=[object])
numerical_features = train.select_dtypes(include=[np.number])


train.describe()


#plot train
plt.figure(figsize=(12, 6))
sns.countplot(data=train, x=target_column)
plt.title('Distribution of Target Variable')
plt.show()



# Basic dataset information
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)
print("\nColumn names:")
print(train.columns.tolist())
print("\nMissing values:")
print(train.isnull().sum())
print("\nData types:")
print(train.dtypes)


# Target variable analysis
print("Target variable distribution:")
print(train['y'].value_counts())
print("\nTarget variable proportions:")
print(train['y'].value_counts(normalize=True))

# Check for class imbalance
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=train, x='y', palette='viridis')
plt.title('Target Variable Distribution')

plt.subplot(1, 2, 2)
train['y'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Target Variable Proportion')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Numerical features analysis
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    train[col].hist(bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Statistical summary
print("Statistical summary of numerical features:")
print(train[numerical_cols].describe())


# Categorical features analysis
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

plt.figure(figsize=(20, 15))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    value_counts = train[col].value_counts()
    plt.bar(range(len(value_counts)), value_counts.values)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)

plt.tight_layout()
plt.show()

# Print unique values for each categorical feature
print("Unique values in categorical features:")
for col in categorical_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts().head())


# Correlation analysis for numerical features
plt.figure(figsize=(12, 8))
correlation_matrix = train[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  correlation_matrix.iloc[i, j]))

print("Highly correlated feature pairs (|correlation| > 0.7):")
for pair in high_corr_pairs:
    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")


# Feature vs Target relationships
# Numerical features vs target
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    sns.boxplot(data=train, x='y', y=col, ax=axes[i])
    axes[i].set_title(f'{col} vs Target')
    axes[i].set_xlabel('Subscription (y)')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# Statistical comparison for numerical features
print("Mean values by target class:")
for col in numerical_cols:
    mean_values = train.groupby('y')[col].mean()
    print(f"\n{col}:")
    print(f"  No subscription: {mean_values[0]:.2f}")
    print(f"  Subscription: {mean_values[1]:.2f}")
    print(f"  Difference: {mean_values[1] - mean_values[0]:.2f}")


# Categorical features vs target
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    # Create cross-tabulation
    ct = pd.crosstab(train[col], train['y'], normalize='index')
    ct.plot(kind='bar', ax=axes[i], stacked=True, color=['lightcoral', 'skyblue'])
    axes[i].set_title(f'{col} vs Target (Proportions)')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Proportion')
    axes[i].legend(['No', 'Yes'])
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Print subscription rates by categorical features
print("Subscription rates by categorical features:")
for col in categorical_cols:
    rates = train.groupby(col)['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
    print(f"\n{col}:")
    print(rates.head())


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create engineered features for the dataset"""
    df = df.copy()
    
    # Duration-based features
    df['duration_minutes'] = df['duration'] / 60
    df['duration_log'] = np.log1p(df['duration'])
    df['is_long_call'] = (df['duration'] > 300).astype(int)
    
    # Age-based features
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 55, 100], labels=['young', 'middle', 'senior'])
    df['is_retirement_age'] = (df['age'] >= 60).astype(int)
    
    # Balance-based features
    df['balance_log'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['has_positive_balance'] = (df['balance'] > 0).astype(int)
    df['balance_category'] = pd.cut(df['balance'], bins=[-np.inf, 0, 1000, 5000, np.inf], 
                                   labels=['negative', 'low', 'medium', 'high'])
    
    # Campaign features
    df['campaign_frequency'] = pd.cut(df['campaign'], bins=[0, 1, 3, 6, np.inf], 
                                     labels=['1', '2-3', '4-6', '7+'])
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['high_campaign_pressure'] = (df['campaign'] > 3).astype(int)
    
    # Previous campaign features
    df['has_previous_contact'] = (df['previous'] > 0).astype(int)
    df['previous_campaign_ratio'] = df['previous'] / (df['campaign'] + 1e-8)
    
    # Temporal features
    peak_months = ['may', 'jun', 'jul', 'aug']
    df['is_peak_month'] = df['month'].isin(peak_months).astype(int)
    df['is_end_of_month'] = (df['day'] > 25).astype(int)
    
    # Interaction features
    df['age_balance_interaction'] = df['age'] * df['balance_log']
    df['duration_campaign_ratio'] = df['duration'] / (df['campaign'] + 1e-8)
    
    # Financial risk score
    risk_features = ['default', 'housing', 'loan']
    for feat in risk_features:
        df[f'{feat}_encoded'] = (df[feat] == 'yes').astype(int)
    df['financial_risk_score'] = df['default_encoded'] + df['housing_encoded'] + df['loan_encoded']
    
    return df

# Apply feature engineering
print("Creating features for train and test sets...")
train_fe = create_features(train)
test_fe = create_features(test)

print("Feature engineering completed!")
print(f"Original features: {train.shape[1]}")
print(f"New features: {train_fe.shape[1]}")
print(f"Added features: {train_fe.shape[1] - train.shape[1]}")


# Analyze class imbalance in detail
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

print("Detailed Class Imbalance Analysis")
print("=" * 40)
y_train = train[target_column]



y_train


# Count classes
class_counts = Counter(y_train)
total_samples = len(y_train)

print(f"Total samples: {total_samples:,}")
print(f"Class 0 (no subscription): {class_counts[0]:,} ({class_counts[0]/total_samples:.1%})")
print(f"Class 1 (subscription): {class_counts[1]:,} ({class_counts[1]/total_samples:.1%})")
print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(f"\nComputed class weights:")
print(f"Class 0 weight: {class_weights[0]:.3f}")
print(f"Class 1 weight: {class_weights[1]:.3f}")

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# Visualize class distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
classes, counts = zip(*class_counts.items())
plt.bar(classes, counts, color=['lightcoral', 'skyblue'])
plt.title('Class Distribution (Absolute)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Subscription', 'Subscription'])

plt.subplot(1, 2, 2)
plt.pie(counts, labels=['No Subscription', 'Subscription'], autopct='%1.1f%%', 
        colors=['lightcoral', 'skyblue'])
plt.title('Class Distribution (Percentage)')

plt.tight_layout()
plt.show()

print("\nClass Imbalance Handling Strategies:")
print("1. ✓ Class weights in Random Forest, LightGBM")
print("2. ✓ Scale_pos_weight in XGBoost") 
print("3. ✓ Auto class weights in CatBoost")
print("4. ✓ Balanced class weights in Logistic Regression")
print("5. ✓ Stratified K-Fold CV")
print("6. ✓ AUC-ROC metric (robust to imbalance)")


# Data preprocessing for models
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def preprocess_data(train_df, test_df, target_col='y'):
    """Preprocess data for ensemble models"""
    
    # Separate features and target
    X_train = train_df.drop([target_col, 'id'], axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(['id'], axis=1)
    
    # Handle categorical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                           'loan', 'contact', 'month', 'poutcome', 'age_group', 
                           'balance_category', 'campaign_frequency']
    
    # Label encode categorical features for tree models
    label_encoders = {}
    for col in categorical_features:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # Create scaled version for neural networks and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, scaler, label_encoders

# Preprocess the data
print("Preprocessing data...")
X_train, X_test, X_train_scaled, X_test_scaled, y_train, scaler, label_encoders = preprocess_data(train_fe, test_fe)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Target distribution:")
print(y_train.value_counts(normalize=True))


# Define ensemble models
def get_models():
    """Define the ensemble of models"""
    
    models = {
        # Tree-based models (Tier 1)
        'lightgbm': lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=1000,
            class_weight='balanced'
        ),
        
        'xgboost': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_estimators=1000,
            scale_pos_weight=scale_pos_weight  # Dynamic calculation for class imbalance
        ),
        
        'catboost': CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_seed=42,
            verbose=False,
            auto_class_weights='Balanced'
        ),
        
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        # Linear/Neural models (Tier 2)
        'logistic': LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=1.0
        ),
        
        'mlp': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    return models

# Initialize models
models = get_models()
print("Models initialized:")
for name in models.keys():
    print(f"- {name}")

# Setup cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
print(f"\nUsing {n_folds}-fold stratified cross-validation")


# Train models and create ensemble
def train_ensemble(models, X_train, y_train, X_test, cv_folds):
    """Train ensemble of models with cross-validation"""
    
    # Store out-of-fold predictions and test predictions
    oof_predictions = np.zeros((len(X_train), len(models)))
    test_predictions = np.zeros((len(X_test), len(models)))
    model_scores = {}
    
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\nTraining {model_name}...")
        
        # Out-of-fold predictions for this model
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Use scaled data for neural networks and SVM
            if model_name in ['logistic', 'mlp']:
                X_fold_train_use = X_train_scaled[train_idx]
                X_fold_val_use = X_train_scaled[val_idx]
                X_test_use = X_test_scaled
            else:
                X_fold_train_use = X_fold_train
                X_fold_val_use = X_fold_val
                X_test_use = X_test
            
            # Train model
            model.fit(X_fold_train_use, y_fold_train)
            
            # Predict validation set
            if hasattr(model, 'predict_proba'):
                val_pred = model.predict_proba(X_fold_val_use)[:, 1]
                test_pred = model.predict_proba(X_test_use)[:, 1]
            else:
                val_pred = model.decision_function(X_fold_val_use)
                test_pred = model.decision_function(X_test_use)
            
            # Store predictions
            oof_preds[val_idx] = val_pred
            test_preds += test_pred / cv_folds.n_splits
            
            # Calculate fold score
            fold_score = roc_auc_score(y_fold_val, val_pred)
            fold_scores.append(fold_score)
            print(f"  Fold {fold + 1}: AUC = {fold_score:.6f}")
        
        # Store out-of-fold predictions
        oof_predictions[:, i] = oof_preds
        test_predictions[:, i] = test_preds
        
        # Calculate overall CV score
        cv_score = roc_auc_score(y_train, oof_preds)
        model_scores[model_name] = {
            'cv_score': cv_score,
            'fold_scores': fold_scores,
            'std': np.std(fold_scores)
        }
        
        print(f"  CV Score: {cv_score:.6f} (+/- {np.std(fold_scores):.6f})")
    
    return oof_predictions, test_predictions, model_scores

# Train the ensemble
print("Starting ensemble training...")
oof_preds, test_preds, scores = train_ensemble(models, X_train, y_train, X_test, skf)

print("\n" + "="*50)
print("ENSEMBLE RESULTS")
print("="*50)
for model_name, score_info in scores.items():
    print(f"{model_name:15}: {score_info['cv_score']:.6f} (+/- {score_info['std']:.6f})")


# Create final ensemble predictions
def create_ensemble_submission(oof_preds, test_preds, y_train, model_scores):
    """Create final ensemble using weighted average based on CV scores"""
    
    # Calculate weights based on CV scores
    weights = []
    for model_name in models.keys():
        weight = model_scores[model_name]['cv_score']
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    print("Model weights:")
    for i, (model_name, weight) in enumerate(zip(models.keys(), weights)):
        print(f"  {model_name:15}: {weight:.4f}")
    
    # Create weighted ensemble predictions
    ensemble_oof = np.average(oof_preds, axis=1, weights=weights)
    ensemble_test = np.average(test_preds, axis=1, weights=weights)
    
    # Calculate ensemble CV score
    ensemble_cv_score = roc_auc_score(y_train, ensemble_oof)
    print(f"\nEnsemble CV Score: {ensemble_cv_score:.6f}")
    
    # Compare with individual models
    print("\nModel comparison:")
    print(f"{'Model':<15} {'CV Score':<10} {'Improvement'}")
    print("-" * 40)
    
    for model_name, score_info in model_scores.items():
        improvement = ensemble_cv_score - score_info['cv_score']
        print(f"{model_name:<15} {score_info['cv_score']:.6f} {improvement:+.6f}")
    
    return ensemble_test, ensemble_cv_score

# Create ensemble predictions
final_predictions, ensemble_score = create_ensemble_submission(oof_preds, test_preds, y_train, scores)

# Create submission file
submission = pd.read_csv(submission_path)
submission['y'] = final_predictions

# Save submission
submission.to_csv('ensemble_submission.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")
print(f"Prediction range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")
print(f"Mean prediction: {final_predictions.mean():.6f}")

# Display first few predictions
print("\nFirst 10 predictions:")
print(submission.head(10))


# Feature importance analysis (using LightGBM as example)
def analyze_feature_importance():
    """Analyze feature importance from the best performing model"""
    
    # Train LightGBM on full dataset for feature importance
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
        n_estimators=1000,
        class_weight='balanced'
    )
    
    lgb_model.fit(X_train, y_train)
    
    # Get feature importance
    importance = lgb_model.feature_importances_
    feature_names = X_train.columns
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title('Top 20 Most Important Features (LightGBM)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    return importance_df

# Analyze feature importance
feature_importance = analyze_feature_importance()

