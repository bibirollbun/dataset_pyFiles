# Diabetes Prediction - Kaggle Playground Series S5E12
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


print("Loading datasets...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print("\nFirst few rows:")
print(train_df.head())


train_df['diagnosed_diabetes'].value_counts()


print(train_df.info())

print("\nNumerical Statistics:")
print(train_df.describe())

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nTarget Distribution:")
print(train_df['diagnosed_diabetes'].value_counts())
print(f"\nClass Balance: {train_df['diagnosed_diabetes'].value_counts(normalize=True)}")


# Visualize target distribution
plt.figure(figsize=(8, 5))
train_df['diagnosed_diabetes'].value_counts().plot(kind='bar', color=['skyblue', 'coral'])
plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


def engineer_features(df):
    """Apply feature engineering transformations"""
    df = df.copy()
    
    # 1. BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], 
                                bins=[0, 18.5, 25, 30, 40],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # 2. Age groups
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 30, 45, 60, 100],
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # 3. Blood pressure categories
    df['bp_category'] = 'Normal'
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 'High'
    df.loc[(df['systolic_bp'] < 90) | (df['diastolic_bp'] < 60), 'bp_category'] = 'Low'
    
    # 4. Cholesterol ratios
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    
    # 5. Lifestyle score (combined metric)
    df['lifestyle_score'] = (
        (df['physical_activity_minutes_per_week'] / 150) +  # WHO recommends 150 min/week
        (df['diet_score'] / 10) +
        (df['sleep_hours_per_day'] / 8) -  # 8 hours recommended
        (df['screen_time_hours_per_day'] / 8) -
        (df['alcohol_consumption_per_week'] / 7)
    )
    
    # 6. Health risk score
    df['health_risk_score'] = (
        df['family_history_diabetes'] +
        df['hypertension_history'] +
        df['cardiovascular_history'] +
        (df['bmi'] > 30).astype(int) +
        (df['smoking_status'] == 'Current').astype(int)
    )
    
    # 7. Metabolic syndrome indicators
    df['metabolic_syndrome_risk'] = (
        (df['bmi'] >= 30).astype(int) +
        (df['systolic_bp'] >= 130).astype(int) +
        (df['triglycerides'] >= 150).astype(int) +
        (df['hdl_cholesterol'] < 40).astype(int)
    )
    
    # 8. Activity to screen time ratio
    df['activity_screen_ratio'] = (df['physical_activity_minutes_per_week'] / 60) / (df['screen_time_hours_per_day'] + 1)
    
    # 9. Age-BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    # 10. Polynomial features for key variables
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_squared'] = df['age'] ** 2
    
    return df


train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

print(f"\nOriginal features: {train_df.shape[1]}")
print(f"After engineering: {train_engineered.shape[1]}")
print(f"New features created: {train_engineered.shape[1] - train_df.shape[1]}")


def encode_categorical(train, test):
    """Encode categorical variables"""
    train = train.copy()
    test = test.copy()
    
    # List of categorical columns
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status', 'bmi_category', 
                       'age_group', 'bp_category']
    
    # Label encoding for ordinal variables
    ordinal_mappings = {
        'education_level': {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3},
        'income_level': {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Higher-Middle': 3, 'High': 4},
        'smoking_status': {'Never': 0, 'Former': 1, 'Current': 2},
        'bmi_category': {'Underweight': -1, 'Normal': 0, 'Overweight': 1, 'Obese': 2},
        'age_group': {'Young': 0, 'Middle': 1, 'Senior': 2, 'Elderly': 3},
        'bp_category': {'Low': 0, 'Normal': 1, 'High': 2}
    }
    
    for col, mapping in ordinal_mappings.items():
        train[f'{col}_encoded'] = train[col].map(mapping)
        test[f'{col}_encoded'] = test[col].map(mapping)
    
    # One-hot encoding for nominal variables
    nominal_cols = ['gender', 'ethnicity', 'employment_status']
    
    for col in nominal_cols:
        dummies_train = pd.get_dummies(train[col], prefix=col, drop_first=True)
        dummies_test = pd.get_dummies(test[col], prefix=col, drop_first=True)
        
        # Ensure same columns in train and test
        for dummy_col in dummies_train.columns:
            if dummy_col not in dummies_test.columns:
                dummies_test[dummy_col] = 0
        
        train = pd.concat([train, dummies_train], axis=1)
        test = pd.concat([test, dummies_test], axis=1)
    
    # Drop original categorical columns
    train = train.drop(columns=categorical_cols)
    test = test.drop(columns=categorical_cols)
    
    return train, test

print("\nEncoding categorical variables...")
train_encoded, test_encoded = encode_categorical(train_engineered, test_engineered)




# Separate features and target
X = train_encoded.drop(columns=['id', 'diagnosed_diabetes'])
y = train_encoded['diagnosed_diabetes']
test_ids = test_encoded['id']
X_test = test_encoded.drop(columns=['id'])

# Ensure X_test has same columns as X
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0

X_test = X_test[X.columns]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Test matrix shape: {X_test.shape}")


# Check for missing values

print(f"\nMissing values in training features: {X.isnull().sum().sum()}")
print(f"Missing values in test features: {X_test.isnull().sum().sum()}")

if X.isnull().sum().sum() > 0:
    print("\nColumns with missing values in training:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

if X_test.isnull().sum().sum() > 0:
    print("\nColumns with missing values in test:")
    print(X_test.isnull().sum()[X_test.isnull().sum() > 0])

# Handle missing values - fill with median for numerical, mode for categorical
from sklearn.impute import SimpleImputer

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# Impute numerical columns with median
if len(numerical_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Impute categorical columns with most frequent
if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

print("\nMissing values handled successfully")
print(f"Remaining NaNs in X: {X.isnull().sum().sum()}")
print(f"Remaining NaNs in X_test: {X_test.isnull().sum().sum()}")

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled successfully")


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=2000, max_depth=150, 
                                           min_samples_split=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=2000, learning_rate=0.01,
                                                    max_depth=50, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01, max_depth=50,
                                 random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, max_depth=50,
                                   random_state=42, verbose=-1)
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"{'='*50}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba_val = model.predict_proba(X_val_scaled)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba_val)
    else:
        roc_auc = None
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'roc_auc': roc_auc
    }
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    if roc_auc:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()



# Voting Classifier (Soft Voting)
print("\nTraining Voting Ensemble (Soft Voting)...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=2000, max_depth=30, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=2000, learning_rate=0.01, max_depth=30, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01, max_depth=30, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, max_depth=30, random_state=42, verbose=-1))
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train)
y_pred_voting_train = voting_clf.predict(X_train_scaled)
y_pred_voting_val = voting_clf.predict(X_val_scaled)
y_pred_proba_voting_val = voting_clf.predict_proba(X_val_scaled)[:, 1]

voting_train_acc = accuracy_score(y_train, y_pred_voting_train)
voting_val_acc = accuracy_score(y_val, y_pred_voting_val)
voting_roc_auc = roc_auc_score(y_val, y_pred_proba_voting_val)

results['Voting Ensemble'] = {
    'model': voting_clf,
    'train_accuracy': voting_train_acc,
    'val_accuracy': voting_val_acc,
    'roc_auc': voting_roc_auc
}

print(f"Training Accuracy: {voting_train_acc:.4f}")
print(f"Validation Accuracy: {voting_val_acc:.4f}")
print(f"ROC-AUC Score: {voting_roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_voting_val)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Voting Ensemble')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Stacking Classifier
print("\nTraining Stacking Ensemble...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)

stacking_clf.fit(X_train_scaled, y_train)
y_pred_stacking_train = stacking_clf.predict(X_train_scaled)
y_pred_stacking_val = stacking_clf.predict(X_val_scaled)
y_pred_proba_stacking_val = stacking_clf.predict_proba(X_val_scaled)[:, 1]

stacking_train_acc = accuracy_score(y_train, y_pred_stacking_train)
stacking_val_acc = accuracy_score(y_val, y_pred_stacking_val)
stacking_roc_auc = roc_auc_score(y_val, y_pred_proba_stacking_val)

results['Stacking Ensemble'] = {
    'model': stacking_clf,
    'train_accuracy': stacking_train_acc,
    'val_accuracy': stacking_val_acc,
    'roc_auc': stacking_roc_auc
}

print(f"Training Accuracy: {stacking_train_acc:.4f}")
print(f"Validation Accuracy: {stacking_val_acc:.4f}")
print(f"ROC-AUC Score: {stacking_roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_stacking_val)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Stacking Ensemble')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# Compare feature importance across different models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

tree_models = {
    'Random Forest': results['Random Forest']['model'],
    'XGBoost': results['XGBoost']['model'],
    'LightGBM': results['LightGBM']['model'],
    'Gradient Boosting': results['Gradient Boosting']['model']
}

for idx, (name, model) in enumerate(tree_models.items()):
    ax = axes[idx // 2, idx % 2]
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        sns.barplot(data=feature_importance, x='importance', y='feature', 
                   palette='viridis', ax=ax)
        ax.set_title(f'Top 20 Features - {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

plt.tight_layout()
plt.show()


# Select best model based on validation ROC-AUC (or accuracy if ROC-AUC not available)
best_model_name = None
best_score = -1
metric_used = 'ROC-AUC'

for name, result in results.items():
    if result['roc_auc'] is not None and result['roc_auc'] > best_score:
        best_score = result['roc_auc']
        best_model_name = name
    elif result['roc_auc'] is None and result['val_accuracy'] > best_score:
        best_score = result['val_accuracy']
        best_model_name = name
        metric_used = 'Validation Accuracy'

print(f"\nBest Model: {best_model_name}")
print(f"Selection Metric: {metric_used}")
print(f"Best Score: {best_score:.4f}")
print(f"\nModel Performance:")
print(f"  - Training Accuracy: {results[best_model_name]['train_accuracy']:.4f}")
print(f"  - Validation Accuracy: {results[best_model_name]['val_accuracy']:.4f}")
if results[best_model_name]['roc_auc']:
    print(f"  - ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")

best_model = results[best_model_name]['model']


# Generate probability predictions (probability of class 1 - having diabetes)
test_predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nPrediction Statistics:")
print(f"  - Min probability: {test_predictions_proba.min():.4f}")
print(f"  - Max probability: {test_predictions_proba.max():.4f}")
print(f"  - Mean probability: {test_predictions_proba.mean():.4f}")
print(f"  - Median probability: {np.median(test_predictions_proba):.4f}")

# Visualize probability distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(test_predictions_proba, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Probability of Diabetes')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(test_predictions_proba, bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
plt.xlabel('Predicted Probability of Diabetes')
plt.ylabel('Cumulative Proportion')
plt.title('Cumulative Distribution of Predicted Probabilities')
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Show probability ranges
print(f"\nProbability Range Distribution:")
ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for low, high in ranges:
    count = ((test_predictions_proba >= low) & (test_predictions_proba < high)).sum()
    pct = count / len(test_predictions_proba) * 100
    print(f"  [{low:.1f} - {high:.1f}): {count:5d} samples ({pct:5.2f}%)")



# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_predictions_proba
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Total predictions: {len(submission)}")
print(f"\nSample predictions:")
print(submission.head(10))
print("\nPrediction summary:")
print(f"  - Mean predicted probability: {test_predictions_proba.mean():.4f}")
print(f"  - Predictions > 0.5 (high risk): {(test_predictions_proba > 0.5).sum()} ({(test_predictions_proba > 0.5).sum()/len(test_predictions_proba)*100:.2f}%)")
print(f"  - Predictions <= 0.5 (low risk): {(test_predictions_proba <= 0.5).sum()} ({(test_predictions_proba <= 0.5).sum()/len(test_predictions_proba)*100:.2f}%)")



summary_df = pd.DataFrame({
    'Model': results.keys(),
    'Train Accuracy': [results[m]['train_accuracy'] for m in results.keys()],
    'Validation Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in results.keys()]
})

print("\n", summary_df.to_string(index=False))
print(f"\n*** Best Model Selected: {best_model_name} ***")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
models_list = list(results.keys())
train_accs = [results[m]['train_accuracy'] for m in models_list]
val_accs = [results[m]['val_accuracy'] for m in models_list]

x = np.arange(len(models_list))
width = 0.35

axes[0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8, color='steelblue')
axes[0].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8, color='coral')
axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Highlight best model
best_idx = models_list.index(best_model_name)
axes[0].axvline(x=best_idx, color='green', linestyle='--', alpha=0.5, linewidth=2.5, label='Best')

# ROC-AUC comparison
roc_scores = [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in models_list]
colors = ['coral' if i == best_idx else 'steelblue' for i in range(len(models_list))]
bars = axes[1].bar(models_list, roc_scores, alpha=0.8, color=colors)
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('ROC-AUC Score', fontsize=11)
axes[1].set_title('Model ROC-AUC Comparison (Best in Orange)', fontsize=13, fontweight='bold')
axes[1].set_xticklabels(models_list, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([min(roc_scores) - 0.02, max(roc_scores) + 0.02])

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, roc_scores)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

