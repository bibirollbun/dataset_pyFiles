import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train_df.head(5)


train_df.info()
train_df.describe()


print(train_df['efs'].value_counts())
print(f"\nClass balance: \n{train_df['efs'].value_counts(normalize=True)}")
plt.figure(figsize=(8, 5))
train_df['efs'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Target Variable Distribution ', fontsize=14, fontweight='bold')
plt.xlabel('EFS Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


def analyze_missing_values(df, name="Dataset"):
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    print(f"Total columns with missing values: {len(missing_df)}")
    print("\nTop 20 with missing val")
    print(missing_df.head(20))
    
    # Visualize missing values
    if len(missing_df) > 0:
        plt.figure(figsize=(12, 6))
        top_missing = missing_df.head(20)
        plt.barh(top_missing['Column'], top_missing['Missing_Percentage'])
        plt.xlabel('Missing Percentage (%)')
        plt.title(f'Top 20 Features {name}', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return missing_df

missing_train = analyze_missing_values(train_df, "Training Set")


numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

if 'efs' in numerical_features:
    numerical_features.remove('efs')
if 'efs_time' in numerical_features:
    numerical_features.remove('efs_time')
if 'efs' in categorical_features:
    categorical_features.remove('efs')
if 'id' in categorical_features:
    categorical_features.remove('id')
if 'id' in numerical_features:
    numerical_features.remove('id')
print(f"Numerical features ({len(numerical_features)}): {numerical_features[:10]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}...")


if len(numerical_features) > 0:
    features_to_plot = numerical_features[:12] # top 12
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(features_to_plot):
        axes[idx].hist(train_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].set_xlabel('')
    
    plt.suptitle('Distribution of Numerical Features ', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


key_categorical = ['prim_disease_hct', 'graft_type', 'donor_related', 'conditioning_intensity', 
                   'sex_match', 'cmv_status', 'race_group', 'ethnicity']

key_categorical = [col for col in key_categorical if col in categorical_features]

if len(key_categorical) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(key_categorical[:8]):
        train_df[col].value_counts().head(10).plot(kind='barh', ax=axes[idx])
        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Count')
    
    plt.suptitle('Distribution of Key Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(df, categorical_col, target_col='efs', top_n=10):
    cross_tab = pd.crosstab(df[categorical_col], df[target_col], normalize='index') * 100
    cross_tab = cross_tab.head(top_n)
    
    cross_tab.plot(kind='barh', stacked=False, figsize=(10, 6))
    plt.title(f'{categorical_col} vs {target_col} (% within each category)', fontweight='bold')
    plt.xlabel('Percentage (%)')
    plt.ylabel(categorical_col)
    plt.legend(title=target_col)
    plt.tight_layout()
    plt.show()

important_cats = ['prim_disease_hct', 'graft_type', 'donor_related', 'conditioning_intensity'] #key feature
important_cats = [col for col in important_cats if col in categorical_features]

for col in important_cats[:3]:  # Plot first 3
    plot_categorical_vs_target(train_df, col)


test_ids = test_df['ID'].values
print(f'Extracted {len(test_ids)} test IDs')
print(f'First few test IDs: {test_ids[:5]}')


def preprocess_data(train, test):
    train_processed = train.copy()
    test_processed = test.copy()
    
    if 'efs' in train_processed.columns:
        y = train_processed['efs'].astype(int)
        train_processed = train_processed.drop(['efs', 'efs_time'], axis=1, errors='ignore')
    else:
        y = None
    
    # Drop ID column if exists
    train_processed = train_processed.drop(['id'], axis=1, errors='ignore')
    test_processed = test_processed.drop(['id'], axis=1, errors='ignore')
    
    categorical_cols = train_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # fill with median
    for col in numerical_cols:
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        test_processed[col].fillna(median_val, inplace=True)
    
    # fill with mode or 'Missing'
    for col in categorical_cols:
        mode_val = train_processed[col].mode()[0] if not train_processed[col].mode().empty else 'Missing'
        train_processed[col].fillna(mode_val, inplace=True)
        test_processed[col].fillna(mode_val, inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to ensure consistency
        combined = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined)
        
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        label_encoders[col] = le
    
    return train_processed, test_processed, y, test_ids, categorical_cols, numerical_cols, label_encoders

X_train_processed, X_test_processed, y_train, test_ids, cat_cols, num_cols, encoders = preprocess_data(train_df, test_df)

print(f"Training features shape: {X_train_processed.shape}")
print(f"Test features shape: {X_test_processed.shape}")
print(f"Target shape: {y_train.shape}")


print(f'Extracted {len(test_ids)} test IDs')
print(f'First few test IDs: {test_ids[:5]}')


X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"\nClass distribution in training set:")
print(pd.Series(y_train_split).value_counts(normalize=True))


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_train_proba = y_train_pred
        y_val_proba = y_val_pred
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    print(f"\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Censoring', 'Event']))
    
    return {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_auc': train_auc,
        'val_auc': val_auc
    }

results = {}


lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
results['Logistic Regression'] = evaluate_model(lr_model, X_train, y_train_split, X_val, y_val, 'Logistic Regression')


# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42, class_weight='balanced')
results['Decision Tree'] = evaluate_model(dt_model, X_train, y_train_split, X_val, y_val, 'Decision Tree')


# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    min_samples_split=10,
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1
)
results['Random Forest'] = evaluate_model(rf_model, X_train, y_train_split, X_val, y_val, 'Random Forest')


scale_pos_weight = (y_train_split == 0).sum() / (y_train_split == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
results['XGBoost'] = evaluate_model(xgb_model, X_train, y_train_split, X_val, y_val, 'XGBoost')


lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
results['LightGBM'] = evaluate_model(lgb_model, X_train, y_train_split, X_val, y_val, 'LightGBM')


cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0
)
results['CatBoost'] = evaluate_model(cat_model, X_train, y_train_split, X_val, y_val, 'CatBoost')


comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_acc'] for m in results],
    'Val Accuracy': [results[m]['val_acc'] for m in results],
    'Train AUC': [results[m]['train_auc'] for m in results],
    'Val AUC': [results[m]['val_auc'] for m in results]
})

comparison_df = comparison_df.sort_values('Val AUC', ascending=False)
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

comparison_df.plot(x='Model', y=['Train Accuracy', 'Val Accuracy'], kind='bar', ax=axes[0])
axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[0].legend(['Train', 'Validation'])
axes[0].grid(axis='y', alpha=0.3)

comparison_df.plot(x='Model', y=['Train AUC', 'Val AUC'], kind='bar', ax=axes[1], color=['lightcoral', 'lightblue'])
axes[1].set_title('Model AUC Comparison', fontweight='bold')
axes[1].set_ylabel('AUC Score')
axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[1].legend(['Train', 'Validation'])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


best_model_name = comparison_df.iloc[0]['Model']
print(f"\nBest model: {best_model_name}")
print(f"Validation AUC: {comparison_df.iloc[0]['Val AUC']:.4f}")


X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([pd.Series(y_train_split), pd.Series(y_val)])


def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'auto_class_weights': 'Balanced',
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) ## 5 folds
    scores = cross_val_score(
        model, X_full_train, y_full_train,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    return scores.mean()


# study_catboost = optuna.create_study(
#     direction='maximize',
#     study_name='CatBoost-HCT-Optimization',
#     sampler=TPESampler(seed=42)
# )

# study_catboost.optimize(objective_catboost, n_trials=30, show_progress_bar=True)



# print(f"\nBest AUC: {study_catboost.best_value:.4f}")
# print(f"Best parameters: {study_catboost.best_params}")


best_params = {'iterations': 809, 'depth': 4, 'learning_rate': 0.07538560751028651, 'l2_leaf_reg': 3.6132164628287775, 'border_count': 155, 'bagging_temperature': 8.737172673321922, 'random_strength': 7.563099140098476} # from optuna
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['eval_metric'] = 'Logloss'

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train_processed, y_train)

# print(f"Cross-validation AUC: {study_catboost.best_value:.4f}")


feature_importance = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))


test_predictions_proba = final_model.predict_proba(X_test_processed)[:, 1]
test_predictions = final_model.predict(X_test_processed)


print(test_ids)


submission = pd.DataFrame({
    'ID': test_ids,
    'prediction': test_predictions_proba  
})

submission.to_csv('submission.csv', index=False)
submission.head(5)

