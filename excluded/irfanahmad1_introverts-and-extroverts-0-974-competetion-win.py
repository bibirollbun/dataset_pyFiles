import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Install required packages
!pip install scikit-learn==1.5.2 koolbox

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from IPython.display import display, HTML

# Machine learning
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Gradient boosting libraries
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Advanced tools
from scipy.special import logit
from koolbox import Trainer
import joblib
import optuna
import glob

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)


class Config:
    """Central configuration for our personality prediction pipeline"""
    
    # File paths
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    # Model settings
    target = 'Personality'
    n_folds = 5
    seed = 42
    n_optuna_trials = 500
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    metric = accuracy_score
    
    # Feature lists
    numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 
                         'Going_outside', 'Friends_circle_size', 'Post_frequency']
    categorical_features = ['Stage_fear', 'Drained_after_socializing']

CFG = Config()

print("âš™ï¸� Configuration loaded!")
print(f"   Target: {CFG.target}")
print(f"   Cross-validation: {CFG.n_folds} folds")
print(f"   Features: {len(CFG.numerical_features)} numerical + {len(CFG.categorical_features)} categorical")


def load_and_explore_data():
    """Load datasets and provide initial exploration"""
    
    # Load data
    train = pd.read_csv(CFG.train_path, index_col='id')
    test = pd.read_csv(CFG.test_path, index_col='id')
    
    # Display shapes
    print("ğŸ“‚ Data Shapes:")
    print(f"   Training: {train.shape}")
    print(f"   Testing: {test.shape}")
    
    # Check data types
    print("\nğŸ“‹ Data Types:")
    print(train.dtypes)
    
    # Missing values analysis
    print("\nâ�“ Missing Values:")
    missing_df = pd.DataFrame({
        'Feature': train.columns,
        'Missing_Count': train.isnull().sum().values,
        'Missing_Percentage': (train.isnull().sum().values / len(train) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        display(missing_df.style.background_gradient(subset=['Missing_Percentage'], cmap='YlOrRd'))
    else:
        print("   âœ… No missing values!")
    
    # Target distribution
    print("\nğŸ�¯ Target Distribution:")
    target_counts = train[CFG.target].value_counts()
    for personality, count in target_counts.items():
        percentage = count / len(train) * 100
        print(f"   {personality}: {count:,} ({percentage:.1f}%)")
    
    # Class imbalance ratio
    imbalance_ratio = target_counts.max() / target_counts.min()
    print(f"\nâš–ï¸� Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    return train, test

# Load the data
train_df, test_df = load_and_explore_data()

# Display sample
print("\nğŸ“� Sample Training Data:")
display(train_df.head().style.set_properties(**{'background-color': '#f0f0f0'}, subset=pd.IndexSlice[:, CFG.target]))


def create_target_visualization(train_df):
    """Create beautiful target distribution plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ğŸ�¯ Personality Type Distribution Analysis', fontsize=16, y=1.02)
    
    # Prepare data
    target_counts = train_df[CFG.target].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        target_counts.values, 
        labels=target_counts.index, 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=90,
        explode=(0.05, 0.05),
        shadow=True
    )
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_weight('bold')
        autotext.set_fontsize(12)
    
    axes[0].set_title('Distribution Breakdown', fontsize=14, pad=20)
    
    # Bar chart with annotations
    bars = axes[1].bar(target_counts.index, target_counts.values, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_title('Sample Counts', fontsize=14, pad=20)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_xlabel('Personality Type', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, target_counts.values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 100,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add grid for better readability
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

create_target_visualization(train_df)


def create_feature_distributions(train_df):
    """Visualize how features differ between personality types"""
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('ğŸ“Š Feature Distributions by Personality Type', fontsize=18, y=1.02)
    axes = axes.ravel()
    
    # Numerical features
    for i, feature in enumerate(CFG.numerical_features):
        ax = axes[i]
        
        # Create violin plot
        data_to_plot = train_df.dropna(subset=[feature])
        sns.violinplot(data=data_to_plot, x=CFG.target, y=feature, ax=ax, palette=['#FF6B6B', '#4ECDC4'])
        
        # Add mean lines
        for j, personality in enumerate(['Extrovert', 'Introvert']):
            mean_val = data_to_plot[data_to_plot[CFG.target] == personality][feature].mean()
            ax.axhline(y=mean_val, xmin=j*0.5 + 0.1, xmax=j*0.5 + 0.4, 
                      color='black', linestyle='--', linewidth=2)
            ax.text(j, mean_val, f'Î¼={mean_val:.2f}', ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
    
    # Categorical features
    for i, feature in enumerate(CFG.categorical_features):
        ax = axes[len(CFG.numerical_features) + i]
        
        # Create stacked bar chart
        ct = pd.crosstab(train_df[feature], train_df[CFG.target], normalize='columns') * 100
        ct.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        
        ax.set_title(f'{feature} Distribution (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage')
        ax.legend(title='Personality', loc='upper right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Remove extra subplots
    for i in range(len(CFG.numerical_features) + len(CFG.categorical_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

create_feature_distributions(train_df)


def create_correlation_heatmap(train_df):
    """Create an enhanced correlation heatmap"""
    
    # Prepare data
    corr_data = train_df.copy()
    
    # Encode categorical variables
    corr_data['Stage_fear'] = corr_data['Stage_fear'].map({'No': 0, 'Yes': 1})
    corr_data['Drained_after_socializing'] = corr_data['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
    corr_data[CFG.target] = corr_data[CFG.target].map({'Extrovert': 0, 'Introvert': 1})
    
    # Calculate correlation
    correlation = corr_data.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(correlation, 
                mask=mask, 
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r', 
                center=0, 
                square=True, 
                linewidths=1,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                annot_kws={"fontsize": 10})
    
    plt.title('ğŸ”¥ Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Highlight target correlations
    target_corr = correlation[CFG.target].drop(CFG.target).sort_values(ascending=False)
    
    plt.tight_layout()
    plt.show()
    
    # Display target correlations
    print("ğŸ�¯ Feature Correlations with Target (Introvert):")
    for feature, corr_value in target_corr.items():
        direction = "â†‘" if corr_value > 0 else "â†“"
        print(f"   {feature}: {corr_value:+.4f} {direction}")
    
    return target_corr

target_correlations = create_correlation_heatmap(train_df)


def calculate_feature_importance(train_df):
    """Calculate and visualize feature importance using mutual information"""
    
    # Prepare data
    X_temp = train_df.drop(CFG.target, axis=1).copy()
    y_temp = train_df[CFG.target].map({'Extrovert': 0, 'Introvert': 1})
    
    # Encode categorical features
    X_temp['Stage_fear'] = X_temp['Stage_fear'].map({'No': 0, 'Yes': 1})
    X_temp['Drained_after_socializing'] = X_temp['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
    
    # Fill missing values
    X_temp = X_temp.fillna(0)
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X_temp, y_temp, random_state=CFG.seed)
    
    # Create dataframe
    mi_df = pd.DataFrame({
        'Feature': X_temp.columns,
        'Mutual_Information': mi_scores
    }).sort_values('Mutual_Information', ascending=False)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mi_df['Feature'], mi_df['Mutual_Information'], 
                    color=['#FF6B6B' if i < 3 else '#4ECDC4' for i in range(len(mi_df))])
    
    plt.title('ğŸ�† Feature Importance Rankings (Mutual Information)', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Mutual Information Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, score in zip(bars, mi_df['Mutual_Information']):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Display importance table
    print("ğŸ“Š Feature Importance Summary:")
    display(mi_df.style.background_gradient(subset=['Mutual_Information'], cmap='YlOrRd'))
    
    return mi_df

feature_importance = calculate_feature_importance(train_df)


def preprocess_data():
    """Complete preprocessing pipeline following the winning strategy"""
    
    print("ğŸ”§ FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load fresh data
    print("\nğŸ“� Step 1: Loading fresh data...")
    train = pd.read_csv(CFG.train_path, index_col='id')
    test = pd.read_csv(CFG.test_path, index_col='id')
    
    # Step 2: Load and merge original dataset (KEY INSIGHT!)
    print("\nğŸ”— Step 2: Merging with original dataset...")
    original = pd.read_csv(CFG.original_path)
    original = original.rename(columns={'Personality': 'match_p'})
    
    # Remove duplicates based on all feature columns
    feature_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
    original = original.drop_duplicates(feature_cols)
    
    print(f"   Original dataset shape after deduplication: {original.shape}")
    
    # Merge with train and test
    train = train.merge(original, how='left', on=feature_cols)
    test = test.merge(original, how='left', on=feature_cols)
    
    # Check merge success
    train_matches = train['match_p'].notna().sum()
    test_matches = test['match_p'].notna().sum()
    print(f"   âœ… Training matches: {train_matches}/{len(train)} ({train_matches/len(train)*100:.1f}%)")
    print(f"   âœ… Testing matches: {test_matches}/{len(test)} ({test_matches/len(test)*100:.1f}%)")
    
    # Step 3: Handle categorical features
    print("\nğŸ�·ï¸� Step 3: Processing categorical features...")
    cat_cols = ["Stage_fear", "Drained_after_socializing"]
    
    # Fill missing values with "missing" category
    train[cat_cols] = train[cat_cols].fillna("missing")
    test[cat_cols] = test[cat_cols].fillna("missing")
    
    # For XGBoost compatibility, we need to encode categoricals
    for col in cat_cols:
        # Create label encoder
        le = LabelEncoder()
        
        # Fit on combined data to ensure consistency
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        
        # Transform
        train[f'{col}_encoded'] = le.transform(train[col])
        test[f'{col}_encoded'] = le.transform(test[col])
    
    # Keep original categorical columns for CatBoost
    train[cat_cols] = train[cat_cols].astype("category")
    test[cat_cols] = test[cat_cols].astype("category")
    
    # Step 4: Encode targets
    print("\nğŸ�¯ Step 4: Encoding target variables...")
    train[CFG.target] = train[CFG.target].map({"Extrovert": 0, "Introvert": 1})
    train["match_p"] = train["match_p"].map({"Extrovert": 0, "Introvert": 1})
    test["match_p"] = test["match_p"].map({"Extrovert": 0, "Introvert": 1})
    
    # Step 5: Create feature sets
    print("\nğŸ“¦ Step 5: Creating feature sets...")
    
    # For CatBoost (uses original categorical columns)
    X_catboost = train.drop(CFG.target, axis=1)
    X_test_catboost = test.copy()
    
    # For XGBoost/LightGBM (uses encoded columns)
    encoded_cols = [col for col in train.columns if col.endswith('_encoded')]
    feature_cols_xgb = CFG.numerical_features + encoded_cols + ['match_p']
    
    X_xgboost = train[feature_cols_xgb]
    X_test_xgboost = test[feature_cols_xgb]
    
    # Target
    y = train[CFG.target]
    
    # Save a copy of test for submission
    _X_test = test.copy()
    
    print("\nâœ… Feature engineering complete!")
    print(f"   CatBoost features: {X_catboost.shape}")
    print(f"   XGBoost features: {X_xgboost.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X_catboost, X_xgboost, y, X_test_catboost, X_test_xgboost, _X_test

# Execute preprocessing
X_catboost, X_xgboost, y, X_test_catboost, X_test_xgboost, _X_test = preprocess_data()


def display_feature_summary():
    """Display a beautiful summary of our features"""
    
    # Create summary table
    summary_data = {
        'Feature Type': ['Numerical', 'Categorical', 'Engineered', 'Total'],
        'Count': [
            len(CFG.numerical_features),
            len(CFG.categorical_features),
            1,  # match_p
            len(CFG.numerical_features) + len(CFG.categorical_features) + 1
        ],
        'Examples': [
            'Time_spent_Alone, Social_event_attendance',
            'Stage_fear, Drained_after_socializing',
            'match_p (from original dataset)',
            'All features combined'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Style the table
    styled_table = summary_df.style\
        .set_properties(**{'background-color': '#f0f0f0'}, subset=['Feature Type'])\
        .set_properties(**{'font-weight': 'bold'}, subset=['Count'])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#e8f5e9')]}
        ])
    
    print("\nğŸ“Š FEATURE ENGINEERING SUMMARY")
    print("=" * 50)
    display(styled_table)
    
    # Key insights
    display(HTML("""
    <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h3>ğŸ’¡ Key Insights from Feature Engineering:</h3>
        <ul style="font-size: 14px; line-height: 1.8;">
            <li><strong>match_p feature:</strong> This is the secret sauce! It comes from merging with the original dataset.</li>
            <li><strong>Missing value strategy:</strong> We treat missing values as a separate "missing" category.</li>
            <li><strong>Dual encoding:</strong> We keep both original categories (for CatBoost) and encoded versions (for XGBoost).</li>
            <li><strong>No imputation:</strong> The models handle missing values natively, preserving information.</li>
        </ul>
    </div>
    """))

display_feature_summary()


def save_submission(name, X_test, test_pred_probs, score, threshold=0.5):
    """Save submission file with proper formatting"""
    sub = pd.read_csv(CFG.sample_sub_path)
    sub[CFG.target] = (test_pred_probs > threshold).astype(int)
    
    # Apply the match_p trick
    sub.loc[X_test.match_p == 0, CFG.target] = 1
    sub.loc[X_test.match_p == 1, CFG.target] = 0
    
    # Map back to string labels
    sub[CFG.target] = sub[CFG.target].map({0: "Extrovert", 1: "Introvert"})
    
    # Save
    filename = 'submission.csv'
    sub.to_csv(filename, index=False)
    
    print(f"ğŸ’¾ Saved: {filename}")
    return sub.head()

# Initialize storage
scores = {}
oof_pred_probs = {}
test_pred_probs = {}

print("ğŸ¤– MODEL TRAINING PIPELINE")
print("=" * 50)
print("We'll train multiple models and ensemble them for best results!")


print("\nğŸ�± Training CatBoost Model...")
print("-" * 40)

# CatBoost parameters (optimized)
catboost_params = {
    'iterations': 1000,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_seed': CFG.seed,
    'verbose': False,
    'cat_features': ['Stage_fear', 'Drained_after_socializing'],
    'early_stopping_rounds': 100
}

# Train with cross-validation
cb_trainer = Trainer(
    CatBoostClassifier(**catboost_params),
    cv=CFG.cv,
    metric=accuracy_score,
    use_early_stopping=True,
    task="binary",
    metric_precision=6,
)

cb_trainer.fit(X_catboost, y)

# Store results
scores["CatBoost"] = cb_trainer.fold_scores
oof_pred_probs["CatBoost"] = cb_trainer.oof_preds
test_pred_probs["CatBoost"] = cb_trainer.predict(X_test_catboost)

print(f"\nâœ… CatBoost Mean CV Score: {np.mean(cb_trainer.fold_scores):.6f} Â± {np.std(cb_trainer.fold_scores):.6f}")


print("\nğŸš€ Training XGBoost Model...")
print("-" * 40)

# XGBoost parameters (optimized)
xgboost_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': CFG.seed,
    'n_jobs': -1,
    'early_stopping_rounds': 100
}

# Train with cross-validation
xgb_trainer = Trainer(
    XGBClassifier(**xgboost_params),
    cv=CFG.cv,
    metric=accuracy_score,
    use_early_stopping=True,
    task="binary",
    metric_precision=6,
)

xgb_trainer.fit(X_xgboost, y)

# Store results
scores["XGBoost"] = xgb_trainer.fold_scores
oof_pred_probs["XGBoost"] = xgb_trainer.oof_preds
test_pred_probs["XGBoost"] = xgb_trainer.predict(X_test_xgboost)

print(f"\nâœ… XGBoost Mean CV Score: {np.mean(xgb_trainer.fold_scores):.6f} Â± {np.std(xgb_trainer.fold_scores):.6f}")


print("\nğŸ’¡ Training LightGBM Model...")
print("-" * 40)

# LightGBM parameters (optimized)
lightgbm_params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'max_depth': 4,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1,
    'min_child_samples': 20,
    'random_state': CFG.seed,
    'n_jobs': -1,
    'verbose': -1
}

# Train with cross-validation
lgb_trainer = Trainer(
    LGBMClassifier(**lightgbm_params),
    cv=CFG.cv,
    metric=accuracy_score,
    use_early_stopping=True,
    task="binary",
    metric_precision=6,
)

lgb_trainer.fit(X_xgboost, y)

# Store results
scores["LightGBM"] = lgb_trainer.fold_scores
oof_pred_probs["LightGBM"] = lgb_trainer.oof_preds
test_pred_probs["LightGBM"] = lgb_trainer.predict(X_test_xgboost)

print(f"\nâœ… LightGBM Mean CV Score: {np.mean(lgb_trainer.fold_scores):.6f} Â± {np.std(lgb_trainer.fold_scores):.6f}")


print("\nğŸ’¡ Training LightGBM (GOSS) Model...")
print("-" * 40)

# LightGBM (GOSS) parameters
lgbm_goss_params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'max_depth': 4,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 1,
    'min_child_samples': 20,
    'boosting_type': 'goss',    # âœ… key difference for GOSS
    'random_state': CFG.seed,
    'n_jobs': -1,
    'verbose': -1
}

# Trainer configuration
lgbm_goss_trainer = Trainer(
    LGBMClassifier(**lgbm_goss_params),
    cv=CFG.cv,
    metric=accuracy_score,      # Use accuracy_score directly
    use_early_stopping=False,   # âœ… GOSS does not support bagging, so keep early_stopping=False
    task="binary",
    metric_precision=6,
)

# Train using X_xgboost (encoded numerical features)
lgbm_goss_trainer.fit(X_xgboost, y)

# Store results
scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
oof_pred_probs["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
test_pred_probs["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test_xgboost)

# Display final CV score
print(f"\nâœ… LightGBM (GOSS) Mean CV Score: {np.mean(lgbm_goss_trainer.fold_scores):.6f} Â± {np.std(lgbm_goss_trainer.fold_scores):.6f}")



print("\nğŸ�¯ Training LightGBM (DART) Model...")
print("-" * 40)

# LightGBM (DART) parameters
lgbm_dart_params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'max_depth': 4,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 1,
    'min_child_samples': 20,
    'boosting_type': 'dart',     # âœ… key difference for DART
    'random_state': CFG.seed,
    'n_jobs': -1,
    'verbose': -1
}

# Trainer configuration
lgbm_dart_trainer = Trainer(
    LGBMClassifier(**lgbm_dart_params),
    cv=CFG.cv,
    metric=accuracy_score,       # âœ… use sklearn's accuracy_score
    use_early_stopping=False,    # âœ… DART often works better without early stopping
    task="binary",
    metric_precision=6,
)

# Train using encoded numerical features
lgbm_dart_trainer.fit(X_xgboost, y)

# Store results
scores["LightGBM (dart)"] = lgbm_dart_trainer.fold_scores
oof_pred_probs["LightGBM (dart)"] = lgbm_dart_trainer.oof_preds
test_pred_probs["LightGBM (dart)"] = lgbm_dart_trainer.predict(X_test_xgboost)

# Display final CV score
print(f"\nâœ… LightGBM (DART) Mean CV Score: {np.mean(lgbm_dart_trainer.fold_scores):.6f} Â± {np.std(lgbm_dart_trainer.fold_scores):.6f}")



def display_model_performance():
    """Create a beautiful summary of model performances"""
    
    # Create performance dataframe
    performance_data = []
    for model_name, fold_scores in scores.items():
        performance_data.append({
            'Model': model_name,
            'Mean Score': np.mean(fold_scores),
            'Std Dev': np.std(fold_scores),
            'Min Score': np.min(fold_scores),
            'Max Score': np.max(fold_scores)
        })
    
    performance_df = pd.DataFrame(performance_data).sort_values('Mean Score', ascending=False)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of mean scores
    bars = ax1.bar(performance_df['Model'], performance_df['Mean Score'], 
                    color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181'], 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, performance_df['Mean Score']):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'{score:.6f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('ğŸ�† Model Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('CV Accuracy Score', fontsize=12)
    ax1.set_ylim(0.965, max(performance_df['Mean Score']) * 1.002)
    ax1.grid(axis='y', alpha=0.3)
    
    # Box plot of fold scores
    fold_data = []
    for model_name, fold_scores in scores.items():
        for score in fold_scores:
            fold_data.append({'Model': model_name, 'Score': score})
    
    fold_df = pd.DataFrame(fold_data)
    sns.boxplot(data=fold_df, x='Model', y='Score', ax=ax2, palette=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181'])
    ax2.set_title('ğŸ“Š Score Distribution Across Folds', fontsize=16, fontweight='bold')
    ax2.set_ylabel('CV Accuracy Score', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display detailed table
    print("\nğŸ“‹ DETAILED MODEL PERFORMANCE")
    print("=" * 60)
    styled_df = performance_df.style\
        .format({'Mean Score': '{:.6f}', 'Std Dev': '{:.6f}', 
                'Min Score': '{:.6f}', 'Max Score': '{:.6f}'})\
        .background_gradient(subset=['Mean Score'], cmap='YlGn')\
        .set_properties(**{'text-align': 'center'})
    
    display(styled_df)
    
    return performance_df

model_performance = display_model_performance()


print("\nâš–ï¸� ENSEMBLE STRATEGY: Weighted Average")
print("=" * 50)

def optimize_weighted_average():
    """Find optimal weights for model averaging"""
    
    def objective(trial):
        # Suggest weights for each model
        weights = np.array([trial.suggest_float(m, -1, 1) for m in oof_pred_probs.keys()])
        weights /= np.sum(weights)  # Normalize
        
        # Calculate weighted predictions
        preds = np.zeros(len(y))
        for m, weight in zip(oof_pred_probs.keys(), weights):
            preds += oof_pred_probs[m] * weight
        
        # Optimize threshold
        threshold = trial.suggest_float('threshold', 0.3, 0.7)
        
        # Calculate score
        return accuracy_score(y, (preds > threshold).astype(int))
    
    print("ğŸ”� Optimizing ensemble weights...")
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True, n_startup_trials=50)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    # Optimize with progress bar
    from tqdm import tqdm
    with tqdm(total=CFG.n_optuna_trials, desc="Weight Optimization") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best': f'{study.best_value:.6f}'})
        
        study.optimize(objective, n_trials=CFG.n_optuna_trials, callbacks=[callback], n_jobs=-1)
    
    return study

# Run optimization
weight_study = optimize_weighted_average()

# Extract best weights
best_weights = np.array([weight_study.best_params[m] for m in oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)

# Create weight dictionary
weight_dict = {
    model: weight for model, weight in sorted(
        zip(oof_pred_probs.keys(), best_weights),
        key=lambda x: x[1],
        reverse=True
    )
}

best_threshold_weighted = weight_study.best_params['threshold']

print(f"\nâœ… Optimal weights found!")
print(f"   Best CV Score: {weight_study.best_value:.6f}")
print(f"   Best threshold: {best_threshold_weighted:.3f}")

# Display weights
print("\nğŸ“Š Model Weights:")
for model, weight in weight_dict.items():
    print(f"   {model}: {weight:.4f}")

# Calculate weighted predictions
weighted_test_preds = np.zeros(len(X_test_xgboost))
for m, weight in weight_dict.items():
    weighted_test_preds += test_pred_probs[m] * weight

# Save weighted submission
scores['WeightedAverage'] = [weight_study.best_value] * CFG.n_folds
save_submission('weighted-average', _X_test, weighted_test_preds, 
                weight_study.best_value, best_threshold_weighted)


def create_final_results_visualization():
    """Create beautiful visualization of all results"""
    
    # Prepare data
    all_scores = pd.DataFrame(scores)
    mean_scores = all_scores.mean().sort_values(ascending=False)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('ğŸ�† FINAL MODEL PERFORMANCE SUMMARY', fontsize=20, fontweight='bold', y=1.02)
    
    # Box plot of all models
    order = mean_scores.index.tolist()
    
    # Create boxplot without ax parameter to get the dictionary
    bp = all_scores[order].boxplot(vert=False, patch_artist=True, return_type='dict')
    
    # Now move the plot to the correct axes
    for artist in bp['whiskers'] + bp['caps'] + bp['medians'] + bp['fliers']:
        artist.remove()
    for box in bp['boxes']:
        box.remove()
    
    # Create the boxplot directly on ax1
    bp = ax1.boxplot([all_scores[col].values for col in order], 
                     vert=False, patch_artist=True, labels=order)
    
    # Color ensemble methods differently
    colors = []
    for model in order:
        if 'Logistic' in model or 'Weighted' in model:
            colors.append('#FFD93D')  # Gold for ensembles
        else:
            colors.append('#6BCF7F')  # Green for base models
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Score Distribution by Model', fontsize=16)
    ax1.set_xlabel('Accuracy Score', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Bar plot with annotations
    bars = ax2.barh(range(len(mean_scores)), mean_scores.values, color=colors)
    ax2.set_yticks(range(len(mean_scores)))
    ax2.set_yticklabels(mean_scores.index)
    ax2.set_xlabel('Mean Accuracy Score', fontsize=12)
    ax2.set_title('Average Performance Ranking', fontsize=16)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, mean_scores.values)):
        ax2.text(score + 0.0001, i, f'{score:.6f}', va='center', fontweight='bold')
    
    # Add grid
    ax2.grid(axis='x', alpha=0.3)
    
    # Set x-axis limits for better visualization
    min_score = mean_scores.min()
    max_score = mean_scores.max()
    ax2.set_xlim(min_score - 0.001, max_score + 0.001)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nğŸ�Š COMPETITION RESULTS SUMMARY")
    print("=" * 60)
    print(f"ğŸ¥‡ Best Model: {mean_scores.index[0]} - Score: {mean_scores.iloc[0]:.6f}")
    print(f"ğŸ¥ˆ Second Best: {mean_scores.index[1]} - Score: {mean_scores.iloc[1]:.6f}")
    print(f"ğŸ¥‰ Third Best: {mean_scores.index[2]} - Score: {mean_scores.iloc[2]:.6f}")
    
    # Achievement check
    if mean_scores.iloc[0] > 0.976:
        display(HTML("""
        <div style="background-color: #4CAF50; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <h2>ğŸ�‰ CONGRATULATIONS! ğŸ�‰</h2>
            <h3>You've achieved the target accuracy of 97.6%+!</h3>
            <p style="font-size: 18px;">Your model is ready to climb the leaderboard! ğŸš€</p>
        </div>
        """))
    
    return mean_scores

final_results = create_final_results_visualization()


print("\nğŸ“¤ PREPARING FINAL SUBMISSION")
print("=" * 50)

# Choose best model for final submission
best_model_name = final_results.index[0]
best_score = final_results.iloc[0]

print(f"ğŸ�† Using {best_model_name} for final submission")
print(f"ğŸ“Š Expected accuracy: {best_score:.6f}")

# Create final submission info
submission_info = {
    'Model': best_model_name,
    'CV_Score': best_score,
    'Features_Used': len(X_xgboost.columns),
    'Training_Samples': len(y),
    'Test_Samples': len(_X_test)
}

# Display submission summary
display(HTML(f"""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px;">
    <h3>ğŸ“‹ Final Submission Summary</h3>
    <table style="width: 100%; font-size: 14px;">
        <tr><td><strong>Model:</strong></td><td>{submission_info['Model']}</td></tr>
        <tr><td><strong>CV Score:</strong></td><td>{submission_info['CV_Score']:.6f}</td></tr>
        <tr><td><strong>Features Used:</strong></td><td>{submission_info['Features_Used']}</td></tr>
        <tr><td><strong>Training Samples:</strong></td><td>{submission_info['Training_Samples']:,}</td></tr>
        <tr><td><strong>Test Samples:</strong></td><td>{submission_info['Test_Samples']:,}</td></tr>
    </table>
</div>
"""))

# Save artifacts for reproducibility
print("\nğŸ’¾ Saving model artifacts...")
joblib.dump(oof_pred_probs, "oof_predictions.pkl")
joblib.dump(test_pred_probs, "test_predictions.pkl")
joblib.dump(scores, "model_scores.pkl")
print("âœ… All artifacts saved!")

print("\nğŸ�¯ NEXT STEPS:")
print("1. Submit the best performing CSV file to Kaggle")
print("2. Monitor your leaderboard position")
print("3. Consider further hyperparameter tuning if needed")
print("4. Share this notebook if you found it helpful! ğŸ˜Š")

# Final celebration
display(HTML("""
<div style="background-color: #FFE082; padding: 30px; border-radius: 15px; text-align: center; margin-top: 30px;">
    <h1>ğŸŒŸ Thank You for Following Along! ğŸŒŸ</h1>
    <p style="font-size: 18px;">If this notebook helped you, please consider upvoting! ğŸ‘�</p>
    <p style="font-size: 16px;">Happy Kaggling! ğŸ�‰</p>
</div>
"""))

