# Install Optuna
!pip install optuna -q

# Simple imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import chi2_contingency
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("âš¡ Libraries loaded!")



# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_ids = test_df['id'].copy()

# Load original data for boost
try:
    original_df = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=';')
    original_df['y'] = original_df['y'].map({'yes': 1, 'no': 0})
    print(f"âœ… Loaded: Train {train_df.shape}, Test {test_df.shape}, Original {original_df.shape}")
except:
    original_df = None
    print("âš ï¸� Original data not found, using synthetic only")


# Remove ID column for analysis
df = train_df.drop('id', axis=1).copy()
# Advanced target analysis with multiple perspectives
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ğŸ�¯ Target Variable Complete Analysis', fontsize=20, fontweight='bold', y=0.98)

# 1. Basic distribution
target_counts = df['y'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4']
bars = axes[0,0].bar(['No (0)', 'Yes (1)'], target_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
axes[0,0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Count', fontsize=12)

# Add value labels on bars
for bar, val in zip(bars, target_counts.values):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                   f'{val:,}\n({val/len(df)*100:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Pie chart with explosion
axes[0,1].pie(target_counts.values, labels=['No Subscription', 'Subscription'], 
              colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05),
              textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0,1].set_title('Target Proportion', fontsize=14, fontweight='bold')

# 3. Class imbalance visualization
imbalance_ratio = target_counts[0] / target_counts[1]
axes[0,2].bar(['Imbalance Ratio'], [imbalance_ratio], color='#FFD93D', alpha=0.8, edgecolor='black', linewidth=2)
axes[0,2].set_title('Class Imbalance Analysis', fontsize=14, fontweight='bold')
axes[0,2].set_ylabel('Ratio (Majority : Minority)', fontsize=12)
axes[0,2].text(0, imbalance_ratio + 0.2, f'{imbalance_ratio:.1f}:1', 
               ha='center', va='bottom', fontweight='bold', fontsize=12)

# 4. Cumulative distribution
cumsum = np.cumsum([target_counts[0], target_counts[1]])
axes[1,0].bar(['No', 'No + Yes'], [target_counts[0], cumsum[1]], 
              color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=2)
axes[1,0].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Cumulative Count', fontsize=12)

# 5. Statistical metrics
metrics = ['Samples', 'Majority %', 'Minority %', 'Imbalance']
values = [len(df), target_counts[0]/len(df)*100, target_counts[1]/len(df)*100, imbalance_ratio]
bars = axes[1,1].bar(metrics, values, color=['#A8E6CF', '#FFB6C1', '#DDA0DD', '#F0E68C'], 
                     alpha=0.8, edgecolor='black', linewidth=2)
axes[1,1].set_title('Key Statistics', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Value', fontsize=12)

# Add value labels
for bar, val in zip(bars, values):
    if val > 1000:
        label = f'{val:,.0f}'
    else:
        label = f'{val:.1f}'
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=10)

# 6. Business impact visualization
success_rate = target_counts[1] / len(df)
failure_rate = target_counts[0] / len(df)
axes[1,2].pie([failure_rate, success_rate], labels=['Campaign Failed', 'Campaign Succeeded'],
              colors=['#FF6B6B', '#4ECDC4'], autopct='%1.2f%%', startangle=90,
              textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1,2].set_title('Campaign Success Rate', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print detailed statistics
print("ğŸ�¯ TARGET VARIABLE INSIGHTS:")
print("="*50)
print(f"ğŸ“Š Total Campaigns: {len(df):,}")
print(f"âœ… Successful Subscriptions: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
print(f"â�Œ Failed Campaigns: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
print(f"âš–ï¸� Class Imbalance Ratio: {imbalance_ratio:.1f}:1")
print(f"ğŸ�¯ Campaign Success Rate: {success_rate*100:.2f}%")


def smart_preprocessing(train_df, test_df, original_df=None):
    # Basic setup
    X_train = train_df.drop(['id', 'y'], axis=1)
    y_train = train_df['y']
    X_test = test_df.drop('id', axis=1)
    
    # Smarter original data integration
    if original_df is not None:
        common_cols = list(set(X_train.columns) & set(original_df.columns))
        
        # Stratified sampling by target (maintain class balance)
        orig_pos = original_df[original_df['y'] == 1]
        orig_neg = original_df[original_df['y'] == 0]
        
        # Sample proportionally to synthetic data distribution
        synth_pos_ratio = y_train.mean()
        sample_size = len(X_train) // 3  # 33% instead of 25%
        
        pos_samples = min(int(sample_size * synth_pos_ratio), len(orig_pos))
        neg_samples = min(sample_size - pos_samples, len(orig_neg))
        
        original_sample = pd.concat([
            orig_pos.sample(n=pos_samples, random_state=42),
            orig_neg.sample(n=neg_samples, random_state=42)
        ], ignore_index=True)
        
        X_train = pd.concat([X_train[common_cols], original_sample[common_cols]], ignore_index=True)
        y_train = pd.concat([y_train, original_sample['y']], ignore_index=True)
        X_test = X_test[common_cols]
        print(f"ğŸ”„ Added {len(original_sample):,} original samples (balanced)")
    
    # Enhanced Feature Engineering
    for df in [X_train, X_test]:
        # Original duration features
        df['duration_log'] = np.log1p(df['duration'])
        df['long_call'] = (df['duration'] > 600).astype(int)
        df['short_call'] = (df['duration'] < 120).astype(int)
        
        # NEW: Advanced duration features
        df['duration_squared'] = df['duration'] ** 2
        df['duration_sqrt'] = np.sqrt(df['duration'])
        df['duration_per_day'] = df['duration'] / (df['day'] + 1)
        df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
        
        # Duration categories (FIX: convert to int immediately)
        df['duration_category'] = np.where(df['duration'] <= 60, 0,
                          np.where(df['duration'] <= 180, 1,
                          np.where(df['duration'] <= 300, 2,
                          np.where(df['duration'] <= 600, 3, 4))))
        
        df['duration_momentum'] = df['duration'] * (df['previous'] + 1)
        df['quality_call'] = ((df['duration'] > 300) & (df['campaign'] <= 2)).astype(int)
        
        # Balance features
        df['balance_log'] = np.log1p(df['balance'] + abs(df['balance'].min()) + 1)
        df['has_positive_balance'] = (df['balance'] > 0).astype(int)
        
        # Campaign features
        df['campaign_efficiency'] = df['duration'] / (df['campaign'] + 1)
        df['few_contacts'] = (df['campaign'] <= 2).astype(int)
        
        # Previous contact features
        df['has_previous'] = (df['previous'] > 0).astype(int)
        df['previous_success'] = (df['poutcome'] == 'success').astype(int)
        
        # Age groups (FIX: convert to int immediately)
        df['age_group'] = np.where(df['age'] <= 30, 0,
                  np.where(df['age'] <= 45, 1,
                  np.where(df['age'] <= 60, 2, 3)))
        
        # Interaction features
        df['age_balance_int'] = df['age'] * df['balance_log']
        df['duration_campaign_int'] = df['duration_log'] * df['campaign']
    
    # FIX: Convert any remaining categorical columns to integers
    for df in [X_train, X_test]:
        categorical_columns = df.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            df[col] = df[col].astype(int)
    
    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    print(f"âœ… Features: {X_train.shape[1]} (added {X_train.shape[1] - train_df.shape[1] + 1} new)")
    return X_train, X_test, y_train

# Apply preprocessing
X_train, X_test, y_train = smart_preprocessing(train_df, test_df, original_df)


# Load data (assuming you already have X_train, X_test, y_train from previous run)
# If not, run your preprocessing code first

print(f"ğŸ“Š Data ready: X_train {X_train.shape}, X_test {X_test.shape}")
print(f"ğŸ�¯ Current best: 0.96924, Target: 0.97646, Gap: {0.97646 - 0.96924:.5f}")


# Train 3 different models (simple and fast)
print("ğŸ¤– Training 3 base models...")
import catboost as cb
import xgboost as xgb
# Model 1: LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=1500,  # More trees
    learning_rate=0.08,  # Slower learning
    max_depth=10,
    num_leaves=50,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
print("âœ… LightGBM trained")

# Model 2: CatBoost  
cat_model = cb.CatBoostClassifier(
    iterations=1500,
    learning_rate=0.08,
    depth=10,
    random_state=42,
    verbose=False
)
cat_model.fit(X_train, y_train)
print("âœ… CatBoost trained")

# Model 3: XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.08,
    max_depth=10,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print("âœ… XGBoost trained")


# Cross-validation function (simple)
def get_cv_score(lgb_weight, cat_weight, xgb_weight):
    """Get CV score for given ensemble weights"""
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Fast 3-fold
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train models on fold
        lgb_fold = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.1, random_state=42, verbose=-1)
        cat_fold = cb.CatBoostClassifier(iterations=800, learning_rate=0.1, random_state=42, verbose=False)
        xgb_fold = xgb.XGBClassifier(n_estimators=800, learning_rate=0.1, random_state=42, verbosity=0)
        
        lgb_fold.fit(X_fold_train, y_fold_train)
        cat_fold.fit(X_fold_train, y_fold_train)
        xgb_fold.fit(X_fold_train, y_fold_train)
        
        # Get fold predictions
        lgb_fold_pred = lgb_fold.predict_proba(X_fold_val)[:, 1]
        cat_fold_pred = cat_fold.predict_proba(X_fold_val)[:, 1]
        xgb_fold_pred = xgb_fold.predict_proba(X_fold_val)[:, 1]
        
        # Ensemble prediction
        ensemble_pred = lgb_weight * lgb_fold_pred + cat_weight * cat_fold_pred + xgb_weight * xgb_fold_pred
        
        # Score
        score = roc_auc_score(y_fold_val, ensemble_pred)
        scores.append(score)
    
    return np.mean(scores)

print("ğŸ”„ CV function ready")


# # Optuna optimization (simple and effective)
# def objective(trial):
#     """Optuna objective function"""
    
#     # Try different ensemble weights
#     lgb_weight = trial.suggest_float('lgb_weight', 0.2, 0.8)
#     cat_weight = trial.suggest_float('cat_weight', 0.1, 0.6)
#     xgb_weight = 1.0 - lgb_weight - cat_weight  # Remaining weight
    
#     # Make sure weights are positive
#     if xgb_weight <= 0:
#         return 0.0
    
#     # Get CV score
#     score = get_cv_score(lgb_weight, cat_weight, xgb_weight)
#     return score

# print("ğŸ�¯ Starting Optuna optimization...")
# print("This will take 5-10 minutes...")

# # Create study
# study = optuna.create_study(direction='maximize')

# # Optimize (50 trials should be enough)
# study.optimize(objective, n_trials=50)

# # Best results
# best_params = study.best_params
# best_score = study.best_value

# print(f"âœ… Optuna optimization complete!")
# print(f"ğŸ�† Best CV Score: {best_score:.6f}")
# print(f"ğŸ�¯ Best Weights:")
# print(f"   LightGBM: {best_params['lgb_weight']:.3f}")
# print(f"   CatBoost: {best_params['cat_weight']:.3f}")
# print(f"   XGBoost: {1.0 - best_params['lgb_weight'] - best_params['cat_weight']:.3f}")


# Create optimized ensemble prediction
print("ğŸš€ Creating final ensemble prediction...")

# FIRST: Generate test predictions from trained models
print("ğŸ“Š Generating test predictions from trained models...")
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
cat_pred = cat_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

# lgb_weight = best_params['lgb_weight']
# cat_weight = best_params['cat_weight']
# xgb_weight = 1.0 - lgb_weight - cat_weight

lgb_weight = 0.438
cat_weight = 0.115
xgb_weight = 0.447
# Final ensemble
final_prediction = lgb_weight * lgb_pred + cat_weight * cat_pred + xgb_weight * xgb_pred

print(f"ğŸ“Š Final Ensemble Stats:")
print(f"   Mean: {final_prediction.mean():.4f}")
print(f"   Range: [{final_prediction.min():.4f}, {final_prediction.max():.4f}]")


# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'y': final_prediction
})

submission.to_csv("submission.csv", index=False)

print("âœ… Optuna ensemble submission created!")
print("\nğŸ“‹ Sample predictions:")
print(submission.head())

