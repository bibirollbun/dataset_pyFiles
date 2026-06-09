import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')




path = '/kaggle/input/playground-series-s5e8/'
train = pd.read_csv(path + 'train.csv', index_col='id')
test = pd.read_csv(path + 'test.csv', index_col='id')
submission = pd.read_csv(path + 'sample_submission.csv', index_col='id')




print("Train data shape:", train.shape)
print("Test data shape:", test.shape)



plt.figure(figsize=(6,4))
sns.countplot(x='y', data=train, palette='Set2')
plt.title('Subscribed (y) Distribution')
plt.xlabel('y')
plt.ylabel('Count')
plt.show()



fig, axes = plt.subplots(1,2,figsize=(12,4))
sns.countplot(y='job', data=train, order=train['job'].value_counts().index, ax=axes[0])
axes[0].set_title('Job Distribution')
sns.countplot(y='education', data=train, order=train['education'].value_counts().index, ax=axes[1])
axes[1].set_title('Education Distribution')
plt.tight_layout()
plt.show()



numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
train[numerical_cols].hist(figsize=(14, 10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Distributions of Original Numerical Features', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] 
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

def preprocess(df):
    # Handle numerical features
    for col in numerical_cols:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q99)
    
    # Transformations
    df['duration_log'] = np.log1p(df['duration'])
    df['campaign_log'] = np.log1p(df['campaign'])
    df['pdays_log'] = np.log1p(df['pdays'] + 1)
    df['previous_log'] = np.log1p(df['previous'])
    df['balance_sqrt'] = np.sqrt(df['balance'] - df['balance'].min() + 1)
    df['age_squared'] = df['age'] ** 2
    
    # Interaction features
    df['balance_duration_ratio'] = df['balance'] / (df['duration'] + 1)
    df['campaign_previous_ratio'] = df['campaign'] / (df['previous'] + 1)
    
    # Handle categorical features
    for feature in cat_cols:
        df[feature] = df[feature].astype("category")
        
    # Date-related features
    if 'month' in df.columns:
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        df['month_num'] = df['month'].map(month_map).astype('int8')
        
    return df

# Apply the preprocessing
train = preprocess(train)
test = preprocess(test)

print("Feature engineering complete. New train shape:", train.shape)



cols = ['duration_log','campaign_log','balance_duration_ratio']
plt.figure(figsize=(6,6))
sns.pairplot(train[cols].sample(500), corner=True, diag_kind='hist')
plt.suptitle('Pairwise Relationships of New Features', y=1.02)
plt.show()



# Target encoding for categorical features
target_enc_cols = [f"{col}_target_enc" for col in cat_cols]
global_means = {}

for col in cat_cols:
    # Calculate global mean for fallback
    global_means[col] = train['y'].mean()
    # Encode training data
    enc_values = train.groupby(col)['y'].mean()
    # Convert to float to avoid categorical type issues
    train[f"{col}_target_enc"] = train[col].map(enc_values).astype(float).fillna(global_means[col])
    # Encode test data
    test[f"{col}_target_enc"] = test[col].map(enc_values).astype(float).fillna(global_means[col])

print("Target encoding complete.")



# Prepare features and target
X = train.drop(columns=['y'])
y = train['y'].values

# Handling class imbalance
scale_pos_weight = (len(y) - y.sum()) / y.sum()
print(f"Scale Pos Weight (for class imbalance): {scale_pos_weight:.2f}")

# Model configuration
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'max_depth': 8,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'subsample_freq': 1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}



# Cross-validation setup
n_splits = 7
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(test.shape[0])
oof_preds = np.zeros(X.shape[0])
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold+1}/{n_splits}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)
    
    # training the model
    model = lgb.train(
        params,
        train_set,
        num_boost_round=15000,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=500)
        ]
    )
    
    # Validation predictions
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    fold_score = log_loss(y_val, val_preds)
    val_scores.append(fold_score)
    print(f"Fold {fold+1} Log Loss: {fold_score:.5f}")
    
    # Test predictions
    test_preds += model.predict(test[X.columns]) / n_splits



imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance('gain')
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=imp.head(20), palette='viridis')
plt.title('Top 20 Features by Gain')
plt.tight_layout()
plt.show()



oof_score = log_loss(y, oof_preds)
print(f"\nOverall OOF Log Loss: {oof_score:.5f}")
print(f"Average Fold Log Loss: {np.mean(val_scores):.5f} Â± {np.std(val_scores):.5f}")



# Create submission
submission['y'] = test_preds



submission = submission.reset_index()



submission.to_csv('submission.csv', index=False)
print("Submission saved!")


submission




