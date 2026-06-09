import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PLAYGROUND SERIES S5E12 - DIABETES PREDICTION")
print("="*80)



print("\nğŸ“‚ Loading data...")

train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
print(f"âœ“ Training data loaded: {train_df.shape}")

test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
print(f"âœ“ Test data loaded: {test_df.shape}")

sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
print(f"âœ“ Sample submission loaded: {sample_sub.shape}")
print(f"  Submission columns: {list(sample_sub.columns)}")

print("\n First few rows of training data:")
print(train_df.head())

print("\n Dataset Info:")
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Number of features: {train_df.shape[1] - 2}")  # Exclude id and target



print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\n Training Data Info:")
print(train_df.info())

print("\n Target Variable Distribution:")
target_dist = train_df['diagnosed_diabetes'].value_counts()
print(target_dist)
print(f"\nDiabetes rate: {train_df['diagnosed_diabetes'].mean():.2%}")

print("\n Missing Values:")
missing = train_df.isnull().sum()
if missing.sum() == 0:
    print("âœ“ No missing values in training data")
else:
    print(missing[missing > 0])

print("\n Statistical Summary:")
print(train_df.describe())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

train_df['diagnosed_diabetes'].value_counts().plot(
    kind='bar', ax=axes[0, 0], color=['lightblue', 'salmon']
)
axes[0, 0].set_title('Target Distribution (diagnosed_diabetes)')
axes[0, 0].set_xlabel('Diagnosed Diabetes')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

numeric_cols = train_df.select_dtypes(include=[np.number]).columns[:15]
corr_matrix = train_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[0, 1], cbar=True)
axes[0, 1].set_title('Feature Correlation Heatmap')

if 'Age' in train_df.columns:
    train_df['Age'].hist(bins=30, ax=axes[1, 0], color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Age Distribution')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Frequency')

if len(numeric_cols) > 1:
    target_corr = train_df[numeric_cols].corrwith(train_df['diagnosed_diabetes']).sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != 'diagnosed_diabetes'][:10]
    target_corr.plot(kind='barh', ax=axes[1, 1], color='teal')
    axes[1, 1].set_title('Top 10 Features Correlated with Target')
    axes[1, 1].set_xlabel('Correlation')

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

target_counts = train_df['diagnosed_diabetes'].value_counts()
axes[0, 0].pie(
    target_counts,
    labels=['No Diabetes (0)', 'Diabetes (1)'],
    autopct='%1.1f%%',
    startangle=90,
    explode=[0.05, 0.05]
)
axes[0, 0].set_title('Diabetes Distribution (Pie Chart)')

if 'Age' in train_df.columns:
    age_group = train_df.groupby('Age')['diagnosed_diabetes'].mean()
    axes[0, 1].plot(age_group.index, age_group.values, marker='o')
    axes[0, 1].set_title('Diabetes Rate by Age')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Diabetes Rate')

if 'Age' in train_df.columns:
    sns.boxplot(
        x='diagnosed_diabetes',
        y='Age',
        data=train_df,
        ax=axes[0, 2]
    )
    axes[0, 2].set_title('Age Distribution vs Diabetes')

if 'BMI' in train_df.columns:
    sns.kdeplot(
        data=train_df,
        x='BMI',
        hue='diagnosed_diabetes',
        fill=True,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('BMI Distribution by Diabetes Status')

binary_cols = [
    col for col in train_df.columns
    if train_df[col].nunique() == 2 and col != 'diagnosed_diabetes'
]

if len(binary_cols) > 0:
    sns.countplot(
        x=binary_cols[0],
        hue='diagnosed_diabetes',
        data=train_df,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title(f'{binary_cols[0]} vs Diabetes')

cum_cases = train_df['diagnosed_diabetes'].cumsum()
axes[1, 2].plot(cum_cases)
axes[1, 2].set_title('Cumulative Diabetes Cases')
axes[1, 2].set_xlabel('Sample Index')
axes[1, 2].set_ylabel('Cumulative Count')

plt.tight_layout()
plt.show()




print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

test_ids = test_df['id'].copy()
X_test_final = test_df.drop('id', axis=1)

print(f"\nâœ“ Training features shape: {X.shape}")
print(f"âœ“ Target shape: {y.shape}")
print(f"âœ“ Test features shape: {X_test_final.shape}")

print("\n Feature Data Types:")
print(X.dtypes.value_counts())

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"\nâš  Found categorical columns: {categorical_cols}")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        X_test_final[col] = le.transform(X_test_final[col].astype(str))
    print("âœ“ Categorical columns encoded")



print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

def create_features(df):
    df_new = df.copy()
    
    if 'Age' in df.columns and 'BMI' in df.columns:
        df_new['Age_BMI_interaction'] = df['Age'] * df['BMI']

    return df_new

X_engineered = create_features(X)
X_test_engineered = create_features(X_test_final)

print(f"âœ“ Features after engineering: {X_engineered.shape[1]}")


X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nâœ“ Training set: {X_train.shape}")
print(f"âœ“ Validation set: {X_val.shape}")


print("\n" + "="*80)
print("FEATURE SCALING")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_engineered)

print("âœ“ Features scaled using StandardScaler")



print("\n" + "="*80)
print("MODEL TRAINING & EVALUATION")
print("="*80)

models = {}
results = {}

print("\n1ï¸�âƒ£ Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_proba = lr.predict_proba(X_val_scaled)[:, 1]
lr_score = roc_auc_score(y_val, lr_proba)
models['Logistic Regression'] = lr
results['Logistic Regression'] = lr_score
print(f"   ROC AUC: {lr_score:.5f}")

print("\n2ï¸�âƒ£ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_proba = rf.predict_proba(X_val_scaled)[:, 1]
rf_score = roc_auc_score(y_val, rf_proba)
models['Random Forest'] = rf
results['Random Forest'] = rf_score
print(f"   ROC AUC: {rf_score:.5f}")

print("\n3ï¸�âƒ£ Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_proba = gb.predict_proba(X_val_scaled)[:, 1]
gb_score = roc_auc_score(y_val, gb_proba)
models['Gradient Boosting'] = gb
results['Gradient Boosting'] = gb_score
print(f"   ROC AUC: {gb_score:.5f}")

print("\n4ï¸�âƒ£ Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_scaled, y_train)
xgb_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
xgb_score = roc_auc_score(y_val, xgb_proba)
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_score
print(f"   ROC AUC: {xgb_score:.5f}")

print("\n5ï¸�âƒ£ Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
lgb_proba = lgb_model.predict_proba(X_val_scaled)[:, 1]
lgb_score = roc_auc_score(y_val, lgb_proba)
models['LightGBM'] = lgb_model
results['LightGBM'] = lgb_score
print(f"   ROC AUC: {lgb_score:.5f}")



from sklearn.metrics import confusion_matrix

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'ROC AUC'])
results_df = results_df.sort_values('ROC AUC', ascending=False)
print("\n", results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_score = results_df.iloc[0]['ROC AUC']
best_model = models[best_model_name]

print(f"\n Best Model: {best_model_name}")
print(f" Validation ROC AUC: {best_score:.5f}")

plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['ROC AUC'], color='skyblue')
plt.xlabel('ROC AUC Score')
plt.title('Model Performance Comparison')
plt.xlim(0.5, 1.0)
for i, v in enumerate(results_df['ROC AUC']):
    plt.text(v + 0.01, i, f'{v:.5f}', va='center')
plt.tight_layout()
plt.show()

y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()



print("\n" + "="*80)
print("CREATING ENSEMBLE MODEL")
print("="*80)

ensemble_proba = (xgb_proba + lgb_proba + gb_proba) / 3
ensemble_score = roc_auc_score(y_val, ensemble_proba)
print(f"\nğŸ“Š Ensemble (XGB+LGB+GB) ROC AUC: {ensemble_score:.5f}")

if ensemble_score > best_score:
    print(f"âœ“ Ensemble improves score by {ensemble_score - best_score:.5f}")
    use_ensemble = True
else:
    print(f"âœ— Ensemble doesn't improve score, using {best_model_name}")
    use_ensemble = False




print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

use_ensemble = True   

best_model_name = "GradientBoosting"
best_model = gb


if use_ensemble:
    print("\n Using Ensemble Model for final predictions...")

    test_pred_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    test_pred_lgb = lgb_model.predict_proba(X_test_scaled)[:, 1]
    test_pred_gb  = gb.predict_proba(X_test_scaled)[:, 1]

   
    test_predictions = (
        test_pred_xgb + test_pred_lgb + test_pred_gb
    ) / 3

else:
    print(f"\n Using {best_model_name} for final predictions...")
    test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"âœ“ Predictions generated for {len(test_predictions)} test samples")



print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nâœ… SUBMISSION FILE CREATED: submission.csv")
print(f"âœ“ Shape: {submission.shape}")
print(f"âœ“ Columns: {list(submission.columns)}")


print("\n" + "="*80)
print("SUBMISSION VERIFICATION")
print("="*80)

print(f"\nâœ“ Sample submission shape: {sample_sub.shape}")
print(f"âœ“ Our submission shape: {submission.shape}")
print(f"âœ“ Shapes match: {sample_sub.shape == submission.shape}")

print(f"\nâœ“ Required columns: {list(sample_sub.columns)}")
print(f"âœ“ Our columns: {list(submission.columns)}")
print(f"âœ“ Columns match: {list(sample_sub.columns) == list(submission.columns)}")

missing_count = submission.isnull().sum().sum()
print(f"\nâœ“ Missing values: {missing_count}")

print(f"\nâœ“ Min probability: {submission['diagnosed_diabetes'].min():.6f}")
print(f"âœ“ Max probability: {submission['diagnosed_diabetes'].max():.6f}")
print(f"âœ“ Mean probability: {submission['diagnosed_diabetes'].mean():.6f}")
print(f"âœ“ All in [0,1] range: {submission['diagnosed_diabetes'].between(0, 1).all()}")

print(f"\nâœ“ Unique IDs: {submission['id'].nunique()} / {len(submission)}")
print(f"âœ“ All IDs unique: {submission['id'].nunique() == len(submission)}")

print("\nğŸ“‹ First 10 predictions:")
print(submission.head(10).to_string(index=False))

print("\nğŸ“‹ Last 5 predictions:")
print(submission.tail(5).to_string(index=False))

print("\nğŸ“Š Prediction Distribution:")
print(f"   0.0 - 0.2: {(submission['diagnosed_diabetes'] < 0.2).sum()}")
print(f"   0.2 - 0.4: {((submission['diagnosed_diabetes'] >= 0.2) & (submission['diagnosed_diabetes'] < 0.4)).sum()}")
print(f"   0.4 - 0.6: {((submission['diagnosed_diabetes'] >= 0.4) & (submission['diagnosed_diabetes'] < 0.6)).sum()}")
print(f"   0.6 - 0.8: {((submission['diagnosed_diabetes'] >= 0.6) & (submission['diagnosed_diabetes'] < 0.8)).sum()}")
print(f"   0.8 - 1.0: {(submission['diagnosed_diabetes'] >= 0.8).sum()}")



print("\n" + "="*80)
print("âœ… EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n Best Model: {best_model_name if not use_ensemble else 'Ensemble'}")
print(f"Validation ROC AUC: {(best_score if not use_ensemble else ensemble_score):.5f}")
print("Submission File: submission.csv")
print(f" Test Predictions: {len(submission)} samples")






