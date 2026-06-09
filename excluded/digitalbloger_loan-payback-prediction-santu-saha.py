import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)



print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")



print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nFirst few rows:")
display(train.head())


print("\nDataset Info:")
train.info()


print("\nMissing Values:")
print(train.isnull().sum())


print("\nStatistical Summary:")
display(train.describe())



print("\n" + "="*60)
print("DATA VISUALIZATIONS")
print("="*60)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


train['loan_paid_back'].value_counts().plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Loan Payback Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Loan Paid Back')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)


train['loan_paid_back'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
axes[1].set_title('Loan Payback Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()



numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
        sns.histplot(data=train, x=col, hue='loan_paid_back', bins=30, ax=axes[idx], palette=['#FF6B6B', '#4ECDC4'])
        axes[idx].set_title(f'{col.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')


fig.delaxes(axes[5])

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 8))
correlation = train[numerical_cols + ['loan_paid_back']].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=1, square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()



def preprocess_data(df, is_train=True):
    df = df.copy()
    
 
    if is_train:
        target = df['loan_paid_back']
        df = df.drop(['loan_paid_back', 'id'], axis=1)
    else:
        test_ids = df['id']
        df = df.drop(['id'], axis=1)
    

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    

    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income_pct'] = (df['loan_amount'] / df['annual_income']) * 100
    df['total_debt'] = df['annual_income'] * df['debt_to_income_ratio']
    df['remaining_income'] = df['annual_income'] - df['total_debt']
    df['can_afford'] = (df['remaining_income'] > df['loan_amount']).astype(int)
    

    df['monthly_payment_est'] = (df['loan_amount'] * (df['interest_rate']/100) / 12)
    df['payment_to_income'] = df['monthly_payment_est'] / (df['annual_income']/12)
    
   
    df['credit_tier'] = pd.cut(df['credit_score'], 
                                bins=[0, 580, 670, 740, 800, 900],
                                labels=[0,1,2,3,4]).astype(int)
    
 
    df['risk_score'] = (df['debt_to_income_ratio'] * 100 + 
                         (800 - df['credit_score'])/10 + 
                         df['interest_rate'])
    
    if is_train:
        return df, target
    else:
        return df, test_ids

print("Preprocessing data...")
X, y = preprocess_data(train, is_train=True)
X_test, test_ids = preprocess_data(test, is_train=False)

print(f"Features shape: {X.shape}")
print(f"New features added!")
print(X.columns.tolist())



print("\nğŸ¤– Training Models...")
print("="*60)


X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n1ï¸�âƒ£ Training LightGBM...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, 
                       valid_sets=[lgb_valid], 
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

lgb_pred_val = lgb_model.predict(X_val)
lgb_score = roc_auc_score(y_val, lgb_pred_val)
print(f"LightGBM Validation AUC: {lgb_score:.5f}")


print("\n2ï¸�âƒ£ Training XGBoost...")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

xgb_train = xgb.DMatrix(X_tr, label=y_tr)
xgb_valid = xgb.DMatrix(X_val, label=y_val)
xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=1000, 
                       evals=[(xgb_valid, 'eval')], 
                       early_stopping_rounds=50, verbose_eval=100)

xgb_pred_val = xgb_model.predict(xgb.DMatrix(X_val))
xgb_score = roc_auc_score(y_val, xgb_pred_val)
print(f"XGBoost Validation AUC: {xgb_score:.5f}")


print("\n3ï¸�âƒ£ Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
cat_pred_val = cat_model.predict_proba(X_val)[:, 1]
cat_score = roc_auc_score(y_val, cat_pred_val)
print(f"CatBoost Validation AUC: {cat_score:.5f}")

print("\n" + "="*60)
print("trained successfully!")



print("\nğŸ�¯ Making Predictions on Test Set...")
print("="*60)


lgb_test_pred = lgb_model.predict(X_test)
xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test))
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]


print("\nğŸ§© Creating Ensemble Predictions...")
total_score = lgb_score + xgb_score + cat_score
w1 = lgb_score / total_score
w2 = xgb_score / total_score
w3 = cat_score / total_score

print(f"Weights: LGB={w1:.3f}, XGB={w2:.3f}, CAT={w3:.3f}")

final_pred = w1 * lgb_test_pred + w2 * xgb_test_pred + w3 * cat_test_pred


final_pred_avg = (lgb_test_pred + xgb_test_pred + cat_test_pred) / 3


submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': final_pred
})


submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file created: submission.csv")
print(f"Shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))
print("\nPrediction statistics:")
print(submission['loan_paid_back'].describe())

print("\n" + "="*60)
print("ğŸ�‰ Done!!")
print("="*60)

