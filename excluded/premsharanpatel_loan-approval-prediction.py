# Import essential libraries for data handling, visualization, and modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Import scikit-learn tools for model evaluation and data splitting

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


# Load data

train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


train


# Display dataset dimensions (rows, columns)
print(train.shape, test.shape)



# Display first few rows of training data
train.head()


# Check target variable distribution
print(train['loan_status'].value_counts(normalize=True))


# Basic Exploratory Data Analysis (EDA)

# Check for missing values

print(train.isnull().sum().sort_values(ascending=False).head(20))


# Count number of numerical and categorical features

print(train.dtypes.value_counts())


# Plot Target distribution
target_distribution = train['loan_status'].value_counts()
target_percentage = train['loan_status'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
axes[0].bar(target_distribution.index, target_distribution.values, color=['#2ca02c', '#d62728'])
axes[0].set_xlabel('Loan Status')
axes[0].set_ylabel('Count')
axes[0].set_title('Target Variable Distribution (Count)')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Rejected (0)', 'Approved (1)'])

# Add count labels
for i, v in enumerate(target_distribution.values):
    axes[0].text(i, v + 100, str(v), ha='center')

# Pie chart
axes[1].pie(target_percentage.values, labels=['Rejected', 'Approved'], 
            autopct='%1.2f%%', colors=['#d62728', '#2ca02c'], startangle=90)
axes[1].set_title('Target Variable Distribution (Percentage)')

plt.tight_layout()
plt.show()

print(f"Target Variable Distribution:")
print(f"Rejected (0): {target_distribution.get(0, 0)} ({target_percentage.get(0, 0):.2f}%)")
print(f"Approved (1): {target_distribution.get(1, 0)} ({target_percentage.get(1, 0):.2f}%)")
print(f"\nClass Imbalance Ratio: {target_distribution.get(0, 0) / target_distribution.get(1, 1):.2f}")


plt.figure(figsize=(7,5))
sns.histplot(train['person_age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Applicant Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()



plt.figure(figsize=(7,5))
sns.histplot(train['loan_int_rate'], bins=30, kde=True, color='purple')
plt.title('Distribution of Loan Interest Rate')
plt.xlabel('Interest Rate')
plt.ylabel('Count')
plt.show()



df = pd.DataFrame({
    'person_home_ownership': ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
    'count': [30594, 24824, 3138, 89],
    'Ratio': [52.168130, 42.329269, 5.350840, 0.151761]
})

# 1. Bar Chart: person_home_ownership vs count

plt.figure(figsize=(8,5))
bars = plt.bar(df['person_home_ownership'], df['count'], color=['skyblue', 'orange', 'green', 'red'])
plt.title('Counts of Person Home Ownership')
plt.xlabel('person_home_ownership')
plt.ylabel('Count')

# Add count labels on top of bars
for bar, count in zip(bars, df['count']):
    plt.text(bar.get_x() + bar.get_width()/2, count + 500, str(count), ha='center', fontsize=10)

plt.show()

# 2. Pie Chart: person_home_ownership vs Ratio

plt.figure(figsize=(6,6))
colors = ['skyblue', 'orange', 'green', 'red']
plt.pie(df['Ratio'], labels=df['person_home_ownership'], autopct='%1.2f%%', startangle=90, colors=colors)
plt.title('Distribution of Person Home Ownership by Ratio')
plt.legend([f'{cat} - {ratio:.2f}%' for cat, ratio in zip(df['person_home_ownership'], df['Ratio'])],
           loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()



# Compute counts and ratios 

counts = train['loan_intent'].value_counts()           
ratios = 100 * counts / counts.sum()                   
categories = ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']                     # category names


# 1. Bar Chart: loan_intent vs count
 
plt.figure(figsize=(10,6))
bars = plt.bar(categories, counts.values, color=plt.cm.tab20.colors)
plt.title('Counts of Loan Intent')
plt.xlabel('loan_intent')
plt.ylabel('Count')

# Add count labels on top of bars
for bar, count in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, count + 200, str(count), ha='center', fontsize=10)

plt.xticks(rotation=45)
plt.show()

# 2. Pie Chart: loan_intent vs ratio

plt.figure(figsize=(8,8))
plt.pie(ratios, labels=categories, autopct='%1.2f%%', startangle=90, colors=plt.cm.tab20.colors)
plt.title('Distribution of Loan Intent by Ratio')

# Legend showing category + ratio
plt.legend([f'{cat} - {ratio:.2f}%' for cat, ratio in zip(categories, ratios)],
           loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()



# Data Preprocessing

# Drop columns if any with too many missing values, or fill missing

for df in [train, test]:
    # Label encode categorical
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Fill missing numerical columns with median values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())



# Separate features and target variable

X = train.drop(['id','loan_status'], axis=1)
y = train['loan_status']
X_test = test.drop(['id'], axis=1)


# Baseline model: LightGBM with simple params and CV

# Define model parameters

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1
}

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    
    print(f"Fold {fold} AUC = {roc_auc_score(y_val, oof_preds[val_idx]):.4f}")

print("Overall AUC:", roc_auc_score(y, oof_preds))


# Add predictions to submission dataframe

submission = pd.DataFrame({
    'id': test['id'],
    'loan_status': test_preds  # probabilities
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv file created successfully!")

