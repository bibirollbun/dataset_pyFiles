import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head()


test.head()


train.info()


test.info()


plt.figure(figsize=(10,6))
ax = sns.countplot(x='loan_paid_back',data=train)
plt.title('distribution of load paid back')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='center', xytext=(0, 10),textcoords='offset points')
plt.show()


train['loan_paid_back'].value_counts()


train['loan_paid_back'].value_counts(normalize=True)


numerical_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                     'loan_amount', 'interest_rate']

print("\n=== NUMERICAL FEATURES DISTRIBUTION ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numerical_features):
    axes[i].hist(train[col], bins=50, alpha=0.7, label='Train')
    axes[i].hist(test[col], bins=50, alpha=0.7, label='Test')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend()
    
# Hide empty subplot
axes[5].set_visible(False)
plt.tight_layout()
plt.show()


train[numerical_features].describe()


test[numerical_features].describe()


categorical_features = ['gender', 'marital_status', 'education_level', 
                       'employment_status', 'loan_purpose', 'grade_subgrade']

print("\n=== CATEGORICAL FEATURES ANALYSIS ===")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_features):
    # Get value counts for train and test
    train_counts = train[col].value_counts()
    test_counts = test[col].value_counts()
    
    # Create combined dataframe for plotting
    plot_df = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).fillna(0)
    
    plot_df.plot(kind='bar', ax=axes[i], title=f'{col} Distribution')
    axes[i].tick_params(axis='x', rotation=45)
    
plt.tight_layout()
plt.show()


for col in categorical_features:
    print(f"\n{col} - Train value counts:")
    print(train[col].value_counts())
    print(f"\n{col} - Test value counts:")
    print(test[col].value_counts())


plt.figure(figsize = (12,8))
correlation_matrix = train[numerical_features + ['loan_paid_back']].corr()
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm',center=0,square=True,fmt='.2f')
plt.title('correlation Matrix')
plt.show()


fig ,axes = plt.subplots(2,3,figsize=(18,12))
axes = axes.ravel()

for i,col in enumerate(numerical_features):
    sns.boxplot(x='loan_paid_back',y=col,data=train,ax=axes[i])
    axes[i].set_title(f'{col} vs Loan Paid Back')

axes[5].set_visible(False)
plt.tight_layout()
plt.show()


fig , axes =plt.subplots(2,3,figsize=(20,12))
axes = axes.ravel()

for i,col in enumerate(categorical_features):
    cross_tab = pd.crosstab(train[col],train['loan_paid_back'],normalize='index')*100
    cross_tab.plot(kind='bar',ax=axes[i],stacked=True)
    axes[i].set_title(f'Loan Paid Back by {col}')
    axes[i].tick_params(axis = 'x',rotation=45)
    axes[i].legend(title='Paid Back', labels=['No','Yes'])
plt.tight_layout()
plt.show()


# Create some potential derived features
train['income_to_loan_ratio'] = train['annual_income'] / train['loan_amount']
train['monthly_payment_estimate'] = (train['loan_amount'] * train['interest_rate'] / 100) / 12
train['credit_income_ratio'] = train['credit_score'] / train['annual_income']

test['income_to_loan_ratio'] = test['annual_income'] / test['loan_amount']
test['monthly_payment_estimate'] = (test['loan_amount'] * test['interest_rate'] / 100) / 12
test['credit_income_ratio'] = test['credit_score'] / test['annual_income']

# Check correlation of new features with target
new_features = ['income_to_loan_ratio', 'monthly_payment_estimate', 'credit_income_ratio']
new_corr = train[new_features + ['loan_paid_back']].corr()['loan_paid_back'].drop('loan_paid_back')

print("Correlation of new features with target:")
print(new_corr)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numerical_features):
    sns.boxplot(data=train, y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    
axes[5].set_visible(False)
plt.tight_layout()
plt.show()


# Summary insights
print("\n=== KEY INSIGHTS SUMMARY ===")
print("1. Data Quality: No missing values in both train and test sets")
print("2. Target Balance: Check if balanced or imbalanced")
print("3. Feature Types: Mix of numerical and categorical features")
print("4. Data Distribution: Compare train vs test distributions")
print("5. Correlations: Identify features strongly correlated with target")
print("6. Categorical Patterns: Look for categories with high/low payback rates")


# Save processed data for modeling
train.to_csv('train_processed.csv', index=False)
test.to_csv('test_processed.csv', index=False)
print("\nProcessed data saved for modeling!")




