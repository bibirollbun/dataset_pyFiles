import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("DataFrame Shapes:")
print(train_df.shape, "|", test_df.shape)


print("# of empty rows in train_df:")
display(train_df.isna().sum())
print("-"*30)
print("# of empty rows in test_df:")
display(test_df.isna().sum())


train_df.head()


# basic statistics of numeric features
train_df.describe(include='number')


# basic statistics of object features
train_df.describe(include='object')


print("dtypes of features:")
train_df.dtypes


fig, axs = plt.subplots(2, 3, figsize=(15, 8))
sns.countplot(x='gender', data=train_df, ax=axs[0,0])
axs[0,0].set_title('Distribution of Gender')
axs[0,0].set_xlabel('Gender')
axs[0,0].set_ylabel('Count')

sns.countplot(x='marital_status', data=train_df, ax=axs[0,1])
axs[0,1].set_title('Distribution of Marital Status')
axs[0,1].set_xlabel('Marital Status')
axs[0,1].set_ylabel('Count')

sns.countplot(x='education_level', data=train_df, ax=axs[0,2])
axs[0,2].set_title('Distribution of Education Level')
axs[0,2].set_xlabel('Education Level')
axs[0,2].set_ylabel('Count')

sns.countplot(x='employment_status', data=train_df, ax=axs[1,0])
axs[1,0].set_title('Distribution of Employment Status')
axs[1,0].set_xlabel('Employment Status')
axs[1,0].set_ylabel('Count')

order = train_df['loan_purpose'].value_counts().index
sns.countplot(x='loan_purpose', data=train_df, order=order, ax=axs[1,1])
axs[1,1].tick_params(axis='x', rotation=45)
axs[1,1].set_title('Distribution of Loan Purpose')
axs[1,1].set_xlabel('Loan Purpose')
axs[1,1].set_ylabel('Count')

order = train_df['grade_subgrade'].value_counts().index
sns.countplot(x='grade_subgrade', data=train_df, order=order, ax=axs[1,2])
axs[1,2].tick_params(axis='x', rotation=90)
axs[1,2].set_title('Distribution of Grade-Subgrade')
axs[1,2].set_xlabel('Grade-Subgrade')
axs[1,2].set_ylabel('Count')

fig.suptitle('Distribution of Categorical Features', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.show()



sns.set_palette("muted")

fig, axs = plt.subplots(2,3, figsize=(15, 8))
sns.histplot(x='annual_income', data=train_df, ax=axs[0,0], kde=True)
axs[0,0].set_title('Distribution of Annual Income')
axs[0,0].set_xlabel('Annual Income')

sns.histplot(x='debt_to_income_ratio', data=train_df, ax=axs[0,1], kde=True)
axs[0,1].set_title('Distribution of Debt to Income Ratio')
axs[0,1].set_xlabel('Debt to Income Ratio')

sns.histplot(x='credit_score', data=train_df, ax=axs[0,2], kde=True)
axs[0,2].set_title('Distribution of Credit Score')
axs[0,2].set_xlabel('Credit Score')

sns.histplot(x='loan_amount', data=train_df, ax=axs[1,0], kde=True)
axs[1,0].set_title('Distribution of Loan Amount')
axs[1,0].set_xlabel('Loan Amount')

sns.histplot(x='interest_rate', data=train_df, ax=axs[1,1], kde=True)
axs[1,1].set_title('Distribution of Interest Rate')
axs[1,1].set_xlabel('Interest Rate')

axs[1,2].remove()

fig.suptitle('Distribution of Numeric Features', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.show()


# to-do: set labels, set titles, set suptitle
fig, axs = plt.subplots(2,3, figsize=(15, 8))
sns.boxplot(x='annual_income', data=train_df, ax=axs[0,0])
axs[0,0].set_title('Boxplot of Annual Income')
axs[0,0].set_xlabel('Annual Income')

sns.boxplot(x='debt_to_income_ratio', data=train_df, ax=axs[0,1])
axs[0,1].set_title('Boxplot of Debt to Income Ratio')
axs[0,1].set_xlabel('Debt to Income Ratio')

sns.boxplot(x='credit_score', data=train_df, ax=axs[0,2])
axs[0,2].set_title('Distribution of Credit Score')
axs[0,2].set_xlabel('Credit Score')

sns.boxplot(x='loan_amount', data=train_df, ax=axs[1,0])
axs[1,0].set_title('Distribution of Loan Amount')
axs[1,0].set_xlabel('Loan Amount')

sns.boxplot(x='interest_rate', data=train_df, ax=axs[1,1])
axs[1,1].set_title('Distribution of Interest Rate')
axs[1,1].set_xlabel('Interest Rate')

axs[1,2].remove()

fig.suptitle('Boxplots of Numeric Features', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.show()



num_features = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
col_skew_values = [train_df[col].skew() for col in num_features]
col_kurtosis_values = [train_df[col].kurtosis() for col in num_features]

print(pd.DataFrame({"skewness": col_skew_values, "kurtosis": col_kurtosis_values}, index=num_features))


palette = sns.color_palette('muted')
palette[0], palette[1] = palette[1], palette[0] 

ax = sns.countplot(x='loan_paid_back', data=train_df, palette=palette)

total = len(train_df)

for p in ax.patches:
    count = p.get_height()
    percentage = 100 * count / total
    ax.annotate(f'{percentage:.1f}%', 
                (p.get_x() + p.get_width() / 2, count),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Count')
ax.set_xlabel('Loan Paid Back')
ax.set_title('Label Distribution', fontweight='bold')
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(12,8))

sns.boxplot(y='annual_income', x='loan_paid_back', data=train_df, ax=axs[0,0], palette=palette)
axs[0,0].set_title("Annual Income vs Loan Repayment Status")
axs[0,0].set_xlabel("Loan Paid Back")
axs[0,0].set_ylabel("Annual Income")

sns.boxplot(y='debt_to_income_ratio', x='loan_paid_back', data=train_df, ax=axs[0,1], palette=palette)
axs[0,1].set_title("Debt-to-Income Ratio vs Loan Repayment Status")
axs[0,1].set_xlabel("Loan Paid Back")
axs[0,1].set_ylabel("Debt-to-Income Ratio")

sns.boxplot(y='credit_score', x='loan_paid_back', data=train_df, ax=axs[1,0], palette=palette)
axs[1,0].set_title("Credit Score vs Loan Repayment Status")
axs[1,0].set_xlabel("Loan Paid Back")
axs[1,0].set_ylabel("Credit Score")

sns.boxplot(y='loan_amount', x='loan_paid_back', data=train_df, ax=axs[1,1], palette=palette)
axs[1,1].set_title("Loan Amount vs Loan Repayment Status")
axs[1,1].set_xlabel("Loan Paid Back")
axs[1,1].set_ylabel("Loan Amount")

plt.suptitle("Distribution of Numeric Features by Loan Repayment Status", fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(15, 8))

pd.crosstab(train_df['gender'], train_df['loan_paid_back'], normalize="index").plot(kind="bar", stacked=True, ax=axs[0,0], color=palette)
axs[0,0].set_title("Loan Paid Back vs Gender")
axs[0,0].set_xlabel("Gender")
axs[0,0].set_ylabel("Proportion")
axs[0,0].legend(title='loan_paid_back', bbox_to_anchor=(1.05, 1), loc='upper left')

pd.crosstab(train_df['marital_status'], train_df['loan_paid_back'], normalize="index").plot(kind="bar", stacked=True, ax=axs[0,1], color=palette)
axs[0,1].set_title("Loan Paid Back vs Marital Status")
axs[0,1].set_xlabel("Marital Status")
axs[0,1].set_ylabel("Proportion")
axs[0,1].legend(title='loan_paid_back', bbox_to_anchor=(1.05, 1), loc='upper left')

pd.crosstab(train_df['education_level'], train_df['loan_paid_back'], normalize="index").plot(kind="bar", stacked=True, ax=axs[0,2], color=palette)
axs[0,2].set_title("Loan Paid Back vs Education Level")
axs[0,2].set_xlabel("Education Level")
axs[0,2].set_ylabel("Proportion")
axs[0,2].legend(title='loan_paid_back', bbox_to_anchor=(1.05, 1), loc='upper left')

pd.crosstab(train_df['employment_status'], train_df['loan_paid_back'], normalize="index").plot(kind="bar", stacked=True, ax=axs[1,0], color=palette)
axs[1,0].set_title("Loan Paid Back vs Employment Status")
axs[1,0].set_xlabel("Employment Status")
axs[1,0].set_ylabel("Proportion")
axs[1,0].legend(title='loan_paid_back', bbox_to_anchor=(1.05, 1), loc='upper left')

pd.crosstab(train_df['loan_purpose'], train_df['loan_paid_back'], normalize="index").plot(kind="bar", stacked=True, ax=axs[1,1], color=palette)
axs[1,1].set_title("Loan Paid Back vs Loan Purpose")
axs[1,1].set_xlabel("Loan Purpose")
axs[1,1].set_ylabel("Proportion")
axs[1,1].legend(title='loan_paid_back', bbox_to_anchor=(1.05, 1), loc='upper left')

axs[1,2].remove()

plt.suptitle("Categorical Features vs Loan Paid Back", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

sns.set_palette('husl')


grade_vs_repayment = pd.crosstab(train_df['grade_subgrade'], train_df['loan_paid_back'], normalize='index')

print(f"grades_subgrades least likely to pay the loan back:")
print(grade_vs_repayment[0].sort_values(ascending=False)[:10])

print("-"*30)

print(f"grades_subgrades most likely to pay the loan back:")
print(grade_vs_repayment[1].sort_values(ascending=False)[:10])


def prepare_features(df):
    df_ = df.copy()

    # One Hot Encoding
    df_['gender'] = df_['gender'].apply(lambda x: int(x=='Male'))
    df_ = pd.get_dummies(df_, columns=['marital_status', 'education_level', 'employment_status', 'loan_purpose'], drop_first=True)

    # Label Encoding
    grade_map = {grade: i for i, grade in enumerate(sorted(df_['grade_subgrade'].unique()))}
    # grade_map_reverse = {i:grade for grade, i in grade_map.items()}

    df_['grade_subgrade'] = df_['grade_subgrade'].map(grade_map)

    return df_ 


def prepare_data(df, test=False):
    prepared_df = prepare_features(df)

    if test:
        X = prepared_df.drop(columns=['id',], axis=1)
        return X
    else:
        X = prepared_df.drop(columns=['id', 'loan_paid_back'], axis=1)
        y = prepared_df['loan_paid_back']
        return X, y


def cv_score(model, X, y, splits=5):
    skf = StratifiedKFold(n_splits=splits)
    scores = []
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
    
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_valid)[:,1]
    
        roc_auc = roc_auc_score(y_valid, preds)
        print(f"[Fold: {i+1}/{splits}] AUC-ROC Score: {roc_auc:.4f}")
        scores.append(roc_auc)
    
    return np.mean(scores)


X, y = prepare_data(train_df)
X_test = prepare_data(test_df, test=True)

num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


model = LogisticRegression()
log_reg_score = cv_score(model, X, y)
print("Average Score across all folds:", round(log_reg_score,4))


model=LogisticRegression()
model.fit(X, y)
preds = model.predict_proba(X_test)[:,1]


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
sample_submission['loan_paid_back'] = preds
sample_submission.to_csv("submission.csv", index=False)

