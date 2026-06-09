import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.head()


df.info()


df.describe()


# numerical columns
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

plt.figure(figsize=(18, 12))

for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# Example: Distribution of 'age' in bins
print(pd.cut(df['age'], bins=[18,25,35,45,55,65,75,85,95]).value_counts())

# Percentage
print(pd.cut(df['age'], bins=[18,25,35,45,55,65,75,85,95]).value_counts(normalize=True) * 100)



num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for col in num_cols:
    print(f"\n--- Distribution of {col} ---")
    
    # for smaller unique values → print exact counts
    if df[col].nunique() < 30:
        print(df[col].value_counts().sort_index())
        print("Percentage:\n", df[col].value_counts(normalize=True).sort_index() * 100)
    else:
        # otherwise bin into 10 groups
        print(pd.cut(df[col], bins=10).value_counts().sort_index())
        print("Percentage:\n", pd.cut(df[col], bins=10).value_counts(normalize=True).sort_index() * 100)



cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    print(f"\n--- Distribution of {col} ---")
    print(df[col].value_counts())
    print("Percentage:\n", df[col].value_counts(normalize=True) * 100)



print(df["y"].value_counts())


numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

correlations = df[numeric_cols + ['y']].corr()['y'].sort_values(ascending=False)
print(correlations)


import scipy.stats as stats

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

cat_corr = {}
for col in categorical_cols:
    cat_corr[col] = cramers_v(df[col], df['y'])

cat_corr = pd.Series(cat_corr).sort_values(ascending=False)
print(cat_corr)



all_corr = pd.concat([correlations.drop('y'), cat_corr])
all_corr.sort_values(ascending=False, inplace=True)
print(all_corr)


# Feature Engineering
df_ = df.copy()

# 1. Age-related features
df['age_bin'] = pd.cut(df['age'], bins=[17, 25, 35, 45, 60, 100],
                       labels=['18-25', '26-35', '36-45', '46-60', '60+'])
df['is_senior'] = (df['age'] >= 60).astype(int)

# 2. Balance-related features
df['balance_bin'] = pd.qcut(df['balance'], q=5, duplicates='drop')
df['positive_balance'] = (df['balance'] > 0).astype(int)

# 3. Contact/campaign features
df['is_first_contact'] = (df['previous'] == 0).astype(int)
df['multiple_contacts'] = (df['campaign'] > 1).astype(int)
df['recently_contacted'] = (df['pdays'] != -1).astype(int)

# 4. Interaction features
df['balance_per_contact'] = df['balance'] / (df['campaign'] + 1)
df['duration_per_call'] = df['duration'] / (df['campaign'] + 1)

# 5. Temporal features
df['is_month_end'] = (df['day'] > 25).astype(int)
df['is_month_start'] = (df['day'] < 5).astype(int)
df['month_num'] = pd.to_datetime(df['month'], format='%b').dt.month  # jan=1, dec=12
df['quarter'] = pd.to_datetime(df['month'], format='%b').dt.quarter

# 6. Risk / loan features
df['any_loan'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)

# 7. Previous outcome encoding
df['prev_success'] = (df['poutcome'] == 'success').astype(int)
df['prev_failure'] = (df['poutcome'] == 'failure').astype(int)

# 8. Combined campaign history
df['contact_history'] = df['previous'] + df['campaign']

# quick check
df.head()



from sklearn.preprocessing import LabelEncoder

df_corr = df.copy()

# encode categorical columns
categorical_cols = df_corr.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    df_corr[col] = le.fit_transform(df_corr[col])

# correlations
correlations = df_corr.corr(numeric_only=True)['y'].sort_values(ascending=False)

print(correlations)


df.info()


del df
del df_
del df_corr


from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report


SEED=229


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_ids = test_df["id"]


def feature_engineering(df):
    # Age bins
    df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,60,100], labels=["young","adult","middle","senior"])
    df["is_senior"] = (df["age"] > 60).astype(int)
    
    # Balance bins & positivity
    df["balance_bin"] = pd.cut(df["balance"], bins=[-10000,0,1000,5000,20000,100000], labels=["debt","low","medium","high","very_high"])
    df["positive_balance"] = (df["balance"] > 0).astype(int)
    
    # Contact-related features
    df["is_first_contact"] = (df["previous"] == 0).astype(int)
    df["multiple_contacts"] = (df["campaign"] > 1).astype(int)
    df["recently_contacted"] = (df["pdays"] != -1).astype(int)
    
    # Ratios
    df["balance_per_contact"] = df["balance"] / (df["campaign"] + 1)
    df["duration_per_call"] = df["duration"] / (df["campaign"] + 1)
    
    # Month-based features
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df["month_num"] = df["month"].map(month_map)
    df["quarter"] = ((df["month_num"]-1)//3 + 1)
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 25).astype(int)
    
    # Loan-related
    df["any_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)
    
    # Previous outcome
    df["prev_success"] = (df["poutcome"] == "success").astype(int)
    df["prev_failure"] = (df["poutcome"] == "failure").astype(int)
    
    # Contact history
    df["contact_history"] = (df["pdays"] > 0).astype(int)
    
    return df

train_df = feature_engineering(train_df)
test_df  = feature_engineering(test_df)


y = train_df["y"]
X = train_df.drop(columns=["id", "y"])
X_test = test_df.drop(columns=["id"])


# combine for encoding
full_data = pd.concat([X, X_test], axis=0)

# encode categorical columns
cat_cols = full_data.select_dtypes(include=["object", "category"]).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col].astype(str))
    le_dict[col] = le


X = full_data.iloc[:len(X), :]
X_test = full_data.iloc[len(X):, :]


model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    use_label_encoder=False,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)


model.fit(X, y, verbose=100)


y_test_pred = model.predict_proba(X_test)[:,1]


submission = pd.DataFrame({
    "id": test_ids,
    "y": y_test_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)


submission.head()




