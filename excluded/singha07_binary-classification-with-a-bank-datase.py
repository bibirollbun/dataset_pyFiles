# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# === PART 1: IMPORTS & DATA LOADING ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')



train.sample(5)


train.info()


test.info()


cats =  train.select_dtypes(include=['object']).columns



# Optional: set a modern style
sns.set_style("whitegrid")
sns.set_context("talk")  # Bigger font and clearer visuals

for col in cats:
    plt.figure(figsize=(8, 6))
    
    # Histogram with KDE overlay
    sns.histplot(data=train, x=col, color='skyblue', kde=True, edgecolor='black', linewidth=0.5)
    
    plt.title(f'Distribution of {col}', fontsize=16, fontweight='bold')
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def feature_engineering(df):
    df['default_encoded'] = df['default'].map({'no': 0, 'yes': 1})
    df['housing_encoded'] = df['housing'].map({'no': 0, 'yes': 1})
    df['loan_encoded'] = df['loan'].map({'no': 0, 'yes': 1})

    df['balance_to_age'] = df['balance'] / (df['age'] + 1)
    df['balance_to_duration'] = df['balance'] / (df['duration'] + 1)
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1)
    df['age_times_balance'] = df['age'] * df['balance']

    df['has_been_contacted'] = (df['pdays'] != -1).astype(int)
    df['contact_success'] = ((df['pdays'] != -1) & (df['poutcome'] == 'success')).astype(int)

    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['is_quarter_end'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)

    df['financial_stability'] = (df['balance'] / (df['age'] + 1)) * (1 - df['default_encoded'])
    df['credit_engagement'] = (df['loan_encoded'] + df['housing_encoded']) * df['previous']

    return df

train_df = feature_engineering(train)
test_df = feature_engineering(test)


train_df.info()


train_df.drop(columns = ['id'], inplace = True)

id = test_df['id'].copy()

test_df.drop(columns = ['id'], inplace = True)


from sklearn.model_selection import train_test_split

X = train_df.drop(columns = ['y'])
y = train_df['y']



cat_cols = X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))



num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])


kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Fold {fold+1} ROC-AUC: {auc:.4f}")
    auc_scores.append(auc)

print(f"Average ROC-AUC: {np.mean(auc_scores):.4f}")


test_probs = model.predict_proba(test_df)[:, 1]
submission = pd.DataFrame({'id': id, 'y': nptest_probs})
submission.to_csv('submission.csv', index=False)


import os
print("Submission file contents:")
print(os.listdir('/kaggle/working'))


