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


df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_submission_sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


df_train.shape


df_train.head()


df_cleaned = df_train.select_dtypes('number')

df_cleaned = df_cleaned.drop(['loan_paid_back','id'],axis=1)

matrix = df_cleaned.corr()

print("Correlation matrix is : ")
print(matrix)


# One hot encode categorical attributes
categorical_cols = ['gender','marital_status','education_level','employment_status','loan_purpose']
df_encoded_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
df_encoded_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Nominal variable grade subgrade.. codify this
grade_map = {grade: i for i, grade in enumerate(sorted(df_encoded_train['grade_subgrade'].unique()))}
df_encoded_train['grade_subgrade'] = df_encoded_train['grade_subgrade'].map(grade_map)
df_encoded_test['grade_subgrade'] = df_encoded_test['grade_subgrade'].map(grade_map)

X = df_encoded_train.drop(['id', 'loan_paid_back'], axis=1)
y = df_encoded_train['loan_paid_back']
X_test = df_encoded_test.drop(['id'], axis=1)

# Scaling -> this seems to be the key that took me from .64 accuracy to .91
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Cross-validation, stratified to help with class imbalance
model = LogisticRegression()
skf = StratifiedKFold(n_splits=5)
scores = []
all_predictions = []

for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    
    fold_preds = model.predict_proba(X_test)[:,1]
    all_predictions.append(fold_preds)
    
    # Calculate validation score
    val_preds = model.predict_proba(X_valid)[:,1]
    score = roc_auc_score(y_valid, val_preds)
    scores.append(score)
    print(f"Fold {i+1} AUC-ROC: {score:.4f}")

positive_class_probabilities = np.mean(all_predictions, axis=0)
print(f"Average CV Score: {np.mean(scores):.4f}")


# Create submission

ids = df_encoded_test['id']
probs_df = pd.DataFrame({
    'id': ids,
    'prob_paid_back': positive_class_probabilities
})
probs_df.to_csv('/kaggle/working/predicted_probabilities.csv', index=False)

