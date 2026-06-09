#[Problem 2] Learning and verification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df1=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

x=df1[['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
y=df1['TARGET'].values
print('describe',x.describe())
print('infor',x.info)
x = pd.get_dummies(x, columns=['NAME_CONTRACT_TYPE', 'CODE_GENDER'])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train = x_train.fillna(0)
x_test = x_test.fillna(0)


model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print('roc_auc_score',roc_auc_score(y_test,y_pred))


#[Problem 3] Estimation on test data
import pandas as pd
from sklearn.preprocessing import StandardScaler

df2 = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

cols_usadas = ['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']
x_test = df2[cols_usadas].copy()

x_test = x_test.fillna(x_test.median(numeric_only=True))

x_test = pd.get_dummies(x_test)

x_train = pd.get_dummies(df1[cols_usadas])
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(x_test)

pred_prob = model.predict(x_test_scaled)
submission = pd.DataFrame({
    'SK_ID_CURR': x_test['SK_ID_CURR'],
    'TARGET':pred_prob
})

submission.to_csv('submission.csv', index=False)
print("File submission.csv created!")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

train['CREDIT_INCOME_RATIO'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']

def run_pattern(features):
    X = train[features].fillna(0)  # Fill missing with 0
    y = train['TARGET']

    # Encode categorical features as numbers (simple)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    return auc

# Define 5 feature sets (patterns)
patterns = [
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH'],
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'CREDIT_INCOME_RATIO'],
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'NAME_CONTRACT_TYPE'],
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'NAME_CONTRACT_TYPE', 'CODE_GENDER'],
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CREDIT_INCOME_RATIO']
]

for i, feats in enumerate(patterns, 1):
    auc = run_pattern(feats)
    print(f'Pattern {i} - Features: {feats} - Validation AUC: {auc:.4f}')





