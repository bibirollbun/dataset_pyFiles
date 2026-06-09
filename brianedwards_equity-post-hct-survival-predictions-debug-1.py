import pandas as pd
from sklearn.linear_model import LinearRegression


train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
test_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv')


train = train_csv.copy(deep=True)
test = test_csv.copy(deep=True)

train = train.set_index('ID')
test = test.set_index('ID')

train = train[['age_at_hct', 'comorbidity_score', 'karnofsky_score', 'efs']].dropna()
test = test[['age_at_hct', 'comorbidity_score', 'karnofsky_score']]


m = LinearRegression()

X = train[['age_at_hct', 'comorbidity_score', 'karnofsky_score']]
y = train['efs']

m.fit(X, y)

submission_csv = pd.DataFrame()
submission_csv.index = train.index
submission_csv['prediction'] = m.predict(train.drop(columns='efs'))
submission_csv.to_csv('submission.csv')


with open('submission.csv') as f:
    submission_csv_contents = f.read().splitlines()
    print(f'{len(submission_csv_contents)} lines ({train.shape[0] + 1} expected)')
    print('\nFirst 10 lines\n--------------')
    print('\n'.join(submission_csv_contents[:10]))




