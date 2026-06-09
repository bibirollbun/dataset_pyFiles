import csv
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter('ignore')


train_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/train.csv')
train_csv = train_csv.set_index('ID')
test_csv = pd.read_csv('../input/equity-post-HCT-survival-predictions/test.csv')
test_csv = test_csv.set_index('ID')
sample_submission = pd.read_csv('../input/equity-post-HCT-survival-predictions/sample_submission.csv')


train = train_csv.drop(columns=['efs', 'efs_time'])
test = test_csv.copy(deep=True)

for col in train.columns:
    if train[col].dtype not in ['object', 'category']:
        nbins = min(20, int(round(train_csv[col].max())) - int(round(train_csv[col].min())))
        _, bins = pd.cut(train[col], nbins, retbins=True)
        bins[0], bins[-1] = -np.inf, np.inf
        labels = [f'{start}-{end}' for start, end in zip(bins[:-1], bins[1:])]
        train[col] = pd.cut(train[col], bins, labels=labels)
        test[col] = pd.cut(test[col], bins, labels=labels)
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    test[col] = test[col].cat.set_categories(train[col].cat.categories)

def clean(df):
    for col in df.columns:
        df[col] = df[col].cat.add_categories('unknown')
        df[col] = df[col].fillna('unknown')
    encoder = OneHotEncoder(handle_unknown='error', sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df)
    return pd.DataFrame(encoded_data, index=df.index, columns=encoder.get_feature_names_out())

train = clean(train)
test = clean(test)

for col in set(train.columns) - set(test.columns):
    test[col] = 0.0

train = train.sort_index(axis=1)
test = test.sort_index(axis=1)


m = Ridge()
X = train.copy(deep=True)
y = train_csv['efs']
m.fit(X, y)
submission = pd.DataFrame()
submission.index = test.index
submission['prediction'] = m.predict(test)
submission.to_csv('submission.csv')




