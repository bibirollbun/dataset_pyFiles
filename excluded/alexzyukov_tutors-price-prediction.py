import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


!pip install -q opendatasets


import opendatasets as od

ds_url = 'https://www.kaggle.com/competitions/tutors-lessons-prices-prediction'
od.download(ds_url)


import os

os.listdir('./tutors-lessons-prices-prediction')


train = pd.read_excel('/content/tutors-lessons-prices-prediction/train.xlsx')
test = pd.read_excel('/content/tutors-lessons-prices-prediction/test.xlsx')
sample_submit = pd.read_csv('/content/tutors-lessons-prices-prediction/sample_submit.csv')


train.head(2)


test.head(2)


sample_submit.head()


train.isna().sum()


train.describe()


plt.figure(figsize=(10, 4))
plt.hist(train['mean_price'], bins=50)
plt.show()


import seaborn as sns

sns.set(rc={'figure.figsize':(5, 5)})
sns.boxplot(train['mean_price'])
plt.show()


train['предмет'].value_counts()


train = pd.get_dummies(train, columns=['предмет'])
train.head(2)


train['status'] = train['status'].apply(
    lambda x: x.split(',') if isinstance(x, str) else np.nan
)


train['status']


train['categories'].head()


import ast
from sklearn.preprocessing import MultiLabelBinarizer

def expand_multilabel_column(df, column_name):
    df[column_name] = df[column_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x if isinstance(x, list) else np.nan)

    binarizer = MultiLabelBinarizer()
    categories_enc = binarizer.fit_transform(df[column_name])
    categories = pd.DataFrame(categories_enc, columns=binarizer.classes_)

    df = pd.concat([df, categories], axis=1)
    return df, binarizer


train, cat_binarizer = expand_multilabel_column(train, 'categories')
train['categories'][0]


train, tags_binarizer = expand_multilabel_column(train, 'tutor_head_tags')
train['tutor_head_tags'][0]


status_mode = str(train['status'].mode()[0])
train['status'] = train['status'].fillna(status_mode)
train, stat_binarizer = expand_multilabel_column(train, 'status')
train['status'][0]


train.head(1)


train['Desc_Education_1'].head()


train['experience'].unique()


train['experience'] = train['experience'].apply(
    lambda x: float(x.split()[0]) if isinstance(x, str) else np.nan
)
train['experience'] = train['experience'].astype('float32')


train['experience'].describe()


numeric_columns = train.select_dtypes(include='number').columns
train = train[numeric_columns]


train.drop(columns=['Unnamed: 0'], inplace=True)


train.head()


train.isna().sum()


exp_mean = train['experience'].mean()
train['experience'] = train['experience'].fillna(exp_mean)


train['tutor_rating'].describe()


plt.hist(train['tutor_rating'], bins=20)
plt.show()


train['tutor_rating'] = train['tutor_rating'].fillna(0)


train.isna().sum()


sns.set(rc={'figure.figsize':(20, 20)})
sns.heatmap(train.corr(), annot=True)
plt.show()


np.random.seed(42)


X = train.drop(columns=['mean_price'])
Y = train['mean_price']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(f'Train size: {X_train.shape}, {y_train.shape}')
print(f'Test size: {X_test.shape}, {y_test.shape}')


!pip install -q catboost


from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, eval_metric='RMSE', random_seed=42, verbose=500)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

y_pred = model.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 score:', r2_score(y_test, y_pred))


from sklearn.model_selection import cross_val_score

model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, eval_metric='RMSE', random_seed=42, verbose=500)
scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')


model_full = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6, eval_metric='RMSE', random_seed=42, verbose=500)
model_full.fit(X, Y)


test_proc = test.copy()
test_proc = pd.get_dummies(test_proc, columns=['предмет'])

test_proc['categories'] = test_proc['categories'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
cats_df = pd.DataFrame(cat_binarizer.transform(test_proc['categories']), columns=cat_binarizer.classes_)
test_proc = pd.concat([test_proc, cats_df], axis=1)

test_proc['tutor_head_tags'] = test_proc['tutor_head_tags'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
tags_df = pd.DataFrame(tags_binarizer.transform(test_proc['tutor_head_tags']), columns=tags_binarizer.classes_)
test_proc = pd.concat([test_proc, tags_df], axis=1)

test_proc['status'] = test_proc['status'].fillna(status_mode)
test_proc['status'] = test_proc['status'].apply(
    lambda x: x.split(',') if isinstance(x, str) else x
)

stat_df = pd.DataFrame(stat_binarizer.transform(test_proc['status']), columns=stat_binarizer.classes_)
test_proc = pd.concat([test_proc, stat_df], axis=1)

test_proc['experience'] = test_proc['experience'].apply(
    lambda x: float(x.split()[0]) if isinstance(x, str) else np.nan
)
test_proc['experience'] = test_proc['experience'].astype('float32')
test_proc['experience'] = test_proc['experience'].fillna(exp_mean)

test_proc['tutor_rating'] = test_proc['tutor_rating'].fillna(0)


test_proc = test_proc.reindex(columns=X.columns, fill_value=0)
test_proc = test_proc.astype(float)

y_test_pred = model_full.predict(test_proc)

submission = pd.DataFrame({'mean_price': y_test_pred})
submission = submission.reset_index()
submission.to_csv('submission.csv', index=False)
submission.head(10)


sample_submit.head(10)


!kaggle competitions submit tutors-lessons-prices-prediction -f submission.csv -m "My submission"

