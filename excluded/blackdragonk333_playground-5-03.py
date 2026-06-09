import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.sample(10)


df.drop(columns=['id'], inplace=True)


df.shape


df.describe()


df.info()


df.corr()


sns.heatmap(df.corr(), vmin=-1, vmax=1)
plt.plot()


df.drop(columns=['maxtemp', 'mintemp'], inplace=True)


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.plot()


df.drop(columns=['temparature'], inplace=True)


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
plt.plot()


df.isna().sum()


for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


dew_square = df['dewpoint'] ** 2


sns.histplot(dew_square, kde=True)


pt = PowerTransformer(method='yeo-johnson')


dew_yeo_johnson = pt.fit_transform(df['dewpoint'].values.reshape(-1, 1))


sns.histplot(dew_yeo_johnson, kde=True)


humidity_cube = df['humidity'] ** 3


sns.histplot(humidity_cube, kde=True)


humidity_yeo_johnson = pt.fit_transform(df['humidity'].values.reshape(-1, 1))


sns.histplot(humidity_yeo_johnson, kde = True)


for col in df.columns:
    plt.figure()
    sns.boxplot(df[col])
    plt.title(f'Box plot of {col}')
    plt.plot()


X = df.drop(columns=['rainfall'])
y = df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y = y.astype(object)


X_train.head()


preprocessor = ColumnTransformer(
    [
        ('transformation', PowerTransformer(method='yeo-johnson'), [1, 2, 3, 4, 5, 6, 7]),
        ('scaling', StandardScaler(), [1, 2, 3, 4, 5,6, 7])
    ]
)


log_reg_pipe = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', LogisticRegression())
    ]
)


log_reg_pipe.fit(X_train, y_train)


y_pred_log_reg = log_reg_pipe.predict(X_test)


accuracy_score(y_test, y_pred_log_reg)


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


X_train.columns


test.columns


ids = test['id']


test = test.drop(columns=['id', 'maxtemp', 'temparature', 'mintemp'])


test.isna().sum()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


test_pred_log_reg = log_reg_pipe.predict(test)


result_log_reg = pd.DataFrame({'id': ids, 'rainfall': test_pred_log_reg})
result_log_reg


result_log_reg.to_csv('submission.csv', index=False)




