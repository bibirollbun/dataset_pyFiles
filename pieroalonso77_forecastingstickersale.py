import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv', index_col = 'id') 
train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col = 'id')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col = 'id')
display(sample.head(3))
display(train_dataset.head(3))
display(test_dataset.head(3))


train_dataset.info()


train_dataset.dropna(inplace = True)
print('New shape, after droping rows with null values', train_dataset.shape)


y = train_dataset['num_sold']
X = train_dataset.drop(columns = 'num_sold')


sns.boxplot(data = list(y))
plt.show()


IQR = np.percentile(y,75) - np.percentile(y,25)
lim_inf, lim_sup = np.percentile(y,25) - 1.5 * IQR, np.percentile(y,75) + 1.5 * IQR
print(lim_inf, lim_sup)


index_exclude = list(y[y>lim_sup].index)
print(len(index_exclude))


train_filtered = train_dataset[~train_dataset.index.isin(index_exclude)]
print('Filtered X shape',train_filtered.shape)
display(train_filtered.head())


sns.histplot(train_filtered['num_sold'])
plt.show()


sns.histplot(train_filtered['num_sold'][train_filtered['num_sold'] <501])
plt.show()


df_num_sold = train_filtered['num_sold'][train_filtered['num_sold'] <501].value_counts().reset_index()
print(df_num_sold.shape)
display(df_num_sold.head())


sns.boxplot(df_num_sold['count'])
plt.show()


iqr2 = np.percentile(df_num_sold['count'],75) - np.percentile(df_num_sold['count'],25)
sup2,inf2  = np.percentile(df_num_sold['count'],75) + 1.5*iqr2, np.percentile(df_num_sold['count'],25)-1.5*iqr2
inf2 = int(np.round(inf2))
sup2 = int(np.round(sup2))
print(inf2, sup2)


num_sold_large = list(df_num_sold['num_sold'][df_num_sold['count']>sup2])
print(len(num_sold_large))
print(num_sold_large[0:5])


df_sample = pd.DataFrame()
for x in num_sold_large:
    df_aux = train_filtered[train_filtered['num_sold']==x]
    nrows = sup2
    df_aux2 = df_aux.sample(n=nrows, random_state=403)
    df_sample = pd.concat([df_sample, df_aux2])


index_to_drop = train_filtered[train_filtered['num_sold'].isin(num_sold_large)].index
print('Shape pre-drop: ', train_filtered.shape)
df_dropped = train_filtered.drop(index = index_to_drop)
print('Shape post-drop: ', df_dropped.shape)
display(df_dropped.head())


train_filtered = pd.concat([df_dropped,df_sample])
print('New shape', train_filtered.shape)
display(train_filtered.head())
display(train_filtered.tail())


sns.histplot(train_filtered['num_sold'][train_filtered['num_sold'] < 501])
plt.show()


sns.histplot(train_filtered['num_sold'])
plt.show()


train_filtered['num_sold'].value_counts()


train_filtered['country'].value_counts()


store_df = train_filtered['store'].value_counts().reset_index()
sns.barplot(store_df, x = 'store', y = 'count')
plt.title('Count of sticker by store')
plt.show()


store_df = train_filtered['product'].value_counts().reset_index()
sns.barplot(store_df, x = 'product', y = 'count')
plt.title('Count of sticker by product')
plt.xticks(rotation=45)
plt.show()


sns.boxplot(data = train_filtered,y='num_sold', x = 'country')
plt.show()


sns.boxplot(data = train_filtered,y='num_sold', x = 'product')
plt.xticks(rotation=45)
plt.show()


sns.boxplot(data = train_filtered,y='num_sold', x = 'store')
plt.show()


def extract_date(df_param):
    df = df_param.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop(columns = 'date', inplace = True)
    return df


train_filtered = extract_date(train_filtered)
train_filtered.head()


df_month =train_filtered.groupby(['month'])['num_sold'].sum().reset_index()
sns.lineplot(data=df_month, x='month', y='num_sold', marker='o', color='blue')
plt.show()


df_year = train_filtered.groupby(['year'])['num_sold'].sum().reset_index()
sns.lineplot(data=df_year, x='year', y='num_sold', marker='o', color='blue')
plt.show()


X_filtered = train_filtered.drop(columns = ['num_sold'])
X_filtered.head()


y_filtered = train_filtered['num_sold']
y_filtered.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=304)
print('X train data', X_train.shape)
print('X test data', X_test.shape)
print('y train data', y_train.shape)
print('y test data', y_test.shape)

display(X_train.head())


numerical_features = ['year','month']
categorical_features = [x for x in X_train.columns if x not in numerical_features]
print(categorical_features)


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])

from sklearn.linear_model import LinearRegression

model_linear = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
    ])


print('X train shape', X_train.shape)
print('y train shape', y_train.shape)


model_linear.fit(X_train, y_train)


y_pred = model_linear.predict(X_test)


df_linear_result = X_test.copy()
df_linear_result['y_test'] = list(y_test)
df_linear_result['y_pred'] = list(y_pred)
df_linear_result.head()


sns.scatterplot(data = df_linear_result, x='y_test', y='y_pred', hue='country', s=10, alpha=0.7)
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.show()


print('Score: ', model_linear.score(X_test, y_test))
print('Correlation: ', np.corrcoef(y_test, y_pred)[0, 1] )


sns.scatterplot(data = df_linear_result, x='y_test', y='y_pred', hue='product', s=10, alpha=0.7)
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.show()


preprocessor_k = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])

model_linear_k = Pipeline([
    ('preprocessor', preprocessor_k),
    ('regressor', LinearRegression())
    ])


X_train_k = X_train[X_train['country']=='Kenya']
y_train_k = y_train[X_train_k.index]

X_test_k = X_test[X_test['country']=='Kenya']
y_test_k = y_test[X_test_k.index]


X_train_k['country'].value_counts()


model_linear_k.fit(X_train_k, y_train_k)


y_pred_k = model_linear_k.predict(X_test_k)


print('Score: ', model_linear_k.score(X_test_k, y_test_k))
print('Correlation: ', np.corrcoef(y_test_k, y_pred_k)[0, 1] )


sns.scatterplot(x = y_test_k, y = y_pred_k)
plt.show()


X_train_nk = X_train[X_train['country']!='Kenya']
y_train_nk = y_train[X_train_nk.index]
X_test_nk = X_test[X_test['country']!='Kenya']
y_test_nk = y_test[X_test_nk.index]


preprocessor_nk = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])

model_linear_nk = Pipeline([
    ('preprocessor', preprocessor_nk),
    ('regressor', LinearRegression())
    ])


model_linear_nk.fit(X_train_nk,y_train_nk)


y_pred_nk = model_linear_nk.predict(X_test_nk)


print('Score: ', model_linear_nk.score(X_test_nk, y_test_nk))
print('Correlation: ', np.corrcoef(y_test_nk, y_pred_nk)[0, 1] )


sns.scatterplot(x = y_test_nk, y = y_pred_nk)
plt.ylabel('Predicted value')
plt.xlabel('Actual value')
plt.show()


sns.scatterplot(data = X_test_nk, x = y_test_nk, y = np.power(y_pred_nk,1.0), hue = 'product')
plt.ylabel('Predicted value')
plt.xlabel('Actual value')
plt.show()


residuals = y_pred_nk - y_test_nk
sns.scatterplot(x= y_test_nk, y= residuals)
plt.show()


sub_k = test_dataset[test_dataset['country']=='Kenya']
sub_nk = test_dataset[test_dataset['country']!='Kenya']
sub_k = extract_date(sub_k)
sub_nk = extract_date(sub_nk)
display(sub_k.head(3))
display(sub_nk.head(3))


sub_k_pred = np.round(model_linear_k.predict(sub_k),0)


sub_nk_pred = np.round(model_linear_nk.predict(sub_nk),0)


sub_k_ = sub_k.copy()
sub_k_['pred'] = list(sub_k_pred)
sub_k_.head()


sub_nk_ = sub_nk.copy()
sub_nk_['pred'] = list(sub_nk_pred)
sub_nk_.head()


sub = pd.concat([sub_nk_, sub_k_], ignore_index=False)
sub.head()


output = sub['pred'].sort_values()
output.head()


output.to_csv('submission.csv')




