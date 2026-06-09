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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

train_df.head()


print(train_df.isnull().any(axis=0))


import seaborn as sns

train_sold = train_df.loc[train_df['num_sold'].notna(),["num_sold"]]
sns.displot(train_sold, x="num_sold", kde=True, binwidth=5)


sns.histplot(np.log(train_sold), x="num_sold", kde=True, binwidth=0.8)



train_df2 = train_df.loc[train_df['num_sold'].isna() == False, : ]

train_df2['num_sold_log'] = np.log(train_df2['num_sold'])

train_df2.describe()


train_df2['date'] = pd.to_datetime(train_df2['date'], format="%Y-%m-%d")


train_df2['year'] = train_df2['date'].dt.year

train_df2['month'] = train_df2['date'].dt.month

dateSoldGroup = train_df2.groupby(['year', 'month'])['num_sold'].sum()

#dateSoldGroup
dateSoldGroup = dateSoldGroup.reset_index()
dateSoldGroup.columns = ['year', 'month', 'total_num_sold']

print(dateSoldGroup.head())


import matplotlib.pyplot as plt

pd.option_context('mode.use_inf_as_na', False)

g = sns.FacetGrid(data=dateSoldGroup, col="year", col_wrap=3, height=4, sharex=False, sharey=False)
g.map(sns.lineplot, "month", "total_num_sold", marker="o")

g.set_titles("Year: {col_name}")
g.set_axis_labels("Month", "Total Sales")
g.set(xticks=range(1, 13), xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Monthly Sales by Year")

plt.show()


numSoldByCountryNTime = train_df2.groupby(['country', 'year', 'month'])['num_sold'].sum().reset_index()

numSoldByCountryNTime['x-axis'] = numSoldByCountryNTime['year'].map(str) + '-' + numSoldByCountryNTime['month'].map(str)

numSoldByCountryNTime.head()


numSoldByCountryNTime = numSoldByCountryNTime.sort_values(by='x-axis')
g = sns.FacetGrid(data=numSoldByCountryNTime, col="country", col_wrap=3, height=4, sharex=True, sharey=True)

g.map(sns.lineplot, "x-axis", "num_sold", marker="o", color="blue")

g.set_titles("Country: {col_name}")
g.set_axis_labels("Time (Year-Month)", "Total sales")


for ax in g.axes.flat:
    xticks = ax.get_xticks()
    if len(xticks) > 0: 
        vtick = xticks[::3]
        ax.set_xticks(vtick)
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Total Sales by Country Over Time", fontsize=16)

plt.show()


numSoldByStorenTime = train_df2.groupby(['store', 'year', 'month'])['num_sold'].sum().reset_index()

numSoldByStorenTime['x-axis'] = numSoldByStorenTime['year'].map(str) + '/' + numSoldByStorenTime['month'].map(str)

numSoldByStorenTime.head()


g = sns.FacetGrid(data=numSoldByStorenTime, col="store", col_wrap=3, height=4, sharex=True, sharey=True)
g.map(sns.lineplot, "x-axis", "num_sold", marker="o", color="blue")

for ax in g.axes.flat:
    ax.set_xticks(ax.get_xticks()[::3])  
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Monthly Sales by Store", fontsize=16)
g.set_axis_labels("Time (Year/Month)", "Total Sales")

plt.show()


numSoldByProductnTime = train_df2.groupby(['product', 'year', 'month'])['num_sold'].sum().reset_index()

numSoldByProductnTime['x-axis'] = numSoldByProductnTime['year'].map(str) + '/' + numSoldByProductnTime['month'].map(str)

numSoldByProductnTime.head()


g = sns.FacetGrid(data=numSoldByProductnTime, col="product", col_wrap=3, height=4, sharex=True, sharey=True)
g.map(sns.lineplot, "x-axis", "num_sold", marker="o", color="blue")

for ax in g.axes.flat:
    ax.set_xticks(ax.get_xticks()[::3])  
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

plt.subplots_adjust(top=0.9)
g.fig.suptitle("Monthly Sales by Store", fontsize=16)
g.set_axis_labels("Time (Year/Month)", "Total Sales")

plt.show()


train_df2['sin_month'] = np.sin(2*np.pi*train_df2['month']/12)
train_df2['cos_month'] = np.cos(2*np.pi*train_df2['month']/12)

train_df2["sin_year"] = np.sin(2 * np.pi * (train_df2["year"] - train_df2["year"].min()) / (train_df2["year"].max()-train_df2["year"].min()))
train_df2["cos_year"] = np.cos(2 * np.pi * (train_df2["year"] - train_df2["year"].min()) / (train_df2["year"].max()-train_df2["year"].min()))

train_df2.head()


from statsmodels.formula.api import ols

fit = ols('num_sold_log~C(country)+C(store)+C(product)+year+month', data = train_df2).fit()

fit.summary()


import statsmodels.formula.api as smf

fit1 = smf.mixedlm('num_sold_log~C(country)+C(store)+C(product)+year+month', data = train_df2, groups="country").fit()

fit1.summary()


mixedmodel1 = smf.mixedlm("num_sold_log ~ C(country):year+C(store)+C(product)+C(country):month",
                    train_df2,
                    groups= "country", re_formula="year+month").fit()

mixedmodel1.summary()


mixedmodel2 = smf.mixedlm("num_sold_log ~ year+C(store)+C(product)+month",
                    train_df2,
                    groups= "country", re_formula="year+month").fit()

mixedmodel2.summary()


formula = """
num_sold_log ~  C(product) + C(store):sin_month + C(store):cos_month + C(store):sin_year + C(store):cos_year
"""

## define linear mixed effect models
mixedModel1 = smf.mixedlm(
    formula=formula,
    data=train_df2,
    groups="country",            
    re_formula="sin_month + cos_month + sin_year + cos_year"  # sin, cos within countries
)

result = mixedModel1.fit()

## summary of model
print(result.summary())


formula = """
num_sold_log ~  C(product) + C(store):sin_month + C(store):cos_month + sin_year+cos_year
"""

mixedModel2 = smf.mixedlm(
    formula=formula,
    data=train_df2,
    groups="country",              
    re_formula="sin_month + cos_month" 
)

result = mixedModel2.fit()

print(result.summary())


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Function to handle date processing
def handling_date(df): 
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df['sin_month'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df["sin_year"] = np.sin(2 * np.pi * (df['date'].dt.year - df['date'].dt.year.min()) / 
                            (df['date'].dt.year.max() - df['date'].dt.year.min()))
    df["cos_year"] = np.cos(2 * np.pi * (df['date'].dt.year - df['date'].dt.year.min()) / 
                            (df['date'].dt.year.max() - df['date'].dt.year.min()))
    df["sin_day"] = np.sin(2 * np.pi *(df['date'].dt.dayofyear / 365))

    df["cos_day"] = np.sin(2 * np.pi *(df['date'].dt.dayofyear / 365))

    print("\n[Step 1: After handling_date]")
    print(df.head())  # Display first 5 rows after this step
    
    return df.drop(columns=["date"])

# Function to log-transform the target variable
def transform_log(y): 
    transformed = np.log(y)
    print("\n[Step 2: After transform_log]")
    print(transformed[:5])  # Display first 5 transformed values
    return transformed

# Define numerical and categorical features
num_features = ['sin_month', 'cos_month']
cat_features = ['country', 'store']
product_feature = ['product']

# One-hot encoding for categorical features
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# One-hot encoding for product (sparse matrix)
product_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

# Define ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),          # Retain sin/cos_month
        ("cat", categorical_transformer, cat_features),  # One-hot encode for country/store
        ("product", product_transformer, product_feature),  # One-hot encode for product
    ],
    remainder="drop",
)


# Complete pipeline
pipeline = Pipeline(
    steps=[
        ("process_date", FunctionTransformer(handling_date)),  # Date handling
        ("preprocessor", preprocessor),  # Feature engineering
        ("interaction", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),  # Interaction terms
        ("regressor", LinearRegression()),  # Linear regression
    ]
)

# Split X and y
X = train_df2.drop(columns=["num_sold", "id"])
y = train_df2["num_sold"]

# Training pipeline
pipeline.fit(X, transform_log(y))

# Predicting
predictions = pipeline.predict(X)

# Output predictions
print("\n[Final Step: Predictions (log-transformed num_sold)]")
print(predictions[:5])  # Display first 5 predictions



print((predictions > 0).sum())

print(predictions.shape)


loss = np.mean((predictions - transform_log(y))**2)

print(np.exp(loss))


## testing sets loading

test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

print(test_df.head())

print(test_df.isna().any())

test_x = test_df.drop(columns=['id'])

ytest_pred = pipeline.predict(test_x)


serial_id = test_df.loc[:, 'id']

y_result = np.exp(ytest_pred)

print(y_result[:5])

result = pd.DataFrame({'num_sold': y_result})

print("submit answers into csv format")
submission = pd.concat([serial_id, result], axis=1)

submission.to_csv('submission_v2.csv', index=False)


train_copy = handling_date(train_df2)

train_copy.drop(columns=['id', 'year', 'month', 'num_sold_log'], inplace=True)
train_copy.head()



# Step 1: One-hot encode 類別欄位
country_code = pd.get_dummies(train_copy['country'], dtype='int')
store_code = pd.get_dummies(train_copy['store'], dtype='int')
product_code = pd.get_dummies(train_copy['product'], dtype='int')

# Step 2: 提取數值特徵
numerical_features = train_copy.loc[:, ["cos_month", "sin_month"]]

# Step 3: 進行交互作用 (逐列相乘)
# store_code 與 numerical_features 交互
interaction_store = store_code.values[:, :, np.newaxis] * numerical_features.values[:, np.newaxis, :]

# Step 4: 將交互作用結果轉為 DataFrame
interaction_store_df = pd.DataFrame(
    interaction_store.reshape(len(train_copy), -1),
    columns=[f"{store}_{feature}" for store in store_code.columns for feature in numerical_features.columns]
)

# Step 5: 顯示結果
print("Store One-hot Encoding:")
print(store_code)
print("\nNumerical Features:")
print(numerical_features)
print("\nInteraction Shape:", interaction_store.shape)
print("\nInteraction DataFrame:")
print(interaction_store_df)

print(interaction_store_df.isnull().any(axis=0))

print(interaction_store_df.shape[0], train_copy.shape[0])


train_copy.isnull().any(axis=0)


# 確保索引一致
country_code.index = train_copy.index
product_code.index = train_copy.index
interaction_store_df.index = train_copy.index

# 合併數據
train_new = pd.concat([train_copy, country_code, product_code, interaction_store_df], axis=1)

# 查看結果
print(train_new.head())

print(train_new.isnull().any(axis=0))


train_copy.drop(columns=['country', 'store', 'product'], inplace=True)

train_new = pd.concat([train_copy, country_code,product_code, interaction_store_df], axis=1)

train_new.head()


X_2 = train_new.drop(columns=['num_sold'])
print(X_2.head())
Y_2 = np.log(train_new['num_sold'])
print(Y_2.head())
model2 = LinearRegression()

model2.fit(X_2, Y_2)

Y_pred_2 = model2.predict(X_2)

print(Y_pred_2[:5])

print(np.exp(np.mean((Y_2 - Y_pred_2)**2)))


test_copy = handling_date(test_df)

country_encode = pd.get_dummies(test_copy['country'], dtype='int')
store_encode = pd.get_dummies(test_copy['store'], dtype='int')
product_encode = pd.get_dummies(test_copy['product'], dtype='int')

month_feature = test_copy.loc[:, ['cos_month', 'sin_month']]

store_month = store_encode.values[:, :, np.newaxis] * month_feature.values[:, np.newaxis, :]

store_month_df = pd.DataFrame(
    store_month.reshape(len(test_copy), -1),
    columns=[f"{store}_{feature}" for store in store_encode.columns for feature in month_feature.columns]
)

print(store_month_df.head())

test_new = pd.concat([test_copy, country_encode, product_encode, store_month_df], axis=1)

print(test_new.head())

print(test_new.isnull().any(axis=0))


X_test = test_new.drop(columns=['id','country', 'product','store'])

print(X_test.columns)
Y_test_pred = model2.predict(X_test)
print(X_test.shape[0])
print(Y_test_pred[(Y_test_pred<0) == False].shape)

print("submit answers into csv format")

serial = test_new['id']

Ytest_result = np.exp(Y_test_pred)

print(Ytest_result[:5])

result2 = pd.DataFrame({'num_sold': Ytest_result})


submissionV3 = pd.concat([serial, result2], axis=1)

submissionV3.to_csv('submission_v3.csv', index=False)

