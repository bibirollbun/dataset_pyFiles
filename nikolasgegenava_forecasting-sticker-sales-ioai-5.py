import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_train.head()


df_train.dropna(inplace=True)
country_sales = df_train.groupby('country')['num_sold'].sum().reset_index()


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world = world.merge(country_sales, left_on='name', right_on='country', how='left')

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
world.boundary.plot(ax=ax)
world.plot(column='num_sold', ax=ax, legend=True,
           legend_kwds={'label': "Total Sales by Country",
                        'orientation': "horizontal"},
           cmap='turbo')

ax.set_title('Sales by Country', fontsize=13)

plt.show()


df_train['country'].unique()


for col in df_train.columns:
    print(df_train[col].unique())


df_train['country'] = df_train['country'].replace({'Canada': 2, 'Finland': 3, 'Italy': 4, 'Kenya': 5, 'Norway': 6, 'Singapore': 7})
df_train['store'] = df_train['store'].replace({'Discount Stickers': 2, 'Stickers for Less': 3, 'Premium Sticker Mart': 4})
df_train['product'] = df_train['product'].replace({'Holographic Goose': 2, 'Kaggle': 3, 'Kaggle Tiers': 4, 'Kerneler': 5, 'Kerneler Dark Mode': 6})


df_train.head()


df_train.shape


df_train['date'] = df_train['date'].astype(str).replace('-', '', regex=True).astype(int)


df_train.head()


import seaborn as sns

df_train = df_train.astype(int)

corr = df_train.corr()

plt.figure(figsize=(20, 9))
k = 18
cols = corr.nlargest(k, 'num_sold')['num_sold'].index
cm = np.corrcoef(df_train[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap="rocket")
plt.show()


df_train.drop(['date'], axis=1, inplace=True)


corr = df_train.corr()

plt.figure(figsize=(20, 9))
k = 18
cols = corr.nlargest(k, 'num_sold')['num_sold'].index
cm = np.corrcoef(df_train[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap="rocket")
plt.show()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

X = df_train.drop(['num_sold', 'id'], axis=1)
Y = df_train['num_sold']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "mape",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False
)

y_pred = model.predict(dval)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f'MAPE: {mape:.4f}')


df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
d1test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

df_test.dropna(inplace=True)
df_test['country'] = df_test['country'].replace({'Canada': 2, 'Finland': 3, 'Italy': 4, 'Kenya': 5, 'Norway': 6, 'Singapore': 7})
df_test['store'] = df_test['store'].replace({'Discount Stickers': 2, 'Stickers for Less': 3, 'Premium Sticker Mart': 4})
df_test['product'] = df_test['product'].replace({'Holographic Goose': 2, 'Kaggle': 3, 'Kaggle Tiers': 4, 'Kerneler': 5, 'Kerneler Dark Mode': 6})
df_test.drop(['date', 'id'], axis=1, inplace=True)

features = ['country', 'store', 'product']
test = df_test[features]

dtest = xgb.DMatrix(test)
test_preds = model.predict(dtest)

submission = pd.DataFrame({
    'id': d1test['id'], 
    'num_sold': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved!")

