import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
import warnings


features = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip")
train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
stores = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
test1 = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")


features.head()


train.info()


features.info()


stores.head()


train_agg = train.groupby(["Store", "Date", "IsHoliday"])["Weekly_Sales"].sum().reset_index()

df = pd.merge(
    train_agg,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)


df = pd.merge(
    df,
    stores,
    on="Store",
    how="left"
)


df.describe()


df['Weekly_Sales'] = df['Weekly_Sales'] / 1e6


(df.isnull().sum()/len(df))*100


df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace = True)


df[df.duplicated(keep=False)]



encoder_type = LabelEncoder()
df['Type'] = encoder_type.fit_transform(df['Type'])
df['IsHoliday'] = encoder_type.fit_transform(df['IsHoliday'])


df = df.drop(columns=['Date'])


target = df.pop('Weekly_Sales')
target.head()


df.head()


df.info()


numerical_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment','Size', 'Type']

melted = df[numerical_cols].melt()

sns.set_theme()

g = sns.FacetGrid(melted, col='variable', col_wrap=3, sharex=False, sharey=False)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')
g.tight_layout()
plt.show()


key_numerical_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
plt.figure(figsize=(15, 10))

for i, col in enumerate(key_numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')

plt.tight_layout()
plt.show()


columns_to_clean = ['Temperature','Unemployment']
z_scores = df[columns_to_clean].apply(zscore)

outliers_mask = (z_scores.abs() > 3).any(axis=1)

data_cleaned = df[~outliers_mask]
target_cleaned = target[~outliers_mask]

print(f'Data before cleaning: {df.shape[0]} rows')
print(f'Data after cleaning: {data_cleaned.shape[0]} rows')


subset = pd.concat([data_cleaned, target_cleaned], axis=1)
corr_matrix = subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Кореляційна матриця')
plt.tight_layout()
plt.show()


high_corr_threshold = 0.6  #High correlation definition
high_corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
seen = set() 

for (col1, col2), corr_value in high_corr_pairs.items():
    if col1 != col2 and corr_value > high_corr_threshold:
        pair = tuple(sorted([col1, col2]))
        if pair not in seen:
            print(f'Висока кореляція між {col1} та {col2}: {corr_value:.2f}')
            seen.add(pair)


data_cleaned = data_cleaned.drop(columns=['Type'])


X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned,
    target_cleaned,
    test_size=0.2,
    random_state=42)


scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled.describe()


model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)
y_pred.head()


r_sq_train = model.score(X_train_scaled, y_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)

print(f'Train Metrics:\nR2: {r_sq_train:.2f} | MAE: {mae_train:.2f} | MAPE: {mape_train:.2f}')


r_sq = model.score(X_test_scaled, y_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Test Metrics:\nR2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')


poly = PolynomialFeatures(3).set_output(transform='pandas')

Xtr = poly.fit_transform(X_train_scaled)
Xts = poly.transform(X_test_scaled)

model_upd = LinearRegression().fit(Xtr, y_train)
y_pred_upd = model_upd.predict(Xts)
y_pred_upd = pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)
y_pred_train_upd = model_upd.predict(Xtr)
y_pred_train_upd = pd.Series(y_pred_train_upd, index=Xtr.index).clip(ymin, ymax)


r_sq_upd_train = model_upd.score(Xtr, y_train)
mae_upd_train = mean_absolute_error(y_train, y_pred_train_upd)
mape_upd_train = mean_absolute_percentage_error(y_train, y_pred_train_upd)

print(f'Train Metrics:\nR2: {r_sq_upd_train:.2f} | MAE: {mae_upd_train:.2f} | MAPE: {mape_upd_train:.2f}')


r_sq_upd = model_upd.score(Xtr, y_train)
mae_upd = mean_absolute_error(y_test, y_pred_upd)
mape_upd = mean_absolute_percentage_error(y_test, y_pred_upd)

print(f'Test Metrics:\nR2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')


test_1 = pd.merge(
    test1,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)
test = pd.merge(
    test_1,
    stores,
    on=["Store"],
    how="left"
)


test.info()


test.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace = True)


test.drop("Date", axis = 1, inplace = True)


test.drop("Type", axis = 1, inplace = True)


test.drop("Dept", axis = 1, inplace = True)


test['CPI'] = test['CPI'].fillna(test['CPI'].mean())
test['Unemployment'] = test['Unemployment'].fillna(test['Unemployment'].mean())


encoder_type = LabelEncoder()
test['IsHoliday'] = encoder_type.fit_transform(test['IsHoliday'])


test.info()


X_test_scaled_final = scaler.transform(test)

poly = PolynomialFeatures(3).set_output(transform='pandas')
X_test_poly_final = poly.fit(X_train_scaled).transform(X_test_scaled_final)
y_test_pred_final = model_upd.predict(X_test_poly_final)
y_test_pred_final = pd.Series(y_test_pred_final).clip(ymin, ymax)

test_full = pd.concat([test, test1[['Dept', 'Date']].reset_index(drop=True)], axis=1)
test_full['Id'] = test_full['Store'].astype(str) + '_' + test_full['Dept'].astype(str) + '_' + test_full['Date'].astype(str)

submiss = pd.DataFrame({
    'Id': test_full['Id'],
    'Weekly_Sales': y_test_pred_final
})

submiss.to_csv('submiss.csv', index=False)



a = pd.read_csv('/kaggle/working/submiss.csv')


a.info()

