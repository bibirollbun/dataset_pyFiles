#import the packages
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


df_train = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
df_test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')


df_train.head(5)


df_test.head(5)


df_train = df_train[['month','sector','amount_new_house_transactions']]
df_train.head()


#cek null values
print(df_train.info())


#check the range of sector
print(df_train['sector'].unique())

df_train["sector_num"] =  df_train["sector"].str.extract(r"(\d+)").astype(int)
print("The minimum sector:", df_train["sector_num"].min())
print("The maximum sector:", df_train["sector_num"].max())


all_sectors = [f"sector {i}" for i in range(1, 97)]

all_combinations = pd.MultiIndex.from_product(
    [df_train["month"].unique(), all_sectors],
    names=["month", "sector"]
).to_frame(index=False)

df_full = pd.merge(
    all_combinations,
    df_train,
    on=["month", "sector"],
    how="left"
)

df_full["amount_new_house_transactions"] = df_full["amount_new_house_transactions"].fillna(0)

print(df_full)


df_train = df_full[['month','sector','amount_new_house_transactions']]
df_train.head()


df_train['month'] = pd.to_datetime(df_train['month'])
df_train.info()


df_train['year'] = df_train['month'].dt.year
df_train['month_num'] = df_train['month'].dt.month
df_train['quarter'] = df_train['month'].dt.quarter


#Encode fitur (month and sector)
le_month = LabelEncoder()
le_sector = LabelEncoder()

df_train['month_enc'] = le_month.fit_transform(df_train['month'])
df_train['sector_enc'] = le_sector.fit_transform(df_train['sector'])


df_train


df_train = df_train.fillna(0)
df_train.info()


#Check the distribution of target variable
df_train['amount_new_house_transactions'].value_counts()


#the column target
col = df_train['amount_new_house_transactions']

#IQR MethodQ1 = df[col].quantile(0.25)
Q1 = df_train['amount_new_house_transactions'].quantile(0.25)
Q3 = df_train['amount_new_house_transactions'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Data outlier
outliers = df_train[(col < lower_bound) | (col > upper_bound)]

print(f"Jumlah outlier: {len(outliers)}")
print(outliers.head())

# --- 2. Visualisasi Boxplot ---
plt.figure(figsize=(8,4))
sns.boxplot(x=df_train['amount_new_house_transactions'])
plt.title(f"(cek outlier)")
plt.show()   # <--- wajib ada ini




#clean the data
df_clean = df_train[(col >= lower_bound) & (col <= upper_bound)]
print("Jumlah data setelah hapus outlier:", len(df_clean))


# --- 2. Visualisasi Boxplot ---
plt.figure(figsize=(8,4))
sns.boxplot(x=df_clean['amount_new_house_transactions'])
plt.title(f"Boxplot {col} (cek outlier)")
plt.show()

#check the dataframe
print(df_clean.info())


df_clean.head()


#Defining target & feature
X = df_clean.drop(columns=["amount_new_house_transactions","month","sector","quarter"])
y = df_clean["amount_new_house_transactions"]

# Split train & valid
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 1000,
}


model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mean_y = y_pred.mean()
rmse_percent = (rmse / mean_y) * 100


print(f"RMSE%: {rmse_percent:.2f}%")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")


# visualizing in a plot
import matplotlib.pyplot as plt

x_ax = range(len(y_test))
plt.figure(figsize=(12, 6))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Real Estate Demand Prediction")
plt.xlabel('X')
plt.ylabel('amount_new_house_transactions')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# plotting feature importance
lgb.plot_importance(model, height=.5)


#copy data test
df_testing = df_test.copy()


print(df_testing.info())
print(df_testing.head())


df_testing['id'] = df_testing['id'].astype(str)


#split the column id
df_testing[['year', 'month', 'sector']] = df_testing['id'].str.split(r'[_ ]', expand=True) [[0,1,3]]

df_testing['month_dt'] = pd.to_datetime(df_testing['month'], format="%b")
df_testing['month_num'] = df_testing['month_dt'].dt.month


df_testing


#Encode fitur (month and sector)
df_testing['month_enc'] = le_month.fit_transform(df_testing['month'])
df_testing['sector_enc'] = le_sector.fit_transform(df_testing['sector'])


df_testing.head()


df_testing['year'] = df_testing['year'].astype(int)


X_testing = df_testing[['year', 'month_num', 'month_enc', 'sector_enc']]
X_testing.info()


# prediksi
y_pred_test = model.predict(X_testing)

print(y_pred[:10])  # lihat 10 hasil pertama


df_test['new_house_transaction_amount'] = y_pred_test


df_test


submission = df_test
submission.to_csv('/kaggle/working/submission.csv', index=False)

