import numpy as np
import pandas as pd 
import os

TRAIN_DATA = "/kaggle/input/playground-series-s5e1/train.csv"
TEST_DATA = "/kaggle/input/playground-series-s5e1/test.csv"

train_df = pd.read_csv(TRAIN_DATA, index_col='id')
test_df = pd.read_csv(TEST_DATA, index_col='id')
train_df.head()


test_df.head()


test_df.info()


train_df.info()


for key, val in dict(train_df.isnull().sum()).items():
    if val > 0:
        print("PERCENTAGE NULL VALUES FOR ", key)
        print(val/train_df.shape[0] * 100, " %")


train_df.dropna(inplace=True)
train_df.isnull().sum()


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
train_df.head()


train_df.info()


cat_cols = train_df.select_dtypes(exclude=[np.number, 'datetime64']).columns
num_cols = train_df.select_dtypes(include=np.number).columns
date_col = train_df.select_dtypes(include=['datetime64']).columns
print(cat_cols, num_cols, date_col)


for col in cat_cols:
    print("VALUE COUNTS FOR ", col)
    print(train_df[col].value_counts())


for col in cat_cols:
    print("VALUE COUNTS FOR ", col)
    print(test_df[col].value_counts())


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in cat_cols:
    train_df["en_"+col] = encoder.fit_transform(train_df[col])
    test_df['en_'+col] = encoder.fit_transform(test_df[col])
train_df.head()


final_cols = [col for col in train_df.columns if col not in cat_cols]
train_df1  = train_df[final_cols]
final_cols.remove('num_sold')
test_df1 = test_df[final_cols]
train_df1.head()


train_df['date'] = pd.to_datetime(train_df['date'])
train_df.sort_values(by='date')
train_df.head()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['num_sold'], axis='columns'), 
                                                    train_df['num_sold'], test_size=0.3, shuffle=False)


pip install statsmodels sktime


from statsmodels.tsa.arima.model import ARIMA


import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to evaluate model performance
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

# Dictionary to store results
results = {}

# ARIMA Model
arima_model = ARIMA(y_train, order=(5,1,0))
arima_fitted = arima_model.fit()
arima_pred = arima_fitted.forecast(steps=len(y_test))
results["ARIMA"] = evaluate(y_test, arima_pred, "ARIMA")


plt.figure(figsize=(12,6))
plt.plot(y_test.index[:20], y_test[:20], label="Actual", color="green", alpha=0.4)
plt.plot(y_test.index[:20], arima_pred[:20], label="ARIMA", linestyle="dashed")
plt.legend()
plt.title("Time Series Forecasting Comparison")
plt.show()

print("\nModel Performance Summary:")
for model, scores in results.items():
    print(f"{model}: MAE={scores[0]:.4f}, RMSE={scores[1]:.4f}")










