import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

sys.path.append('kaggle/input/russian-car-plates-prices-prediction/')
from supplemental_english import REGION_CODES, GOVERNMENT_CODES



df_raw = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
print(df_raw.info())
df_raw.head(10)


region_codes_to_region = {}
for key, val in REGION_CODES.items():
    if type(val) == list:
        for v in val:
            region_codes_to_region[v] = key
    else:
        region_codes_to_region[val] = key
        
def get_region(plate):
    region_code = plate[6:]
    if region_codes_to_region[region_code]:
        return region_codes_to_region[region_code]
    else:
        return np.nan 

def get_government(plate):
    num_range = int(plate[1:4])
    letters = plate[0] + plate[4:6]
    regionCodes = plate[6:]
    
    for key, value in GOVERNMENT_CODES.items():
        key_letter, (start, end), key_region = key
        if letters == key_letter and regionCodes == key_region and int(start) <= num_range <= int(end):
            return value
    return (np.nan, np.nan, np.nan, np.nan)



def preprocess(df_raw):
    df = df_raw.copy()
    df["region"] = df["plate"].apply(get_region)
    df[['text', 'forbidenToBuy', 'advantageOnRoad', 'significance']] = df['plate'].apply(get_government).apply(pd.Series)
    
    df.drop("text", axis=1, inplace=True)
    df.drop("id", axis=1, inplace=True)
    df.fillna(0, inplace=True) # Most of the newly added columns have NaN values, we will replace it with 0, since they are mostly binary and categorical(significance)
    
    df["date"] = df["date"].astype('datetime64[ns]')
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df.drop("date", axis=1, inplace=True)
    
    df["same_number_first"] = df["plate"].apply(lambda x: x[1] == x[2] == x[3])
    df["same_number_second"] = df["plate"].apply(lambda x: x[6] == x[7] == x[8] if len(x) == 9 else x[6] == x[7])
    df["matching_numbers"] = df["plate"].apply(lambda x: x[1:4] == x[6:])
    df["palindrom_numbers_first"] = df["plate"].apply(lambda x: x[1:4] == x[1:4][::-1])
    df["palindrom_numbers_second"] = df["plate"].apply(lambda x: x[6:] == x[6:][::-1])
    df["palidrom_all_numbers"] = df["plate"].apply(lambda x: x[1:4] + x[6:] == str(x[1:4] + x[6:])[::-1])
    df["cool_number"] = df["plate"].apply(lambda x: x[1:4] in ["001", "007", "100", "200", "300", "400", "500", "600", "700", "800", "900"])
    
    df["one_unique_letter"] = df["plate"].apply(lambda x: len(set(x[0] + x[4:6])) == 1)
    
    df["plate"] = df["plate"].astype('category')
    df["region"] = df["region"].astype('category')
    df["forbidenToBuy"] = df["forbidenToBuy"].astype('bool')
    df["advantageOnRoad"] = df["advantageOnRoad"].astype('bool')
    df["significance"] = df["significance"].astype('int')
    df["year"] = df["year"].astype('int')
    df["month"] = df["month"].astype('int')
    df["day"] = df["day"].astype('int')
    df["day_of_week"] = df["day_of_week"].astype('int')
    df["same_number_first"] = df["same_number_first"].astype('bool')
    df["same_number_second"] = df["same_number_second"].astype('bool')
    df["matching_numbers"] = df["matching_numbers"].astype('bool')
    df["palindrom_numbers_first"] = df["palindrom_numbers_first"].astype('bool')
    df["palindrom_numbers_second"] = df["palindrom_numbers_second"].astype('bool')
    df["palidrom_all_numbers"] = df["palidrom_all_numbers"].astype('bool')
    df["one_unique_letter"] = df["one_unique_letter"].astype('bool')
    
    return df



df = preprocess(df_raw)
display(df)
display(df.info())


plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='significance', y='price', estimator='mean')
plt.title('Average Price by Significance')
plt.xlabel('Significance')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


columns = [
    "forbidenToBuy", "advantageOnRoad", "same_number_first", "same_number_second",
    "matching_numbers", "palindrom_numbers_first", "palindrom_numbers_second", "palidrom_all_numbers", "one_unique_letter"
]

data = []
for col in columns:
    true_count = df[col].sum()
    false_count = (~df[col]).sum()  
    data.append({"Feature": col, "Value": "True", "Count": true_count})
    data.append({"Feature": col, "Value": "False", "Count": false_count})

dftest = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=dftest, x='Feature', y='Count', hue='Value')
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
plt.xticks(rotation=45)
plt.title("True/False Value Counts per Feature")
plt.tight_layout()
plt.show()


df.loc[:, ~df.columns.isin(["plate", "region"])].corr()["price"].sort_values(ascending=False)


X = df.copy()
X.drop("plate", axis=1, inplace=True)

X_train = X[~((X["year"] == 2024) & ((X["month"] == 11) | (X["month"] == 12) ))].copy()
y_train_log = np.log1p(X_train["price"])
X_train.drop("price", axis=1, inplace=True)

X_val = X[(X["year"] == 2024) & ((X["month"] == 11) | ( (X["month"] == 12) & (X["day"].isin(range(5)))))].copy()
y_val_log = np.log1p(X_val["price"])
X_val.drop("price", axis=1, inplace=True)

X_test = X[(X["year"] == 2024) & (X["month"] == 12) & ~(X["day"].isin(range(5)))].copy()
y_test_log = np.log1p(X_test["price"])
X_test.drop("price", axis=1, inplace=True)


print(f"Size train - X: {X_train.shape}, y: {y_train_log.shape}")
print(f"Size val - X: {X_val.shape}, y: {y_val_log.shape}")
print(f"Size test - X: {X_test.shape}, y: {y_test_log.shape}")


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid


param_grid = {
    "eta" : [0.005 ,0.05, 0.1],
    "gamma" : [0, 0.5, 1],
    "max_depth" : [3, 6, 8],
    "subsample" : [0.7, 0.5,  0.3],
    "n_estimators" : [100, 1000, 10000]
}


param_comb = ParameterGrid(param_grid)
best_score = float('inf')
best_param = None
for par in param_comb:
    clf = XGBRegressor(eval_metric="rmse", enable_categorical= True, **par)
    clf.fit(X_train, y_train_log, verbose=10)
    
    rmse_score = np.sqrt(mean_squared_error(np.exp(y_val_log), np.exp(clf.predict(X_val))))
    if rmse_score < best_score:
        best_score = rmse_score
        best_param = par
    
    print(f"RMSE: {rmse_score} for param: {par}")




best_param


clf = XGBRegressor(eval_metric="rmse", enable_categorical= True, **best_param)
clf.fit(X_train, y_train_log, verbose=10)
pred_val = clf.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(np.exp(y_val_log), np.exp(pred_val)))

pred_test = clf.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(np.exp(y_test_log), np.exp(pred_test)))

print(f"Validation RMSE: {rmse_val}")
print(f"Test RMSE: {rmse_test}")


xgb.plot_importance(clf)


def calculate_smape(actual, predicted):
    return np.mean( np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual))/2))*100
    

print(f"SMAPE for Validation: {calculate_smape(np.exp(y_val_log), np.exp(pred_val)):.2f}")
print(f"SMAPE for Test: {calculate_smape(np.exp(y_test_log), np.exp(pred_test)):.2f}")


df_test_raw = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
df_test = preprocess(df_test_raw)

EXCLUDE_COLS = ["id", "price", "plate"]
INCLUDE_COLS = [c for c in df_test.columns if c not in EXCLUDE_COLS]

y_log = np.log1p(X["price"])
X.drop("price", axis=1, inplace=True)

clf = XGBRegressor(eval_metric="rmse", enable_categorical= True, **best_param)
clf.fit(X, y_log, verbose=10)

predicted = np.exp(clf.predict(df_test[INCLUDE_COLS]))

submission = pd.DataFrame({
    'id': df_test_raw['id'],
    'price': predicted
})
submission.to_csv('submission.csv', index=False)

