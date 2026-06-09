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


train = pd.read_csv("/kaggle/input/playground-series-s3e1/train.csv")
train


train.info()


testset = pd.read_csv("/kaggle/input/playground-series-s3e1/test.csv")
testset


testset.info()


trainset = train.sample(frac = 0.8)
valset = train.drop(trainset.index)


def prepare_data(data, status="Train", info_dct = None):
    if status == "Train":
        info_dct = {}
        
    #dropping unnecessary columns
    uncs_columns = ["id"]

    if status == "Train":
        data = data.drop(uncs_columns, axis = 1)
        info_dct["uncs_columns"] = uncs_columns
    elif status == "Test":
        uncs_columns = info_dct.get("uncs_columns")
        data = data.drop(uncs_columns, axis = 1)
    if status == "Test" and info_dct.get("drop_MedInc"):
        data = data.drop(columns=["MedInc"])

    #replacing missing values with medians

    if status == "Train": 
        missing_dct = {}
        columns = data.columns
        for i in columns:
            col = data.loc[:,[i]]
            median = col.median()
            missing_dct[i] = median
            data.loc[:, i] = data.loc[:, i].fillna(median)

            info_dct["Missings"] = missing_dct
    elif status == "Test":
        missing_dct = info_dct.get("Missings")
        columns = data.columns
        for i in columns:
            median = missing_dct.get(i)
            data.loc[:, i] = data.loc[:, i].fillna(median)


    #replacing duplicates

    if status == "Train":
        for col in data.select_dtypes(include="number").columns:
            dup_ratio = data[col].value_counts(normalize=True, dropna=False).iloc[0]
            info_dct[f"{col}_dup_ratio"] = dup_ratio
    
            if dup_ratio >= 0.75:
                median = data[col].median()
                data[col] = median
                info_dct[f"replace_{col}"] = True
            else:
                info_dct[f"replace_{col}"] = False

    #scaling

    if status == "Train":
        medinc_mean = data["MedInc"].mean()
        medinc_std = data["MedInc"].std()
        houseAge_mean = data["HouseAge"].mean()
        houseAge_std = data["HouseAge"].std()
        avgrooms_mean = data["AveRooms"].mean()
        avgrooms_std = data["AveRooms"].std()
        avgbdrms_mean = data["AveBedrms"].mean()
        avgbdrms_std = data["AveBedrms"].std()
        population_mean = data["Population"].mean()
        population_std = data["Population"].std()
        avgocups_mean = data["AveOccup"].mean()
        avgocups_std = data["AveOccup"].std()
        lat_mean = data["Latitude"].mean()
        lat_std = data["Latitude"].std()
        long_mean = data["Longitude"].mean()
        long_std = data["Longitude"].std()
        

        info_dct["MedInc_Scale"] = medinc_mean, medinc_std
        info_dct["HouseAge_Scale"] = houseAge_mean, houseAge_std
        info_dct["AveRooms_Scale"] = avgrooms_mean, avgrooms_std
        info_dct["AveBedrms_Scale"] = avgbdrms_mean, avgbdrms_std
        info_dct["Population_Scale"] = population_mean, population_std
        info_dct["AveOccup_Scale"] = avgocups_mean, avgocups_std
        info_dct["Latitude_Scale"] = lat_mean, lat_std
        info_dct["Longitude_Scale"] = long_mean, long_std
        
    
    else:
        medinc_mean, medinc_std = info_dct["MedInc_Scale"]
        houseAge_mean, houseAge_std = info_dct["HouseAge_Scale"]
        avgrooms_mean, avgrooms_std = info_dct["AveRooms_Scale"]
        avgbdrms_mean, avgbdrms_std = info_dct["AveBedrms_Scale"]
        population_mean, population_std = info_dct["Population_Scale"]
        avgocups_mean, avgocups_std = info_dct["AveOccup_Scale"]
        lat_mean, lat_std = info_dct["Latitude_Scale"]
        long_mean, long_std = info_dct["Longitude_Scale"]
        

    scales = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
    
    for col in scales:
        if col in data.columns:
            mean, std = info_dct[f"{col}_Scale"]
            if std != 0:
                data[col] = (data[col] - mean) / std

    return data, info_dct


train.isnull().values.any()


clean_trainset, info_dct = prepare_data(trainset, "Train")

clean_valset, _ = prepare_data(valset, "Test", info_dct)

clean_testset, _ = prepare_data(testset, "Test", info_dct)


for col in clean_valset.select_dtypes(include="number").columns:
    dup_ratio = clean_valset[col].value_counts(normalize=True, dropna=False).iloc[0]
    print(f"{col}: {dup_ratio:.2%} duplicate values")


clean_trainset.describe()


clean_valset.describe()


clean_testset.describe()


target = 'MedHouseVal'

X_train = clean_trainset.drop(columns=[target])
y_train = clean_trainset[target]

X_val = clean_valset.drop(columns=[target])
y_val = clean_valset[target]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lin = LinearRegression()
lin.fit(X_train, y_train)

x_train_pred_lin_def = lin.predict(X_train)
y_val_pred_lin_def = lin.predict(X_val)


mse_lin_def = mean_squared_error(y_val, y_val_pred_lin_def)
mae_lin_def = mean_absolute_error(y_val, y_val_pred_lin_def)
rmse_lin_def = mean_squared_error(y_val, y_val_pred_lin_def, squared=False)
r2_lin_def = r2_score(y_val, y_val_pred_lin_def)

print(mse_lin_def)
print(mae_lin_def)
print(rmse_lin_def)
print(r2_lin_def)


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_val, y_val_pred_lin_def, alpha=0.5, label="LinearRegression")
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}
lin_grid = {
    'fit_intercept': [True, False]
}

lin_search = GridSearchCV(lin, lin_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
lin_search.fit(X_train, y_train)
y_val_pred_lin_tuned = lin_search.predict(X_val)


mse_lin_tuned = mean_squared_error(y_val, y_val_pred_lin_tuned)
mae_lin_tuned = mean_absolute_error(y_val, y_val_pred_lin_tuned)
rmse_lin_tuned = mean_squared_error(y_val, y_val_pred_lin_tuned, squared=False)
r2_lin_tuned = r2_score(y_val, y_val_pred_lin_tuned)

print(mse_lin_tuned)
print(mae_lin_tuned)
print(rmse_lin_tuned)
print(r2_lin_tuned)


plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_pred_lin_tuned, alpha=0.5, label="Tuned LinearRegression")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values (Tuned Model)")
plt.legend()
plt.show()


compare_df = pd.DataFrame({
    "Model": ["LinearRegression (Default)", "LinearRegression (Tuned)"],
    "MSE": [mse_lin_def, mse_lin_tuned],
    "MAE": [mae_lin_def, mae_lin_tuned],
    "RMSE": [rmse_lin_def, rmse_lin_tuned],
    "R2": [r2_lin_def, r2_lin_tuned]
})

print(compare_df)


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

x_train_tree_pred_def = tree.predict(X_train)
y_val_pred_tree_def = tree.predict(X_val)


mse_tree_def = mean_squared_error(y_val, y_val_pred_tree_def)
mae_tree_def = mean_absolute_error(y_val, y_val_pred_tree_def)
rmse_tree_def = mean_squared_error(y_val, y_val_pred_tree_def, squared=False)
r2_tree_def = r2_score(y_val, y_val_pred_tree_def)

print(mse_tree_def)
print(mae_tree_def)
print(rmse_tree_def)
print(r2_tree_def)


plt.figure()
plt.scatter(y_val, y_val_pred_tree_def, alpha=0.5, label="DecisionTree")
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

tree_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_tuned = DecisionTreeRegressor(random_state=42)
tree_search = GridSearchCV(tree_tuned, tree_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
tree_search.fit(X_train, y_train)
y_val_pred_tree_tuned = tree_search.predict(X_val)


mse_tree_tuned = mean_squared_error(y_val, y_val_pred_tree_tuned)
mae_tree_tuned = mean_absolute_error(y_val, y_val_pred_tree_tuned)
rmse_tree_tuned = mean_squared_error(y_val, y_val_pred_tree_tuned, squared=False)
r2_tree_tuned = r2_score(y_val, y_val_pred_tree_tuned)

print(mse_tree_tuned)
print(mae_lin_tuned)
print(rmse_lin_tuned)
print(r2_lin_tuned)


plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_pred_tree_tuned, alpha=0.5, label="Tuned DecisionTree")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values (Tuned Model)")
plt.legend()
plt.show()


compare_df = pd.DataFrame({
    "Model": ["DecisionTree (Default)", "DecisionTree (Tuned)"],
    "MSE": [mse_tree_def, mse_tree_tuned],
    "MAE": [mae_tree_def, mae_tree_tuned],
    "RMSE": [rmse_tree_def, rmse_tree_tuned],
    "R2": [r2_tree_def, r2_tree_tuned]
})

print(compare_df)


from sklearn.linear_model import Ridge

rdg = Ridge()
rdg.fit(X_train, y_train)

x_train_rdg_pred = rdg.predict(X_train)
y_val_pred_rdg_def = rdg.predict(X_val)


mse_rdg_def = mean_squared_error(y_val, y_val_pred_rdg_def)
mae_rdg_def = mean_absolute_error(y_val, y_val_pred_rdg_def)
rmse_rdg_def = mean_squared_error(y_val, y_val_pred_rdg_def, squared=False)
r2_rdg_def = r2_score(y_val, y_val_pred_rdg_def)

print(mse_rdg_def)
print(mae_rdg_def)
print(rmse_rdg_def)
print(r2_rdg_def)


plt.figure()
plt.scatter(y_val, y_val_pred_rdg_def, alpha=0.5, label="Ridge")
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()



param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

rdg_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

rdg_search = GridSearchCV(rdg, rdg_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
rdg_search.fit(X_train, y_train)
y_val_pred_rdg_tuned = rdg_search.predict(X_val)


mse_rdg_tuned = mean_squared_error(y_val, y_val_pred_rdg_tuned)
mae_rdg_tuned = mean_absolute_error(y_val, y_val_pred_rdg_tuned)
rmse_rdg_tuned = mean_squared_error(y_val, y_val_pred_rdg_tuned, squared=False)
r2_rdg_tuned = r2_score(y_val, y_val_pred_rdg_tuned)

print(mse_rdg_tuned)
print(mae_rdg_tuned)
print(rmse_rdg_tuned)
print(r2_rdg_tuned)


plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_pred_rdg_tuned, alpha=0.5, label="Tuned Ridge")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values (Tuned Model)")
plt.legend()
plt.show()


compare_df = pd.DataFrame({
    "Model": ["Ridge (Default)", "Ridge (Tuned)"],
    "MSE": [mse_rdg_def, mse_rdg_tuned],
    "MAE": [mae_rdg_def, mae_rdg_tuned],
    "RMSE": [rmse_rdg_def, rmse_rdg_tuned],
    "R2": [r2_rdg_def, r2_rdg_tuned]
})

print(compare_df)


metrics_df = pd.DataFrame({
    "Model": ["Tuned LinearRegression", "Tuned DecisionTree", "Tuned Ridge"],
    "MSE": [mse_lin_tuned, mse_tree_tuned, mse_rdg_tuned
    ],
    "MAE": [
        mae_lin_tuned, mae_tree_tuned, mae_rdg_tuned
    ],
    "RMSE": [
        rmse_lin_tuned, rmse_tree_tuned, rmse_rdg_tuned
    ],
    "R2": [
        r2_lin_tuned, r2_tree_tuned, r2_rdg_tuned
    ]
})

print(metrics_df)

