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


dataset_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")


test_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")


from sklearn.model_selection import train_test_split


def prepocessor(X, train, scaler_train) :

    dataset_df = X.copy()
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    import pandas as pd
    
    ordenc = OrdinalEncoder()
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    dataset_df["sale_date"] = pd.to_datetime(dataset_df["sale_date"])
    dataset_df["sale_year"] = dataset_df["sale_date"].dt.year
    dataset_df["sale_month"] = dataset_df["sale_date"].dt.month
    dataset_df["sale_day"] = dataset_df["sale_date"].dt.day
    dataset_df = dataset_df.drop(columns=['sale_date'])

    
    dataset_df["encoded_city"] = ordenc.fit_transform(dataset_df[["city"]])
    dataset_df = dataset_df.drop(columns=["city"])
    
    zoning_freq = dataset_df['zoning'].value_counts(normalize=True)
    dataset_df['zoning_freq'] = dataset_df['zoning'].map(zoning_freq)
    dataset_df = dataset_df.drop(columns = ["zoning"])

    dataset_df = dataset_df.drop(columns = ["subdivision"])

    dataset_df["encoded_submarket"] = ordenc.fit_transform(dataset_df[["submarket"]])
    dataset_df = dataset_df.drop(columns=["submarket"])

    #dataset_df = dataset_df.dropna(subset=['encoded_submarket'])
    dataset_df['encoded_submarket'] = imputer.fit_transform(dataset_df[['encoded_submarket']])

    X_return = dataset_df.drop(["sale_nbr", "sale_warning", "join_status"], axis=1)


    if train:
        y = X_return["sale_price"]
        X_return = X_return.drop(columns = ["sale_price", "id"])
        
        scaler = MinMaxScaler()
        X_return = scaler.fit_transform(X_return)
        
        return X_return, scaler, y
    else :
        id_given_test = X_return["id"]
        X_return = X_return.drop(columns=["id"])
        
        scaler = scaler_train
        X_return = scaler.transform(X_return)
        
        return X_return, id_given_test

    


X, scaler_train, y = prepocessor(dataset_df, True, None)


X_train, X_test, y_sale_price_train, y_sale_price_test = train_test_split(X, y, test_size = 0.2, random_state=42)


from sklearn.ensemble import HistGradientBoostingRegressor


# hgbr_lower = HistGradientBoostingRegressor(
#     loss="quantile",
#     quantile=0.05,
#     learning_rate=0.03,
#     max_depth=7,
#     min_samples_leaf=50,
#     max_iter=300,
#     l2_regularization=0.1,
#     max_bins=255,
#     early_stopping=False,
#     random_state=42
# )

# hgbr_upper = HistGradientBoostingRegressor(
#     loss="quantile",
#     quantile=0.95,
#     learning_rate=0.03,
#     max_depth=7,
#     min_samples_leaf=50,
#     max_iter=300,
#     l2_regularization=0.1,
#     max_bins=255,
#     early_stopping=False,
#     random_state=42
# )

# # gbr_price = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, random_state=42)
# # gbr_price.fit(X_train, y_sale_price_train)


hgbr_lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.05, random_state=42)
hgbr_upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, random_state=42)


from sklearn.model_selection import RandomizedSearchCV


param_grid = {
    'learning_rate': [0.01, 0.03, 0.1],
    'max_depth': [5, 7, 10, None],
    'min_samples_leaf': [20, 50, 100],
    'max_iter': [200, 300, 500],
    'l2_regularization': [0.0, 0.1, 1.0],
    'max_bins': [255, 512],  # higher if categorical encoded
    'early_stopping': [False]  # disable for full runs
}


search_lower = RandomizedSearchCV(
    estimator=hgbr_lower,
    param_distributions=param_grid,
    n_iter=30,
    scoring=None,
    cv=3,
    verbose=2,
    n_jobs=-1
)

search_lower.fit(X_train, y_sale_price_train)
best_model_lower = search_lower.best_estimator_
print(best_model_lower)


search_upper = RandomizedSearchCV(
    estimator=hgbr_upper,
    param_distributions=param_grid,
    n_iter=30,
    scoring=None,
    cv=3,
    verbose=2,
    n_jobs=-1
)

search_upper.fit(X_train, y_sale_price_train)
best_model_upper = search_upper.best_estimator_
print(best_model_upper)


hgbr_lower = HistGradientBoostingRegressor(early_stopping=False, l2_regularization=0.1,
                              loss='quantile', max_depth=7, max_iter=500,
                              min_samples_leaf=50, quantile=0.05,
                              random_state=42, max_bins=255, learning_rate=0.1)
hgbr_upper = HistGradientBoostingRegressor(early_stopping=False, l2_regularization=1.0,
                              loss='quantile', max_depth=7, max_iter=500,
                              min_samples_leaf=50, quantile=0.95,
                              random_state=42, max_bins=255, learning_rate=0.1)


hgbr_lower.fit(X_train, y_sale_price_train)
hgbr_upper.fit(X_train, y_sale_price_train)


lower_test = hgbr_lower.predict(X_test)
upper_test = hgbr_upper.predict(X_test)
# price = gbr_price.predict(X_test)


# print(type(y_sale_price_test.to_numpy()))
# print(type(lower))


accuracy = (np.sum((lower_test < y_sale_price_test.to_numpy()) * (y_sale_price_test.to_numpy() < upper_test)))/float(y_sale_price_test.shape[0])
print(accuracy)


X_given_test, id_given_test = prepocessor(test_df, False, scaler_train)
# id_given_test = X_given_test["id"]
id_given_test = id_given_test.to_numpy()
# X_given_test = X_given_test.drop(columns=["id"])


lower_given_test = hgbr_lower.predict(X_given_test)
upper_given_test = hgbr_upper.predict(X_given_test)


result = pd.DataFrame({
    'id' : id_given_test,
    'pi_lower' : lower_given_test,
    'pi_upper' : upper_given_test
})


print(result)


result.to_csv('submission_3_pi_house_price.csv', index=False)



import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


print(type(y_sale_price_test), type(lower_test))
indices = np.random.choice(len(y_sale_price_test), size=100, replace=False)
y_sale_price_test = y_sale_price_test.to_numpy()
print(type(y_sale_price_test))


print(type(X_test))
# print(X_test)
print(X_test[:,2][indices].shape, y_sale_price_test[indices].shape)


print("Training data test set")
plt.figure(figsize=(5, 5))
plt.scatter(X_test[:,2][indices], y_sale_price_test[indices], color='red')
plt.scatter(X_test[:,2][indices], lower_test[indices], color='blue')
plt.scatter(X_test[:,2][indices], upper_test[indices], color='green')
# plt.plot(X_test[indices], lower_test[indices])
# plt.plot(X_test[indices], upper_test[indices])
plt.title("y_sale_price_test vs prediction")
plt.xlabel("y_sale_price_test")
plt.ylabel("prediction")
plt.grid(True)
plt.show()

