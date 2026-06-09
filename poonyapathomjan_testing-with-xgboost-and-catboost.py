import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import warnings


warnings.filterwarnings("ignore")


data_test = pd.read_csv(r"...\test.csv")
data_train = pd.read_csv(r"...\train.csv")


data_train.head()


data_test.head()


data_train.describe()


data_train["BMI"] = data_train["Weight"]/((data_train["Height"]/100)**2)
data_train["Intensity"] = data_train["Duration"]*data_train["Heart_Rate"]

data_test["BMI"] = data_test["Weight"]/((data_test["Height"]/100)**2)
data_test["Intensity"] = data_test["Duration"]*data_test["Heart_Rate"]


data_train.describe(exclude=np.number)


from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = data_train.drop('Calories', axis=1), data_train[['Calories']]


# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')


X.dtypes


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


X_test.dtypes


X_train["Sex"] = X_train["Sex"].astype("category")
X_test["Sex"] = X_test["Sex"].astype("category")


model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=10000,
    learning_rate=0.01,
    enable_categorical=True,
)
model.fit(X.drop(columns = ["id"]), y)


rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("RMSLE:", rmsle)



data_test["Sex"] = data_test["Sex"].astype("category")


data_test


model.predict(data_test.drop(columns = ["id"]))


rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("RMSLE:", rmsle)


data_test["Calories"] = model.predict(data_test.drop(columns = ["id"]))


data_test["Calories"] = data_test["Calories"].apply(lambda x: round(x, 3))


data_test["Calories"] = data_test["Calories"].apply(lambda x: x*-1 if x<=0 else x)


data_test


data_test[["id", "Calories"]].to_csv(r"...\Predict Calorie Expenditure\result\Predict Calorie Expenditure 10.csv", index = False)


data_test.shape[0]


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np


data_test = pd.read_csv(r"...\Predict Calorie Expenditure\test.csv")
data_train = pd.read_csv(r"...\Predict Calorie Expenditure\train.csv")


def evaluate(model, x_val, y_val):
    y_pred = model.predict(x_val)
    r2 = metrics.r2_score(y_val, y_pred)
    mse = metrics.mean_squared_error(y_val, y_pred)
    mae = metrics.mean_absolute_error(y_val, y_pred)
    msle = metrics.mean_squared_log_error(y_val, y_pred)
    # mape = np.mean(tf.keras.metrics.mean_absolute_percentage_error(y_val, y_pred).numpy())
    rmse = np.sqrt(mse)
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    # rmlse_score = rmlse(y_val, y_pred).numpy()
    print("R2 Score:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("MSLE:", msle)
    print("RMSE:", rmse)
    # print("RMLSE", rmlse_score)
    return {"r2": r2, "mse": mse, "mae": mae, "msle": msle, "rmse": rmse, "rmlse": rmsle}


data_train.head()


correlation_scores = data_train.drop(columns = ["Sex"]).corr()
correlation_scores


data_train["BMI"] = data_train["Weight"]/((data_train["Height"]/100)**2)


data_train["Intensity"] = data_train["Duration"]*data_train["Heart_Rate"]


data_train.drop(columns = ["Sex"]).corr()["Calories"].sort_values(key = lambda x: abs(x), ascending=False)


data_train.columns


train_use = data_train[['id', 'Sex', 'Age', 'Body_Temp', 'Calories', 'BMI', 'Intensity']]

train_use.head()


train_use_X, train_use_y = train_use.drop(columns = ["Calories"]), train_use["Calories"]


val_X = train_use_X[112500:]
val_y = train_use_y[112500:]



import catboost
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
begin = time.time()
parameters = {
    "depth": [4, 5, 6, 7, 8, 9],
    "learning_rate": [0.01, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
    "iterations": [500], 
}
def train_with_catboost(hyperparameters, X_train, X_val, y_train, y_val):
    keys = hyperparameters.keys()
    best_index = {key:0 for key in keys}
    best_cat = None
    best_score = 10e8
    for (index, key) in enumerate(keys):
        print("Find best parameter for %s" %(key))
        items = hyperparameters[key]
        best_parameter = None
        temp_best = 10e8
        for (key_index, item) in enumerate(items):
            iterations = hyperparameters["iterations"][best_index["iterations"]] if key != "iterations" else item
            learning_rate = hyperparameters["learning_rate"][best_index["learning_rate"]] if key != "learning_rate" else item
            depth = hyperparameters["depth"][best_index["depth"]] if key != "depth" else item
            print("Training with iterations: %d learning_rate: %.2f depth:%d"%(iterations, learning_rate, depth))
            cat = catboost.CatBoostRegressor(
                iterations = iterations, 
                learning_rate = learning_rate,
                depth = depth,
                verbose=500
            )
            cat.fit(X_train, y_train, verbose=False)
            result = evaluate(cat, X_val, y_val)
            score = result["rmlse"]
            if score < temp_best:
                temp_best = score
                best_index[key] = key_index
                best_parameter = item
            if score < best_score:
                best_score = score
                best_cat = cat
        print("Best Parameter for %s: "%(key), best_parameter)
    best_parameters = {
        "iterations": hyperparameters["iterations"][best_index["iterations"]],
        "learning_rate": hyperparameters["learning_rate"][best_index["learning_rate"]],
        "depth": hyperparameters["depth"][best_index["depth"]]
    }
    return best_cat, best_score, best_parameters
best_cat, best_score, best_parameters = train_with_catboost(parameters, train_use_X.drop(columns = ["Sex", "id"]), val_X.drop(columns = ["Sex", "id"]), train_use_y, val_y)
print("Best RMLSE: ", best_score)
print("Best Parameters: ", best_parameters)
elapsed = time.time() - begin 
print("Elapsed time: ", elapsed)
# submit(best_cat, test_features, test_ids, "submission_cat.csv")


cat = catboost.CatBoostRegressor(
                iterations = 500, 
                learning_rate = 0.14,
                depth = 9,
                verbose=500
            )

cat.fit(train_use_X.drop(columns = ["Sex", "id"]), train_use_y, verbose=False)


train_use_X.head()


test_use = data_test.drop(columns = ["Sex", "id"])
test_use["BMI"] = test_use["Weight"]/(test_use["Height"]/100)**2
test_use["Intensity"] = test_use["Duration"]*test_use["Heart_Rate"]
test_use.drop(columns = ["Height", "Weight", "Duration", "Heart_Rate"], inplace = True)


test_use.head()


y_pred = cat.predict(test_use)
y_pred


data_test["Calories"] = y_pred


data_test["Calories"] = data_test["Calories"].apply(lambda x: x*-1 if x<=0 else x)


data_test["Calories"] = data_test["Calories"].apply(lambda x: round(x,2))


data_test[["id", "Calories"]].to_csv(r"...\Predict Calorie Expenditure.csv", index = False)




