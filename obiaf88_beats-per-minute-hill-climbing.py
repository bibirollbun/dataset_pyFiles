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


from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import tree
import random
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv(r'/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e9/test.csv')


train.isnull().sum()


test.isnull().sum()


train.head()


test.head()


train.shape, test.shape


sc = StandardScaler()


train_scaled = sc.fit_transform(train[[col for col in train.columns if col not in ['id','BeatsPerMinute']]])


test_scaled = sc.transform(test[[col for col in test.columns if col not in ['id']]])


target_scaled = sc.fit_transform(train[['BeatsPerMinute']])


train_scaled.shape, test_scaled.shape


poly = PolynomialFeatures(2)


train_scaled_poly = poly.fit_transform(train_scaled)


test_scaled_poly = poly.transform(test_scaled)


train_scaled_poly.shape, test_scaled_poly.shape


models = {
    'ridge_regr' : lambda alpha: Ridge(alpha=alpha),
    'rf_regressor' : lambda n_estimators: RandomForestRegressor(n_estimators= n_estimators),
    'tree': lambda max_depth: tree.DecisionTreeRegressor(max_depth = max_depth)
}


param_ranges = {
    "ridge_regr": {"alpha": [0.01, 0.1, 1, 10]},
    "rf_regressor": {"n_estimators": [10, 50, 100,500]},
    'tree' : {'max_depth': [3,4,5,10]}
}


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


rmse_scorer = make_scorer(rmse, greater_is_better=False)


def evaluate(model):
    scores = cross_val_score(model,X,y,cv = 3, scoring=rmse_scorer)
    return np.mean(scores)


current_model_name = random.choice(list(models.keys())) 
current_param_name  = list(param_ranges[current_model_name].keys())[0]
current_param_value = random.choice(param_ranges[current_model_name][current_param_name])


current_model = models[current_model_name](current_param_value)


X = train_scaled_poly
y = target_scaled


current_score = evaluate(current_model)


print(f"Start: {current_model_name}({current_param_name}={current_param_value}) => {current_score:.5f}")


for step in range(5):
    
    print(f"Running step {step}")
    
    new_model_name = current_model_name
    print(f"Actual model name:{new_model_name}")
    new_param_value = current_param_value
    print(f"Actual model param value:{new_param_value} ")
    
    rand = random.random()
    print(f"Random : {rand}")
    if rand <= 0.25:
        new_model_name = random.choice(list(models.keys()))
        new_param_name = list(param_ranges[new_model_name].keys())[0]
        new_param_value = random.choice(param_ranges[new_model_name][new_param_name])
        print(f"Changing model to {new_model_name} with params {new_param_value}")
    else:
        print("Maintaining same model but maybe trying a different parameter")
        new_param_name = list(param_ranges[new_model_name].keys())[0]
        new_param_value = random.choice(param_ranges[new_model_name][new_param_name])
        print(f"New param: {new_param_value}")
                                            
    new_model = models[new_model_name](new_param_value)
    new_score = evaluate(new_model)
    
    print(f"Current model score : {current_score}")
    print(f"New model score : {new_score}")
    
    if new_score > current_score:
            current_model_name = new_model_name
            current_param_value = new_param_value
            current_score = new_score
            print(f"Step {step+1}: {current_model_name}({new_param_name}={new_param_value}) => {current_score:.5f}")



print("\nBest found:")
print(f"{current_model_name} with {current_param_name}={current_param_value} => Score {current_score:.5f}")    


predictions = pd.DataFrame(sc.inverse_transform(current_model.fit(X,y).predict(test_scaled_poly)), columns = ['BeatsPerMinute'])


submission = pd.concat([test['id'], predictions], axis = 1)


submission


submission.to_csv('submission.csv', index=False)
print("Submission created")

