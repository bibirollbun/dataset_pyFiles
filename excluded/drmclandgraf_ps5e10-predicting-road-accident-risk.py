# This is the standard code, when creating a new notebook on Kaggle with some extra libraries loaded

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train.head()


train.dtypes


train.describe()


for column in train.select_dtypes(include=['object', "boolean"]).columns:
    display(train[column].value_counts())

display(train["num_lanes"].value_counts())
display(train["speed_limit"].value_counts())
display(train["num_reported_accidents"].value_counts())


sns.kdeplot(x = train["curvature"], fill = True)
plt.title("curvature")
plt.show()
sns.kdeplot(x = train["accident_risk"], fill = True) 
plt.title("accident_risk")
plt.show()


train = train.drop("id", axis = 1)
train, test = train_test_split(train, test_size = 0.2, random_state = 13)
train, val = train_test_split(train, test_size = 0.25, random_state = 13)
train.head()


s = ((train.dtypes == "object") | (train.dtypes ==  "bool"))
object_cols = list(s[s].index)
sns.heatmap(train.drop(object_cols, axis =1).corr().round(2), annot = True)


import numpy as np
from scipy import stats
import itertools
# from https://towardsdatascience.com/a-definitive-guide-to-effect-size-9bc93f00db86/ by Eryk Lewinson
def get_cramer_v(x, y):
    n = len(x)
    cont_table = pd.crosstab(x, y)
    chi_2 = stats.chi2_contingency(cont_table, correction=False)[0]
    v = np.sqrt(chi_2 / (n * (np.min(cont_table.shape) - 1)))
    return v

#code from https://stackoverflow.com/questions/52083501/how-to-compute-correlation-ratio-or-eta-in-python by Kiryl
def correlation_ratio(categories, values):
    categories = np.array(categories)
    values = np.array(values)
    
    ssw = 0
    ssb = 0
    for category in set(categories):
        subgroup = values[np.where(categories == category)[0]]
        ssw += sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5

def score_maker(data):
    cols_save = data.columns
    types = data.dtypes
    k = len(cols_save)
    res_df = pd.DataFrame([[None]*k for i in range(k)],cols_save, cols_save)
    for feature1 in cols_save:
        for feature2 in cols_save:
            if feature1 == feature2:
                res_df.loc[feature1, feature2] = 1
            elif types.loc[feature1] in ["int64", "float64"]:
                if types.loc[feature2] in ["int64", "float64"]:
                    res_df.loc[feature1, feature2] = train[[feature1, feature2]].corr().iloc[1,0]
                if types.loc[feature2] in ["bool", "object"]:
                    res_df.loc[feature1, feature2] = correlation_ratio(train[feature2], train[feature1])
            elif types.loc[feature1] in ["bool", "object"]:
                if types.loc[feature2] in ["int64", "float64"]:
                    res_df.loc[feature1, feature2] = correlation_ratio(train[feature1], train[feature2])
                if types.loc[feature2] in ["bool", "object"]:
                    res_df.loc[feature1, feature2] = get_cramer_v(train[feature2], train[feature1])
    return(res_df)
        


scores = score_maker(train)
plt.figure(figsize=(13, 13))
sns.heatmap(scores[scores.columns].astype(float).round(2), annot = True)


pd.crosstab(train["road_type"], train["time_of_day"])


sns.violinplot(x = train["speed_limit"],y =train["accident_risk"])


from catboost import CatBoostRegressor

#parameters = {'iterations': [int(x) for x in np.logspace(np.log10(100),np.log10(2000), num = 15)],
#             'learning_rate': [x for x in np.logspace(np.log10(0.1), np.log10(0.9), num = 10)],
#             'depth': [4,6,8,10,12],
#             'l2_leaf_reg': [x for x in np.logspace(np.log10(100),np.log10(2000), num = 15)],
#             'colsample_bylevel': [x for x in np.logspace(np.log10(0.1), np.log10(0.9), num = 10)]}
#cat_reg = CatBoostRegressor(loss_function = "RMSE", cat_features = object_cols, verbose = 0,
#                            early_stopping_rounds = 50, random_state = 13)
#randomized_search_results = cat_reg.randomized_search(param_distributions = parameters, X = train.drop("accident_risk", axis = 1), 
#                                                      y = train["accident_risk"], n_iter = 20, cv = 3)

#randomized_search_results["params"]


cat_model = CatBoostRegressor(loss_function = "RMSE", cat_features = object_cols, verbose = 0,
                             depth = 6, learning_rate = 0.9, l2_leaf_reg = 686.0826, iterations = 1303,
                              early_stopping_rounds = 50, random_state = 13)
cat_model.fit(X = train.drop("accident_risk", axis = 1), y = train["accident_risk"])
val_predictions = cat_model.predict(val.drop("accident_risk", axis = 1))
print("The test predictions range from {} to {}.".format(min(val_predictions), max(val_predictions)))
rmse = np.sqrt(mean_squared_error(val["accident_risk"], val_predictions))
print("Estimated RMSE: {}".format(round(rmse,4)))


from sklearn.inspection import permutation_importance
r = permutation_importance(cat_model, val.drop("accident_risk", axis = 1), val["accident_risk"],
                           n_repeats=30,
                           random_state=13)

imp = pd.DataFrame({"Mean": r["importances_mean"], "std": r ["importances_std"]}, 
                   index = val.drop("accident_risk", axis = 1).columns)
imp = imp.sort_values(by = ["Mean"], ascending = False)
plt.figure(figsize=(13,7))
sns.barplot(y = imp.index,x = imp["Mean"], 
           xerr = imp["std"])


import shap
explainer = shap.Explainer(cat_model, random_state = 13)
shap_values = explainer(val.drop("accident_risk", axis = 1))
shap.plots.bar(shap_values)

