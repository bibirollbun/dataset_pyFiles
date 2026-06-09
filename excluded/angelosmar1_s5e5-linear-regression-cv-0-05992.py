import os, time, gc
from itertools import combinations

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

print(f"numpy version {np.__version__}")
print(f"pd version {pd.__version__}")
print(f"sklearn version {sklearn.__version__}")


data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col="id")
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col="id")


y = np.log1p(data["Calories"])
X = data.drop(columns="Calories")
X_test = test


# combine train and test data
combined = pd.concat([X, X_test])

sex = combined["Sex"].map({"female": 0, "male": 1})
combined = combined.drop(columns="Sex")


squares = (combined * combined).add_prefix("sq_")
logs = np.log(combined).add_prefix("log_")
invs = (1 / combined).add_prefix("inv_")
sqrts = np.sqrt(combined).add_prefix("sqrt_")
inv_squares = (1 / (combined ** 2)).add_prefix("inv_sq_")
sq_logs = (logs * logs).add_prefix("sq_")
semi_logs = (combined * np.log(combined)).add_prefix("semi_log_")
semi_sqrts = (combined * np.sqrt(combined)).add_prefix("semi_sqrt_")

products2 = {}
for comb in combinations(combined.columns, 2):
    products2["_mul_".join(comb)] = combined[list(comb)].prod(axis=1)
products2 = pd.DataFrame(products2)

products3 = {}
for comb in combinations(combined.columns, 3):
    products3["_mul_".join(comb)] = combined[list(comb)].prod(axis=1)
products3 = pd.DataFrame(products3)

products4 = {}
for comb in combinations(combined.columns, 4):
    products4["_mul_".join(comb)] = combined[list(comb)].prod(axis=1)
products4 = pd.DataFrame(products4)

ratios = {}
for col1, col2 in combinations(combined.columns, 2):
    ratios[col1 + "_div_" + col2] = combined[col1] / combined[col2]
    ratios[col2 + "_div_" + col1] = combined[col2] / combined[col1]
ratios = pd.DataFrame(ratios)

mul2div1 = {}   # product of 2 feats divided by another feat
for col1, col2, col3 in combinations(combined.columns, 3):
    mul2div1[col1 + "_mul_" + col2 + "_div_" + col3] = combined[col1] * combined[col2] / combined[col3]
    mul2div1[col1 + "_mul_" + col3 + "_div_" + col2] = combined[col1] * combined[col3] / combined[col2]
    mul2div1[col3 + "_mul_" + col2 + "_div_" + col1] = combined[col3] * combined[col2] / combined[col1]
mul2div1 = pd.DataFrame(mul2div1)

div_by_sq = {}  # divide a feature by the square of another
for col1, col2 in combinations(combined.columns, 2):
    div_by_sq[col1 + "_div_sq_" + col2] = combined[col1] / (combined[col2] * combined[col2])
    div_by_sq[col2 + "_div_sq_" + col1] = combined[col2] / (combined[col1] * combined[col1])
div_by_sq = pd.DataFrame(div_by_sq)

mul2div1sq = {}   # product of 2 feats divided by the square of another
for col1, col2, col3 in combinations(combined.columns, 3):
    mul2div1sq[col1 + "_mul_" + col2 + "_div_sq_" + col3] = combined[col1] * combined[col2] / (combined[col3] ** 2)
    mul2div1sq[col1 + "_mul_" + col3 + "_div_sq_" + col2] = combined[col1] * combined[col3] / (combined[col2] ** 2)
    mul2div1sq[col3 + "_mul_" + col2 + "_div_sq_" + col1] = combined[col3] * combined[col2] / (combined[col1] ** 2)
mul2div1sq = pd.DataFrame(mul2div1sq)

mul3div1 = {}   # product of 3 feats divided by another
for col1, col2, col3, col4 in combinations(combined.columns, 4):
    mul3div1["(" + col1 + col2 + col3 + ")_div_" + col4] = (combined[col1] * combined[col2] * combined[col3]) / combined[col4]
    mul3div1["(" + col1 + col4 + col3 + ")_div_" + col2] = (combined[col1] * combined[col4] * combined[col3]) / combined[col2]
    mul3div1["(" + col1 + col2 + col4 + ")_div_" + col3] = (combined[col1] * combined[col2] * combined[col4]) / combined[col3]
    mul3div1["(" + col4 + col2 + col3 + ")_div_" + col1] = (combined[col4] * combined[col2] * combined[col3]) / combined[col1]
mul3div1 = pd.DataFrame(mul3div1)

mul3div1sq = {}   # product of 3 feats divided by the square of another
for col1, col2, col3, col4 in combinations(combined.columns, 4):
    mul3div1sq["(" + col1 + col2 + col3 + ")_div_sq_" + col4] = (combined[col1] * combined[col2] * combined[col3]) / (combined[col4] ** 2)
    mul3div1sq["(" + col1 + col4 + col3 + ")_div_sq_" + col2] = (combined[col1] * combined[col4] * combined[col3]) / (combined[col2] ** 2)
    mul3div1sq["(" + col1 + col2 + col4 + ")_div_sq_" + col3] = (combined[col1] * combined[col2] * combined[col4]) / (combined[col3] ** 2)
    mul3div1sq["(" + col4 + col2 + col3 + ")_div_sq_" + col1] = (combined[col4] * combined[col2] * combined[col3]) / (combined[col1] ** 2)
mul3div1sq = pd.DataFrame(mul3div1sq)

ratios_squared = (ratios ** 2).add_suffix("_squared")
mul2div1sq_inv = (1 / mul2div1sq).add_suffix("_inversed")
mul2div1sq_squared = (mul2div1sq ** 2).add_suffix("_squared")

# add sex column interactions
combined_mul_sex = combined.multiply(sex, axis=0).add_suffix("_mul_sex")
products2_mul_sex = products2.multiply(sex, axis=0).add_suffix("_mul_sex")
products3_mul_sex = products3.multiply(sex, axis=0).add_suffix("_mul_sex")
products4_mul_sex = products4.multiply(sex, axis=0).add_suffix("_mul_sex")
mul2div1sq_mul_sex = mul2div1sq.multiply(sex, axis=0).add_suffix("_mul_sex")
mul3div1_mul_sex = mul3div1.multiply(sex, axis=0).add_suffix("_mul_sex")
mul3div1sq_mul_sex = mul3div1sq.multiply(sex, axis=0).add_suffix("_mul_sex")



# final selected batches

final_features = [combined,
                  sex,
                  invs,
                  products3,
                  combined_mul_sex,
                  products2_mul_sex, 
                  ratios,
                  mul2div1sq,
                  mul2div1sq_mul_sex, 
                  sq_logs,
                  semi_logs,
                  div_by_sq,
                  mul3div1,
                  mul3div1_mul_sex, 
                  mul3div1sq,
                  mul3div1sq_mul_sex, 
                  ratios_squared, 
                  mul2div1sq_inv, 
                  inv_squares,
                  mul2div1sq_squared, 
                ]

combined = pd.concat(final_features, axis=1)


# scale data and split to train and test

mean = combined.mean()
std = combined.std()
combined = (combined - mean) / std

X, X_test = combined.loc[X.index], combined.loc[X_test.index]

del combined


model = LinearRegression()

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=5)

time0 = time.perf_counter()

oof_preds = np.zeros(shape=(X.shape[0],))
test_preds = np.zeros(shape=(X_test.shape[0],))

scores = []


for fold_number, (train_indices, val_indices) in enumerate(kfold.split(X)):

    print(f"fold {fold_number + 1}")
    
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    oof_preds[val_indices] = val_preds

    cur_test_preds = model.predict(X_test)
    test_preds += cur_test_preds

    score = np.sqrt(mean_squared_error(y_val, val_preds))
    print(score)
    scores.append(score)


test_preds /= num_folds

time1 = time.perf_counter()

cv_score = np.mean(scores)
cv_std = np.std(scores)

print("----------------------------------")
print(f"time {round(time1 - time0, 2)}s")
print("cv score ", cv_score)
print("cv std", cv_std)


oof_preds = pd.DataFrame(np.expm1(oof_preds).clip(0, 314), index=X.index)
test_preds = pd.DataFrame(np.expm1(test_preds).clip(0, 314), index=X_test.index)


plt.hist(oof_preds, bins=100)
plt.title("oof preds")
plt.show()


plt.hist(test_preds, bins=100)
plt.title("test preds")
plt.show()


oof_preds.to_csv("oof.csv")
test_preds.to_csv("submission.csv")


model.fit(X, y)

weights_dict = dict(zip(model.feature_names_in_, model.coef_))
weights = pd.DataFrame.from_dict(weights_dict, orient='index', columns=["weight"])


model.intercept_


weights.sort_values(by="weight", ascending=False, key=lambda x: abs(x)).head(15)

