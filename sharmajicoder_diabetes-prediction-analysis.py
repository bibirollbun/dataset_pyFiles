import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import copy


import warnings
warnings.filterwarnings("ignore")


df_train        =     pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test         =     pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_submission   =     pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


df_train.head()


df_train.info()


df_train.drop(columns = "id", inplace = True)


df_train.isnull().sum()


print(pd.DataFrame({"min": df_train.select_dtypes("int").min(), "max":df_train.select_dtypes("int").max()}))


df_train["diagnosed_diabetes"].value_counts(normalize = True) * 100


int_columns_list = df_train.select_dtypes("int64").columns.tolist()

for column in int_columns_list:
    df_train[column] = df_train[column].astype(np.uint16)


print(pd.DataFrame({"min": df_train.select_dtypes("float").min(), "max":df_train.select_dtypes("float").max()}))


int_columns_list = df_train.select_dtypes("float64").columns.tolist()

for column in int_columns_list:
    df_train[column] = df_train[column].astype(np.float16)


df_train.info()


#checking duplicates rows and columns

df_train.duplicated().sum()


df_train.describe()


object_columns_list = df_train.select_dtypes("object").columns.tolist()

rows = 2
fig, axes = plt.subplots(rows, 3, figsize = (25, 7 * rows))
axes = axes.flatten()
for ax, col in zip(axes, object_columns_list):
    ax.bar(
        df_train[col].value_counts().index, df_train[col].value_counts().values,
        color = "gray",
        width = 0.8,
        label = f"{col}",
        alpha = 1,
        linewidth = 1,
        edgecolor = "black"
    )
    ax.set_title(f"Bar plot of {col}", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    ax.set_xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    ax.set_ylabel("Count", fontsize = 16, fontweight = "bold", color = "black")
    ax.legend(shadow = True, fontsize = 12)
    ax.grid(True, alpha = 0.5, linewidth = 1.0)
plt.show()


object_columns_list = df_train.select_dtypes("object").columns.tolist()

for col in object_columns_list:
    print(df_train[col].value_counts(normalize = True) * 100)
    print("-" * 30)


float_columns_list = df_train.select_dtypes("float16").columns.tolist()

rows = 2
fig, axes = plt.subplots(rows, 3, figsize = (25, 7 * rows))
axes = axes.flatten()
for ax, col in zip(axes, float_columns_list):
    ax.hist(
        df_train[col],
        bins = int(np.sqrt(df_train[col].nunique())) + 100,
        color = "gray",
        label = f"{col}"
    )
    ax.set_title(f"Hist plot of {col}", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    ax.set_xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    ax.set_ylabel("Count", fontsize = 16, fontweight = "bold", color = "black")
    ax.legend(shadow = True, fontsize = 12)
plt.show()


int_columns_list = df_train.select_dtypes("uint16").columns.tolist()

rows = 5
fig, axes = plt.subplots(rows, 3, figsize = (25, 7 * rows))
axes = axes.flatten()
for ax, col in zip(axes, int_columns_list):
    ax.hist(
        df_train[col],
        bins = int(np.sqrt(df_train[col].nunique())) + 100,
        color = "gray",
        label = f"{col}"
    )
    ax.set_title(f"Hist plot of {col}", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    ax.set_xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    ax.set_ylabel("Count", fontsize = 16, fontweight = "bold", color = "black")
    ax.legend(shadow = True, fontsize = 12)
plt.show()


sns.heatmap(df_train.select_dtypes(["float16", "uint16"]).corr(), cmap = "Blues")


encoder = LabelEncoder()

object_columns_list = df_train.select_dtypes("object").columns.tolist()

for col in object_columns_list:
    df_train[col] = encoder.fit_transform(df_train[col])


df_train.head()


X = df_train.drop(columns = "diagnosed_diabetes")
Y = df_train["diagnosed_diabetes"]


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

auc_scores = []
models = []

best_auc = -1
fold = 1
best_model1 = None

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model3 = XGBClassifier(
        n_estimators = 800, # number of trees
        learning_rate = 0.05,
        # max_depth = 6, # depth of each tree
        # min_child_weight = 1, # minimum sum of weights in a child
        # gamma = 0.0, # minimum loss reduction to make a split
        # subsamples = 0.8, # row sampling per tree
        # colsample_bytree = 0.8, # feature sampling per tree 
        # reg_alpha = 0.1, # L1 regularization
        # reg_lambda = 1.0, # L2 regularization
        # objective = "binary:logistic", #classification
        # random_state = 42, # shuffling
        # n_jobs = 1 # use all CPU
    )

    model3.fit(X_train, Y_train)
    
    Y_pred3 = model3.predict_proba(X_test)[:,1]

    auc =roc_auc_score(Y_test, Y_pred3)
    print(f"roc-auc score of {fold}: {auc}")
    auc_scores.append(auc)
    models.append(copy.deepcopy(model3))

    if auc > best_auc:
        best_auc = auc
        best_model1 = copy.deepcopy(model3)

    fold = fold + 1
    
print(f"Final Average roc-auc-score: {np.mean(auc_scores)}")
print(f"Best auc-roc score: {best_auc}")


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

auc_scores = []
models = []

best_auc = -1
fold = 1
best_model2 = None

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model4 = LGBMClassifier(
        n_estimators = 800, # number of trees
        learning_rate = 0.05,
        # max_depth = 6, # depth of each tree
        # num_leaves = 31,
        # subsample = 0.8, # row sampling per tree
        # colsample_bytree = 0.8, # feature sampling per tree 
        # reg_alpha = 0.1, # L1 regularization
        # reg_lambda = 1.0, # L2 regularization
        # boosting_type = "gbdt",
        # objective = "binary", #classification
        # random_state = 42, # shuffling
        # n_jobs = 1 # use all CPU
    )

    model4.fit(X_train, Y_train)
    
    Y_pred4 = model4.predict_proba(X_test)[:,1]

    auc =roc_auc_score(Y_test, Y_pred4)
    print(f"roc-auc score of {fold}: {auc}")
    auc_scores.append(auc)
    models.append(copy.deepcopy(model4))

    if auc > best_auc:
        best_auc = auc
        best_model2 = copy.deepcopy(model4)

    fold = fold + 1
    
print(f"Final Average roc-auc-score: {np.mean(auc_scores)}")
print(f"Best auc-roc score: {best_auc}")


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

auc_scores = []
models = []

best_auc = -1
fold = 1
best_model3 = None

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    model5 = CatBoostClassifier(
        iterations = 800,
        learning_rate = 0.05,
        # depth = 6,
        # loss_function = "Logloss",
        # eval_metric = "AUC",
        # l2_leaf_reg = 3,
        # random_state = 42,
        # verbose = False,
        # task_type = "CPU"
    )

    model5.fit(X_train, Y_train)
    
    Y_pred5 = model5.predict_proba(X_test)[:, 1]

    auc =roc_auc_score(Y_test, Y_pred5)
    print(f"roc-auc score of {fold}: {auc}")
    auc_scores.append(auc)
    models.append(copy.deepcopy(model5))

    if auc > best_auc:
        best_auc = auc
        best_model3 = copy.deepcopy(model5)

    fold = fold + 1
    
print(f"Final Average roc-auc-score: {np.mean(auc_scores)}")
print(f"Best auc-roc score: {best_auc}")


Y_pred3


Y_pred4


Y_pred5


Y_pred_combine = (Y_pred3 + Y_pred4 + Y_pred5) / 3


Y_pred_combine


roc_auc_score(Y_test, Y_pred_combine)


df_test.head()


df_test.info()


test_id_column = df_test["id"]


df_test.drop(columns = "id", inplace = True)


df_test.isnull().sum()


print(pd.DataFrame({"min": df_test.select_dtypes("int").min(), "max":df_test.select_dtypes("int").max()}))


int_columns_list = df_test.select_dtypes("int64").columns.tolist()

for column in int_columns_list:
    df_test[column] = df_test[column].astype(np.uint16)


print(pd.DataFrame({"min": df_test.select_dtypes("float").min(), "max":df_test.select_dtypes("float").max()}))


int_columns_list = df_test.select_dtypes("float64").columns.tolist()

for column in int_columns_list:
    df_test[column] = df_test[column].astype(np.float16)


df_test.info()


encoder = LabelEncoder()

object_columns_list = df_test.select_dtypes("object").columns.tolist()

for col in object_columns_list:
    df_test[col] = encoder.fit_transform(df_test[col])


df_test.head()


model1_prediction = best_model1.predict(df_test)


model2_prediction = best_model2.predict(df_test)


model3_prediction = best_model3.predict(df_test)


model1_prediction


model2_prediction


model3_prediction


final_prediction = model1_prediction*0.2 + model2_prediction*0.5 + model3_prediction*0.3


final_prediction


df_submission = pd.DataFrame({
    "id": test_id_column,
    "diagnosed_diabetes": final_prediction
})
df_submission.to_csv("submission.csv", index = False)
df_submission.to_csv("/kaggle/working/submission.csv", index = False)

