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


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


full = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv')


cols = full.columns.tolist()
cols = cols[0].split(";")
cols = [str(cols[i][1:len(cols[i])-1]) for i in range(len(cols))]
cols[0] = 'age'
print(cols)


full_data = [full.iloc[i].tolist()[0].split(";") for i in range(full.shape[0])]


def rem_inv_comma(lis):
    p = []
    for i in lis:
        if type(i) == str:
            try:
                i = float(i)
            except ValueError:
                i = i[1:len(i)-1]
            p.insert(len(p), i)
    lis = p
    return lis

new = []
for j in full_data:
    new.insert(len(new), rem_inv_comma(j))

index = [i for i in range(750000,795211)]

full = pd.DataFrame(new, columns = cols, index = index)


#save the fixed dataset
full.to_csv("bank-dataset-fixed.csv",index = [i for i in range(1,45212)])


full['y'] = full['y'].map({'yes':1,'no':0})
full


train = pd.concat([train, full])
train = train.drop(['id'], axis = 1)


print(train.info())
train.head(5)


print(test.info())
test.sample(5)


print("TrainSet:\n",train.isnull().sum())
print("\n\nTestSet:\n",train.isnull().sum())


# we can see all unique values in categorical features
#function to reference all encoded objects after preprocessing
def unique():
    for i in train.columns[1:]:
        if type(train[i].iloc[0]) == str:
            print(f"All unique {i}: {sorted(train[i].unique())}", sep = "\n")

unique()


type(train['age'].iloc[0])


#I use labelEncoder to encode the categorical values since there is no problem of missing values
def encode(data):
    en = LabelEncoder()
    if type(data.iloc[0]) == str:
        return en.fit_transform(data)
    else:
        return data

for i in train.columns:
    train[i] = encode(train[i])

for i in test.columns:
    test[i] = encode(test[i])

# train = en.fit_transform(train)
# test = en.fit_transform(train)

print(train.info(),train.tail(5), sep = "\n")
print(test.info(),test.head(5), sep = "\n")


train


#correlation heatmap
fig, ax = plt.subplots(figsize=(40,40))
heatmap = sns.heatmap(train.corr(), annot = True, cmap = 'coolwarm', ax = ax)


# histogram for age
fig, ax = plt.subplots(figsize=(20,10))

sns.histplot(data = train, x = 'age', ax = ax, kde = True)


#job barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['job'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['job','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")





# Class imbalance check
print("Percentage of people who didn't opt for bank term deposit:",round(len(train[train['y'] == 0]) / train.shape[0], 3)*100,"%")
print("Percentage of people who opted for bank term deposit:",round(len(train[train['y'] == 1]) / train.shape[0], 3)*100, "%")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['marital'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['marital','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['education'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['education','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['default'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['default','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


len(sorted(train['balance']))


#marital barplot alongwith target comparison
fig, ax = plt.subplots(figsize=(10,5))
bins = [-20000, 0, 20000, 40000, 60000, 80000, 100000]
bins_label = ['-2000-0','0-20000','20000-40000','40000-60000','60000-80000','80000-100000']

train['balance_group'] = pd.cut(train['balance'], bins = bins, labels = bins_label, include_lowest = True)

grouped = train.groupby(['balance_group','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax)
ax.set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['housing'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['housing','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['loan'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['loan','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(figsize=(20,10))

grouped = train.groupby(['balance_group','loan','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax)
ax.set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['contact'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['contact','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['day'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['day','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['month'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['month','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(figsize=(20,10))

bins = [0,1000,2000,3000,4000,5000]
labels = ['0-1000','1000-2000','2000-3000','3000-4000','4000-5000']

train['duration_groups'] = pd.cut(train['duration'], bins = bins, labels = labels, include_lowest = True)

grouped = train.groupby(['duration_groups','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax)
ax.set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['campaign'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['campaign','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


train


#marital barplot alongwith target comparison
fig, ax = plt.subplots(figsize=(20,10))

bins = [-200,0,200,400,600,800]
labels = ['-200-0','0-200','200-400','400-600','600-800']

train['pdays_groups'] = pd.cut(train['pdays'], bins = bins, labels = labels, include_lowest = True)

grouped = train.groupby(['pdays_groups','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax)
ax.set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['previous'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['previous','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


#marital barplot alongwith target comparison
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax = ax.flatten()

val = train['poutcome'].value_counts()

sns.barplot(x = val.index, y = val.values, ax = ax[0])
ax[0].set_ylabel("No. of customers")

grouped = train.groupby(['poutcome','y']).size().unstack(fill_value = 0)
grouped.plot(kind = 'bar', stacked = True, ax = ax[1])
ax[1].set_ylabel("No. of customers")


train


# concat both the testdata and traindata
data = pd.concat([train.drop(['y','balance_group','duration_groups','pdays_groups'], axis = 1), test])
Y = train['y']


def feature_engineering():
    "All features must be encoded to integer values before attempting feature engineering"
    #dual feature interactions
    cols = data.columns
    
    for i in cols:
        for j in cols:
            if i != j:
                data[f"{i}+{j}"] = data[i] * data[j]

    #squared
    for i in cols:
        data[f"{i}_squared"] = data[i] * data[i]

    # will go for tri features if necessary

feature_engineering()


X = data[:795211]
test = data[795211:]


# import basic models
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier,early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
import optuna


for i in full.columns:
    full[i] = encode(full[i])

X_train_full, X_test_full, Y_train_full, Y_test_full = train_test_split(full.drop(['y'], axis = 1),
                                                                       full['y'],
                                                                       test_size = 0.3,
                                                                       random_state = 42)



# data splits
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                   Y,
                                                   test_size = 0.2,
                                                   random_state = 42)


#correlation features
cols_corr = ['job','marital','education','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
X_train_corr = X_train[cols_corr]
X_test_corr = X_test[cols_corr]


class_weights = {0:0.57, 1:4.13}


def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
    }
    
    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in cv.split(X, Y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]
        
        model = LGBMClassifier(**params, n_estimators=1000, n_jobs = 4)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)
            ]
        )
        preds = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, preds))
    
    return np.mean(aucs)


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
optuna.logging.set_verbosity(optuna.logging.INFO)
study.optimize(objective_lgbm, n_trials=10)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)



best_params = study.best_params
model = LGBMClassifier(**best_params, n_estimators=1000, n_jobs = 4)
model.fit(
            X_train, Y_train,
            eval_set=[(X_test, Y_test)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)
            ]
        )

preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:,1]

print(f"Model: LGBMClassifier")
print(f"Acuracy Score : {accuracy_score(Y_test, preds)}")
print(f"ROC_AUC Score : {roc_auc_score(Y_test, preds_proba)}")
print(f"Precision Score : {precision_score(Y_test, preds)}")
print(f"Recall Score : {recall_score(Y_test, preds)}")
print(f"Classification Report : \n{classification_report(Y_test, preds)}")
cm = confusion_matrix(Y_test, preds)
cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_disp.plot()
plt.show()


#model.fit(X, Y)
preds = model.predict_proba(test)[:,1]


#submission file
output = pd.DataFrame({'id':test['id'],'y':(preds>=0.5).astype(int)})

output.to_csv('submission.csv', index = False)

