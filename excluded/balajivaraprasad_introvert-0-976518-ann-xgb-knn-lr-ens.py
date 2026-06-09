import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import optuna

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


class vars:
    train_path = "/kaggle/input/playground-series-s5e7/train.csv"
    test_path = "/kaggle/input/playground-series-s5e7/test.csv"
    sam_sub_path = "/kaggle/input/playground-series-s5e7/sample_submission.csv"
    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv"
    model_path = '/kaggle/input/introverts-5hl-nn/pytorch/default/1/model_weights.pth'

    FOLDS = 10
    EPOCHS = 250 


train = pd.read_csv(vars.train_path, index_col='id')
test = pd.read_csv(vars.test_path, index_col='id')
original = pd.read_csv(vars.original_path)
sam_sub = pd.read_csv(vars.sam_sub_path)


off_pred_probs = pd.DataFrame({'Index': range(1, train.shape[0]+1)})
test_pred_probs = pd.DataFrame({'Index': range(1, test.shape[0]+1)})


train.shape, test.shape


cols = train.columns
vals = train.isna().sum()
per = train.isna().sum()*100/train.shape[0]
print("\033[1mPercentage of Missing Values\033[0m".center(50))
print(pd.DataFrame({"Count": vals, 'Percentage': per.round(2)}))


cols = test.columns
vals = test.isna().sum()
per = test.isna().sum()*100/test.shape[0]
print("\033[1mPercentage of Missing Values\033[0m".center(50))
print(pd.DataFrame({"Count": vals, 'Percentage': per.round(2)}))


cols = original.columns
vals = original.isna().sum()
per = original.isna().sum()*100/test.shape[0]
print("\033[1mPercentage of Missing Values\033[0m".center(50))
print(pd.DataFrame({"Count": vals, 'Percentage': per.round(2)}))


# # too much data loss, need to impute
# train.dropna(inplace = True)
# test.dropna(inplace = True)
# train.shape, test.shape
# # ((10189, 9), (3397, 8))


cont_vars = train.select_dtypes(include = "float64").columns.to_list()


fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20, 6));
fig.suptitle("Dist. of Variables in Train Data", fontsize = 20);
vals = list(range(5)) # , ax = axes[next(index)]
index = iter(vals);
for i in cont_vars:
    sns.distplot(train[i], ax = axes[next(index)]);
plt.tight_layout();
fig.show()


fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20, 6))
fig.suptitle("Dist. of Variables in Test Data", fontsize = 20)
vals = list(range(5))
index = iter(vals)
for i in cont_vars:
    sns.distplot(test[i], ax = axes[next(index)])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize=(20, 6))
fig.suptitle("Dist. of Variables in Original Data", fontsize = 20)
vals = list(range(5))
index = iter(vals)
for i in cont_vars:
    sns.distplot(original[i], ax = axes[next(index)])
plt.tight_layout()
plt.show()


train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No':0})
test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, "No": 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, "No": 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, "No": 0})
original['Stage_fear'] = original['Stage_fear'].map({'Yes': 1, "No": 0})
original['Drained_after_socializing'] = original['Drained_after_socializing'].map({'Yes': 1, "No": 0})


train['Personality'] = train['Personality'].map({'Extrovert': 0, 'Introvert': 1})
original['Personality'] = original['Personality'].map({'Extrovert': 0, 'Introvert': 1})


# MICE Imputation
# ```````````````
imputer = IterativeImputer(max_iter=50, random_state=0)
imputed_data = imputer.fit_transform(original)
imp_original = pd.DataFrame(imputed_data, columns=train.columns)

# Fill 0
# ``````
# imp_original = original.fillna(-1)


# MICE Imputation
# ```````````````
imputer = IterativeImputer(max_iter=50, random_state=0)
imputed_data = imputer.fit_transform(train)
imp_train = pd.DataFrame(imputed_data, columns=train.columns)

# # Fill 0
# # ``````
# imp_train = train.fillna(-1)


# MICE Imputation
# ```````````````
imputer = IterativeImputer(max_iter=50, random_state=0)
imputed_data = imputer.fit_transform(test)
imp_test = pd.DataFrame(imputed_data, columns=test.columns)

# Fill 0
# ``````
# imp_test = test.fillna(-1)


fin_train = pd.concat([imp_train, imp_original, imp_original], axis = 0)


X = imp_train.drop(imp_train.columns[-1], axis = 1)
y = imp_train[imp_train.columns[-1]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


lr = LogisticRegression()
lr.fit(X_train, y_train)


pred = lr.predict(X_test)


print("Logistic Regression Accuracy : ", accuracy_score(pred, y_test))


off_pred_probs['LR'] = lr.predict(X)
test_pred_probs['LR'] = lr.predict(imp_test)


pred = lr.predict(imp_test)
sam_lr = sam_sub.copy()
sam_lr['Personality'] = pred
sam_lr['Personality'] = sam_lr['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_lr.head()


sam_lr.to_csv("Logistic_sub.csv", index = False)


knn = KNeighborsClassifier(n_neighbors=10)  # You can choose k as per your use case
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)
print("KNN Accuracy : ", accuracy_score(y_test, y_pred))


off_pred_probs['KNN'] = knn.predict(X)
test_pred_probs['KNN'] = knn.predict(imp_test)


pred = knn.predict(imp_test)
sam_knn = sam_sub.copy()
sam_knn['Personality'] = pred
sam_knn['Personality'] = sam_knn['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_knn.head()


sam_knn.to_csv("KNN_sub.csv", index = False)


# def objective(trial):
#     train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
#     params = {
#         "objective": "binary:logistic",
#         "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
#         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
#         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#         "max_depth": trial.suggest_int("max
#         _depth", 2, 12),
#         "eta": trial.suggest_float("eta", 1e-4, 1.0, log=True),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#     }
#     dtrain = xgb.DMatrix(train_x, label=train_y)
#     dvalid = xgb.DMatrix(valid_x, label=valid_y)
#     bst = xgb.train(params, dtrain, evals=[(dvalid, "eval")], num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
#     preds = bst.predict(dvalid)
#     pred_labels = (preds > 0.5).astype(int)
#     accuracy = accuracy_score(valid_y, pred_labels)
#     return accuracy


# ## Hyperparameter Tuning using Bayesian Search
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
# study.optimize(objective, n_trials=250, show_progress_bar = False)


# # Print the best result
# print(f'Best trial accuracy: {study.best_trial.value}')
# print(f'Best hyperparameters: {study.best_trial.params}')


# if study.best_trial.value >= 0.9771107752105377:
#     params = study.best_trial.params
# else:
#     params = {'booster': 'gbtree', 
#           'lambda': 0.005553898958169121, 
#           'alpha': 0.030364530410431377, 
#           'max_depth': 6, 
#           'eta': 0.6076967918765325, 
#           'subsample': 0.9540478939663687, 
#           'colsample_bytree': 0.8426790562963757
#          }


params = {'booster': 'gbtree', 
          'lambda': 0.005553898958169121, 
          'alpha': 0.030364530410431377, 
          'max_depth': 6, 
          'eta': 0.6076967918765325, 
          'subsample': 0.9540478939663687, 
          'colsample_bytree': 0.8426790562963757
         }


train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)
bst = xgb.train(params, dtrain, 
                evals=[(dvalid, "eval")], 
                num_boost_round=100, 
                early_stopping_rounds=10, 
                verbose_eval=False)


preds = bst.predict(dvalid)
pred_labels = (preds > 0.5).astype(int)
pred_labels
print("XGB Accuracy:", accuracy_score(valid_y, pred_labels))


off_pred_probs['XGB'] = bst.predict(xgb.DMatrix(X))
test_pred_probs['XGB'] = bst.predict(xgb.DMatrix(imp_test))


Xgb_test = xgb.DMatrix(imp_test)


preds = bst.predict(Xgb_test)
pred_labels = (preds > 0.5).astype(int)
sam_xgb = sam_sub.copy()
sam_xgb['Personality'] = pred_labels
sam_xgb['Personality'] = sam_xgb['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_xgb.head()


sam_xgb.to_csv("XGB_sub.csv", index = False)


lgbm_params = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "colsample_bytree": 0.4366677273946288,
    "learning_rate": 0.016164161953515117,
    "max_depth": 12,
    "min_child_samples": 67,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 243,
    "random_state": 42,
    "reg_alpha": 6.38288560443373,
    "reg_lambda": 9.392999314379155,
    "subsample": 0.7989164499431718,
    "verbose": -1,
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}
lgbm_goss_params = {
    "boosting_type": "goss",
    "device": "gpu",
    "colsample_bytree": 0.32751831793031183,
    "learning_rate": 0.006700715059604966,
    "max_depth": 12,
    "min_child_samples": 84,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 229,
    "random_state": 42,
    "reg_alpha": 6.879977008084246,
    "reg_lambda": 4.739518466581721,
    "subsample": 0.5411572049978781,
    "verbosity": -1,
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}


lgbm_model = LGBMClassifier(**lgbm_params, verbosity = -1)
lgbm_goss_model = LGBMClassifier(**lgbm_goss_params)


train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)


lgbm_model.fit(train_x, train_y)


lgbm_goss_model.fit(train_x, train_y)


preds = lgbm_model.predict(valid_x)
pred_labels = (preds > 0.5).astype(int)
pred_labels
print("LGBM GBDT Accuracy:", accuracy_score(valid_y, pred_labels))


off_pred_probs['LGBM_GBDT'] = lgbm_model.predict(X)
test_pred_probs['LGBM_GBDT'] = lgbm_model.predict(imp_test)


preds = lgbm_goss_model.predict(valid_x)
pred_labels = (preds > 0.5).astype(int)
pred_labels
print("LGBM goss Accuracy:", accuracy_score(valid_y, pred_labels))


off_pred_probs['LGBM_GOSS'] = lgbm_goss_model.predict(X)
test_pred_probs['LGBM_GOSS'] = lgbm_goss_model.predict(imp_test)


preds = lgbm_model.predict(imp_test)
pred_labels = (preds > 0.5).astype(int)
sam_lgbm = sam_sub.copy()
sam_lgbm['Personality'] = pred_labels
sam_lgbm['Personality'] = sam_lgbm['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_lgbm.head()


sam_lgbm.to_csv("LGBM_GBDT_sub.csv", index = False)


preds = lgbm_goss_model.predict(imp_test)
pred_labels = (preds > 0.5).astype(int)
sam_lgbm_goss = sam_sub.copy()
sam_lgbm_goss['Personality'] = pred_labels
sam_lgbm_goss['Personality'] = sam_lgbm_goss['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_lgbm_goss.head()


sam_lgbm_goss.to_csv("LGBM_GOSS_sub.csv", index = False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32))
y_train = torch.from_numpy(y_train.to_numpy().astype(np.int64))
X_test  = torch.from_numpy(X_test.to_numpy().astype(np.float32))
y_test  = torch.from_numpy(y_test.to_numpy().astype(np.int64))


test_imp = torch.from_numpy(imp_test.to_numpy().astype(np.float32))


class custom_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = custom_dataset(X_train, y_train)
test_dataset = custom_dataset(X_test, y_test)


dl_train = DataLoader(train_dataset, batch_size=128, shuffle=True,)
dl_test  = DataLoader( test_dataset, batch_size=128, shuffle=False)


X_train.shape


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    def forward(self, X):
        return self.network(X)


# def objective(trial):
#     train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
#     train_x = torch.from_numpy(train_x.to_numpy().astype(np.float32))
#     train_y = torch.from_numpy(train_y.to_numpy().astype(np.int64))
#     valid_x  = torch.from_numpy(valid_x.to_numpy().astype(np.float32))
#     valid_y  = torch.from_numpy(valid_y.to_numpy().astype(np.int64))
    
#     params = {
#         "learning_rate": trial.suggest_float('learning_rate', 0.0001, 0.1, log = True),
#         "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1, log=True),
#     }
    
#     model = Model(train_x.shape[1])
#     model.load_state_dict(torch.load('/kaggle/input/introverts-5hl-nn/pytorch/default/1/model_weights.pth', map_location=device))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr = params['learning_rate'], weight_decay = params['weight_decay'])
#     model = nn.DataParallel(model)
#     model.to(device)
#     for epoch in range(10):
#         total_epoch_loss = 0
#         for batch_features, batch_labels in dl_train:
#             batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#             output = model(batch_features)
#             loss = criterion(output, batch_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_epoch_loss = total_epoch_loss + loss.item()
#     model.eval()
#     preds = torch.max(model(valid_x.to(device)), 1).indices
#     pred_labels = preds.cpu().numpy()
#     accuracy = accuracy_score(valid_y, pred_labels)
#     return accuracy


# ## Hyperparameter Tuning using Bayesian Search
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
# study.optimize(objective, n_trials=250, show_progress_bar = False)


# Print the best result
# print(f'Best trial accuracy: {study.best_trial.value}')
# print(f'Best hyperparameters: {study.best_trial.params}')


# Best trial accuracy: 
# 0.9762470308788599
# Best hyperparameters: 
# {
#     'learning_rate': 0.0046665002218051034, 
#     'weight_decay': 0.0034367269000430438
# }


model = Model(train.shape[1]-1)
model.load_state_dict(torch.load(vars.model_path, map_location=device))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.0046665002218051034, weight_decay = 0.0034367269000430438)


from torchinfo import summary

summary(model)


model = nn.DataParallel(model)
model.to(device)


for epoch in range(vars.EPOCHS):
    total_epoch_loss = 0
    for batch_features, batch_labels in dl_train:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        output = model(batch_features)
        loss = criterion(output, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_epoch_loss = total_epoch_loss + loss.item()
    avg_loss = total_epoch_loss/len(dl_train)
    if (epoch+1)%10 == 0:
        print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')


model.eval()


outputs = model(torch.from_numpy(X.to_numpy().astype(np.float32)))
_, predicted = torch.max(outputs, 1)


off_pred_probs['ANN'] = predicted.cpu().numpy()


outputs = model(torch.from_numpy(imp_test.to_numpy().astype(np.float32)))
_, predicted = torch.max(outputs, 1)


test_pred_probs['ANN'] = predicted.cpu().numpy()


total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in dl_test:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
print(correct/total)


# evaluation code
total = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in dl_train:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
print(correct/total)


# Save model's state_dict
# torch.save(model.state_dict(), 'model_weights.pth')


preds = torch.max(model(test_imp.to(device)), 1).indices
pred_vals = preds.cpu().numpy()
sam_ann = sam_sub.copy()
sam_ann['Personality'] = pred_vals
sam_ann['Personality'] = sam_ann['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_ann.head()


# sam_ann.to_csv("submission.csv", index = False)


lr_X = off_pred_probs.drop('Index', axis = 1)


lr = LogisticRegression()
lr.fit(lr_X, y)


off_pred_probs.head()


test_pred_probs.head()


pred = lr.predict(test_pred_probs.drop('Index', axis = 1))
sam_ens = sam_sub.copy()
sam_ens['Personality'] = pred_vals
sam_ens['Personality'] = sam_ens['Personality'].map({0: 'Extrovert', 1 : 'Introvert'})
sam_ens.head()


sam_ens.to_csv("Ensemble_sub.csv", index = False)

