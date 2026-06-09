import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) 

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col = 'id')
sam_sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep = ';')


# original.y = original.y.map({"no": 0, "yes": 1})
# train = pd.concat([train, original], axis = 0)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


train.shape, test.shape, original.shape, sam_sub.shape


int_cols = train.select_dtypes(int).columns.to_list()
int_cols.remove('y')
cat_cols = train.select_dtypes(object).columns


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


counts = train.y.value_counts()
print(f"{color.BOLD}Ratio of Imbalance : {color.END}", 
      np.round(counts.values[0]/counts.values[1], 3))


plt.figure(figsize = (6, 6));
sns.histplot(x = train.y.astype(str), 
             hue = train.y.astype(str), 
             palette = 'Accent',
             shrink=0.8
            );
plt.title("Distribution of Outcome Variable", 
          loc = 'right',
          color = "#25344d", 
          fontweight = 'extra bold', 
          size = 12, 
          fontfamily = "Helvetica"
         );
plt.xlabel('Term Despsit Subscription', 
           fontweight = 'bold', 
           color = "#7294cf",
           size = 12, 
           labelpad = 10
          );
plt.xticks(ticks = [0, 1], 
           labels = ['No', 'Yes'],
           fontweight = 'bold'
          );
plt.text(0, 699434/2, str(699434), ha='center', va='bottom')
plt.text(1, 95777/2-1000, str(95777), ha='center', va='bottom')
plt.tight_layout();
plt.legend([])
plt.margins(0.1);
plt.ylabel(ylabel = f"Ratio of Imbalance : {np.round(counts.values[0]/counts.values[1], 3)}");
plt.yticks([]);
plt.show();


print(f"{color.BOLD}Number of Categories{color.END}".center(40))
print("`"*35)
for i in cat_cols:
    print(f"      {i} ".ljust(20) + f" : {len(train[i].unique())}")


# for i in cat_cols:
#     train[i] = train[i].astype('category')
# for i in cat_cols:
#     test[i] = test[i].astype('category')


corr_mat = train.select_dtypes(int).corr();
plt.figure(figsize=(6, 6));
sns.heatmap(corr_mat, 
            annot = True, 
            annot_kws={"size": 10}, 
            fmt = ".2f", 
            mask = np.triu(np.ones((8, 8)), k = 1),
            cbar = False,
            cmap = 'Accent'
           );
plt.title("Correlation Plot", 
          size = 12, 
          loc = 'right', 
          color = '#4f2e2b', 
          fontfamily = "Helvetica",
          fontweight = 'extra bold'
         );
plt.tight_layout();
plt.xticks(size = 10);
plt.yticks(size = 10);
plt.show();


fig, axes = plt.subplots(5, 2, figsize=(15, 24));
fig.suptitle("Categorical Columns - EDA", fontsize=16, fontweight="bold", y = 0.92)
axes = np.array(axes).reshape(-1);
cols = iter(cat_cols);
for i, ax in enumerate(axes.flatten()):
    if i == 9:
        break;
    col_name = next(cols);
    sns.histplot(x = train[col_name], 
        hue = train.y.astype(str), 
        palette = 'Accent',
        multiple = 'stack',
        ax = ax
    );
    ax.set_title(f"Distribution of Outcome Variable on {col_name}", 
              loc = 'right',
              color = "#25344d", 
              fontweight = 'extra bold', 
              size = 7, 
              fontfamily = "Helvetica"
             );

    n_groups = len(train[col_name].unique())
    counts = {g: [] for g in train[col_name].unique()}
    groups = list(train[col_name].unique())
    
    bin_edges = [p.get_x() for p in ax.patches] + [ax.patches[-1].get_x() + ax.patches[-1].get_width()]
    bin_edges = bin_edges[:int(len(bin_edges)/2)]
    n_groups = len(train[col_name].unique())
    for i in range(0, len(ax.patches), n_groups):
        bin_patches = ax.patches[i:i+n_groups]
        for j, patch in enumerate(bin_patches):
            counts[groups[j]].append(patch.get_height())
        bin_edges.append(bin_patches[0].get_x())

    for i, key in zip(range(len(bin_edges)), counts.keys()):
        count_tot = counts[key][0]+counts[key][1]
        ax.text(bin_edges[i], 
                count_tot, 
                f"    Yes : {int(counts[key][0]*100/count_tot)} \n    No  : {int(counts[key][1]*100/count_tot)} \n",
               fontdict = {'fontsize': 6, 'fontweight' : 'bold'})
    ax.margins(y=0.1)
    ax.tick_params(axis='x', labelrotation=10, labelsize = 6);
    ax.legend_.remove()
    ax.set_ylabel(None);
    ax.set_xlabel(None);
    # ax.set_yticks([]);
    ax.tick_params(axis="y", labelsize=8)
plt.show()


fig, axes = plt.subplots(2, 4, figsize=(15, 15));
fig.suptitle("Integer Columns - EDA", fontsize=16, fontweight="bold", y = 0.92)
axes = np.array(axes).reshape(-1);
cols = iter(int_cols);
for i, ax in enumerate(axes.flatten()):
    if i == 7:
        break;
    col_name = next(cols);
    sns.boxplot(y = train[col_name], 
        x = train.y.astype(str), 
        palette = 'Accent',
        ax = ax
    );
    ax.set_title(f"Distribution of Outcome Variable on {col_name}", 
              loc = 'right',
              color = "#25344d", 
              fontweight = 'extra bold', 
              size = 7, 
              fontfamily = "Helvetica"
             );
    ax.tick_params(axis='x', labelrotation=10, labelsize = 6);
    ax.set_ylabel(None);
    ax.set_xlabel(None);
    # ax.set_yticks([]);
    ax.tick_params(axis="y", labelsize=8)
plt.show()


fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 8))
fig.suptitle("`pday` histplot across dependent varaible", y = 0.92, color = 'teal', fontweight = 'bold')
sns.distplot(train.pdays[train.y == 0], color = 'blue', ax = ax[0])
sns.distplot(train.pdays[train.y == 1], color = 'red', ax = ax[1])
plt.show()


fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 8))
fig.suptitle("`previous` histplot across dependent varaible", y = 0.92, color = 'teal', fontweight = 'bold')
sns.distplot(train.previous[train.y == 0], color = 'blue', ax = ax[0])
sns.distplot(train.previous[train.y == 1], color = 'red', ax = ax[1])
plt.show()


temp_train = train.copy(True)
for i in int_cols:
    Q1 = temp_train[i].quantile(0.25)
    Q3 = temp_train[i].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    temp_train[i] = temp_train[i].mask(
        (temp_train[i] < lower_bound) | (temp_train[i] > upper_bound), np.nan
    )


fig, axes = plt.subplots(2, 4, figsize=(15, 15));
fig.suptitle("Integer Columns with no outliers - EDA", fontsize=16, fontweight="bold", y = 0.92)
axes = np.array(axes).reshape(-1);
cols = iter(int_cols);
for i, ax in enumerate(axes.flatten()):
    if i == 7:
        break;
    col_name = next(cols);
    sns.boxplot(y = temp_train[col_name], 
        x = temp_train.y.astype(str), 
        palette = 'Accent',
        ax = ax
    );
    ax.set_title(f"Distribution of Outcome Variable on {col_name}", 
              loc = 'right',
              color = "#25344d", 
              fontweight = 'extra bold', 
              size = 7, 
              fontfamily = "Helvetica"
             );
    ax.tick_params(axis='x', labelrotation=10, labelsize = 6);
    ax.set_ylabel(None);
    ax.set_xlabel(None);
    # ax.set_yticks([]);
    ax.tick_params(axis="y", labelsize=8)
plt.show()


from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, auc


encoder_list = []
for i in cat_cols:
    le = LabelEncoder()
    train[i] = le.fit_transform(train[i])
    test[i] = le.transform(test[i])
    encoder_list.append(le)


X = train.drop(['y'], axis = 1)
y = train['y']


from sklearn.feature_selection import mutual_info_classif


mi_scores = mutual_info_classif(X.select_dtypes(exclude = [object]), y, random_state=42)


mi_series = pd.Series(mi_scores, index=X.select_dtypes(exclude = [object]).columns)
mi_series = mi_series.sort_values(ascending=True)

print(mi_series)


X['train'] = True
test['train'] = False
ind = pd.concat([X, test], axis = 0)


int_feat = ['balance', 'duration', 'campaign', 'pdays', 'previous']
cat_feat = ['default', 'housing', 'loan', 'contact', 'poutcome']


ind[int_feat].describe()


from itertools import product, combinations
        
def pair_int_cat_prod():
    for i, j in product(int_feat, cat_feat):
        if i != j:
            ind[f"{i}_{j}"] = ind[i]+ind[j]

def add_int_feats():
    ind["pdays_previous"] = ind["pdays"] + ind['previous']

def add_int_sqrt():
    for i in [ 'duration', 'campaign']:
        ind[f"{i}_sqrt"] = np.sqrt(ind[i]).round(2)
def add_int_logs():
    for i in ['duration', 'campaign']:
        ind[f"{i}_logs"] = np.log1p(ind[i]).round(2)
def new_int_col():
    ind['balance_duration_add'] = ind['balance'] + ind['duration']    


pair_int_cat_prod();
add_int_feats();
add_int_sqrt();
add_int_logs();
new_int_col();


X = ind[ind['train'] == True].drop(['train'], axis = 1)
test = ind[ind['train'] == False].drop(['train'], axis = 1)


mi_scores = mutual_info_classif(X.select_dtypes(exclude = [object]), y, random_state=42)


mi_series = pd.Series(mi_scores, index=X.select_dtypes(exclude = [object]).columns)
mi_series = mi_series.sort_values(ascending=True)

print(mi_series)


X = X.astype('category')
test = test.astype('category')


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# class Trainer:
#     def __init__(self, model, FOLDS):
#         self.model = model
#         self.FOLDS = FOLDS

#     def fit(self, X, y, test, mod = 'xgb'):
#         skf = StratifiedKFold(n_splits = self.FOLDS, shuffle = True)
#         auc_scores = []
#         test_pred_probs = np.zeros(test.shape[0])
#         for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
#             print(f"FOLD {fold + 1} \n" + "="*10)
#             X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#             X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
#             off_pred_probs = np.zeros(X_valid.shape[0])
#             model = clone(self.model)
#             if mod == 'xgb':
#                 model.fit(
#                     X_train, 
#                     y_train, 
#                     eval_set = [(X_train, y_train), (X_valid, y_valid)], 
#                     verbose = 100
#                 )
#             elif mod == 'cat':
#                 model.fit(
#                     X_train, 
#                     y_train, 
#                     cat_features=cat_cols,
#                     eval_set = [(X_valid, y_valid)]
#                 )
#             elif mod == 'lgbm':
#                 model.fit(
#                         X_train, 
#                         y_train, 
#                         eval_set=[(X_valid, y_valid)], 
#                         callbacks=[lgb.early_stopping(350), lgb.log_evaluation(period=500)
# ]
#                 )
#             oof_pred_probs = model.predict_proba(X_valid)[:, 1]
#             auc = roc_auc_score(y_valid, oof_pred_probs)
#             auc_scores.append( auc )
#             print(f"FOLD {fold+1} AUC: {auc}\n\n")
#             test_pred_probs += model.predict_proba(test)[:, 1]/self.FOLDS
#         print(f"\n \nCV Average AUC : {np.mean(auc_scores)}")
#         if mod == 'xgb':
#             self.submit(test_pred_probs, 'xgb')
#         elif mod == 'cat':
#             self.submit(test_pred_probs, 'cat')
#         elif mod == 'lgbm':
#             self.submit(test_pred_probs, 'lgbm')
            
#     def submit(self, probs, mod):
#         sam_sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
#         sam_sub.y = probs
#         sam_sub.to_csv(f'{mod}_submission.csv', index = False)


# import xgboost as xgb


# xgb_params = {'max_depth': 13, 
#               'learning_rate': 0.01036808915308291, 
#               'min_child_weight': 7, 
#               'subsample': 0.4406011562109482,
#               'colsample_bytree': 0.8033679369123714, 
#               'gamma': 2.4652180617514747, 
#               'reg_alpha': 2.1421895943084053,
#               'reg_lambda': 1.5758614095439158, 
#               'n_estimators': 2000, 
#               'enable_categorical':True,
#               'eval_metric' : 'auc',
#               'tree_method' : 'hist',
#               'booster': 'dart',
#               'device' : 'cuda',
#               'early_stopping_rounds': 100
#              }
# xgb_model = xgb.XGBClassifier(**xgb_params)


# model_trainer = Trainer(xgb_model, 5)
# model_trainer.fit(X, y, test)


# xgb_model.fit(
#         X_train, 
#         y_train, 
#         eval_set = [(X_train, y_train), (X_test, y_test)], 
#         verbose = 100
# )


# probs =xgb_model.predict_proba(test)[:, 1]
# sam_sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
# sam_sub.y = probs
# sam_sub.to_csv('XGBoost_submission_Orig_AllCat.csv', index = False)


# from catboost import CatBoostClassifier


# cat_params = {'n_estimators' : 10000,
#               'early_stopping_rounds': 200,
#               'learning_rate': 0.0652,
#               'l2_leaf_reg': 0.886,
#               'bagging_temperature': 0.131,
#               'random_strength': 0.992,
#               'depth': 7,
#               'min_data_in_leaf': 8,
#               'task_type': "GPU",
#               'verbose' : 500,
#               'eval_metric' : 'Accuracy',
#              }
# cat_model = CatBoostClassifier(**cat_params)


# X = X.astype(str)


# cat_cols = X.select_dtypes(object).columns.to_list()


# for i in ['balance_duration_add', 'duration_sqrt', 'campaign_sqrt', 'duration_logs', 'campaign_logs']:
#     cat_cols.remove(i)


# model_trainer = Trainer(cat_model, 5)
# model_trainer.fit(X, y, test, 'cat')


# import lightgbm as lgb
# from lightgbm import LGBMClassifier

# print(lgb.__version__)


# lgbm_params = {
#     "n_estimators" : 10000,
#     "learning_rate" : 0.06,
#     "num_leaves" : 100,
#     "max_depth" : 15,
#     "min_child_samples" : 9,
#     "subsample" : 0.8,
#     "colsample_bytree" : 0.5,
#     "reg_alpha" : 0.78,
#     "reg_lambda" : 3.0,
#     "max_bin" : 4523,
#     "random_state" : 42,
#     "verbosity" : -1,
#     'metric': 'AUC',
#     'objective': 'binary',
# }
# lgbm_model = lgb.LGBMClassifier(**lgbm_params)

# # Running lgbm model on GPU is complex
# # Reference
# # 1. https://www.kaggle.com/code/abhishek/running-lightgbm-on-gpu/notebook (Didn't work)


# X = X.astype('category')


# model_trainer = Trainer(lgbm_model, 5)
# model_trainer.fit(X, y, test, 'lgbm')


# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32))
# y_train = torch.from_numpy(y_train.to_numpy().astype(np.int64))
# X_test  = torch.from_numpy(X_test.to_numpy().astype(np.float32))
# y_test  = torch.from_numpy(y_test.to_numpy().astype(np.int64))


# class cdataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, index):
#         return self.X[index], self.y[index]


# train_dataset = cdataset(X_train, y_train)
# test_dataset = cdataset(X_test, y_test)


# dl_train = DataLoader(train_dataset, batch_size=128, shuffle=True,)
# dl_test  = DataLoader( test_dataset, batch_size=128, shuffle=False)


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(61, 48),
#             nn.LeakyReLU(),
#             nn.Linear(48, 36),
#             nn.LeakyReLU(),
#             nn.Linear(36, 24),
#             nn.ReLU(),
#             nn.Linear(24, 12),
#             nn.ReLU(),
#             nn.Linear(12, 2),
#         )
#     def forward(self, X):
#         return self.network(X)


# model = Model()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr = 0.0046, weight_decay = 0.0034)


# model = nn.DataParallel(model)
# model.to(device)


# for epoch in range(100):
#     total_epoch_loss = 0
#     for batch_features, batch_labels in dl_train:
#         batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#         output = model(batch_features)
#         loss = criterion(output, batch_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_epoch_loss = total_epoch_loss + loss.item()
#     avg_loss = total_epoch_loss/len(dl_train)
#     print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')
#     if (epoch+1)%25 == 0:
#         print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')

