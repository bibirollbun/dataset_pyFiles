import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) 

%matplotlib inline


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col = 'id')
orig_train = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
orig_test = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Test.csv").drop('BeatsPerMinute', axis = 1)


train = pd.concat([train, orig_train])


train.shape, test.shape, orig_train.shape


mask = np.triu(np.ones(shape = (10, 10)))


plt.figure(figsize = (15, 6))
sns.heatmap(train.corr(), annot = True, mask = mask);
plt.title('Correlation Plot', pad = 20, fontdict = {'size': 20, 'weight': 'bold'});
plt.show();


# fig, axes = plt.subplots(2, 5, figsize=(24, 10));
# fig.suptitle("TRAIN - Box Plots", fontsize=18, fontweight="bold", y = 1.01, )
# axes = np.array(axes).reshape(-1);
# cols = iter(train.columns.to_list());
# for i, ax in enumerate(axes.flatten()):
#     col = next(cols)
#     sns.boxplot(train[col], ax = ax)
#     ax.set_xlabel(col, fontdict = {'size' : 12})
#     ax.set_xticks([])
# plt.tight_layout();
# plt.show()


# fig, axes = plt.subplots(2, 5, figsize=(24, 10));
# fig.suptitle("ORIGINAL - Box Plots", fontsize=18, fontweight="bold", y = 1.01, )
# axes = np.array(axes).reshape(-1);
# cols = iter(orig_train.columns.to_list());
# for i, ax in enumerate(axes.flatten()):
#     col = next(cols)
#     sns.boxplot(orig_train[col], ax = ax)
#     ax.set_xlabel(col, fontdict = {'size' : 12})
#     ax.set_xticks([])
# plt.tight_layout();
# plt.show()


# # Transformations and Correlations
# temp_y = train['BeatsPerMinute']
# corr_df = pd.DataFrame(index = ['None', 'Log', 'Inverse', 'Exp', "Square", 'Cube'])
# for i in train.columns:
#     corr_lst = []
#     temp_x = train[i]
#     temp_corr = np.corrcoef(temp_x, temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     temp_corr = np.corrcoef(np.log1p(temp_x), temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     temp_corr = np.corrcoef(1/temp_x, temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     temp_corr = np.corrcoef(np.exp(temp_x), temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     temp_corr = np.corrcoef(temp_x**2, temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     temp_corr = np.corrcoef(temp_x**3, temp_y)[1, 0]
#     corr_lst.append(temp_corr)
#     corr_df[i] = corr_lst
# corr_df


cols = train.columns.to_list()
cols.remove('BeatsPerMinute')


# Transformations and Correlations
temp_y = np.log(train['BeatsPerMinute'])
corr_df = pd.DataFrame(index = ['None', 'Log', 'Inverse', 'Exp', "Square", 'Cube'])
for i in cols:
    corr_lst = []
    temp_x = train[i]
    temp_corr = np.corrcoef(temp_x, temp_y)[1, 0]
    corr_lst.append(temp_corr)
    temp_corr = np.corrcoef(np.log1p(temp_x), temp_y)[1, 0]
    corr_lst.append(temp_corr)
    temp_corr = np.corrcoef(1/temp_x, temp_y)[1, 0]
    corr_lst.append(temp_corr)
    temp_corr = np.corrcoef(np.exp(temp_x), temp_y)[1, 0]
    corr_lst.append(temp_corr)
    temp_corr = np.corrcoef(temp_x**2, temp_y)[1, 0]
    corr_lst.append(temp_corr)
    temp_corr = np.corrcoef(temp_x**3, temp_y)[1, 0]
    corr_lst.append(temp_corr)
    corr_df[i] = corr_lst
corr_df


plt.figure(figsize = (15, 7))
sns.heatmap(corr_df, annot = True)


# # Modification 1 - Score(27.05918)

# def modif1(data):
#     data['RhythmScore'] = 1/data['RhythmScore']
#     data['AudioLoudness'] = 1/data['AudioLoudness']
#     data['VocalContent'] = data['VocalContent']**3
#     data['AcousticQuality'] = 1/data['AcousticQuality']
#     data['InstrumentalScore'] = 1/data['InstrumentalScore']
#     data['LivePerformanceLikelihood'] = 1/data['LivePerformanceLikelihood']
#     data['MoodScore'] = data['MoodScore']**3
#     data['Energy'] = np.log1p(data['Energy'])

# modif1(train)
# modif1(test)


# # Modification 2
# train['BeatsPerMinute'] = np.log(train['BeatsPerMinute'])


## Miscellaneous
from sklearn.metrics import r2_score

def rmse(a1, a2):
    return np.sqrt(sum((a1-a2)**2/len(a1)))


import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone


class Trainer:
    def __init__(self, model = None, FOLDS = 10):
        self.model = model
        self.FOLDS = FOLDS
        self.test_preds = np.zeros(174722)
        self.rmse_list = []
        self.r2_list = []
        
    def fit(self, X, y, test, mod = None):
        indexes = (np.array(X.index)%self.FOLDS).astype(int)
        model = None
        y_pred = []
        if mod == 'lr':
            X = sm.add_constant(X)
            test = sm.add_constant(test)
        for i in range(self.FOLDS):
            X_train, X_test, y_train, y_test = X[indexes != i], X[indexes == i], y[indexes != i], y[indexes == i]
            if mod == 'lr':
                model = sm.OLS(y_train, X_train).fit()
                y_pred = model.predict(X_test).to_list()
            elif mod == 'knn':
                model = clone(self.model)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            self.eval(y_test, y_pred, i, model, test)
        print(f"\n\nCV RMSE : {np.mean(self.rmse_list):5f} - CV R2 : {np.mean(self.r2_list):5f}")
        if mod == 'exp':
            self.save_sub(np.exp(self.test_preds))
        else:
            self.save_sub(self.test_preds)
    def eval(self, y_test, y_pred, i, model, test):
        rmse_val = rmse(y_test, y_pred)
        r_score = r2_score(y_test, y_pred)
        self.r2_list.append(r_score)
        self.rmse_list.append(rmse_val)
        print(f"FOLD {i+1} - RMSE : {rmse_val:5f} - R2 : {r_score:5f}")
        self.test_preds += np.array(model.predict(test))/self.FOLDS
        
    def save_sub(self, preds):
        sam_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
        sam_sub['BeatsPerMinute'] = preds
        sam_sub.to_csv("submission.csv", index = False)


x = train.drop(['BeatsPerMinute'], axis = 1)
y = train['BeatsPerMinute']


# mod1 = Trainer()
# mod1.fit(x, y, test, 'lr') # , mod = 'exp'


from sklearn.neighbors import KNeighborsRegressor


knn_model = KNeighborsRegressor(n_neighbors=500)
# knn.fit(x, y)


knn_t = Trainer(knn_model, 10)


knn_t.fit(x, y, test, 'knn')

