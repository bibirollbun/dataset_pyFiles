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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE, SequentialFeatureSelector, RFECV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score,KFold
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


df_train = pd.read_csv("/kaggle/input/playground-series-s3e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s3e1/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s3e1/sample_submission.csv")


print(f"shape of training set:{df_train.shape}")
print(f"shape of training set:{df_test.shape}")
print(f"shape of training set:{df_sub.shape}")


df_train.head()


df_train.isnull().sum()


df_train.isna().sum()


df_train.info()


for col in df_train.columns:
    print(f"Unique Values {len(df_train[col].unique())} in column {col}")


num_feat = df_train.select_dtypes(include=np.number)
corr = df_train.corr()
fig, ax = plt.subplots(figsize=(14,7))
plt.title("Heatmap for the correlation", y=1, size=16)
sns.heatmap(corr,square=True, vmax=0.7)



sns.scatterplot(x="MedInc", y="MedHouseVal", data=df_train)


sns.scatterplot(x="AveRooms", y="MedHouseVal", data=df_train)


n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}
# features= df_train.columns
cols=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude',"MedHouseVal"]
fig, ax = plt.subplots(3,3, figsize=(16, 10))
ax = ax.flatten()

for i, column in enumerate(cols):
    sns.histplot(
        df_train[column], label='Train',
        ax=ax[i], color='red', **histplot_hyperparams
    )


t = sns.distplot(df_train["AveRooms"], 
                 label="Skewness: %.2f"%(df_train["AveRooms"].skew()))
t.legend()



Log_Ave = df_train["AveRooms"].map(lambda i: np.log(i) if i > 0 else 0)
t = sns.distplot(Log_Ave, label="Skewness: %.2f"%(Log_Ave.skew()))
t.legend()


Boxcox = stats.boxcox(df_train["AveRooms"])
Boxcox = pd.Series(Boxcox[0])
t = sns.distplot(Boxcox, label="Skewness: %.2f"%(Boxcox.skew()))
t.legend()


sqrt_Ave = df_train["AveRooms"].map(lambda i: np.sqrt(i))
t = sns.distplot(sqrt_Ave, label="Skewness: %.2f"%(sqrt_Ave.skew()))
t.legend()


def out_iqr(df, column):
    global lower, upper
    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    print('The IQR is',iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    df1 = df[df[column] < lower]
    df2 = df[df[column] > upper]
    return print('Total number of outliers are', df1.shape[0]+ df2.shape[0])


out_iqr(df_train, "AveRooms")


plt.figure(figsize = (10,6))
sns.distplot(df_train.AveRooms, kde=False)
plt.axvspan(xmin = lower,xmax= df_train.AveRooms.min(),alpha=0.2, color='red')
plt.axvspan(xmin = upper,xmax= df_train.AveRooms.max(),alpha=0.2, color='red')


plt.figure(figsize = (10,5))
sns.distplot(df_train['MedInc'])


def z_score(df, col):
    global zscore
    outlier_list = []

    mean = df[col].mean()
    std = df[col].std()
    threshold = 3

    zscore = abs(df[col] - mean) / std



    df1 = df[zscore > threshold]

    return print('Total number of outliers is:', df1.shape[0])


z_score(df_train, "MedInc")


plt.figure(figsize = (10,5))
sns.distplot(zscore)
plt.axvspan(xmin = 3 ,xmax= max(zscore),alpha=0.2, color='red')


X = df_train[['MedInc', 'HouseAge', 'AveRooms']].values
model = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = model.labels_
model


pd.Series(labels).value_counts().sort_index()


plt.figure(figsize=(12,12))
ax = plt.axes(projection ='3d')
unique_labels = set(labels)
colors = ['blue', 'red', "green"]
sample_mask = []
for color,label in zip(colors, unique_labels):
    sample_mask = [True if l == label else False for l in labels]
    plt.plot(X[:,0][sample_mask], X[:,1][sample_mask], X[:,2][sample_mask], 'o', color=color);
# ax.scatter3D()
plt.xlabel('MedInc');
plt.ylabel('HouseAge');
plt.ylabel('AveRooms');


lof = LocalOutlierFactor(n_neighbors=50, contamination="auto")
lof_val= df_train[['AveRooms', 'Population']].values
y_pred = lof.fit_predict(lof_val)
plt.figure(figsize=(12,12))
in_mask = [True if l == 1 else False for l in y_pred]
out_mask = [True if l == -1 else False for l in y_pred]

plt.title("Local Outlier Factor (LOF)")
# inliers
a = plt.scatter(lof_val[in_mask, 0], lof_val[in_mask, 1], c = 'blue',
                edgecolor = 'k', s = 30)
# outliers
b = plt.scatter(lof_val[out_mask, 0], lof_val[out_mask, 1], c = 'red',
                edgecolor = 'k', s = 30)
plt.axis('tight')
plt.xlabel('AveBedrms');
plt.ylabel('Population');
plt.show()


n_bins = 50

# features= df_train.columns
cols=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude',"MedHouseVal"]
fig, ax = plt.subplots(3,3, figsize=(16, 10))
ax = ax.flatten()

for i, column in enumerate(cols):
    sns.boxplot(
        df_train[column], 
        ax=ax[i], color='red'
    )


sns.boxplot(x='AveRooms',data=df_train)


first_quartile=df_train.AveRooms.quantile(0.25)
third_quartile=df_train.AveRooms.quantile(0.75)
IQR = third_quartile - first_quartile


new_boundary = third_quartile + 3 * IQR 

df_train.drop( df_train [ df_train ["AveRooms"] > new_boundary].index , axis=0 , inplace=True)


sns.boxplot(x='AveRooms',data=df_train)


df_train.columns


df_ios = df_train.copy()
df_ios = df_ios.drop(['MedHouseVal'], axis=1)


model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=float(0.1),
    max_features=float(1.0)
).fit(df_ios)


score = model.decision_function(df_ios)
outlier = model.predict(df_ios)

df_ios['scores']=score
df_ios['anomaly']=outlier

df_ios.head(10)


len(df_train.index)


anomaly = df_ios.loc[df_ios['anomaly']==-1]
anomaly_index = list(anomaly.index)
print('Total number of outliers is:', len(anomaly))


df_train.columns


X = df_train[['id', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
       'AveOccup', 'Latitude', 'Longitude']]
y = df_train['MedHouseVal']


model = LinearRegression()


rfe = RFE(model, n_features_to_select=5, verbose=1)

rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
print(f"Selected features: {selected_features}")


sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward', scoring='r2')

sfs.fit(X, y)

selected_features = X.columns[sfs.get_support()]
print(f"Selected features: {selected_features}")


sfs_backward = SequentialFeatureSelector(model, n_features_to_select=5, direction='backward', scoring='r2')

sfs_backward.fit(X, y)

selected_features_backward = X.columns[sfs_backward.get_support()]
print(f"Selected features: {selected_features_backward}")


rfecv = RFECV(estimator=model, step=1, cv=5)   #5-fold cross-validation
rfecv = rfecv.fit(X, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X.columns[rfecv.support_])


features = ['MedInc', 'AveBedrms', 'Latitude', 'Longitude']
X = X[features]


kf = KFold(n_splits=5, shuffle=True, random_state=42)


preds = []
met_score = []
feature_importance_df = pd.DataFrame()


for fold, (idx_train, idx_valid) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    
    model = XGBRegressor(booster='gbtree',
                         eval_metric='rmse',
                         early_stopping_rounds=100)
    
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose=False)
    
    pred_valid = model.predict(X_valid)
    score = mean_squared_error(y_valid, pred_valid, squared=False)
    met_score.append(score)
    print(f"Fold: {fold + 1} Score: {score}")
    print('||'*20)
    
    test_preds = model.predict(df_test[features])
    preds.append(test_preds)

print(f"Overall Validation Score: {np.mean(met_score)}")


preds2 = []
met_score2 = []
for fold, (idx_train, idx_valid) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    
    model = LGBMRegressor(learning_rate=0.025, n_estimators=100_000, metric='rmse')
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgbm.early_stopping(500, verbose=True)])
    
    pred_valid = model.predict(X_valid)
    score = mean_squared_error(y_valid, pred_valid, squared=False)
    met_score2.append(score)
    print(f"Fold: {fold + 1} Score: {score}")
    print('||'*20)
    
    test_preds = model.predict(df_test[features])
    preds2.append(test_preds)

print(f"Overall Validation Score: {np.mean(met_score2)}")


preds3 = []
met_score3 = []
for fold, (idx_train, idx_valid) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    
    model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
    model.fit(X_train, y_train)
    
    pred_valid = model.predict(X_valid)
    score = mean_squared_error(y_valid, pred_valid, squared=False)
    met_score3.append(score)
    print(f"Fold: {fold + 1} Score: {score}")
    print('||'*20)
    
    test_preds = model.predict(df_test[features])
    preds3.append(test_preds)
print(f"Overall Validation Score: {np.mean(met_score3)}")


prediction1 = np.mean(np.column_stack(preds),axis=1)
prediction2 =np.mean(np.column_stack(preds2),axis=1)
prediction3 =np.mean(np.column_stack(preds3),axis=1)
final_pred=(prediction1+prediction2+prediction3)/3


final_pred


df_sub['MedHouseVal'] = final_pred
df_sub.to_csv('submission.csv', index=False)
df_sub.head()




