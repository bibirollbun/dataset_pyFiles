import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings .filterwarnings('ignore')


test_raw = pd.read_csv('/kaggle/input/predict-the-building-safety-under-the-earthquake/test.csv')
print(test_raw.shape)
test_raw.head().T


train_raw = pd.read_csv('/kaggle/input/predict-the-building-safety-under-the-earthquake/train.csv')
print(train_raw.shape)
train_raw.head().T


# target, feature, and test_df
train_drift = train_raw.rename(columns={'Max drift mm': 'drift'})
target = train_raw['Max drift mm']
feature = train_raw.drop(['Max drift mm' ], axis=1) # 'Period s', 'Floor mass kg'

# test_df = test_df.drop(['Period s', 'Floor mass kg'], axis=1)
target.shape, feature.shape, test_raw.shape


train_drift['drift'].value_counts()


pd.crosstab(train_drift['drift'],train_drift['Number of floors']).head(20)


fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(20, 12))
sns.distplot(train_drift['drift'], ax=ax[0, 0]) # target
sns.distplot(train_drift['Period s'], ax=ax[0, 1])
sns.distplot(train_drift['Magnitude'], ax=ax[0, 2])
sns.distplot(train_drift['PGA g'], ax=ax[0, 3])
sns.distplot(train_drift['Floor mass kg'], ax=ax[0, 4])
sns.distplot(train_drift['LLRS tributary width m'], ax=ax[1, 0])
sns.distplot(train_drift['Final Dead Load'], ax=ax[1, 1])
sns.distplot(train_drift['Span width m'], ax=ax[1, 2])
sns.distplot(train_drift['Floor height m'], ax=ax[1, 3])
sns.distplot(train_drift['soil_class__C'], ax=ax[1, 4])
sns.distplot(train_drift['Number of floors'], ax=ax[2, 0])
sns.distplot(train_drift['Final Live Load'], ax=ax[2, 1])
plt.show()


# correlation thereshold, 0.90
corr_rate_threshold = 0.90
cor_matrix = feature.corr().abs()

# remove mirror and diagonal values
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Drop columns with higher correlation than rate_corr_threshold
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= corr_rate_threshold)]

selected_feature = feature.drop(feature[to_drop], axis=1)

print(selected_feature.shape)
selected_feature.head()
# removed "Columns 4-6 I mm4*10^6", "soil_class__C"


feature[to_drop].columns


train_df = train_raw.copy()
test_df = test_raw.copy()
train_df.shape, test_df.shape


# log transformation
plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.title("histogram of log transformed 'Max drift mm'")
train_df['Max drift mm_log'] = np.log1p(train_df['Max drift mm'])
sns.histplot(train_df['Max drift mm_log'] , kde=True)

plt.subplot(1, 2, 2)
from scipy.stats import probplot
probplot(train_df['Max drift mm_log'], plot=plt)

plt.show()

from scipy import stats
stats.shapiro(train_df['Max drift mm_log'])
# it is not a normal distribution but let's use the log transformed data.


# target, feature, and test_df

target = train_df['Max drift mm_log']
feature = train_df.drop(['Max drift mm', 'Max drift mm_log', 
                         ], axis=1) # 'Period s', 'Floor mass kg'

# test_df = test_df.drop(['Period s', 'Floor mass kg'], axis=1)
target.shape, feature.shape, test_df.shape


# correlation thereshold, 0.90
corr_rate_threshold = 0.90
cor_matrix = feature.corr().abs()

# remove mirror and diagonal values
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Drop columns with higher correlation than rate_corr_threshold
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= corr_rate_threshold)]

selected_feature = feature.drop(feature[to_drop], axis=1)

print(selected_feature.shape)
selected_feature.head()
# removed "Columns 4-6 I mm4*10^6", "soil_class__C"


feature[to_drop].columns


selected_test = test_df.drop(['Columns 4-6 I mm4*10^6', 'soil_class__C'], axis=1)
selected_test.shape


# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(selected_feature)

feature_std = scaler.transform(selected_feature)
test_std = scaler.transform(selected_test)

feature_std = pd.DataFrame(data=feature_std, columns=selected_feature.columns)
test_std = pd.DataFrame(data=test_std, columns=selected_test.columns)

print(feature_std.shape, test_std.shape)
feature_std.head()


# X_train, y_train
from sklearn.model_selection import train_test_split, KFold

kf = KFold(n_splits = 15, shuffle = True, random_state = 2304)
for tr_idx, te_idx in kf.split(feature_std):
    X_train, X_val = feature_std.iloc[tr_idx], feature_std.iloc[te_idx]
    y_train, y_val = target.iloc[tr_idx], target.iloc[te_idx]
    
X_train.shape, X_val.shape, y_train.shape, y_val.shape


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])
num_pipeline


def prepare_model(algorithm, X_train, y_train): 
    model = Pipeline(steps=[('preprocessing', num_pipeline),('algorithm', algorithm)])
    model.fit(X_train, y_train)
    return model


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

algorithms = [RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), 
              BaggingRegressor(), SVR(), DecisionTreeRegressor(), ExtraTreeRegressor(), 
              LinearRegression(), SGDRegressor(), KNeighborsRegressor()]


names = []
times = []
mse = []
rmse = []

for algorithm in algorithms:
    name = type(algorithm).__name__
    names.append(name)
    start_time = time.time()
    model = prepare_model(algorithm, X_train, y_train)
    pred = model.predict(X_val)
    end_time = time.time()
    times.append(end_time - start_time)
    mse.append(mean_squared_error(y_val, pred))
    rmse.append(np.sqrt(mean_squared_error(y_val, pred)))

print('Regression Results in Algorithms') 
results_dict = {'Algorithm': names, 'MSE': mse, 'RMSE': rmse, 'Time': times}
pd.DataFrame(results_dict).sort_values(by='RMSE', ascending=1)

# RandomForestRegressor is confirmed as the best model.
# 	Algorithm	MSE	RMSE	Time
# 0	RandomForestRegressor	0.036412	0.190819	1.926576
# 3	BaggingRegressor	0.038858	0.197125	0.201391
# 2	GradientBoostingRegressor	0.064235	0.253447	0.618242
# 5	DecisionTreeRegressor	0.093151	0.305206	0.035081
# 1	AdaBoostRegressor	0.109033	0.330201	0.385344
# 6	ExtraTreeRegressor	0.122748	0.350354	0.019221
# 4	SVR	0.147058	0.383482	0.556077
# 7	LinearRegression	0.405456	0.636754	0.012403
# 9	KNeighborsRegressor	0.405684	0.636933	0.028097
# 8	SGDRegressor	0.407533	0.638383	0.023873


# 1st model
rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)

print(rfr.score(X_train, y_train))
print(rfr.score(X_val, y_val))

rfr_pred = rfr.predict(X_val)
rfr_rmse = np.sqrt(mean_squared_error(y_val, rfr_pred))
print(rfr_rmse)


plt.figure(figsize=(4, 3))
sns.regplot(x = rfr_pred, y = y_val)
plt.show()


# 3rd model
gbr = GradientBoostingRegressor(random_state=2304)

gbr.fit(X_train, y_train)
print(gbr.score(X_train, y_train))
print(gbr.score(X_val, y_val))

gbr_pred = gbr.predict(X_val)
gbr_rmse = np.sqrt(mean_squared_error(y_val, gbr_pred))
print(gbr_rmse)


# 4th model
etr = ExtraTreeRegressor(random_state=2304)

etr.fit(X_train, y_train)
print(etr.score(X_train, y_train))
print(etr.score(X_val, y_val))

etr_pred = etr.predict(X_val)
etr_rmse = np.sqrt(mean_squared_error(y_val, etr_pred))
print(etr_rmse)


# aggregate result in order to sort 
result_drift = train_drift.groupby(["drift"])['Number of floors'].mean().reset_index().sort_values('drift')
# make plot
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='drift',y='Number of floors',data=train_drift,order=result_drift['drift'],ax=ax,alpha=.75)
ax.axhline(0.049,linestyle='dashed', c='black',alpha = .3)


