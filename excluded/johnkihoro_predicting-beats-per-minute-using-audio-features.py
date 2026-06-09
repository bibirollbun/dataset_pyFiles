import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Setting plot defaults
sns.set_style('darkgrid')
plt.rc('axes',
       labelweight='normal',
       titleweight='bold')


data1 = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
data1


print(data1.shape)
data1.info()


data1.describe()


data2 = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
print(data2.shape)
data2


data2.describe()


data = pd.concat([data1, data2], ignore_index=True)
print(data.shape)
data


y = data['BeatsPerMinute']
X = data.drop('BeatsPerMinute', axis=1)
X['TrackDuration_Mins'] = X['TrackDurationMs']/60000
X.drop('TrackDurationMs', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)
X_train.head()


X.head()


sns.histplot(data, x='BeatsPerMinute')
plt.title('Distribution of BeatsPerMinute')
plt.ylabel('Count')
plt.xlabel('BeatsPerMinute');


sns.boxplot(x=y)


mi_score = mutual_info_regression(X, y, random_state=7, n_neighbors=500)
mi_score = pd.Series(mi_score, index=X.columns)
mi_score = mi_score[mi_score>0]
mi_score


X = X[mi_score.index]
X_train = X_train[mi_score.index]
X_test = X_test[mi_score.index]
X_train


plt.figure(figsize=(15, 8))

for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 4, i)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.hist(x=X[col])
    plt.title(col)

plt.tight_layout()
plt.show()


X['AudioLoudness'] = np.log(X['AudioLoudness']*-1)
X['AcousticQuality'] = np.log(X['AcousticQuality'])
X['LivePerformanceLikelihood'] = np.log(X['LivePerformanceLikelihood'])


plt.figure(figsize=(15, 8))

for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 7, i)
    plt.xlabel(col)
    sns.boxplot(y=X[col])
    plt.title(col)

plt.tight_layout()
plt.show()


# Handling outluers by clipping
cols_to_clip = ['RhythmScore', 'AcousticQuality', 'TrackDuration_Mins']
for col in cols_to_clip:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(.75)
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    X[col] = X[col].clip(lower_bound, upper_bound)


plt.figure(figsize=(15, 8))

for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 7, i)
    sns.boxplot(y=X[col])
    plt.title(col)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=True, cmap='viridis');


model = GradientBoostingRegressor(n_estimators=300, random_state=63)
model.fit(X_train, y_train)


y_preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print(f'RMSE is {rmse}')


residuals = y_test - y_preds
plt.plot(y_preds, residuals, linestyle='None', marker='.',  color='black')
plt.title('Residual plot')
plt.xlabel('PredictedBeatsPerMinute')
plt.axhline(y=0, linestyle='--');


model.fit(X, y)


test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col=0)
test_data['TrackDuration_Mins'] = test_data['TrackDurationMs']/60000
test_data.drop('TrackDurationMs', axis=1, inplace=True)
test_data = test_data[mi_score.index]
test_data.head()


test_data['AudioLoudness'] = np.log(test_data['AudioLoudness']*-1)
test_data['AcousticQuality'] = np.log(test_data['AcousticQuality'])
test_data['LivePerformanceLikelihood'] = np.log(test_data['LivePerformanceLikelihood'])


for col in cols_to_clip:
    q1 = test_data[col].quantile(0.25)
    q3 = test_data[col].quantile(.75)
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    test_data[col] = test_data[col].clip(lower_bound, upper_bound)



test_preds = model.predict(test_data)
test_preds


sub = pd.DataFrame({'id':test_data.index, 'BeatsPerMinute':test_preds})


sub.to_csv('s9 sub.csv', index=False)


np.mean(test_preds)


feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)
feature_importance


importances = model.feature_importances_
features = X_train.columns 


indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(range(len(importances)), importances[indices], align="center")
plt.yticks(range(len(importances)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Bar Plot")
plt.tight_layout()
plt.show()


