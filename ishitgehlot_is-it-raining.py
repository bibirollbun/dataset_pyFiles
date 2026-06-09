import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_1samp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.head()


train.info()


train.isnull().sum()


train.describe()


cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']


plt.figure(figsize = (15,12))
for i in list(enumerate(cols)):
  plt.subplot(3,4, i[0]+1)
  sns.histplot(data = train, x = i[1], kde = True)


plt.figure(figsize = (14,11))
for i in enumerate(cols):
  plt.subplot(3,4, i[0]+1)
  sns.histplot(data = train, x = i[1], hue = 'rainfall', kde = True)


plt.figure(figsize = (12,12))
for i in enumerate(cols):
  plt.subplot(4,3,i[0]+1)
  sns.boxplot(data = train, x = i[1])


train['sin_day'] = np.sin( (2 * np.pi * train['day']) / 365)
train['cos_day'] = np.cos((2 * np.pi * train['day']) / 365)
test['sin_day'] = np.sin( (2 * np.pi * test['day']) / 365)
test['cos_day'] = np.cos((2 * np.pi * test['day']) / 365)


train.drop(columns='day', inplace = True)
test.drop(columns='day', inplace = True)
train.drop(columns='id', inplace = True)
test.drop(columns='id', inplace = True)


train.head()


train.info()


sns.countplot(x = train['rainfall'])


train.columns


train['cloud_humid'] = train['cloud'] * train['humidity']
test['cloud_humid'] = test['cloud'] * test['humidity']


train['dew_pt_dep'] = train['temparature'] - train['dewpoint']
test['dew_pt_dep'] = test['temparature'] - test['dewpoint']


train['temp_variation'] = train['maxtemp'] - train['mintemp']
test['temp_variation'] = test['maxtemp'] - test['mintemp']


train['feel_like_temp'] = train['temparature'] + 0.33*train['humidity'] - 0.7*train['windspeed'] - 4
test['feel_like_temp'] = test['temparature'] + 0.33*test['humidity'] - 0.7*test['windspeed'] - 4


train.head()


sns.histplot(x =train['cloud_humid'], hue = train['rainfall'])


temp_at_rain = train[train['rainfall'] != 0]['temparature']


sns.kdeplot(x = temp_at_rain)


shap = shapiro(temp_at_rain)
shap


samples = []
for i in range(15):
  samples.append(temp_at_rain.sample(30).values)


samples = np.array(samples)


# sample distribution of sampling distribution
samples = samples.mean(axis = 1)


sns.kdeplot(x = samples)


tstat , pval = ttest_1samp(temp_at_rain, 24)
print(f't_statistic = {tstat} p_value = {pval}', end= '\n')


train.head()


X = train.drop(columns='rainfall')
y = train['rainfall']


xtrain, xtest, ytrain, ytest = train_test_split(X,y , random_state=42, test_size= 0.2)


pipe = Pipeline(
    [
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression())
    ]
)


pipe.fit(xtrain, ytrain)


ypred = pipe.predict(xtest)


roc_auc_score(ytest, ypred)


smote = SMOTE(sampling_strategy='auto', random_state=42)


X_smote, ysmote = smote.fit_resample(X,y)


x_train , x_test, y_train, y_test = train_test_split(X_smote, ysmote, random_state=42 , test_size=0.2)


pipe2 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(n_estimators=55))
])


pipe2.fit(x_train, y_train)


y_pred = pipe2.predict_proba(x_test)[:,1]


roc_auc_score(y_test, y_pred)


test['winddirection'].fillna(test['winddirection'].mean(), inplace = True)


# logistic 83.76
# gradient boost 85.89
# DT 85.78 maxdepth 9
# random forest :- 89.83 , n_estim 50
# SVC :- 89.71
# randomforest :- n_estim = 55, reaching to 90


import xgboost as  xgb
# from sklearn.tree import DecisionTreeClassifier
pipe3 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', xgb.XGBClassifier())
])


pipe3.fit(x_train, y_train)


preds = pipe3.predict_proba(x_test)[:,1]


roc_auc_score(y_test, preds)


pred = pipe3.predict_proba(test)[:,1]


submission['rainfall'] = pred


from sklearn.metrics import roc_curve, auc


# roc_curve(y_test, preds)
fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
#precision, recall, thresholds_pr = precision_recall_curve(y_test, rf_pred_proba)


#print("AUC:", roc_auc)

#plt.figure(figsize=(12, 5))
fig = plt.figure(figsize=(12, 5), dpi=100) 
# fig.patch.set_facecolor(background_color) 

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


submission


submission.to_csv('submission.csv', index = False)
print('success!')

