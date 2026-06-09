import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
print(train.shape)
train.head()


train.describe()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(test.shape)
test.head()


X_train = train.drop(['id', 'Calories'], axis = 1)
X_test = test.drop('id', axis = 1)
y_train = train['Calories']
y_train_log = np.log1p(y_train)
X_train = pd.get_dummies(X_train, drop_first = True)
X_test = pd.get_dummies(X_test, drop_first = True)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
X = pd.concat([X_train, X_test])


X.isna().sum()


train_num = train.drop(['id','Sex'], axis = 1)
corr = train_num.corr()
mask = np.triu(corr.corr())
sns.heatmap(corr, annot = True, annot_kws={"size": 8}, fmt = '.3f', cmap = 'coolwarm', square = True, mask = mask, cbar = False)


import statsmodels.api as sm

X_train_c = sm.add_constant(X_train)
lin_model_0 = sm.OLS(y_train_log, X_train_c).fit()
lin_model_0.summary()


from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_num = X_train.drop('Sex_male', axis = 1)

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train_num.values, i) for i in range(X_train_num.shape[1])]
vif["features"] = X_train_num.columns 

vif = vif.sort_values(by="VIF Factor", ascending=False)
vif = vif.reset_index().drop(columns='index')
vif[vif["VIF Factor"]>10]


X_train_1 = X_train[['Heart_Rate', 'Duration']]
X_train_1_c = sm.add_constant(X_train_1)
lin_model_1 = sm.OLS(y_train_log, X_train_1_c).fit()
lin_model_1.summary()


vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train_1.values, i) for i in range(X_train_1.shape[1])]
vif["features"] = X_train_1.columns 

vif = vif.sort_values(by="VIF Factor", ascending=False)
vif = vif.reset_index().drop(columns='index')
vif[vif["VIF Factor"]>10]


from sklearn.model_selection import train_test_split
X1_train, X1_val, y1_train, y1_val = train_test_split(X_train_1, y_train_log, test_size = 0.2, random_state = 32)


X1_train_c = sm.add_constant(X1_train)
X1_val_c = sm.add_constant(X1_val)

lin_model_val = sm.OLS(y1_train, X1_train_c).fit()

y_pred = lin_model_val.predict(X1_val_c)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y1_val, y_pred)
r2 = r2_score(y1_val, y_pred)
n = X1_val_c.shape[0]
p = X1_val_c.shape[1]-1
r2_adj = 1-(1-r2)*(n-1)/(n-p-1)

print(f"검증 데이터 MSE: {mse:.3f}")
print(f"검증 데이터 R-Squared: {r2:.3f}")
print(f"검증 데이터 Adj R-Squared: {r2_adj:.3f}")


X_test_1 = X_test[['Heart_Rate', 'Duration']]
X_test_1_c = sm.add_constant(X_test_1)
pred_log = lin_model_1.predict(X_test_1_c)
pred = np.expm1(pred_log)
sub = pd.DataFrame()
sub['id'] = test['id']
sub['Calories'] = pred
sub['Calories'] = sub['Calories'].apply(lambda x: 1 if x<0 else x)
sub.to_csv('submission.csv', index = False)

