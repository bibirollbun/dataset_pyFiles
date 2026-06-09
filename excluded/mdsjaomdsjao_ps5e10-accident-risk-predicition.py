import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',)
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head(3)


train.sample(2)


test.head(3)


train.isnull().sum()


train.info()


train["accident_risk"].describe()


train.columns


num = ['num_lanes','speed_limit']
boolean = ['road_signs_present','public_road','holiday','school_season']
categorical = ['road_type','lighting','weather','time_of_day']


names = [ 'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents','accident_risk']

fig, axs = plt.subplots(1, 5, figsize=(8, 3)) 
for i in range(0,5):
    axs[i].hist(train[names[i]], bins=20, color='lightblue', edgecolor='black')
    axs[i].set_title(names[i])

plt.suptitle('Feature values Distribution', fontsize= 15)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10,6))


cols = boolean+num+['accident_risk']
corr = train[cols].corr()

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap',fontsize = 15, pad=10)
plt.tight_layout()
plt.show()


X = train.drop(['id','accident_risk'], axis=1)
y = train['accident_risk']

X_test = test.copy().drop(columns=['id'], axis=1)


from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


#preprocessor

prep = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                                    ('num', StandardScaler(), num)],
                                    remainder='passthrough')


model = Pipeline(steps = [('preprocessor', prep),
                            ('model', LinearRegression(fit_intercept = True))
                            ])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=42, test_size = 0.3)

model.fit(X_train, y_train)


y_pred = model.predict(X_val)


#some metrics

from sklearn.metrics import mean_squared_error, r2_score

print(f'MODEL [LINEAR REGRESSION] mean_squared_error: {round(mean_squared_error(y_val, y_pred),3)}\nMODEL [LINEAR REGRESSION] nr2_score: {round(r2_score(y_val, y_pred),3)}')


y_pred = model.predict(X_test)


submission['accident_risk'] = y_pred

submission.to_csv('submission.csv',index = False)




submission.head(4)

