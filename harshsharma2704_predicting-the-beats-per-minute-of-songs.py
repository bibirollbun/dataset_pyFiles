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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test =pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print(df.isna().mean()*100)
print()
print(df.info())


df.columns


eps = 1e-6

# ----- Train Features -----
df["LogDuration"] = np.log1p(df["TrackDurationMs"])
df["vocal_energy"] = df["VocalContent"] * df["Energy"]
df["loudness_squared"] = df["AudioLoudness"] ** 2
df["vocal_density"] = df["VocalContent"] / (df["TrackDurationMs"] + eps)
df["mood_rhythm"] = df["MoodScore"] * df["RhythmScore"]
df["rhythm_instrumental"] = df["RhythmScore"] * df["InstrumentalScore"]
df["instrumental_ratio"] = df["InstrumentalScore"] / (df["AcousticQuality"] + eps)
df["rhythm_normalized"] = df["RhythmScore"] / (df["LogDuration"] + eps)
df["TrackDurationMin"] = df["TrackDurationMs"] / 60000.0
df["TrackDurationSec"] = df["TrackDurationMs"] / 1000.0
df["Rhythm_Energy"] = df["RhythmScore"] * df["Energy"]
df["Acoustic_Vocal"] = df["AcousticQuality"] * df["VocalContent"]
df["Mood_Live"] = df["MoodScore"] * df["LivePerformanceLikelihood"]
df["Energy_over_Rhythm"] = df["Energy"] / (df["RhythmScore"] + eps)
df["Vocal_over_Acoustic"] = df["VocalContent"] / (df["AcousticQuality"] + eps)

# ----- Test Features -----
test["LogDuration"] = np.log1p(test["TrackDurationMs"])
test["vocal_energy"] = test["VocalContent"] * test["Energy"]
test["loudness_squared"] = test["AudioLoudness"] ** 2
test["vocal_density"] = test["VocalContent"] / (test["TrackDurationMs"] + eps)
test["mood_rhythm"] = test["MoodScore"] * test["RhythmScore"]
test["rhythm_instrumental"] = test["RhythmScore"] * test["InstrumentalScore"]
test["instrumental_ratio"] = test["InstrumentalScore"] / (test["AcousticQuality"] + eps)
test["rhythm_normalized"] = test["RhythmScore"] / (test["LogDuration"] + eps)
test["TrackDurationMin"] = test["TrackDurationMs"] / 60000.0
test["TrackDurationSec"] = test["TrackDurationMs"] / 1000.0
test["Rhythm_Energy"] = test["RhythmScore"] * test["Energy"]
test["Acoustic_Vocal"] = test["AcousticQuality"] * test["VocalContent"]
test["Mood_Live"] = test["MoodScore"] * test["LivePerformanceLikelihood"]
test["Energy_over_Rhythm"] = test["Energy"] / (test["RhythmScore"] + eps)
test["Vocal_over_Acoustic"] = test["VocalContent"] / (test["AcousticQuality"] + eps)

# ----- Prepare X, y -----
X = df.drop(columns=["BeatsPerMinute", "id"])
y = df["BeatsPerMinute"]

test_id = test["id"]
test.drop(columns=["id"], inplace=True)



num_cols = X.select_dtypes(include=[np.number]).columns


for i in num_cols:
    plt.figure(figsize=(15,8))
    plt.subplot(131)
    sns.boxplot(x=X[i])
    plt.subplot(132)
    sns.kdeplot(x=X[i])
    plt.subplot(133)
    sns.histplot(x=df[i],bins=30)


for i in num_cols:
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    sns.scatterplot(x=df[i],y=df['BeatsPerMinute'])
    plt.subplot(122)
    sns.histplot(x=X[i],bins=30)


for i in num_cols:
    Q1 = X[i].quantile(0.25)
    Q3 = X[i].quantile(0.75)
    iqr = Q3-Q1
    lower = Q1-1.5*iqr
    upper = Q3+ 1.5*iqr
    X[i].clip(lower = lower,upper = upper)
    test[i].clip(lower=lower,upper=upper)


X_train,X_valid,y_train,y_valid  = train_test_split(X,y,test_size=0.2,random_state=6)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,RobustScaler,PowerTransformer

num_pipe = Pipeline([
    ('scale_power',PowerTransformer(method='yeo-johnson')),
    # ('scale',RobustScaler())
])

preprocess = ColumnTransformer([
    ('num',num_pipe,num_cols)
])


import tensorflow
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,Input


model = Sequential([
    Input(shape=(24,)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(6,activation='relu'),
    Dense(5,activation='relu'),
    Dense(1,activation='linear')
])

model.summary()



model.compile(loss='mse',optimizer='Adam',metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train,y_train,epochs=15,validation_split=0.2)


y_pred = model.predict(X_valid)


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_valid,y_pred))


y_train_pred = model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_train_pred))


from catboost import CatBoostRegressor

model_cat= Pipeline([
    ('pre',preprocess),
    ('algo',CatBoostRegressor(n_estimators=350,learning_rate=0.15,depth=4,verbose=0))
])

model_cat.fit(X_train,y_train)
y_pred_cat = model_cat.predict(X_valid)
y_pred_train = model_cat.predict(X_train)
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_cat)))
print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


feature_values = model_cat.named_steps['algo'].feature_importances_
sns.barplot(y=X_train.columns,x=feature_values)


model_gb= Pipeline([
    ('pre',preprocess),
    ('algo',HistGradientBoostingRegressor(max_iter=250,max_depth=4,learning_rate=0.05))
])

model_gb.fit(X_train,y_train)
y_pred_gb = model_gb.predict(X_valid)
y_pred_train = model_gb.predict(X_train)
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_gb)))
print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


from xgboost import XGBRegressor

model_xgb= Pipeline([
    ('pre',preprocess),
    ('algo',XGBRegressor(n_estimators=350,min_child_weight=10,reg_alpha=10,reg_lambda=0,subsample=0.8,max_depth=5,learning_rate=0.05))
])

model_xgb.fit(X_train,y_train)
y_pred_xgb = model_xgb.predict(X_valid)
y_pred_train = model_xgb.predict(X_train)
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_xgb)))
print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


# feature_values = model_xgb.named_steps['algo'].feature_importances_

# sns.barplot(y=X_train.columns,x=feature_values)


# import optuna
# from xgboost import XGBRegressor
# from sklearn.linear_model import SGDRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from lightgbm import LGBMRegressor
# X_train,X_valid,y_train,y_valid  = train_test_split(X,y,test_size=0.2,random_state=6)

# def objective(trial):
#     params={
#         'learning_rate':trial.suggest_float('learning_rate',0.01,0.4),
#         'n_estimators':trial.suggest_int('n_estimators',150,650),
#         'max_depth':trial.suggest_int('max_depth',3,12),
#         'n_jobs':-1
#     }

#     model = Pipeline([
#         ('scale',PowerTransformer(method='yeo-johnson')),
#         ('algo',XGBRegressor(**params))
#     ])

#     model.fit(X_train,y_train)
#     y_valid_pred = model.predict(X_valid)
#     rmse_valid = np.sqrt(mean_squared_error(y_valid,y_valid_pred))
#     return rmse_valid

# study= optuna.create_study(direction='minimize')
# study.optimize(objective,n_trials=50)
# print("Best trial finished with value: ", study.best_value)
# print("Best hyperparameters found: ", study.best_params)



# from sklearn.ensemble import RandomForestRegressor
# model_rf = Pipeline([
#     ('preprocess',preprocess),
#     ('algo',RandomForestRegressor(n_estimators=250,max_depth=4,bootstrap=True,oob_score=True,n_jobs=-1))
# ])

# model_rf.fit(X_train,y_train)
# y_pred_train = model_rf.predict(X_train)
# y_pred_valid = model_rf.predict(X_valid)
# print('validation score',np.sqrt((mean_squared_error(y_train,y_pred_train))))
# print('training sccore',np.sqrt(mean_squared_error(y_train,y_pred_valid)))


# model_rf.named_steps['algo'].oob_score_


from lightgbm import LGBMRegressor
model_lgb= Pipeline([
    ('pre',preprocess),
    ('algo',LGBMRegressor(n_estimators=350,max_depth=8,min_child_samples=20,subsample=0.85,reg_alpha=0.5,reg_lambda=0.5,n_jobs=-1,learning_rate=0.025,verbose=-1,num_leaves=47))
])

model_lgb.fit(X_train,y_train)
y_pred_lgb = model_lgb.predict(X_valid)
y_pred_train = model_lgb.predict(X_train)
print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_lgb)))
print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


score = cross_val_score(model_lgb,X,y,cv=5,scoring='neg_root_mean_squared_error')
-score.mean()


feature_values = model_lgb.named_steps['algo'].feature_importances_
sns.barplot(y=X_train.columns,x=feature_values)


# from sklearn.neural_network import MLPRegressor

# model_mlp= Pipeline([
#     ('pre',preprocess),
#     ('algo',MLPRegressor(hidden_layer_sizes=(50,20),activation='relu',alpha=0.1,learning_rate_init=0.15,max_iter=500,solver='adam'))
# ])

# model_mlp.fit(X_train,y_train)
# y_pred_mlp = model_mlp.predict(X_valid)
# y_pred_train = model_mlp.predict(X_train)
# print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_mlp)))
# print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


# from sklearn.tree import DecisionTreeRegressor
# model_dt= Pipeline([
#     ('pre',preprocess),
#     ('algo',DecisionTreeRegressor(criterion='squared_error',max_depth=5,min_samples_leaf=50,min_samples_split=20))
# ])

# model_dt.fit(X_train,y_train)
# y_pred_dt = model_dt.predict(X_valid)
# y_pred_train = model_dt.predict(X_train)
# print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_dt)))
# print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


# from sklearn.ensemble  import BaggingRegressor
# from sklearn.linear_model import SGDRegressor,LogisticRegression
# from sklearn.tree import DecisionTreeRegressor
# model_bag= Pipeline([
#     ('pre',preprocess),
#     ('algo',BaggingRegressor(estimator=RandomForestRegressor(criterion='squared_error',max_depth=5,min_samples_leaf=20,min_samples_split=80),max_features=0.8,bootstrap=True,oob_score=True))
# ])

# model_bag.fit(X_train,y_train)
# y_pred_bag = model_bag.predict(X_valid)
# y_pred_train = model_bag.predict(X_train)
# print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_bag)))
# print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))



# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import SGDRegressor,Ridge

# estimators = [
#     ('xgb',XGBRegressor(n_estimators=450,max_depth=5,learning_rate=0.01)),
#     ('lgb',LGBMRegressor(n_estimators=500,max_depth=5,learning_rate=0.08,verbose=-1)),
#     ('cat',CatBoostRegressor(n_estimators=550,learning_rate=0.22,depth=4,verbose=0))
# ]

# model_stack= Pipeline([
#     ('pre',preprocess),
#     ('algo',StackingRegressor(estimators=estimators,final_estimator=Ridge(),cv=4,n_jobs=-1))
# ])


# model_stack.fit(X_train,y_train)
# y_pred_stack = model_stack.predict(X_valid)
# y_pred_train = model_stack.predict(X_train)
# print('validation score',np.sqrt(mean_squared_error(y_valid,y_pred_stack)))
# print('training score',np.sqrt(mean_squared_error(y_train,y_pred_train)))


test_pred = model.predict(test)
submit = pd.DataFrame({'id':test_id,'BeatsPerMinute':test_pred.ravel()})
submit.to_csv('submission.csv',index=False)


result = pd.read_csv('/kaggle/working/submission.csv')
result.head()




