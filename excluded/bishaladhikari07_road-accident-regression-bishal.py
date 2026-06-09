# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_train.head() 
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_test.head()
df=pd.concat([df_train,df_test],ignore_index=True)
df.shape


df.head()


# plt.figure(figsize=(8,6))
# sns.histplot(data=df, x='num_lanes', bins=10, kde=True)
# plt.title('Accident Frequency by Time of Day')
# plt.xlabel('Time of Day')
# plt.ylabel('Number of Accidents')
# plt.show()



plt.figure(figsize=(8,6))
sns.countplot(data=df, x='time_of_day')
plt.title('Accident Frequency by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Accidents')
plt.show()



# corr = df.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='num_lanes', y='num_reported_accidents')
plt.title('Number of Lanes vs accident_risk')
plt.show()



df['weather'].value_counts()


df['curve_speed'] = df['curvature'] * df['speed_limit']
df['high_speed_limit'] = (df['speed_limit'] > 47).astype(int)
df['high_curvature'] = (df['curvature'] > 0.6).astype(int)

# df['curve_accident_ratio']=df['num_reported_accidents']/(df['curve_speed']==0?0.001:df['curve_speed'])
df['curve_accident_ratio'] = df['num_reported_accidents'] / df['curve_speed'].replace(0, 0.001)



# Example: bad_weather = True if weather is 'rainy' or 'foggy'
df['bad_weather'] = df['weather'].isin(['rainy', 'foggy']).astype(int)
# Assume df['road'] has values like 'urban', 'highway', 'rural'
df['developedarearoad'] = df['road_type'].isin(['urban', 'highway']).astype(int)
df['badlighting']=df['lighting'].isin(['dim','night']).astype(int)



df['time_of_day'].value_counts()





binaryCol=['road_signs_present','public_road','holiday','school_season']

for col in binaryCol:
    df[col]=df[col].map({True:1,False:0})


df_train=df[:517754]
df_test=df[517754:]





df_train_x=df_train.drop(['accident_risk'],axis=1)
# df_train_x.head()
df_train_y=df_train['accident_risk']
df_train_y.head()


df_train_x.shape


df_train_y.shape


df_test_x=df_test.drop(['accident_risk'],axis=1)
df_test_x.head()


df_train_x_MI=df_train_x.drop(['school_season','holiday','public_road','road_signs_present','num_lanes','time_of_day','id'
],axis=1)
df_test_x_MI=df_test_x.drop(['school_season','holiday','public_road','road_signs_present','num_lanes','time_of_day','id'
],axis=1)

# road_type


# curve_speed               0.334587
# curvature                 0.296545
# speed_limit               0.151669
# curve_accident_ratio      0.146754
# lighting                  0.121404
# num_reported_accidents    0.062134
# weather                   0.048963
# bad_weather               0.039637
# badlighting               0.021265
# road_type                 0.013194
# developedarearoad         0.008718
# num_lanes                 0.007583
# school_season             0.000000
# holiday                   0.000000
# public_road               0.000000
# road_signs_present        0.000000
# time_of_day               0.000000
# Name: MI Scores, dtype: float64


# Utility functions from Tutorial
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd

def make_mi_scores(X, y, n_neighbors=2, sample_size=10000):
    X = X.copy()
    
    # ðŸ”¹ Drop the ID column (it has no predictive power)
    X = X.drop(columns=['id'], errors='ignore')
    
    # ðŸ”¹ Random sample to speed up (optional)
    if len(X) > sample_size:
        idx = np.random.choice(X.index, sample_size, replace=False)
        X = X.loc[idx]
        y = y.loc[idx]
    
    # ðŸ”¹ Factorize categorical columns (convert object â†’ numeric codes)
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    
    # ðŸ”¹ Drop constant columns (same value everywhere)
    X = X.loc[:, X.nunique() > 1]
    
    # ðŸ”¹ Detect discrete features
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    # ðŸ”¹ Calculate Mutual Information
    mi_scores = mutual_info_regression(
        X, y,
        discrete_features=discrete_features,
        random_state=0,
        n_neighbors=n_neighbors
    )
    
    # ðŸ”¹ Return as a sorted Series
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


df_train_x.dtypes


df_train_x.isna().sum()


make_mi_scores(df_train_x_MI[0:5000],df_train_y[0:5000])


df_train_x_MI[0:5000]


# df_train_x_MI['curvature'].max()


df_train_y[0:5000]


df['speed_limit'].value_counts()


df_train_x_MI.head()





# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# numerical_col=['curvature','num_reported_accidents','speed_limit']

# categorical_features=["lighting","weather","time_of_day"]

# preprocessor=ColumnTransformer([
#     ('cat',OneHotEncoder(),categorical_features),
#     ('num',StandardScaler(),numerical_col)
# ],remainder='passthrough')

# df_train_x_transformed=preprocessor.fit_transform(df_train_x_MI)
# df_test_x_transformed=preprocessor.transform(df_test_x_MI)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Explicitly drop or encode all categorical features
numerical_col = ['curvature', 'num_reported_accidents', 'speed_limit', 'public_road','curve_speed']  # include binary numeric
categorical_features = ['lighting', 'weather', 'time_of_day']

# Make sure only columns that exist are included
numerical_col = [col for col in numerical_col if col in df_train_x_MI.columns]
categorical_features = [col for col in categorical_features if col in df_train_x_MI.columns]

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_col)
], remainder='drop')  # DROP any remaining columns like id, road_type, etc.

# Fit-transform
df_train_x_transformed = preprocessor.fit_transform(df_train_x_MI)
df_test_x_transformed = preprocessor.transform(df_test_x_MI)



 from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df_train_x_transformed,df_train_y,test_size=0.2,random_state=42)

LRMODEL=LinearRegression().fit(X_train,y_train)


LRMODEL.score(X_test,y_test)


LRMODEL.score(X_train,y_train)


# from xgboost import XGBRegressor
# XGMODEL=XGBRegressor().fit(X_train,y_train)
# XGMODEL.score(X_test,y_test)
0.8848648426158833



from xgboost import XGBRegressor

XGMODEL = XGBRegressor(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    subsample=1.0,
    reg_lambda=1.0,   # L2 regularization (Ridge)
    reg_alpha=0.1     # L1 regularization (Lasso)
)

XGMODEL.fit(X_train, y_train)
score = XGMODEL.score(X_test, y_test)
print("RÂ² Score:", score)



XGMODEL.score(X_train,y_train)


from sklearn.metrics import mean_squared_error, r2_score
y_lrpred=LRMODEL.predict(X_test)
mse = mean_squared_error(y_test, y_lrpred)
r2 = r2_score(y_test, y_lrpred)
print("Test linear model  MSE:", mse)
print("Test linear model R2:", r2)


xgpred=XGMODEL.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, xgpred)
r2 = r2_score(y_test, xgpred
             )
print("Test XG MSE:", mse)
print("Test  XG R2:", r2)


# Test MSE: 0.0031766106259266667
# Test R2: 0.8849562408733649


in 100 estmator
E: 0.0031673639455314802
Test R2: 0.8852911175697477


Test XG MSE: 0.003179134332367689
Test  XG R2: 0.8848648426158833


# from sklearn.model_selection import GridSearchCV

# params_grid={
#     'model__n_estimators':[100,200,300],
#     'model__max_depth':[3,5,7],
#     'model__learning_rate':[0.01,0.1,0.3],
#     'model__subsample':[0.8,1.0],
#     'model__colsample_bytree':[0.8,1.0]
# }

# grid_search=GridSearchCV(XGBRegressor(),params_grid,cv=3,scoring='r2',n_jobs=-1)

# grid_search.fit(X_train,y_train)
# print("Best Parameters:", grid_search.best_params_)
# print("Best CV Accuracy:", grid_search.best_score_)
from sklearn.model_selection import GridSearchCV
params_grid = {
    'n_estimators':[100,150,400],
    'max_depth':[3,5,12],
    'learning_rate':[0.001,0.1,0.3],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}

grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                           params_grid, cv=3, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)



grid_search


# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 400, 'subsample': 1.0}



best_model = grid_search.best_estimator_



best_model.score(X_train,y_train)


best_model.score(X_test,y_test)


y_preds=best_model.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_preds)
r2 = r2_score(y_test, y_preds)
print("Test MSE:", mse)
print("Test R2:", r2)


# Test MSE: 0.0031766106259266667
# Test R2: 0.8849562408733649


in 100 estmator
E: 0.0031673639455314802
Test R2: 0.8852911175697477


submission=pd.DataFrame({
    "id":df_test['id'],
    "accident_risk":best_model.predict(df_test_x_transformed)
})
print(submission)
submission.to_csv('submission.csv',index=False)
submission.head(2)




