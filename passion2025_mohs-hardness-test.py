import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')


df_train.head()


df_test.head()


print(df_train.shape)
print(df_test.shape)


df_train.hist(bins=50, figsize=(20, 15)) #用hist畫出直方圖
plt.show()


df_test.hist(bins=50, figsize=(20, 15)) #用hist畫出直方圖
plt.show()


df_train.info()


df_train.isna().sum()


df_test.info()


df_test.isna().sum()


df_train.describe()


from pandas.plotting import scatter_matrix

scatter_matrix(df_train, figsize=(25, 25), alpha=0.9, #用scatter_matrix對train_set畫數值型資料的圖
        diagonal="kde", marker='.') #diagonal="kde"自己分佈畫在對角線畫一個密度函數的估計
plt.show()


import seaborn as sns

#correlation matrix

df = pd.DataFrame(df_train)
corr_matrix = df.corr()
#print(corr_matrix)

#sns.heatmap(corr_matrix, annot=True)
#plt.show()
plt.figure(figsize=(13,13))
sns.heatmap(df.corr(method="pearson"), vmin=-1, vmax=1,annot=True,fmt=".4f")


corr = df_train.corr(numeric_only = True) #用corr這個函數來看correlation(相關係數)
corr['Hardness'].sort_values(ascending=False) #只把features跟y的相關係數叫出來看，sort_values(ascending=False)做排序由大到小
#跟y相關係數最高為el_neg_chi_Average


df_train.plot(kind='scatter', x='el_neg_chi_Average', y='Hardness', alpha=0.2)
plt.show()


x_train = df_train.drop(['id','Hardness'], axis=1) #x訓練資料
y_train = df_train['Hardness'].copy()


print(x_train.shape)
print(y_train.shape)


x_train


y_train


testing_data = df_test.drop(['id'], axis=1)


print(testing_data.shape)


testing_data


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
x_train_prep = std_scaler.fit_transform(x_train)
x_test_prep = std_scaler.transform(x_test)
x_train_prep, x_test_prep


testing_data_prep = std_scaler.fit_transform(x_test)
testing_data_prep


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train_prep, y_train)
lin_reg.score(x_train_prep, y_train)


lin_reg.score(x_test_prep, y_test)


# predict
y_pred = lin_reg.predict(x_test_prep)
y_pred


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


from sklearn.metrics import median_absolute_error

MedAE = median_absolute_error(y_test, y_pred)
print("MedAE:", MedAE)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pip_lin_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scal', StandardScaler()),
    ('poly_lin_reg', LinearRegression())
    ])
pip_lin_reg.fit(x_train, y_train)
pip_lin_reg.score(x_train, y_train)


pip_lin_reg.score(x_test, y_test)


# predict
y_pred = lin_reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


MedAE = median_absolute_error(y_test, y_pred)
print("MedAE:", MedAE)


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

Rid = Ridge()
# set_params
param_grid = {'alpha': [0.1, 1, 10],
         'solver':['svd','cholesky','lsqr','sparse_cg']
              }
# GridSearchCV 
grid_search_rid = GridSearchCV(Rid, param_grid, cv=5)
grid_search_rid.fit(x_train_prep, y_train)


#mean validation score
grid_search_rid.best_score_


# best_params
grid_search_rid.best_params_


# model
model_rid = grid_search_rid.best_estimator_ #用所有training data與best_params_訓練出的model
model_rid.score(x_test_prep, y_test)


# predict
y_pred = model_rid.predict(x_test_prep)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.linear_model import SGDRegressor

Sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=.1)
# set_params
param_grid = {'alpha':[0.1, 1, 10],
        'penalty':['l1', 'l2', 'elasticnet']     
              }
# GridSearchCV
grid_search_sgd = GridSearchCV(Sgd, param_grid, cv=5)
grid_search_sgd.fit(x_train_prep, y_train.ravel())


#mean validation score
grid_search_sgd.best_score_


# best_params
grid_search_sgd.best_params_


# model
model_Sgd = grid_search_sgd.best_estimator_ #用所有training data與best_params_訓練出的model
model_Sgd.score(x_test_prep, y_test)


# predict
y_pred = model_Sgd.predict(x_test_prep)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

ridge_sgd = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scal', StandardScaler()),
    ('ridge', SGDRegressor(penalty='l2', random_state=1)) #ridge的penalty要用l2
])
# set_params
param_grid = {'poly__degree':[2, 3, 4, 5],
        'ridge__alpha':[0.1, 1, 10],
              }
# GridSearchCV
grid_search_ridge_sgd = GridSearchCV(ridge_sgd, param_grid, cv=5)
grid_search_ridge_sgd.fit(x_train, y_train.ravel())


#mean validation score
grid_search_ridge_sgd.best_score_


# best_params
grid_search_ridge_sgd.best_params_


# model
model_ridge_sgd = grid_search_ridge_sgd.best_estimator_ #用所有training data與best_params_訓練出的model
model_ridge_sgd.score(x_test, y_test)


# predict
y_pred = model_ridge_sgd.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


poly_sgd = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scal', StandardScaler()),
    ('sgd', SGDRegressor(max_iter=1000, tol=1e-3, eta0=.1, random_state=1)) #ridge的penalty要用l2
])
# set_params
param_grid = {'poly__degree':[2, 3, 4, 5],
        'sgd__alpha':[0.1, 1, 10],
        'sgd__penalty':['l1', 'l2', 'elasticnet']
              }
# GridSearchCV
grid_search_poly_sgd = GridSearchCV(poly_sgd, param_grid, cv=5)
grid_search_poly_sgd.fit(x_train, y_train.ravel())


#mean validation score
grid_search_poly_sgd.best_score_


# best_params
grid_search_ridge_sgd.best_params_


# model
model_poly_sgd = grid_search_poly_sgd.best_estimator_ #用所有training data與best_params_訓練出的model
model_poly_sgd.score(x_test, y_test)


# predict
y_pred = model_ridge_sgd.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.linear_model import Lasso
lasso_reg = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scal', StandardScaler()),
    ('lasso', Lasso(random_state=1))
])
# set_param
param_grid = {'poly__degree':[2, 3, 4, 5],
        'lasso__alpha':[0.1, 1, 10],
              }
# GridSearchCV
grid_search_lasso_reg = GridSearchCV(lasso_reg, param_grid, cv=5)
grid_search_lasso_reg.fit(x_train, y_train)


#mean validation score
grid_search_lasso_reg.best_score_


# best_params
grid_search_lasso_reg.best_params_


# model
model_lasso_reg = grid_search_lasso_reg.best_estimator_ #用所有training data與best_params_訓練出的model
model_lasso_reg.score(x_test, y_test)


# predict
y_pred = model_lasso_reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.svm import LinearSVR

lin_svr = LinearSVR(epsilon=0.5, random_state=1)
lin_svr.fit(x_train_prep, y_train)
lin_svr.score(x_train_prep, y_train)


lin_svr.score(x_test_prep, y_test)


# predict
y_pred = lin_reg.predict(x_test_prep)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn import svm

# kernel='linear'
svr_linear = svm.SVR(C=1, kernel='linear')
svr_linear.fit(x_train_prep, y_train)
svr_linear.score(x_train_prep, y_train)


svr_linear.score(x_test_prep, y_test)


# predict
y_pred = svr_linear.predict(x_test_prep)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


svr_non_lin = svm.SVR(epsilon=0.1)

# set_params
param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 5, 10],
    'gamma': ['scale', 'auto']
}

# GridSearchCV
grid_search_svr_non_lin = GridSearchCV(svr_non_lin, param_grid, cv=5)
grid_search_svr_non_lin.fit(x_train_prep, y_train)


#mean validation score
grid_search_svr_non_lin.best_score_


# best_params
grid_search_svr_non_lin.best_params_


# model
model_svr_non_lin = grid_search_svr_non_lin.best_estimator_ #用所有training data與best_params_訓練出的model
model_svr_non_lin.score(x_test_prep, y_test)


# predict
y_pred = model_svr_non_lin.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.neural_network import MLPRegressor

mlp_reg = MLPRegressor()

# set_param
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.01, 0.1, 1]
}

# GridSearchCV 
grid_search_mlp = GridSearchCV(mlp_reg, param_grid, cv=5)
grid_search_mlp.fit(x_train_prep, y_train)


#mean validation score
grid_search_mlp.best_score_


# best_params
grid_search_mlp.best_params_


# model
model_mlp_reg = grid_search_mlp.best_estimator_ #用所有training data與best_params_訓練出的model
model_mlp_reg.score(x_test_prep, y_test)


# predict
y_pred = model_mlp_reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model_nn = Sequential()
model_nn.add(Dense(10, activation='tanh', input_shape=(11,))) #用11個feature來看，第一層hidden layer:10
#model.add(Dense(28, activation='tanh')) 
model_nn.add(Dense(60, activation='relu')) #第二層hidden layer:60
model_nn.add(Dense(1))  #output layer
model_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model_nn.summary()


#Fit the model
history = model_nn.fit(x_train_prep, y_train, epochs=35,batch_size=16)


plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'])


#Evaluate the model
model_nn.evaluate(x_test_prep, y_test, verbose=0)


plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy'])


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=1)
# set_params
param_grid = {
    'max_depth': [5, 6, 10],
}

# GridSearchCV
grid_search_tree_reg = GridSearchCV(tree_reg, param_grid, cv=5)
grid_search_tree_reg.fit(x_train, y_train)


#mean validation score
grid_search_tree_reg.best_score_


# best_params
grid_search_tree_reg.best_params_


# model
model_tree_reg = grid_search_tree_reg.best_estimator_ #用所有training data與best_params_訓練出的model
model_tree_reg.score(x_test, y_test)


# predict
y_pred = model_tree_reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

tree_reg = DecisionTreeRegressor(max_depth=6)

ada_reg = AdaBoostRegressor(estimator=tree_reg, random_state=1)
#estimator設定為tree_reg
#n_estimators=200代表執行200次DecisionTreeRegressor

# set_params
param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.1, 0.5]
}

# GridSearchCV
grid_search_ada_reg = GridSearchCV(ada_reg, param_grid, cv=5)
grid_search_ada_reg.fit(x_train, y_train)


#mean validation score
grid_search_ada_reg.best_score_


# best_params
grid_search_ada_reg.best_params_


# model
model_ada_reg = grid_search_ada_reg.best_estimator_ #用所有training data與best_params_訓練出的model
model_ada_reg.score(x_test, y_test)


# predict
y_pred = model_ada_reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


#先決定要用哪幾個方法來做ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

svr_pip = Pipeline([
    ('scaler', StandardScaler()),
    ('svr_non', svm.SVR(epsilon=0.1,
    C=5, gamma='auto', kernel='rbf'))
])


estimators = [
    ('svr_pip', svr_pip),
    ('dt', DecisionTreeRegressor(random_state=1)),
    ('rf', RandomForestRegressor(random_state=1))
] 


reg = StackingRegressor(
    estimators=estimators,
    final_estimator=MLPRegressor(activation = "relu", alpha = 0.1, hidden_layer_sizes = (20,),
                  learning_rate_init = 0.01, max_iter = 2000, random_state=1))
#estimators為剛前面定義的那兩個方法
#最後blender就是final_estimator用RandomForestRegressor


reg.fit(x_train, y_train).score(x_test, y_test)


# predict
y_pred = reg.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


from xgboost import XGBRegressor

xgb = XGBRegressor()
# set_param
param_grid = {'learning_rate': [0.1, 0.5, 1],
         'n_estimators': [50, 100, 200],
         'max_depth': [3,4,5,6,7,8],
         'min_child_weight': [0.1, 0.5, 0.6, 1]
         }
         
# GridSearchCV
grid_search_xgb = GridSearchCV(xgb, param_grid, cv=5)
grid_search_xgb.fit(x_train, y_train)


#mean validation score
grid_search_xgb.best_score_


# best_params
grid_search_xgb.best_params_


# model
#the best model
model_xgb = grid_search_xgb.best_estimator_ #用所有training data與best_params_訓練出的model
model_xgb.score(x_test, y_test)


# predict
y_pred = model_xgb.predict(x_test)
y_pred


mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)


imp = model_xgb.feature_importances_
imp


features = list(x_train)[:11]  #11個features
sorted(zip(imp, features), reverse=True)


ind = np.argsort(imp) #argsort把最小到最大的features的index找出來
plt.title('Feature Importances')
plt.barh(range(len(ind)), imp[ind], color='g', align='center') #畫橫線的長條圖
plt.yticks(range(len(ind)), [features[i] for i in ind]) #畫y座標
plt.xlabel('Relative Importance') #畫x label
plt.show()


pip_final = Pipeline([
    ('xgb', model_xgb)
]) #把資料處理和模型部分結合起來
pip_final.fit(x_train, y_train) #直接把x的訓練資料跟y的訓練資料，代進去fit而得到最終的模型

final_model = pip_final


import joblib
joblib.dump(final_model, 'final_model_reg.pkl') 


model_reg_loaded = joblib.load('final_model_reg.pkl')


y_pred_testing = model_reg_loaded.predict(testing_data)
y_pred_testing


sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')
sample_submission['Hardness'] = y_pred_testing
sample_submission.to_csv("submission.csv", index = False)
sample_submission


'''''from lightgbm import LGBMRegressor

lgmb = LGBMRegressor()
# set_param
param_grid = {'learning_rate': [0.01 , 0.1, 0.5],
         'n_estimators': [100, 200, 300],
         'max_depth': [3,4,5,6],
         'num_leaves':[1, 2],
         'min_child_weight': [0.1, 0.5, 0.6, 1],
         'min_child_samples':[100, 200, 300],
         'reg_alpha':[0.1, 0.5, 1],
         'reg_lambda':[0.1, 0.5, 1]
         }
# GridSearchCV
grid_search_lgmb = GridSearchCV(lgmb, param_grid, cv=5)
grid_search_lgmb.fit(x_train, y_train)


'''#mean validation score
grid_search_lgmb.best_score_


'''# best_params
grid_search_lgmb.best_params_


'''# model
model_lgmb = grid_search_lgmb.best_estimator_ #用所有training data與best_params_訓練出的model
model_lgmb.score(x_test, y_test)


'''mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


'''medae = median_absolute_error(y_test, y_pred)
print("MedAE:", medae)

