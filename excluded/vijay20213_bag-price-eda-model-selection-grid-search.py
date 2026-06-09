import numpy as np
import pandas as pd 

TRAIN_PATH = "/kaggle/input/playground-series-s5e2/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e2/test.csv"
SAMPLE_SUBMISSIONS = "/kaggle/input/playground-series-s5e2/sample_submission.csv"


train_df = pd.read_csv(TRAIN_PATH, index_col='id')
test_df = pd.read_csv(TEST_PATH, index_col='id')
train_df.head()


test_df.head()


print("PERCENTAGE TRAINING DATA = ", train_df.shape[0]/(train_df.shape[0] + test_df.shape[0]) * 100, "%")
print("PERCENTAGE TESTING DATA = ", test_df.shape[0]/(train_df.shape[0] + test_df.shape[0]) * 100, "%")


print("TRAIN DATA INFORMATION")
train_df.info()


print("TESTING DATA INFORMATION")
test_df.info()


print("NULL PERCENTAGE IN TRAIN DATA")
train_nulls = {}
for col in train_df.columns:
    nulls = round((train_df[col].isnull().sum() / (train_df[col].isnull().sum() + train_df.shape[0]) * 100),2)
    null_counts = nulls / 100 * train_df.shape[0]
    train_nulls[col] = [nulls, null_counts]
display(pd.DataFrame(train_nulls, index = ['%', 'count']))

print("-" * 100)

test_nulls = {}
print("NULL PERCENTAGE IN TEST DATA")
for col in test_df.columns:
    nulls = round((test_df[col].isnull().sum() / (test_df[col].isnull().sum() + test_df.shape[0]) * 100), 2)
    null_counts = nulls / 100 * train_df.shape[0]
    test_nulls[col] = [nulls, null_counts]
    
display(pd.DataFrame(test_nulls,  index = ['%', 'count']))



cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',  'Waterproof', 'Style', 'Color']
num_cols = ['Weight Capacity (kg)']

# for numerical columns
for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# for categorical columns
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

print("---------------------NULLS IN TRAIN------------------------")
print(train_df.isnull().sum())
print("---------------------NULLS IN TEST------------------------")
print(test_df.isnull().sum())


import plotly.graph_objects as go
import pandas as pd
import plotly.subplots as sp

for i in range(0, len(cat_cols), 2):
    cols = cat_cols[i:i+2]
    fig = sp.make_subplots(rows=1, cols=len(cols), subplot_titles=[f'Value Counts for {col}' for col in cols])
    
    for j, col in enumerate(cols):
        value_counts = train_df[col].value_counts()
        fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values, name=col), row=1, col=j+1)
    
    fig.update_layout(title_text='Value Counts for Categorical Columns', showlegend=True)
    fig.show()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in cat_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.fit_transform(test_df[col])
train_df.head()


# dedscribe the data
train_df.describe()


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
for col in num_cols:
    train_df[col] = scaler.fit_transform(train_df[[col]])
    test_df[col] = scaler.fit_transform(test_df[[col]])
train_df.head()


import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

mean = train_df['Price'].mean()
std_dev = train_df['Price'].std()

x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)

y = norm.pdf(x, mean, std_dev)

histogram = go.Histogram(
    x=train_df['Price'],  
    nbinsx=30,  
    histnorm='probability density', 
    opacity=0.6,  
    name="Price Histogram",  
    marker_color='orange'  
)

normal_curve = go.Scatter(
    x=x,  
    y=y,  
    mode='lines', 
    name=f"Normal Distribution (μ={mean:.2f}, σ={std_dev:.2f})", 
    line=dict(color='blue') 
)

fig = go.Figure()

fig.add_trace(histogram)
fig.add_trace(normal_curve)

fig.update_layout(
    title="Price Distribution with Histogram", 
    xaxis_title="Bag Price",  
    yaxis_title="Probability Density", 
    legend_title="Legend", 
    template="plotly_white", 
    showlegend=True  
)

fig.show()


from sklearn.model_selection import train_test_split, GridSearchCV
X = train_df.drop(['Price'], axis='columns')
y = train_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
X_train.shape, y_test.shape


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
import lightgbm as lgb


train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "regression",   
    "metric": "rmse",            
    "boosting_type": "gbdt",    
    "learning_rate": 0.05,       
    "num_leaves": 31,           
    "max_depth": -1,            
    "verbose": -1                
}

lgbm_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,        
)

y_train_pred = lgbm_model.predict(X_train, num_iteration=lgbm_model.best_iteration)
y_test_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")



models = {
    'LinReg': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(),
    'ExtraTrees': ExtraTreesRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'LGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'KNN': KNeighborsRegressor(),
    'MLP': MLPRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Bagging': BaggingRegressor()
}

results = {}
for model_name, model in models.items():
    print(model_name + " training initiated ...")
    trained_model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results[model_name] = (mse, rmse)
    print(str(model_name) + " training Done ...")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

models_results = list(results.keys())
mse = [results[model][0] for model in models_results]
rmse = [results[model][1] for model in models_results]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Train Score", "Test Score", "MSE"),
    horizontal_spacing=0.2
)

fig.add_trace(
    go.Bar(x=models_results, y=mse, name="Train Score", marker_color="blue"),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=models_results, y=rmse, name="Test Score", marker_color="green"),
    row=1, col=2
)

fig.update_layout(
    title_text="Model Performance Comparison",
    showlegend=False,
    height=500,
    width=1200,
)

fig.update_xaxes(title_text="Models", row=1, col=1)
fig.update_xaxes(title_text="Models", row=1, col=2)

fig.update_yaxes(title_text="MSE", row=1, col=1)
fig.update_yaxes(title_text="RMSE", row=1, col=2)

fig.show()


import matplotlib.pyplot as plt
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

plt.plot(y_pred[:100], color ='blue',linewidth=0, marker='*')
plt.plot(y_test[:100], color='red', linewidth=0, marker='*')
plt.title('prediction comparision')
plt.show()















# for model in models:    
#     print("GRID SRARCH ON MODEL ----------------", model)
#     grid_search = GridSearchCV(
#         estimator=model,
#         scoring='neg_mean_squared_error',
#         cv=5)
#     grid_search.fit(X_train, y_train)

#     best_params = grid_search.best_params_
#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     result[str(model)] = (best_params, best_model, y_pred, mse)
#     print("GRID SRARCH DONE FOR MODEL --------------", model)

