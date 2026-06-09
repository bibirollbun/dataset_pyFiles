import numpy as np
import pandas as pd
from IPython.display import display , HTML
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


df  = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df.set_index('id', inplace=True)


df


df['country'].value_counts()


df['store'].value_counts()


df['product'].value_counts()


df.info()


numerical_features = df.select_dtypes(include = [np.number])
numerical_features.describe().T


categorical_features = df.select_dtypes(include = [object])
categorical_features.describe().T


null_values = df.isnull().sum()
null_values


missing_percentage = df.isnull().sum() / len(df) *100
missing_percentage


df = df.dropna(subset=['num_sold'])


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='num_sold', stat='density', kde=True, 
            color='blue', alpha=0.6, bins=30 )
plt.title(f'Distribution of Sticker sold')
plt.xlabel('Number of Stickers sold')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.show()


mu , sigma = stats.norm.fit(df['num_sold'])

hist_data = go.Histogram( x = df['num_sold'] , nbinsx = 50 , name = 'histogram' , opacity = 0.75 , histnorm = 'probability density' ,
                        marker = dict( color = 'purple'))

x_norm = np.linspace(df['num_sold'].min() ,  df['num_sold'].max() , 100)
y_norm = stats.norm.pdf(x_norm , mu , sigma)


norm_data = go.Scatter(x = x_norm  , y= y_norm , mode = 'lines' , name = f'Normal Distribution : ( mu : {mu : .2f} , sigma = {sigma : .2f})'  , line = dict(color = "green"))

fig = go.Figure(data = [hist_data  , norm_data])

fig.update_layout(
    title = "Number of Stickers sold Prediction" ,
    xaxis_title = "Number of Stickers sold" ,
    yaxis_title = "Density" ,
    plot_bgcolor = 'rgba(32 , 32 , 32 ,1)',
    paper_bgcolor = 'rgba(32 ,32 , 32 ,1)' ,
    font = dict(color = 'white')
)


qq_data = stats.probplot(df['num_sold'], dist="norm")

qq_fig = px.scatter(x=qq_data[0][0], y=qq_data[0][1], 
                   labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'}, 
                   color_discrete_sequence=["purple"])

qq_fig.update_layout(
    title="Q-Q plot",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)

slope, intercept, r_value, p_value, std_err = stats.linregress(qq_data[0][0], qq_data[0][1])
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

line_data = go.Scatter(x=line_x, y=line_y, mode="lines", name="Normal Line", 
                      line=dict(color="green"))

qq_fig.add_trace(line_data)
qq_fig.show()


country_sell = df.groupby('country')['num_sold'].mean()

text_values = [f"{value:.0f}" for value in country_sell.values]

fig2 = px.bar( df , x =  country_sell.index , y = country_sell.values  , color =  country_sell.index  , text = text_values  ) 
fig2.update_layout( uniformtext_minsize = 10 , uniformtext_mode = 'hide' ,
                    title = "Number of Stickers sold by country" ,
                    xaxis_title = "Country" ,
                    yaxis_title = "average stickers sold" ,
                  ) 
fig2.show()


store_sell = df.groupby('store')['num_sold'].mean()

text_values1 = [f"{value:.0f}" for value in store_sell.values]

fig2 = px.bar(df ,  x =  store_sell.index , y =  store_sell.values , color = store_sell.index , text =  text_values1 ) 
fig2.update_layout( uniformtext_minsize = 10 , uniformtext_mode = 'hide' ,
                    title = 'Average Strickers sold by store' ,
                    xaxis_title = "store" ,
                    yaxis_title = "average stickers sold" ,
                  ) 
fig2.show()


product_sell = df.groupby('product')['num_sold'].mean()

text_values2 = [f"{value:.0f}" for value in product_sell.values]

fig2 = px.bar( df , x =  product_sell.index , y =  product_sell.values , color = product_sell.index , text =  text_values2 ) 
fig2.update_layout( uniformtext_minsize = 10 , uniformtext_mode = 'hide' ,
                    title = 'Average Strickers sold by product' ,
                    xaxis_title = "product" ,
                    yaxis_title = "average stickers sold" ,
                  ) 
fig2.show()


df_dates = df.copy()

df_dates['date'] = pd.to_datetime(df_dates['date'], format='%Y-%m-%d')


print("New date type:", df_dates['date'].dtype)

df_dates['year'] = df_dates['date'].dt.year
df_dates['month'] = df_dates['date'].dt.month
df_dates['day'] = df_dates['date'].dt.day
df_dates['day_of_week'] = df_dates['date'].dt.dayofweek
df_dates['quarter'] = df_dates['date'].dt.quarter
df_dates['is_weekend'] = df_dates['date'].dt.dayofweek >= 5


df_dates['month_sin'] = np.sin(2 * np.pi * df_dates['month']/12)
df_dates['month_cos'] = np.cos(2 * np.pi * df_dates['month']/12)
df_dates['dow_sin'] = np.sin(2 * np.pi * df_dates['day_of_week']/7)
df_dates['dow_cos'] = np.cos(2 * np.pi * df_dates['day_of_week']/7)


print("\nFirst few rows of transformed data:")
print(df_dates[['date', 'year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend']].head())


df_dates


df_dates = df_dates.drop(['date'], axis=1)


df_dates.info()


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import make_scorer


categorical_transformer = Pipeline(steps = [
                 ('imputer' , SimpleImputer(strategy = 'constant' , fill_value = 'missing') ) ,
                ( 'onehot' , OneHotEncoder(handle_unknown = 'ignore' , sparse = False))
    
])


categorical_columns = df_dates.select_dtypes(include = ['object' , 'category'] ).columns


preprocessor = ColumnTransformer(
    transformers = [
        ('cat' , categorical_transformer , categorical_columns) 
      ], remainder = 'passthrough'
)


pipeline = Pipeline( steps = [
    ('preprocessor' ,  preprocessor ) 
])


X = df_dates.drop('num_sold' , axis =1 )
Y = df_dates['num_sold']


X


X_processed = pipeline.fit_transform(X)


from xgboost import XGBRegressor


model = XGBRegressor(
    tree_method='gpu_hist',  
    predictor='gpu_predictor'
)


X_train, X_test, y_train, y_test = train_test_split(X_processed , Y, test_size=0.15, random_state=42)


param_grid = {
    'regressor__tree_method': ['gpu_hist'],
    'regressor__predictor': ['gpu_predictor'],
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 4, 5, 6],
    'regressor__learning_rate': [0.01, 0.1, 0.3],
    'regressor__subsample': [0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.8, 0.9, 1.0],
    'regressor__min_child_weight': [1, 3, 5]
}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 10

custom_mape_scorer = make_scorer(mape, greater_is_better=False)


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring= custom_mape_scorer,
    n_jobs=-1,
    verbose=1
)


import torch
print("Is GPU available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


grid_search.fit(X_train, y_train)


print("Best parameters:", grid_search.best_params_)
print("Best MAPE:", -grid_search.best_score_)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
test_mape = mape(y_test, y_pred)
print(f"Test set MAPE: {test_mape:.2f}%")


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test


test_update = test.set_index('id')


test_update


# First, create a copy of the test data
test_dates = test_update.copy()

# Convert dates to datetime using the correct format
test_dates['date'] = pd.to_datetime(test_dates['date'], format='%Y-%m-%d')

# Extract date features 
test_dates['year'] = test_dates['date'].dt.year
test_dates['month'] = test_dates['date'].dt.month
test_dates['day'] = test_dates['date'].dt.day
test_dates['day_of_week'] = test_dates['date'].dt.dayofweek
test_dates['quarter'] = test_dates['date'].dt.quarter
test_dates['is_weekend'] = test_dates['date'].dt.dayofweek >= 5

# Create cyclical features
test_dates['month_sin'] = np.sin(2 * np.pi * test_dates['month']/12)
test_dates['month_cos'] = np.cos(2 * np.pi * test_dates['month']/12)
test_dates['dow_sin'] = np.sin(2 * np.pi * test_dates['day_of_week']/7)
test_dates['dow_cos'] = np.cos(2 * np.pi * test_dates['day_of_week']/7)

# Drop the original date column as we've extracted all features
test_dates = test_dates.drop(['date'], axis=1)

# Show first few rows to verify
print("\nFirst few rows of transformed test data:")
print(test_dates.head())


categorical_columns_test = df_dates.select_dtypes(include = ['object' , 'category'] ).columns


preprocessor_test = ColumnTransformer(
    transformers = [
        ('cat' , categorical_transformer , categorical_columns_test) 
      ], remainder = 'passthrough'
)


pipeline_test = Pipeline( steps = [
    ('preprocessor' ,  preprocessor_test ) 
])


X_sub = test_dates


X_sub_pro = pipeline.fit_transform(X_sub)


X_sub_pro[0]


y_sub = best_model.predict(X_sub_pro)


submission = pd.DataFrame({
    'id': test['id'], 
    'num_sold': y_sub
})


submission.to_csv('submission.csv', index=False)

