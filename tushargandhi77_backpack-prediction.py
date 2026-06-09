import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from  sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
import plotly.express as px
import plotly.graph_objects as go
from xarray.util.generate_ops import inplace


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.head()


df.isna().sum()


( df.isna().sum() / df.shape[0] )*100


categorical = ['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']
numeric_col = ['Compartments','Weight Capacity (kg)']

for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_col:
    if df[col].skew() > 1 or df[col].skew() < -1:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mean())


df.isna().sum()


# from ydata_profiling import ProfileReport
# prof = ProfileReport(df)
# prof.to_file(output_file='output.html')


for col in categorical:
    fig = px.histogram(df,x=col,title=f"Distribution of col {col}")
    fig.show()


numeric_col = ['Compartments','Weight Capacity (kg)','Price']
for col in numeric_col:
    fig = px.histogram(df, x=col, nbins=30, marginal='box', title=f'Distribution of {col}')
    fig.show()


for col in numeric_col:
    sns.displot(x=df[col])
    plt.show()


for col in numeric_col:
    skewness = df[col].skew()
    print(f"{col}: Skewness = {skewness:.2f}")



for col in numeric_col:
    sns.boxplot(df[col])
    plt.show()


for col in numeric_col:
    fig = px.violin(df, y=col, box=True, points="all", title=f'Violin Plot for {col}')
    fig.show()


import scipy.stats as stats

for col in numeric_col:
    plt.figure()
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {col}')
    plt.show()


sns.heatmap(df[numeric_col].corr(),annot=True)


sns.pairplot(df[numeric_col])


skew_kurt = pd.DataFrame({
    'Feature': numeric_col,
    'Skewness': [df[col].skew() for col in numeric_col],
    'Kurtosis': [df[col].kurt() for col in numeric_col]
})
display(skew_kurt.sort_values(by='Skewness', key=abs, ascending=False))



fig = px.scatter_3d(df, 
                    x='Size', 
                    y='Laptop Compartment', 
                    z='Weight Capacity (kg)', 
                    color='Brand', 
                    title='3D Plot: Size vs Laptop Compartment vs Capacity')
fig.show()



df.head()


df.drop(columns=['id'],inplace=True)


ohe_columns = ['Laptop Compartment','Waterproof']

for col in ohe_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    


df['Brand'].value_counts()


brand_avg_price = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
print(brand_avg_price)


def ordering_brand(str):
    if str == 'Adidas':
        return 1
    elif str == 'Nike':
        return 2
    elif str == 'Puma':
        return 3
    elif str == 'Jansport':
        return 4
    else:
        return 5


df['Brand'] = df['Brand'].apply(ordering_brand)


df['Brand'].value_counts()


material_price = df.groupby('Material')['Price'].mean().sort_values(ascending=False)
print(material_price)



def ordering_material(str):
    if str == 'Canvas':
        return 4
    if str == 'Polyester':
        return 3
    if str == 'Nylon':
        return 2
    else:
        return 1


df['Material'] = df['Material'].apply(ordering_material)


material_price = df.groupby('Material')['Price'].mean().sort_values(ascending=False)
print(material_price)


size_price = df.groupby('Size')['Price'].mean().sort_values(ascending=False)
print(size_price)


def ordering_size(str):
    if str == 'Large':
        return 3
    if str == 'Small':
        return 2
    else:
        return 1


df['Size'] = df['Size'].apply(ordering_size)


size_price = df.groupby('Size')['Price'].mean().sort_values(ascending=False)
print(size_price)


df.head()


style_price = df.groupby('Style')['Price'].mean().sort_values(ascending=False)
print(style_price)


def ordering_style(str):
    if str == 'Messenger':
        return 3
    if str == 'Backpack':
        return 2
    else:
        return 1


df['Style'] = df['Style'].apply(ordering_style)


style_price = df.groupby('Style')['Price'].mean().sort_values(ascending=False)
print(style_price)


color_price = df.groupby('Color')['Price'].mean().sort_values(ascending=False)
print(color_price)


def ordering_color(str):
    if str == 'Green':
        return 6
    if str == 'Blue':
        return 5
    if str == 'Pink':
        return 4
    if str == 'Red':
        return 3
    if str == 'Gray':
        return 2
    else:
        return 1


df['Color'] = df['Color'].apply(ordering_color)


color_price = df.groupby('Color')['Price'].mean().sort_values(ascending=False)
print(color_price)


df.head()


X = df.drop(columns=['Price'])
y = df['Price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape,X_test.shape


y_train.shape,y_test.shape


lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print(f"Linear Regression Score: ",r2_score(y_test,y_pred))


dt = RandomForestRegressor(n_estimators=60,min_samples_split=15,max_depth=5,min_samples_leaf=5,random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
r2_score(y_test,y_pred)


# Random forest Regressor
rf_params = {
    'n_estimators':[10,50,60,100],
    'max_depth':[5,10,15,20,None],
    'min_samples_split':[2,5,10,15],
    'min_samples_leaf':[1,2,5,10]
}

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(rf,rf_params,n_iter=10,cv=3,verbose=2,n_jobs=-1)
rf_random.fit(X_train,y_train)
print(f"Random Forest Best Score:{rf_random.best_score_}")
print(f"Random forest Best Params: {rf_random.best_params_}")


# Ada Booost Params
ada_params = {
    'n_estimators':[10,50,60,100],
    'learning_rate':[0.01,0.05,0.1,0.5,1]
}

ada = AdaBoostRegressor(random_state=42)
ada_random = RandomizedSearchCV(ada,ada_params,n_iter=10,cv=3,verbose=2,n_jobs=-1)
ada_random.fit(X_train,y_train)
print(f"ADA BOOST  Best Score:{ada_random.best_score_}")
print(f"ADA BOOST Best Params: {ada_random.best_params_}")


# Gradient Boosting params

gb_params = {
    'n_estimators':[10,50,60,100],
    'learning_rate':[0.01,0.05,0.1,0.5,1],
    'max_depth':[3,5,10,15,None],
    'min_samples_split':[2,5,10,15],
    'min_samples_leaf':[1,2,5]
}

gb = GradientBoostingRegressor(random_state=42)
gb_random = RandomizedSearchCV(gb,gb_params,n_iter=10,cv=3,verbose=2,n_jobs=-1)
gb_random.fit(X_train,y_train)
print(f"Gradient Boosting Best Score:{gb_random.best_score_}")
print(f"Gradient Boosting Best Params: {gb_random.best_params_}")




