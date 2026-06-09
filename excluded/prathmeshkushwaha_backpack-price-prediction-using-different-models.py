import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.shape


df.info()


print("Percentage Distribution of Null Values")
display(df.isnull().sum()*100/len(df))


null_percentage = (df.isnull().sum() * 100) / len(df)
null_percentage = null_percentage[null_percentage > 0] 

if not null_percentage.empty:
    fig = px.pie(
        names=null_percentage.index,
        values=null_percentage.values,
        hole=0.4,
        title="Percentage Distribution of Null Values",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),  
        height=500,  
        width=600,
        title_x=0.5  
    )

    fig.show()
else:
    print("No missing values in the dataset!")


df.duplicated().sum()


df.columns


df_cat = df.select_dtypes(include=["object"])
for column in df_cat.columns:
    print(column,": ",df[column].unique())


color_counts = df["Color"].value_counts().reset_index()
color_counts.columns = ["Color", "Count"]
fig = px.treemap(
    color_counts,
    path=["Color"],
    values="Count",
    title="Color Distribution Tree Map",
    color="Count",
    color_continuous_scale="viridis"  
)

fig.update_layout(margin=dict(l=10, r=10, t=50, b=10)) 
fig.show()


sns.histplot(df['Price'],palette='Blues')


df_cat = df.select_dtypes(include=["object"])
if column in df_cat.columns:
    df_grouped = df.groupby(column)["Price"].mean().reset_index()

    fig = px.sunburst(
        df_grouped,
        path=[column],  # Hierarchical path
        values="Price", 
        title=f"Average Price Distribution across {column}",
        color="Price",
        color_continuous_scale="viridis"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),  
        height=500,  
        width=600,
        title_x=0.5  
    )

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))  # Reducing white space
    fig.show()
else:
    print(f"Column '{column}' not found in dataset. Choose a valid categorical column. ")


for column in df_cat.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column], y=df['Price'], palette="deep")
    plt.xticks(rotation=45)
    plt.title(f'Price Distribution by {column}')
    plt.show()


import scipy.stats as stats


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k-1, r-1)))) if min(k-1, r-1) > 0 else 0

def cramers_v_matrix(df, categorical_columns):
    n = len(categorical_columns)
    matrix = np.zeros((n, n))  

    for i in range(n):
        for j in range(i, n):  
            if i == j:
                matrix[i, j] = 1.0 
            else:
                value = cramers_v(df[categorical_columns[i]], df[categorical_columns[j]])
                matrix[i, j] = value
                matrix[j, i] = value  

    return pd.DataFrame(matrix, index=categorical_columns, columns=categorical_columns)


categorical_columns = ['Brand', 'Material', 'Size','Waterproof', 'Style', 'Color']
cramers_v_corr_matrix = cramers_v_matrix(df, categorical_columns)
print(cramers_v_corr_matrix)


plt.figure(figsize=(10, 6))
sns.heatmap(cramers_v_corr_matrix, annot=True, cmap=sns.light_palette("seagreen", as_cmap=True), fmt=".2f", linewidths=0.5)

plt.title("CramÃ©r's V Correlation Heatmap for Categorical Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


numerical_columns = ['Compartments','Weight Capacity (kg)', 'Price']
numerical_corr_matrix = df[numerical_columns].corr(method='pearson')
print(numerical_corr_matrix)

numerical_corr_spearman = df[numerical_columns].corr(method='spearman')
print(numerical_corr_spearman)


plt.figure(figsize=(8, 5))
sns.heatmap(numerical_corr_matrix, annot=True, cmap=sns.color_palette("dark:salmon_r", as_cmap=True), fmt=".2f", linewidths=0.5)

plt.title("Pearson Correlation Heatmap (Numerical Features)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(numerical_corr_spearman, annot=True, cmap=sns.color_palette("dark:salmon_r", as_cmap=True), fmt=".2f", linewidths=0.5)

plt.title("Spearman Correlation Heatmap (Numerical Features)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


display(df['Waterproof'].value_counts())
display(df['Waterproof'].isnull().sum())


categories = df['Waterproof'].dropna().unique().tolist()
categories
df.loc[df['Waterproof'].isnull(), 'Waterproof'] = np.random.choice(categories, size=df['Waterproof'].isnull().sum())


display(df['Waterproof'].value_counts())
display(df['Waterproof'].isnull().sum())


df['Brand'] = df['Brand'].replace('nan', np.nan)


display(df['Brand'].value_counts())
display(df['Brand'].isnull().sum())


categories = df['Brand'].dropna().unique().tolist()
categories


df.loc[df['Brand'].isnull(), 'Brand'] = np.random.choice(categories, size=df['Brand'].isnull().sum())


display(df['Brand'].value_counts())
display(df['Brand'].isnull().sum())


display(df['Material'].value_counts())
print("Null Values: ",df['Material'].isnull().sum())


material_counts ={
    "Polyester":79630,
    "Leather":73416,
    "Nylon":70603,
    "Canvas":68004
}
total_count = sum(material_counts.values())
probabilities = [count / total_count for count in material_counts.values()]
materials = list(material_counts.keys())

num_nulls = df['Material'].isnull().sum()

df.loc[df['Material'].isnull(), 'Material'] = np.random.choice(materials, size = num_nulls, p=probabilities)    


display(df['Material'].value_counts())
print("Null Values: ",df['Material'].isnull().sum())


display(df['Size'].value_counts())
print("Null Values: ",df['Size'].isnull().sum())


size_counts = {
    "Medium": 101906,
    "Large": 98643,
    "Small": 92856
}

total_count = sum(size_counts.values())
probabilities = [count / total_count for count in size_counts.values()]
size_categories = list(size_counts.keys())

num_nulls = df['Size'].isnull().sum()

df.loc[df['Size'].isnull(), 'Size'] = np.random.choice(size_categories, size=num_nulls, p=probabilities)


display(df['Size'].value_counts())
print("Null Values: ",df['Size'].isnull().sum())


display(df['Laptop Compartment'].value_counts())
print("Null Values: ",df['Laptop Compartment'].isnull().sum())


value_counts = df['Laptop Compartment'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

df.loc[df['Laptop Compartment'].isnull(), 'Laptop Compartment'] = np.random.choice(
    categories, 
    size=df['Laptop Compartment'].isnull().sum(), 
    p=probabilities  
)


display(df['Laptop Compartment'].value_counts())
print("Null Values: ",df['Laptop Compartment'].isnull().sum())


display(df['Style'].value_counts())
print("Null Values of Style: ",df['Style'].isnull().sum())


display(df['Color'].value_counts())
print("Null Values of Color: ",df['Color'].isnull().sum())


value_counts = df['Style'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

df.loc[df['Style'].isnull(), 'Style'] = np.random.choice(
    categories, 
    size=df['Style'].isnull().sum(), 
    p=probabilities  
)


value_counts = df['Color'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

df.loc[df['Color'].isnull(), 'Color'] = np.random.choice(
    categories, 
    size=df['Color'].isnull().sum(), 
    p=probabilities  
)


display(df['Style'].value_counts())
print("Null Values of Style: ",df['Style'].isnull().sum())


display(df['Color'].value_counts())
print("Null Values of Color: ",df['Color'].isnull().sum())


print("Null Values: ",df['Weight Capacity (kg)'].isnull().sum())


df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median(), inplace=True)


print("Null Values: ",df['Weight Capacity (kg)'].isnull().sum())


df.isnull().sum()


def detect_outliers_iqr(df):
    outliers_dict = {}
    
    for col in df.select_dtypes(include=[np.number]):  # Select only numerical columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_dict[col] = outliers.tolist()  
    return outliers_dict

outliers_iqr = detect_outliers_iqr(df)
print("Outliers in each column using IQR:\n", outliers_iqr)


df = pd.get_dummies(df, columns=['Brand', 'Material', 'Size', 'Style', 'Color'], drop_first=True)


df.head()


label_encoder = LabelEncoder()
df['Laptop Compartment'] = label_encoder.fit_transform(df['Laptop Compartment'])
df['Waterproof'] = label_encoder.fit_transform(df['Waterproof'])


num_cols = ["Weight Capacity (kg)","Compartments"]

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


df = df.drop(columns=['id'])


df.columns


X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


print(X_train.dtypes)
print(y_train.dtypes)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"RÂ² Score: {r2}")
print(f"RMSE: {rmse}")


models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}


for name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model: {name}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RÂ² Score: {r2:.4f}, RMSE: {rmse:.4f}")
    print("-" * 20)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "LightGBM": LGBMRegressor(n_estimators=100),
    "CatBoost": CatBoostRegressor(n_estimators=100, verbose=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model: {name}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RÂ² Score: {r2:.4f}, RMSE: {rmse:.4f}")
    print("-" * 20)


stacking_model = StackingRegressor(
    estimators=[("ridge", Ridge()), ("lightgbm", LGBMRegressor())],
    final_estimator=CatBoostRegressor(verbose=0)
)
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)
print("Stacking Regressor Results:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_stack):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_stack)):.2f}\n")


models = ['Linear Regression','Decision Tree','Random Forest','Gradient Boosting','XGBoost','Ridge Regression','Lasso Regression','Elastic Net','LightGBM','CatBoost','Stacking Regressor']
rmse_values = [38.92,55.95,40.17,38.91,39.11,38.92,38.93,38.93,38.92,39.02,38.95]

min_rmse = min(rmse_values)
min_index = rmse_values.index(min_rmse)
min_model = models[min_index]

plt.figure(figsize=(10, 5))
plt.plot(models, rmse_values, marker='o', linestyle='-', color='b', markersize=8)
plt.scatter(min_index, min_rmse, color='red', s=100, label=f"Lowest RMSE: {min_rmse:.2f} ({min_model})")
plt.xlabel("Model")
plt.xticks(rotation=45) 
plt.ylabel("RMSE Value")
plt.title("RMSE Trend Across Models")
plt.legend()
plt.grid(True)
plt.show()


X_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


X_test


X_test.isnull().sum()


X_test['Weight Capacity (kg)'].fillna(X_test['Weight Capacity (kg)'].median(), inplace=True)


categories = X_test['Waterproof'].dropna().unique().tolist()
X_test.loc[X_test['Waterproof'].isnull(), 'Waterproof'] = np.random.choice(categories, size=X_test['Waterproof'].isnull().sum())


categories = X_test['Brand'].dropna().unique().tolist()
X_test.loc[X_test['Brand'].isnull(), 'Brand'] = np.random.choice(categories, size=X_test['Brand'].isnull().sum())


X_test['Material'].value_counts()


material_counts ={
    "Polyester":54534,
    "Leather":50382,
    "Nylon":48903,
    "Canvas":46181
}
total_count = sum(material_counts.values())
probabilities = [count / total_count for count in material_counts.values()]
materials = list(material_counts.keys())

num_nulls = X_test['Material'].isnull().sum()

X_test.loc[X_test['Material'].isnull(), 'Material'] = np.random.choice(materials, size = num_nulls, p=probabilities)    


X_test['Size'].value_counts()


size_counts = {
    "Medium": 69250,
    "Large": 67219,
    "Small": 63531
}

total_count = sum(size_counts.values())
probabilities = [count / total_count for count in size_counts.values()]
size_categories = list(size_counts.keys())

num_nulls = X_test['Size'].isnull().sum()

X_test.loc[X_test['Size'].isnull(), 'Size'] = np.random.choice(size_categories, size=num_nulls, p=probabilities)


value_counts = X_test['Laptop Compartment'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

X_test.loc[X_test['Laptop Compartment'].isnull(), 'Laptop Compartment'] = np.random.choice(
    categories, 
    size=X_test['Laptop Compartment'].isnull().sum(), 
    p=probabilities  
)


value_counts = X_test['Style'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

X_test.loc[X_test['Style'].isnull(), 'Style'] = np.random.choice(
    categories, 
    size=X_test['Style'].isnull().sum(), 
    p=probabilities  
)


value_counts = X_test['Color'].value_counts(dropna=True)

categories = value_counts.index.tolist() 
probabilities = value_counts.values / value_counts.values.sum()  

X_test.loc[X_test['Color'].isnull(), 'Color'] = np.random.choice(
    categories, 
    size=X_test['Color'].isnull().sum(), 
    p=probabilities  
)


X_test.isnull().sum()


categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight Capacity (kg)"]


X_test = pd.get_dummies(X_test, columns=['Brand', 'Material', 'Size', 'Style', 'Color'], drop_first=True)


X_test


label_encoder = LabelEncoder()
X_test['Laptop Compartment'] = label_encoder.fit_transform(X_test['Laptop Compartment'])
X_test['Waterproof'] = label_encoder.fit_transform(X_test['Waterproof'])


X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


id = X_test['id']


X_test = X_test.drop(columns={'id'})


id


from sklearn.model_selection import GridSearchCV


gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)


gb_model.fit(X,y)


y_pred = gb_model.predict(X_test)


y_pred


submission = pd.DataFrame({'id': id, 'Price': y_pred})


submission


submission.to_csv('submission.csv', index=False)

