import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_train


df_train.info()


### Converting date to datetime type
df_train['date'] = pd.to_datetime(df_train['date'])


df_train.duplicated().sum(), df_train.drop('id', axis=1).duplicated().sum()


pd.DataFrame({'Nulls': df_train.isna().sum(), '%': df_train.isna().sum() / df_train.shape[0]})


df_train.dropna(inplace=True, axis=0)


df_train.describe()


for col in df_train.columns:
    unqs = df_train[col].unique()
    print(col, unqs if len(unqs) < 10 else '[...]', len(unqs))


df_train['date'].min(), df_train['date'].max()


df_train.groupby(['country', 'store'])['date'].diff()


# Country wise sales

display(pd.DataFrame({'num_sold': df_train.groupby('country')['num_sold'].sum(), '%': df_train.groupby('country')['num_sold'].sum() / df_train['num_sold'].sum() * 100}).sort_values('%', ascending=False))

sns.barplot(df_train.groupby('country')['num_sold'].sum().reset_index(), x='country', y='num_sold')
plt.show()


# Store wise sales

display(pd.DataFrame({'num_sold': df_train.groupby('store')['num_sold'].sum(), '%': df_train.groupby('store')['num_sold'].sum() / df_train['num_sold'].sum() * 100}).sort_values('%', ascending=False))

sns.barplot(df_train.groupby('store')['num_sold'].sum().reset_index(), x='store', y='num_sold')
plt.show()


# Product wise sales

display(pd.DataFrame({'num_sold': df_train.groupby('product')['num_sold'].sum(), '%': df_train.groupby('product')['num_sold'].sum() / df_train['num_sold'].sum() * 100}).sort_values('%', ascending=False))

sns.barplot(df_train.groupby('product')['num_sold'].sum().reset_index(), x='product', y='num_sold')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure 'date' is in datetime format
df_train["date"] = pd.to_datetime(df_train["date"])

# Loop through each country
for country in df_train["country"].unique():
    stores = df_train[df_train["country"] == country]["store"].unique()
    
    # Create a figure with subplots for each store
    fig, axes = plt.subplots(len(stores), 1, figsize=(15, 4 * len(stores)), sharex=True)

    if len(stores) == 1:  # Ensure axes is always a list
        axes = [axes]

    # Loop through each store
    for i, store in enumerate(stores):
        subset = df_train[(df_train["country"] == country) & (df_train["store"] == store)].copy()
        
        # Calculate 7-day rolling average per product
        subset["rolling_avg"] = subset.groupby("product")["num_sold"].transform(lambda x: x.rolling(7, min_periods=1).mean())
        
        # Plot the rolling average trend for each product
        sns.lineplot(data=subset, x="date", y="rolling_avg", hue="product", ax=axes[i], linewidth=2)
        
        # For each product, compute and plot the regression line on the rolling average
        for product in subset["product"].unique():
            product_subset = subset[subset["product"] == product].sort_values("date")
            # Convert dates to ordinal numbers for regression calculation
            x = product_subset["date"].map(lambda d: d.toordinal()).values
            y = product_subset["rolling_avg"].values
            if len(x) > 1:  # Only compute regression if there are multiple points
                coef = np.polyfit(x, y, 1)
                poly1d_fn = np.poly1d(coef)
                # Plot the regression line using the original date values
                axes[i].plot(product_subset["date"], poly1d_fn(x), linestyle="--", color="black")
        
        axes[i].set_title(f"{country} - {store}")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Number Sold (Trend)")
    
    plt.suptitle(f"Sales Trends for {country}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


df_train.drop(columns=['id', 'date'], axis=1)


df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['dayOfYear'] = df_train['date'].dt.dayofyear
df_train['weekday'] = df_train['date'].dt.weekday


def regression_pipeline(df):    
    X = df.drop('num_sold', axis=1)
    y = df['num_sold']
    
    # Identify categorical and numerical features
    categorical_features = ["country", "store", "product"]
    numerical_features = ["year", "month", "day", "dayOfYear", "weekday"]
    
    # Preprocessing
    categorical_transformer = OneHotEncoder()
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Decision Tree Regressor with fixed parameters
    regressor = DecisionTreeRegressor(
        min_samples_split=10, 
        max_features='auto', 
        max_depth=20
    )
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print("Decision Tree Regression Results:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"R^2 Score: {r2}")
    
    # Return results
    return model


best_model = regression_pipeline(df_train)
best_model





df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_test


### Converting date to datetime type
df_test['date'] = pd.to_datetime(df_test['date'])


df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['dayOfYear'] = df_test['date'].dt.dayofyear
df_test['weekday'] = df_test['date'].dt.weekday


y_pred = best_model.predict(df_test.drop(columns=['id', 'date']))
y_pred


submission = pd.DataFrame({'id': df_test['id'], 'num_sold': y_pred})
submission


submission.to_csv('submission.csv', index=False)




