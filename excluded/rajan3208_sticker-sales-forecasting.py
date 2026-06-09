pip install pandas numpy matplotlib seaborn scikit-learn



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Load the data
train_df = pd.read_csv('/kaggle/input/dataset/train.csv')
test_df = pd.read_csv('/kaggle/input/dataset/test.csv')



def preprocess_data(df, is_train=True):
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract additional date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Encode categorical variables
    le_country = LabelEncoder()
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    
    df['country_encoded'] = le_country.fit_transform(df['country'])
    df['store_encoded'] = le_store.fit_transform(df['store'])
    df['product_encoded'] = le_product.fit_transform(df['product'])
    
    if is_train:
        return df, le_country, le_store, le_product
    else:
        return df



train_df, le_country, le_store, le_product = preprocess_data(train_df, is_train=True)
test_df = preprocess_data(test_df, is_train=False)



features = ['country_encoded', 'store_encoded', 'product_encoded', 
            'year', 'month', 'day', 'dayofweek', 'is_weekend']
target = 'num_sold'

X = train_df[features]
y = train_df[target]



print("Missing values in target variable:", y.isnull().sum())

X = X[y.notna()]
y = y.dropna()



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")



test_features = test_df[features]
test_predictions = rf_model.predict(test_features)



submission_df = test_df[['id', 'date', 'country', 'store', 'product']].copy()
submission_df['num_sold'] = test_predictions
submission_df = submission_df[['id', 'num_sold']] 
submission_df.to_csv('kaggle_sticker_predictions.csv', index=False)


def create_sales_visualization(train_df, rf_model, features):
    plt.figure(figsize=(15, 10))
    
    # Sales by Country
    plt.subplot(2, 2, 1)
    train_df.groupby('country')['num_sold'].sum().plot(kind='bar')
    plt.title('Total Sales by Country')
    plt.xlabel('Country')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    
    # Sales by Product
    plt.subplot(2, 2, 2)
    train_df.groupby('product')['num_sold'].sum().plot(kind='bar')
    plt.title('Total Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    
    # Monthly Sales Trend
    plt.subplot(2, 2, 3)
    monthly_sales = train_df.groupby(pd.Grouper(key='date', freq='M'))['num_sold'].sum()
    monthly_sales.plot(kind='line')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    
    # Feature Importance
    plt.subplot(2, 2, 4)
    feature_importance = pd.Series(rf_model.feature_importances_, index=features)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    plt.close()



create_sales_visualization(train_df, rf_model, features)



print("Analysis complete. Submission file of kaggle sticker forecasting and visualizations saved.")





