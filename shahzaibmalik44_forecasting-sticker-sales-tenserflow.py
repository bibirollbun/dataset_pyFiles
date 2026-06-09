import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import time


train = pd.read_csv(r"/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e1/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e1/sample_submission.csv")
print("Test shape:", train.shape )
print("Test shape:", test.shape )
train.head()


def display_dataset_info(dataset, name):
    # Define color codes
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    print(f"{GREEN}-----------------------------------------------------------------{RESET}")
    print(f"{BLUE}{name} DataFrame Shape: Rows = {dataset.shape[0]}, Columns = {dataset.shape[1]}{RESET}")
    
    # Numerical and categorical columns information
    num_cols = dataset.select_dtypes(include='number')
    cat_cols = dataset.select_dtypes(exclude='number')
    print(f"{YELLOW}{name} DataFrame numeric columns size = {len(num_cols.columns)}, categorical columns size = {len(cat_cols.columns)}{RESET}")
    
    # Missing values information
    total_missing = dataset.isnull().sum().sum()
    if total_missing > 0:
        missing_perc = (total_missing / (dataset.shape[0] * dataset.shape[1])) * 100
        print(f"{RED}There are a total of {total_missing} missing values in the {name} DataFrame ({missing_perc:.2f}% of all values).{RESET}")
        print(f"{RED}Missing values per column:{RESET}")
        print(dataset.isnull().sum().sort_values(ascending=False).head(10))
    else:
        print(f"{GREEN}There are no missing values in the {name} DataFrame.{RESET}")
    
    # Duplicate rows information
    total_duplicates = dataset.duplicated().sum()
    if total_duplicates > 0:
        print(f"{RED}There are {total_duplicates} duplicate rows in the {name} DataFrame.{RESET}")
    else:
        print(f"{GREEN}There are no duplicate rows in the {name} DataFrame.{RESET}")   
    
    # Check for column data types
    print(f"\n{YELLOW}Column data types:{RESET}")
    print(dataset.dtypes.value_counts())   
    
    print(f"{GREEN}-----------------------------------------------------------------{RESET}")
    print(f"{BLUE}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{RESET}")



%%time
display_dataset_info(train, "Train")  
display_dataset_info(test, "Test")


%%time
def create_important_features(df):
    
    # 2. Product Sales Share (for each product, relative to store's total sales)
    store_sales = df.groupby('store')['num_sold'].transform('sum')
    df['product_sales_share'] = df['num_sold'] / store_sales
    
    # 3. Store Sales Share (for each store, relative to total sales)
    total_sales = df['num_sold'].sum()
    df['store_sales_share'] = df.groupby('store')['num_sold'].transform('sum') / total_sales
    
    # 4. Rolling Average (7-day moving average of sales)
    df['rolling_avg_7'] = df.groupby('product')['num_sold'].transform(lambda x: x.rolling(7).mean())
    
    # 5. Cumulative Sales
    df['cumulative_sales'] = df.groupby('product')['num_sold'].cumsum()
    
    return df

train = create_important_features(train)
train.head()



def remove_non(df):
    return df.dropna()
    
train, test = map(remove_non, [train, test])   
train = train.drop('id', axis = 1)


# Set style
sns.set(style="whitegrid")

# 2. Top-selling Products
top_products = train.groupby('product')['num_sold'].sum().sort_values(ascending=False).head(10)

# Plot top-selling products
plt.figure(figsize=(12, 6))
top_products.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Top 10 Selling Products', fontsize=16)
plt.xlabel('Product', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjusts layout to prevent overlapping
plt.show()

# 3. Store Performance
store_sales = train.groupby('store')['num_sold'].sum().sort_values(ascending=False)

# Plot store performance
plt.figure(figsize=(12, 6))
store_sales.plot(kind='bar', color='royalblue', edgecolor='black')
plt.title('Store Sales Performance', fontsize=16)
plt.xlabel('Store', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 4. Sales Distribution by Country
country_sales = train.groupby('country')['num_sold'].sum()

# Plot sales by country
plt.figure(figsize=(6, 7))
country_sales.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("Set3", len(country_sales)), 
                   title='Sales Distribution by Country', fontsize=14, wedgeprops={'edgecolor': 'black'})
plt.ylabel('')
plt.title('Sales Distribution by Country', fontsize=16)
plt.tight_layout()
plt.show()



train['date'] = pd.to_datetime(train['date'])  
train['month'] = train['date'].dt.to_period('M')

# Aggregate sales by month
monthly_sales = train.groupby('month')['num_sold'].sum()

# Plot sales trend
monthly_sales.plot(kind='line', figsize=(10, 6), title='Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


# Encoding categorical variables using factorize in a loop
cat_cols = list(train.select_dtypes(include=['object']).columns)
for col in cat_cols:
    print(f"{col}, ",end="")
    train[col],_ = train[col].factorize()
    train[col] -= train[col].min()
    test[col],_ = test[col].factorize()
    test[col] -= test[col].min()


X = train.drop(['num_sold','date','month'], axis=1)
y = train['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define a deeper model with multiple layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]), 
    tf.keras.layers.Dense(32, activation='relu'),  
    tf.keras.layers.Dense(16, activation='relu'),  
    tf.keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train_tensor, y_train_tensor, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test_tensor, y_test_tensor)
print(f"Loss: {loss}, MAE: {mae}")


# Make predictions
predictions = model.predict(X_test_tensor)
print("Predictions:", predictions[:10])  

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

