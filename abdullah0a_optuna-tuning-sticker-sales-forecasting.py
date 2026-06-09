import pandas as pd


train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train.shape


test.shape


train.head()


train.info()


train['country'].value_counts()


train['store'].value_counts()


train.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')



# Convert date to datetime format
train['date'] = pd.to_datetime(train['date'])

# Plotting sales over time
plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='num_sold', data=train, marker='o')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,6))
country_sales = train.groupby('country')['num_sold'].sum().reset_index()
sns.barplot(x='country', y='num_sold', data=country_sales)
plt.title('Total Sales by Country')
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.show()


plt.figure(figsize=(10,6))
store_sales = train.groupby('store')['num_sold'].sum().reset_index()
sns.barplot(x='store', y='num_sold', data=store_sales)
plt.title('Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()



plt.figure(figsize=(10,6))
product_sales = train.groupby('product')['num_sold'].sum().reset_index()
sns.barplot(x='product', y='num_sold', data=product_sales)
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()



plt.figure(figsize=(10,6))
sns.histplot(train['num_sold'], bins=20, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Number of Sales')
plt.ylabel('Frequency')
plt.show()



plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()



train=train.dropna(subset=['num_sold'])


# Convert 'date' to datetime format in both train and test datasets
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Extract date features for train dataset
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.day_name()
train['weekday'] = train['date'].dt.weekday  # 0 = Monday, 6 = Sunday
train['is_weekend'] = (train['weekday'] >= 5).astype(int)  # 1 = Weekend
train['quarter'] = train['date'].dt.quarter

# Extract date features for test dataset
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.day_name()
test['weekday'] = test['date'].dt.weekday  # 0 = Monday, 6 = Sunday
test['is_weekend'] = (test['weekday'] >= 5).astype(int)  # 1 = Weekend
test['quarter'] = test['date'].dt.quarter



import numpy as np
from scipy import stats

#train.drop(columns=['date'], inplace=True)

numeric_train = train.select_dtypes(include=[np.number])

# Calculate Z-scores
z_scores = np.abs(stats.zscore(numeric_train))

# Set a threshold for outliers (typically 3)
threshold = 2

# Filter the rows that are not outliers (all columns must be below the threshold)
train = train[(z_scores < threshold).all(axis=1)]


train.shape


train['store_product'] = train['store'] + '-' + train['product']
test['store_product'] = test['store'] + '-' + test['product']


train=pd.get_dummies(train, columns=['country', 'store','store_product','product','day_of_week'])
test=pd.get_dummies(test, columns=['country', 'store','store_product', 'product','day_of_week'])


train.head()


"""from sklearn.preprocessing import LabelEncoder

#categorical_cols = train.select_dtypes(include=['object', 'category']).columns
# Initialize the label encoder
le = LabelEncoder()

train['store_product']=le.fit_transform(train['store_product'])
test['store_product'] =le.fit_transform(test['store_product'])"""


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np


train.columns


features=['year',
       'is_weekend', 'quarter', 'country_Canada', 'country_Finland',
       'country_Italy', 'country_Kenya', 'country_Norway', 'country_Singapore',
       'store_Discount Stickers', 'store_Premium Sticker Mart',
       'store_Stickers for Less',
       'store_product_Discount Stickers-Holographic Goose',
       'store_product_Discount Stickers-Kaggle',
       'store_product_Discount Stickers-Kaggle Tiers',
       'store_product_Discount Stickers-Kerneler',
       'store_product_Discount Stickers-Kerneler Dark Mode',
       'store_product_Premium Sticker Mart-Holographic Goose',
       'store_product_Premium Sticker Mart-Kaggle',
       'store_product_Premium Sticker Mart-Kaggle Tiers',
       'store_product_Premium Sticker Mart-Kerneler',
       'store_product_Premium Sticker Mart-Kerneler Dark Mode',
       'store_product_Stickers for Less-Holographic Goose',
       'store_product_Stickers for Less-Kaggle',
       'store_product_Stickers for Less-Kaggle Tiers',
       'store_product_Stickers for Less-Kerneler',
       'store_product_Stickers for Less-Kerneler Dark Mode',
       'product_Holographic Goose', 'product_Kaggle', 'product_Kaggle Tiers',
       'product_Kerneler', 'product_Kerneler Dark Mode', 'day_of_week_Friday',
       'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday',
       'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday']


train['num_sold'] = np.log1p(train['num_sold'])

X =train[features]
y =train['num_sold']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the XGBoost Regressor
import xgboost as xgb
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    n_estimators=1000,  # Number of trees
    max_depth=4,  # Maximum depth of each tree
    learning_rate=0.003,  # Step size at each iteration
    subsample=0.7,  # Fraction of samples used for each tree
    colsample_bytree=0.7,  # Fraction of features used for each tree
    random_state=42
)

model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_percentage_error

# Actual values (True values from the test set)
y_true = y_test

# Make predictions with the ensemble model
y_pred = model.predict(X_test)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f'MAPE: {mape * 100:.2f}%')


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_percentage_error')

# Convert negative MAPE to positive MAPE
cv_scores = -cv_scores
print(f'MAPE scores from cross-validation: {cv_scores}')
print(f'Mean MAPE: {cv_scores.mean() * 100:.2f}%')



test_features=test[features]
test_predictions =model.predict(test_features)

# Prepare the submission file
submission = pd.DataFrame({'id': test['id'], 'num_sold': test_predictions})
submission.to_csv('submission.csv', index=False)


submission.head()

