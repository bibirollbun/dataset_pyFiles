import numpy as np #  for linear algebra
import pandas as pd # for data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 


#defining the file paths
train_file = '/kaggle/input/playground-series-s5e1/train.csv'
test_file = '/kaggle/input/playground-series-s5e1/test.csv'
sample_submission_file = "/kaggle/input/playground-series-s5e1/sample_submission.csv"

# loading files to pandas DataFram

train_Data = pd.read_csv(train_file)
test_Data = pd.read_csv(test_file)
sample_submission_Data = pd.read_csv(sample_submission_file)

#displaying basic information

train_Data.info()
train_Data.head()




# handleing Missing Values 
missing_value = train_Data.isnull().sum()

# converting Date column to datetime format
train_Data['date'] = pd.to_datetime(train_Data['date'])

#check if their is any unique value in categorical vriable
unique_values = {
    'country': train_Data['country'].unique(),
    'store': train_Data['store'].unique(),
    'product': train_Data['product'].unique()
}

# statistiacl summery for num_sold
stat_num_sold = train_Data['num_sold'].describe()

missing_value , unique_values , stat_num_sold


import matplotlib.pyplot as plt
import seaborn as sns

# Handling missing values - Drop for now (can later consider imputation)
train_Data_clean = train_Data.dropna(subset=['num_sold'])

# Plot sales distribution
plt.figure(figsize=(10, 5))
sns.histplot(train_Data_clean['num_sold'], bins=50, kde=True, color='blue')
plt.xlabel("Number of Stickers Sold")
plt.ylabel("Frequency")
plt.title("Distribution of Stickers Sold")
plt.show()

# Boxplot to check outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=train_Data_clean['num_sold'], color='orange')
plt.xlabel("Number of Stickers Sold")
plt.title("Boxplot of Stickers Sold")
plt.show()



# Aggregate daily sales
daily_sales = train_Data_clean.groupby('date')['num_sold'].sum()

# Plot sales trend over time
plt.figure(figsize=(12, 5))
sns.lineplot(x=daily_sales.index, y=daily_sales.values, color="purple")
plt.xlabel("Date")
plt.ylabel("Total Stickers Sold")
plt.title("Daily Sales Trend Over Time")
plt.xticks(rotation=45)
plt.show()



# Extract year and month for trend analysis
train_Data_clean["year"] = train_Data_clean["date"].dt.year
train_Data_clean["month"] = train_Data_clean["date"].dt.month

# Aggregate sales by year and month
monthly_sales = train_Data_clean.groupby(["year", "month"])["num_sold"].sum().reset_index()

# Pivot for visualization
monthly_sales["date"] = pd.to_datetime(monthly_sales[["year", "month"]].assign(day=1))
monthly_sales = monthly_sales.sort_values("date")

# Plot Monthly Sales Trend
plt.figure(figsize=(12, 5))
sns.lineplot(x=monthly_sales["date"], y=monthly_sales["num_sold"], color="green")
plt.xlabel("Date")
plt.ylabel("Total Stickers Sold")
plt.title("Monthly Sales Trend Over Years")
plt.xticks(rotation=45)
plt.show()



# Aggregate sales by country
country_sales = train_Data_clean.groupby("country")["num_sold"].sum().reset_index()

# Plot sales distribution by country
plt.figure(figsize=(10, 5))
sns.barplot(x="country", y="num_sold", data=country_sales, palette="viridis")
plt.xlabel("Country")
plt.ylabel("Total Stickers Sold")
plt.title("Total Sales by Country")
plt.xticks(rotation=45)
plt.show()



# Aggregate sales by store type
store_sales = train_Data_clean.groupby("store")["num_sold"].sum().reset_index()

# Plot sales distribution by store type
plt.figure(figsize=(10, 5))
sns.barplot(x="store", y="num_sold", data=store_sales, palette="coolwarm")
plt.xlabel("Store Type")
plt.ylabel("Total Stickers Sold")
plt.title("Total Sales by Store Type")
plt.xticks(rotation=45)
plt.show()



# Aggregate sales by product type
product_sales = train_Data_clean.groupby("product")["num_sold"].sum().reset_index()

# Plot sales distribution by product type
plt.figure(figsize=(10, 5))
sns.barplot(x="product", y="num_sold", data=product_sales, palette="magma")
plt.xlabel("Product Type")
plt.ylabel("Total Stickers Sold")
plt.title("Total Sales by Product Type")
plt.xticks(rotation=45)
plt.show()


# Extract weekday information (0=Monday, 6=Sunday)
train_Data_clean["weekday"] = train_Data_clean["date"].dt.weekday

# Aggregate sales by weekday
weekday_sales = train_Data_clean.groupby("weekday")["num_sold"].sum().reset_index()

# Map weekday numbers to names
weekday_sales["weekday"] = weekday_sales["weekday"].map({
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
})

# Sort by weekday order
weekday_sales = weekday_sales.sort_values("weekday", key=lambda x: x.map({
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}))

# Plot sales by weekday
plt.figure(figsize=(10, 5))
sns.barplot(x="weekday", y="num_sold", data=weekday_sales, palette="Blues")
plt.xlabel("Weekday")
plt.ylabel("Total Stickers Sold")
plt.title("Total Sales by Day of the Week")
plt.xticks(rotation=45)
plt.show()



# Extract year and month for yearly trend analysis
yearly_sales = train_Data_clean.groupby("year")["num_sold"].sum().reset_index()

# Plot yearly sales trend
plt.figure(figsize=(10, 5))
sns.lineplot(x="year", y="num_sold", data=yearly_sales, marker="o", color="green")
plt.xlabel("Year")
plt.ylabel("Total Stickers Sold")
plt.title("Total Sales Over the Years")
plt.xticks(yearly_sales["year"])
plt.show()



# Check if there is a holiday column in the dataset
train_Data_clean.columns



# Function to create lag features
def create_lag_features(df, lags=[1, 7, 30]):
    df = df.sort_values(by=["country", "store", "product", "date"]) 
    for lag in lags:
        df[f"num_sold_lag_{lag}"] = df.groupby(["country", "store", "product"])["num_sold"].shift(lag)
    return df

# Apply the function to training data
train_Data_clean = create_lag_features(train_Data_clean, lags=[1, 7])

# Display updated training data with lag features
train_Data_clean.head(10)



#NaN values for the first few rows (because there's no past data).
train_Data_clean.fillna(0, inplace=True)  # Replace NaNs with 0
train_Data_clean.head(10)


from sklearn.preprocessing import LabelEncoder

# Columns to encode
categorical_cols = ["country", "store", "product"]

# Apply label encoding
label_encoders = {}  # Store encoders for later use (if needed)

for col in categorical_cols:
    le = LabelEncoder()
    train_Data_clean[col] = le.fit_transform(train_Data_clean[col])  # Encode train data
    label_encoders[col] = le  # Save encoder for future use (test set)

# Check encoded values
train_Data_clean.head()




for col in categorical_cols:
    try:
        test_Data[col] = label_encoders[col].transform(test_Data[col])  # Use saved encoder
    except KeyError as e:
        print(f"Column '{col}' has unseen labels in test data: {str(e)}")
        # Handle the error: either skip this column or use a fallback strategy

# Alternatively, you can catch ValueError for handling unseen labels


train_Data_clean.columns


# Sort by date (important for time series)
train_Data_clean = train_Data_clean.sort_values("date")

# Define the cutoff date (e.g., last few months for validation)
split_date = "2013-07-01"  # Adjust as needed based on data

# Create training and validation sets
train_data = train_Data_clean[train_Data_clean["date"] < split_date]
valid_data = train_Data_clean[train_Data_clean["date"] >= split_date]

# Define features and target
features = ["country", "store", "product", "year", "month", "weekday", "num_sold_lag_1", "num_sold_lag_7"]  # Include lag features
target = "num_sold"

X_train, y_train = train_data[features], train_data[target]
X_valid, y_valid = valid_data[features], valid_data[target]

print("Train set size:", X_train.shape)
print("Validation set size:", X_valid.shape)



import xgboost as xgb
from sklearn.metrics import mean_squared_error



# Define Features and Target
features = ["country", "store", "product", "year", "month", "weekday", "num_sold_lag_1", "num_sold_lag_7"]  
target = "num_sold"

# Convert features to numpy arrays
X_train, y_train = train_data[features], train_data[target]
X_valid, y_valid = valid_data[features], valid_data[target]



import xgboost as xgb

# Define the model with early stopping in the constructor
xgb_model = xgb.XGBRegressor(
    n_estimators=500,    # Number of boosting rounds
    learning_rate=0.05,  # Step size shrinkage
    max_depth=6,         # Depth of trees
    subsample=0.8,       # Fraction of samples used
    colsample_bytree=0.8, # Fraction of features used
    random_state=42
)

xgb_model.set_params(early_stopping_rounds=50)

xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=True)




# Predict on Validation Set
y_pred = xgb_model.predict(X_valid)

# Compute RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")



print(f"Mean of y_valid: {y_valid.mean():.4f}")
print(f"Std Dev of y_valid: {y_valid.std():.4f}")




print(test_Data.columns)  # Check what columns exist
print(features)         # Check the list of features




# Ensure 'date' column is in datetime format
test_Data["date"] = pd.to_datetime(test_Data["date"])

# Extract features
test_Data["year"] = test_Data["date"].dt.year
test_Data["month"] = test_Data["date"].dt.month
test_Data["weekday"] = test_Data["date"].dt.weekday
test_Data.head()


# Combine train and test data to create lag features
combined_df = pd.concat([train_Data, test_Data], sort=False)

# Ensure the data is sorted by date, store, product, and country
combined_df = combined_df.sort_values(by=["country", "store", "product", "date"])

# Create lag features
for lag in [1, 7]:  
    combined_df[f"num_sold_lag_{lag}"] = combined_df.groupby(["country", "store", "product"])["num_sold"].shift(lag)

# Split test data again
test_Data = combined_df[combined_df["date"] >= test_Data["date"].min()]

test_Data = test_Data.copy()

# Fill missing values (because test set doesn't have past sales)
test_Data.fillna(0, inplace=True)

test_Data.columns


# Ensure categorical variables in test set are encoded the same way
for col in ["country", "store", "product"]:
    test_Data[col] = test_Data[col].astype("category").cat.codes

# Extract features from test set
X_test = test_Data[features]

# Make predictions
test_Data["num_sold"] = xgb_model.predict(X_test)
test_Data.head()


# Prepare submission file
submission = test_Data[["id", "num_sold"]]
submission.to_csv("submission.csv", index=False)
submission.head(25)
print("Submission file saved!")


