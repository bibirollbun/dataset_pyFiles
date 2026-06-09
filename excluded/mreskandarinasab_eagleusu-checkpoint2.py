import numpy as np
import pandas as pd


path_data = "ashrae_data/"

train_data = pd.read_csv(path_data + 'train.csv')
test_data = pd.read_csv(path_data + 'test.csv')
building_metadata = pd.read_csv(path_data + 'building_metadata.csv')
weather_train = pd.read_csv(path_data + 'weather_train.csv')
weather_test = pd.read_csv(path_data + 'weather_test.csv')
sample_submission = pd.read_csv(path_data + 'sample_submission.csv')


display(train_data.head())
display(test_data.head())
display(sample_submission)


display(building_metadata.head())


display(weather_train.head())


display(weather_test.head())


display(sample_submission.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
sns.set_style("whitegrid")

### 1. Class Distribution of Unique Buildings
plt.figure(figsize=(12, 5))
sns.countplot(x=building_metadata["site_id"], palette="coolwarm")
plt.xlabel("Site ID")
plt.ylabel("Count")
plt.title("Class Distribution of Unique Buildings")
plt.show()

### 2. Class Distribution of Unique Meters
# Mapping meter types to categories
meter_mapping = {0: "Electricity", 1: "ChilledWater", 2: "Steam", 3: "HotWater"}
train_data["meter_category"] = train_data["meter"].map(meter_mapping)
test_data["meter_category"] = test_data["meter"].map(meter_mapping)
train_data.drop(columns=["meter"], inplace=True)
test_data.drop(columns=["meter"], inplace=True)

plt.figure(figsize=(8, 5))
sns.countplot(x=train_data["meter_category"], palette="viridis", order=["Electricity", "ChilledWater", "Steam", "HotWater"])
plt.xlabel("Meter Type")
plt.ylabel("Count")
plt.title("Class Distribution of Unique Meters")
plt.show()

### 3. Class Distribution of Unique Primary Use of Buildings
plt.figure(figsize=(12, 5))
sns.countplot(y=building_metadata["primary_use"], palette="magma", order=building_metadata["primary_use"].value_counts().index)
plt.xlabel("Count")
plt.ylabel("Primary Use Category")
plt.title("Class Distribution of Unique Primary Use of Buildings")
plt.show()


# 1. Number of unique buildings
num_unique_buildings = building_metadata["building_id"].nunique()
print(f"Number of Unique Buildings: {num_unique_buildings}")

# 2. Number of unique meters
# num_unique_meters = train_data["meter"].nunique()
# print(f"Number of Unique Meters: {num_unique_meters}")

# 3. Number of unique primary use of buildings
num_unique_primary_use = building_metadata["primary_use"].nunique()
print(f"Number of Unique Primary Use Categories: {num_unique_primary_use}")

# 4. Distribution of square footage of buildings
plt.figure(figsize=(12, 5))
sns.histplot(building_metadata["square_feet"].dropna(), bins=50, kde=True, color="blue")
plt.xlabel("Square Feet")
plt.ylabel("Frequency")
plt.title("Distribution of Building Square Footage")
plt.show()

# 5. Distribution of year built of buildings
plt.figure(figsize=(12, 5))
sns.histplot(building_metadata["year_built"].dropna(), bins=50, kde=True, color="green")
plt.xlabel("Year Built")
plt.ylabel("Frequency")
plt.title("Distribution of Building Construction Year")
plt.show()

# 6. Distribution of unique site IDs
plt.figure(figsize=(8, 5))
sns.countplot(x=building_metadata["site_id"], palette="coolwarm")
plt.xlabel("Site ID")
plt.ylabel("Count")
plt.title("Distribution of Unique Site IDs")
plt.show()

# 7. Correlation Heatmap for Weather Data (Dropping Non-Numeric Columns)
weather_train_numeric = weather_train.select_dtypes(include=[np.number])  # Keep only numeric columns

plt.figure(figsize=(10, 6))
sns.heatmap(weather_train_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Weather Variables")
plt.show()

# 8. Distribution plots for selected weather features
weather_features = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr",
                    "sea_level_pressure", "wind_direction", "wind_speed"]

for feature in weather_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(weather_train[feature].dropna(), bins=50, kde=True, color="purple")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature.replace('_', ' ').title()}")
    plt.show()


def prepare_data(X, building_data, weather_data, test=False):
    """
    Preparing vfinal dataset with all features.
    """
    
    X = X.merge(building_data, on="building_id", how="left")
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")
    
    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)
    
    if not test:
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)
        
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
    
    X["hour"] = X.timestamp.dt.hour
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = (X.timestamp.dt.date.astype("str").isin(holidays)).astype(int)
    
    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]

    X.drop(drop_features, axis=1, inplace=True)

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        X.drop("meter_reading", axis=1, inplace=True)
        return X, y


train_full, y_train_full = prepare_data(train_data, building_metadata, weather_train)

# Applying One Hot Encoding to Categorical Features
train_full = pd.get_dummies(train_full, columns=["primary_use"])
train_full = pd.get_dummies(train_full, columns=["meter_category"])

display(y_train_full)


display(train_full)


test_full, row_ids = prepare_data(test_data, building_metadata, weather_test, test=True)

# Applying One Hot Encoding to Categorical Features
test_full = pd.get_dummies(test_full, columns=["primary_use"])
test_full = pd.get_dummies(test_full, columns=["meter_category"])

display(test_full)


from scipy.stats import zscore

def normalize_features(df):
    """
    Normalize each feature individually using Z-score normalization.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select numerical columns
    df[numeric_cols] = df[numeric_cols].apply(zscore)  # Apply Z-score normalization
    return df


def impute_missing_values(df):
    """
    Perform mean imputation for missing values in numeric features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select numerical columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))  # Mean imputation
    return df


def remove_dew_temperature(df):
    """
    Remove the feature 'dew_temperature' since it has correlation with 'site_id'.
    """
    if "dew_temperature" in df.columns:
        df.drop(columns=["dew_temperature"], inplace=True)
    return df


# Apply preprocessing functions to train and test datasets
train_full = impute_missing_values(train_full)
train_full = normalize_features(train_full)
train_full = remove_dew_temperature(train_full)

test_full = impute_missing_values(test_full)
test_full = normalize_features(test_full)
test_full = remove_dew_temperature(test_full)


display(train_full)


mean_meter_reading = y_train_full.mean()
print(mean_meter_reading)


num_rows = len(test_full)  # Get the number of rows in test_full
mean_meter_reading = y_train_full.mean()  # Compute the mean meter reading

# Create the submission DataFrame
submission_df = pd.DataFrame({
    "row_id": range(num_rows),  # Sequential row IDs
    "meter_reading": mean_meter_reading  # Fill with mean meter reading
})

# Save to CSV file
submission_df.to_csv("submission_dummy.csv", index=False)

print("CSV file 'submission_dummy.csv' created successfully!")


train_full_np = np.array(train_full, dtype=float)
y_train_full_np = np.array(y_train_full, dtype=float)
test_full_np = np.array(test_full, dtype=float)


print (train_full_np.shape[1])


# Building the model
m = np.zeros(train_full_np.shape[1])
c = 0

L = 0.01  # The learning Rate
epochs = 30  # The number of iterations to perform gradient descent

n = float(len(y_train_full_np))  # Number of elements in y_train_full_np

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = np.dot(train_full_np, m) + c  # The current predicted value of Y
    D_m = (-2/n) * np.dot(train_full_np.T, (y_train_full_np - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * np.sum(y_train_full_np - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print(m, c)



# Predicting the meter readings for the test set
Y_pred_test = np.dot(test_full_np, m) + c

# Converting the predictions back from log scale
Y_pred_test = np.expm1(Y_pred_test)

# Calculating the Mean Squared Error for the training set
Y_pred_train = np.dot(train_full_np, m) + c
train_error = np.mean((y_train_full_np - Y_pred_train)**2)
print("Training MSE:", train_error)

# Creating the submission DataFrame
submission_df = pd.DataFrame({
    "row_id": row_ids,
    "meter_reading": Y_pred_test
})

# Save to CSV file
submission_df.to_csv("submission_simpleLinear.csv", index=False)

print("CSV file 'submission_simpleLinear.csv' created successfully!")


from sklearn.linear_model import SGDRegressor

# Lasso using Stochastic Gradient Descent
lasso_sgd = SGDRegressor(penalty="l1", alpha=0.001, eta0=0.01, learning_rate="constant", max_iter=30)
lasso_sgd.fit(train_full_np, y_train_full_np)

print("Coefficients:", lasso_sgd.coef_)
print("Intercept:", lasso_sgd.intercept_)

# Predicting the meter readings for the training set
Y_pred_train = lasso_sgd.predict(train_full_np)

# Calculating the Mean Squared Error for the training set
train_error = np.mean((y_train_full_np - Y_pred_train)**2)
print("Training MSE:", train_error)


# Predicting the meter readings for the test set
Y_pred_test = lasso_sgd.predict(test_full_np)

# Converting the predictions back from log scale
Y_pred_test = np.expm1(Y_pred_test)

# Creating the submission DataFrame
submission_df = pd.DataFrame({
    "row_id": row_ids,
    "meter_reading": Y_pred_test
})

# Save to CSV file
submission_df.to_csv("submission_lasso.csv", index=False)

print("CSV file 'submission_lasso.csv' created successfully!")


from sklearn.linear_model import SGDRegressor

ridge_sgd = SGDRegressor(penalty="l2", alpha=0.001, learning_rate="constant", eta0=0.01, max_iter=30)
ridge_sgd.fit(train_full_np, y_train_full_np)

print("Coefficients:", ridge_sgd.coef_)
print("Intercept:", ridge_sgd.intercept_)
# Predicting the meter readings for the training set
Y_pred_train = ridge_sgd.predict(train_full_np)

# Calculating the Mean Squared Error for the training set
train_error = np.mean((y_train_full_np - Y_pred_train)**2)
print("Training MSE:", train_error)

# Predicting the meter readings for the test set
Y_pred_test = ridge_sgd.predict(test_full_np)

# Converting the predictions back from log scale
Y_pred_test = np.expm1(Y_pred_test)

# Creating the submission DataFrame
submission_df = pd.DataFrame({
    "row_id": row_ids,
    "meter_reading": Y_pred_test
})

# Save to CSV file
submission_df.to_csv("submission_ridge.csv", index=False)

print("CSV file 'submission_ridge.csv' created successfully!")

