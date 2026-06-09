import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression



print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-Learn version:", sklearn.__version__)

# Verify Matplotlib
plt.figure(figsize=(3,3))
plt.plot([1, 2, 3], [4, 5, 6], marker='o', linestyle='--', color='b')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Matplotlib Verification")
plt.grid(True)
plt.show()

# Verify Seaborn
sns.set_style("darkgrid")
sample_data = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 25, 30]})
plt.figure(figsize=(4,3))
sns.lineplot(data=sample_data, x='x', y='y', marker='o', color='r')
plt.title("Seaborn Verification")
plt.show()



# Load the training dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")

# Display basic information
data_info = pd.DataFrame({
    "Column Name": train_df.columns,
    "Data Type": train_df.dtypes.values,
    "Non-Null Count": train_df.count().values,
    "Missing Values": train_df.isnull().sum().values
})

# Display data info as a table
print("\nğŸ“Œ Dataset Information")
display(data_info.style.set_table_attributes("style='display:inline'"))
#set_caption("ğŸ“Š Dataset Information"))

# Display the first few rows of the dataset
print("\nğŸ“Œ First 5 Rows")
display(train_df.head().style.set_table_attributes("style='display:inline'"))
#set_caption("ğŸ“‹ First 5 Rows"))

# Pictorial Representation: Show sample data (Histogram)
print("\nğŸ“Œ Histogram Representation of Price")
plt.figure(figsize=(6,3))
plt.hist(train_df['Price'], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Histogram of Price Distribution")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Check for missing values (Bar Chart)
print("\nğŸ“Œ Bar Chart Representation of Missing Values")
missing_values = train_df.isnull().sum()
plt.figure(figsize=(6,3))
plt.bar(missing_values.index, missing_values.values, color="crimson")
plt.xticks(rotation=90)
plt.ylabel("Count of Missing Values")
plt.title("Missing Values in Each Column")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Load the extra training dataset
extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Display basic information
extra_data_info = pd.DataFrame({
    "Column Name": extra_train_df.columns,
    "Data Type": extra_train_df.dtypes.values,
    "Non-Null Count": extra_train_df.count().values,
    "Missing Values": extra_train_df.isnull().sum().values
})

# Display data info as a table
print("\nğŸ“Œ Dataset Information for training_extra.csv")
display(extra_data_info.style.set_table_attributes("style='display:inline'"))

# Display the first few rows of the dataset
print("\nğŸ“Œ First 5 Rows of training_extra.csv")
display(extra_train_df.head().style.set_table_attributes("style='display:inline'"))

# Pictorial Representation: Show sample data (Histogram)
print("\nğŸ“Œ Histogram Representation of Price (training_extra.csv)")
plt.figure(figsize=(6,3))
plt.hist(extra_train_df['Price'], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Histogram of Price Distribution (training_extra.csv)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Check for missing values (Bar Chart)
print("\nğŸ“Œ Bar Chart Representation of Missing Values (training_extra.csv)")
extra_missing_values = extra_train_df.isnull().sum()
plt.figure(figsize=(6,3))
plt.bar(extra_missing_values.index, extra_missing_values.values, color="crimson")
plt.xticks(rotation=90)
plt.ylabel("Count of Missing Values")
plt.title("Missing Values in Each Column (training_extra.csv)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Load the test dataset
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Display basic information
test_data_info = pd.DataFrame({
    "Column Name": test_df.columns,
    "Data Type": test_df.dtypes.values,
    "Non-Null Count": test_df.count().values,
    "Missing Values": test_df.isnull().sum().values
})

# Display data info as a table
print("\nğŸ“Œ Dataset Information for test.csv")
display(test_data_info.style.set_table_attributes("style='display:inline'"))

# Display the first few rows of the dataset
print("\nğŸ“Œ First 5 Rows of test.csv")
display(test_df.head().style.set_table_attributes("style='display:inline'"))

# Check for missing values (Bar Chart)
print("\nğŸ“Œ Bar Chart Representation of Missing Values (test.csv)")
test_missing_values = test_df.isnull().sum()
plt.figure(figsize=(6,3))
plt.bar(test_missing_values.index, test_missing_values.values, color="crimson")
plt.xticks(rotation=90)
plt.ylabel("Count of Missing Values")
plt.title("Missing Values in Each Column (test.csv)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



from sklearn.impute import SimpleImputer

# Identify numeric and categorical columns
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('Price', errors='ignore')
categorical_cols = train_df.select_dtypes(include=['object']).columns

# Define imputers
num_imputer = SimpleImputer(strategy='median')  # Median for numeric
cat_imputer = SimpleImputer(strategy='most_frequent')  # Mode for categorical

# Apply imputers on train, extra_train, and test data
for df in [train_df, extra_train_df, test_df]:
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\nâœ… Missing Values Filled Successfully!")



# Define the function to check missing values
def check_missing_values(df, df_name):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0] 
    
    if missing_values.empty:
        print(f"\nâœ… No missing values in {df_name}!")
    else:
        print(f"\nâš ï¸� Missing values found in {df_name}:\n{missing_values}")

# Now call the function
check_missing_values(train_df, "Train Data (Cleaned)")
check_missing_values(extra_train_df, "Extra Train Data (Cleaned)")
check_missing_values(test_df, "Test Data (Cleaned)")



# Separate features (X) and target label (Y)
target_column = "Price" 

# train_X contains all columns except target
train_X = train_df.drop(columns=[target_column])

# train_Y contains only the target column
train_Y = train_df[target_column]

# Display first few rows
print("\nğŸ“Œ First 5 Rows of train_X (Features)")
display(train_X.head().style.set_table_attributes("style='display:inline'"))

print("\nğŸ“Œ First 5 Rows of train_Y (Target)")
display(train_Y.head().to_frame().style.set_table_attributes("style='display:inline'"))



from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns
categorical_cols = train_X.select_dtypes(include=["object"]).columns

# Apply One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # âœ… Correct
encoded_cats = encoder.fit_transform(train_X[categorical_cols])

# Convert to DataFrame and drop original categorical columns
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
train_X = train_X.drop(columns=categorical_cols).reset_index(drop=True)
train_X = pd.concat([train_X, encoded_df], axis=1)

print("\nâœ… Categorical columns successfully encoded!")



# Initialize Linear Regression model
model = LinearRegression()

# Train the model again
model.fit(train_X, train_Y)

print("\nâœ… Model training completed successfully!")



from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoding using the same encoder used for train data
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoder.fit(train_df[categorical_cols])  # Fit only on train data categories

# Transform categorical features in test data
encoded_test = encoder.transform(test_df[categorical_cols])

# Convert to DataFrame and match column names
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

# Reset index to align with test_df
encoded_test_df.index = test_df.index

# Drop original categorical columns and add encoded columns
test_X = test_df.drop(columns=categorical_cols).join(encoded_test_df)

# Ensure columns are in the same order as training data
test_X = test_X[train_X.columns]

# Now, predict prices using the trained model
test_predictions = model.predict(test_X)

# Round predicted prices to 3 decimal places
test_df['Price'] = test_predictions.round(3)

# Save the results with 'id' and 'Price' as column names
test_df[['id', 'Price']].to_csv("predicted_prices.csv", index=False)

print("\nâœ… Predictions saved successfully in 'predicted_prices.csv' with rounded prices!")


