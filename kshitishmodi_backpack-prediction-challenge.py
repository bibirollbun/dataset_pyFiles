# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_original_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_ex_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


print(train_original_df.info())
print(train_original_df.isnull().sum())
print(train_original_df.head())


print(test_df.info())
print(test_df.isnull().sum())
print(test_df.head())


print(train_ex_df.info())
print(train_ex_df.isnull().sum())
print(train_ex_df.head())


train_df=pd.concat([train_original_df, train_ex_df], ignore_index=True)
train_df.shape


train_df.shape, train_ex_df.shape, train_original_df.shape, test_df.shape


# Drop ID column (since it's not a numerical feature)
id_column = "id"  # Change this if necessary
if id_column in train_df.columns:
    train_df.drop(columns=[id_column], axis=1, inplace=True)
    
train_df.shape


#for train
train_df['Compartments'] = train_df['Compartments'].astype('int8') # because only 1 to 10 values
object_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
int_columns = [col for col in train_df.columns if train_df[col].dtype == 'int8']
float_columns = [col for col in train_df.columns if train_df[col].dtype == 'float']

for col in object_columns:
    train_df[col]=train_df[col].astype('category')
for col in float_columns:
    train_df[col]=train_df[col].astype('float32')


print(object_columns, int_columns, float_columns)


#for test
test_df['Compartments'] = test_df['Compartments'].astype('int8') # because only 1 to 10 values
object_columns = [col for col in test_df.columns if test_df[col].dtype == 'object']
int_columns = [col for col in test_df.columns if test_df[col].dtype == 'int8']
float_columns = [col for col in test_df.columns if test_df[col].dtype == 'float']

for col in object_columns:
    test_df[col]=test_df[col].astype('category')
for col in float_columns:
    test_df[col]=test_df[col].astype('float32')


print(object_columns, int_columns, float_columns)


train_df.info(memory_usage='deep')


test_df.info(memory_usage='deep')


for column in train_df.columns:
    unique_values = train_df[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


for column in test_df.columns:
    unique_values = test_df[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


unique_counts = train_df.nunique()
print(unique_counts)


unique_counts = test_df.nunique()
print(unique_counts)



train_df.isnull().sum()


test_df.isnull().sum()


# Show rows where 'Brand' is null
#no_brand_material_value = test_df[train_df['Brand'].isnull() & test_df['Material'].isnull() & train_df['Size'].isnull() & train_df['Color'].isnull()]

# Display the rows with missing 'Brand' values
#no_brand_material_value


#no_weight = train_df[train_df['Weight Capacity (kg)'].isnull()]
#no_weight


# Count the number of null values in each row and filter out rows with at least 3 null values
train_df = train_df[train_df.isnull().sum(axis=1) < 3]
train_df.shape


# Count the number of null values in each row and filter out rows with at least 3 null values
test_df = test_df[test_df.isnull().sum(axis=1) < 3]
test_df.shape


#for test
# Ensure 'Size' and 'Laptop Compartment' include 'Other' in their unique values
test_df['Size'] = test_df['Size'].astype('category')
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].astype('category')
test_df['Style'] = test_df['Style'].astype('category')

# Add 'Other' to the categories
test_df['Size'] = test_df['Size'].cat.add_categories('Other')
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].cat.add_categories('Other')
test_df['Style'] = test_df['Style'].cat.add_categories('Other')

# Fill NaN values with 'Other'
test_df['Size'] = test_df['Size'].fillna('Other')
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].fillna('Other')
test_df['Style'] = test_df['Style'].fillna('Other')

# Check the updated columns
test_df.isnull().sum()


#for train
# Ensure 'Size' and 'Laptop Compartment' include 'Other' in their unique values
train_df['Size'] = train_df['Size'].astype('category')
train_df['Laptop Compartment'] = train_df['Laptop Compartment'].astype('category')
train_df['Style'] = train_df['Style'].astype('category')

# Add 'Other' to the categories
train_df['Size'] = train_df['Size'].cat.add_categories('Other')
train_df['Laptop Compartment'] = train_df['Laptop Compartment'].cat.add_categories('Other')
train_df['Style'] = train_df['Style'].cat.add_categories('Other')

# Fill NaN values with 'Other'
train_df['Size'] = train_df['Size'].fillna('Other')
train_df['Laptop Compartment'] = train_df['Laptop Compartment'].fillna('Other')
train_df['Style'] = train_df['Style'].fillna('Other')

# Check the updated columns
train_df.isnull().sum()


# Compute the mean while skipping NaN values
#avg_capacity = train_df['Weight Capacity (kg)'].astype('float32').mean(skipna=True)

# Display the calculated mean
#print("Mean is : ",avg_capacity)
# Fill NaN values with the computed mean
#train_df['Weight Capacity (kg)'] = train_df['Weight Capacity (kg)'].fillna(avg_capacity)


# Remove rows where 'Weight Capacity' is NaN
train_df = train_df.dropna(subset=['Weight Capacity (kg)'])


test_df = test_df.dropna(subset=['Weight Capacity (kg)'])


# Count occurrences of each color (excluding NaN values)
color_counts = train_df['Color'].value_counts(normalize=True)

# Display probability percentages
print(color_counts * 100)  # Convert to percentage

# Get the list of unique colors and their probabilities
colors = color_counts.index.tolist()  # ['Black', 'Blue', 'Gray', 'Green', 'Pink', 'Red']
probabilities = color_counts.values   # Corresponding probabilities

# Randomly assign missing values using np.random.choice()
train_df.loc[train_df['Color'].isna(), 'Color'] = np.random.choice(colors, 
                                                                   size=train_df['Color'].isna().sum(), 
                                                                   p=probabilities)

print(train_df['Color'].isna().sum())  # Should be 0 if all NaNs were replaced


# Count occurrences of each color (excluding NaN values)
color_counts = test_df['Color'].value_counts(normalize=True)

# Display probability percentages
print(color_counts * 100)  # Convert to percentage

# Get the list of unique colors and their probabilities
colors = color_counts.index.tolist()  # ['Black', 'Blue', 'Gray', 'Green', 'Pink', 'Red']
probabilities = color_counts.values   # Corresponding probabilities

# Randomly assign missing values using np.random.choice()
test_df.loc[test_df['Color'].isna(), 'Color'] = np.random.choice(colors, 
                                                                   size=test_df['Color'].isna().sum(), 
                                                                   p=probabilities)

print(test_df['Color'].isna().sum())  # Should be 0 if all NaNs were replaced


#for test
import pandas as pd
from sklearn.impute import KNNImputer

# Create a copy of the dataset
df_knn = test_df.copy()

# One-hot encode Brand & Material (drop_first=True prevents dummy variable trap)
df_knn = pd.get_dummies(df_knn, columns=['Brand', 'Material'], drop_first=False)

# Store original column names before imputation
brand_columns = [col for col in df_knn.columns if col.startswith('Brand_')]
material_columns = [col for col in df_knn.columns if col.startswith('Material_')]

# Initialize KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN Imputer only on one-hot encoded Brand & Material columns
df_knn[brand_columns + material_columns] = knn_imputer.fit_transform(df_knn[brand_columns + material_columns])

# Convert Brand back to categorical labels
df_knn['Brand'] = df_knn[brand_columns].idxmax(axis=1).str.replace('Brand_', '')

# Convert Material back to categorical labels
df_knn['Material'] = df_knn[material_columns].idxmax(axis=1).str.replace('Material_', '')

# Drop one-hot encoded columns
df_knn.drop(columns=brand_columns + material_columns, inplace=True)

# Assign imputed values back to the original dataset
test_df['Brand'] = df_knn['Brand']
test_df['Material'] = df_knn['Material']

test_df.isnull().sum()


#for train
import pandas as pd
from sklearn.impute import KNNImputer

# Create a copy of the dataset
df_knn = train_df.copy()

# One-hot encode Brand & Material (drop_first=True prevents dummy variable trap)
df_knn = pd.get_dummies(df_knn, columns=['Brand', 'Material'], drop_first=False)

# Store original column names before imputation
brand_columns = [col for col in df_knn.columns if col.startswith('Brand_')]
material_columns = [col for col in df_knn.columns if col.startswith('Material_')]

# Initialize KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN Imputer only on one-hot encoded Brand & Material columns
df_knn[brand_columns + material_columns] = knn_imputer.fit_transform(df_knn[brand_columns + material_columns])

# Convert Brand back to categorical labels
df_knn['Brand'] = df_knn[brand_columns].idxmax(axis=1).str.replace('Brand_', '')

# Convert Material back to categorical labels
df_knn['Material'] = df_knn[material_columns].idxmax(axis=1).str.replace('Material_', '')

# Drop one-hot encoded columns
df_knn.drop(columns=brand_columns + material_columns, inplace=True)

# Assign imputed values back to the original dataset
train_df['Brand'] = df_knn['Brand']
train_df['Material'] = df_knn['Material']

train_df.isnull().sum()


#for test
from sklearn.linear_model import LogisticRegression

features = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Style', 'Color', 'Weight Capacity (kg)']

train_data = test_df.dropna(subset=['Waterproof'])
test_data = test_df[test_df['Waterproof'].isna()]

train_X = pd.get_dummies(train_data[features], drop_first=True)
train_y = train_data['Waterproof'].map({'Yes': 1, 'No': 0})  # Convert to binary

log_model = LogisticRegression()
log_model.fit(train_X, train_y)

test_X = pd.get_dummies(test_data[features], drop_first=True).reindex(columns=train_X.columns, fill_value=0)
predicted_waterproof = log_model.predict(test_X)

# Convert NumPy array to categorical labels
test_df.loc[test_df['Waterproof'].isna(), 'Waterproof'] = ['Yes' if x == 1 else 'No' for x in predicted_waterproof]

test_df.isnull().sum()


#for train
from sklearn.linear_model import LogisticRegression

features = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Style', 'Color', 'Weight Capacity (kg)']

train_data = train_df.dropna(subset=['Waterproof'])
test_data = train_df[train_df['Waterproof'].isna()]

train_X = pd.get_dummies(train_data[features], drop_first=True)
train_y = train_data['Waterproof'].map({'Yes': 1, 'No': 0})  # Convert to binary

log_model = LogisticRegression()
log_model.fit(train_X, train_y)

test_X = pd.get_dummies(test_data[features], drop_first=True).reindex(columns=train_X.columns, fill_value=0)
predicted_waterproof = log_model.predict(test_X)

# Convert NumPy array to categorical labels
train_df.loc[train_df['Waterproof'].isna(), 'Waterproof'] = ['Yes' if x == 1 else 'No' for x in predicted_waterproof]

train_df.isnull().sum()


# Display basic information
print("Dataset Info:")
print(train_df.info())

# Summary statistics
print("\nSummary Statistics:")
print(train_df.describe())

# Check for missing values
print("\nMissing Values:")
print(train_df.isnull().sum())


# Display basic information
print("Dataset Info:")
print(test_df.info())

# Summary statistics
print("\nSummary Statistics:")
print(test_df.describe())

# Check for missing values
print("\nMissing Values:")
print(test_df.isnull().sum())


# Histogram of numerical features
train_df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplot for detecting outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Select only numerical columns for correlation
num_cols = train_df.select_dtypes(include=['int8', 'float32']).columns

# Compute correlation matrix
corr_matrix = train_df[num_cols].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Numerical Features Only)")
plt.show()



# Split back into train and test
'''train_data = train_df.iloc[:len(train_original_df)]
test_data = train_df.iloc[len(train_original_df):].drop(columns=['Price'], errors='ignore')'''


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data
X = train_data.drop(columns=['Price'])
y = train_data['Price']
X = pd.get_dummies(X, drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with parameter tuning
param_grid_ridge = {'alpha': np.logspace(-3, 3, 10)}
param_grid_lasso = {'alpha': np.logspace(-3, 3, 10)}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": GridSearchCV(Ridge(), param_grid=param_grid_ridge, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1),
    "Lasso Regression": GridSearchCV(Lasso(), param_grid=param_grid_lasso, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train & evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'RÂ²': r2}
    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, RÂ²: {r2:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Plot heatmap of correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Plot histogram of residuals
y_pred_final = models["Linear Regression"].predict(X_val)
residuals = y_val - y_pred_final

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="skyblue")
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Store RMSE values
rmse_scores = {
    "Linear Regression": 38.9211,
    "Ridge Regression": 38.9217,
    "Lasso Regression": 38.9215,
    "Gradient Boosting": 38.9171
}

# Convert to DataFrame for easy plotting
import pandas as pd
rmse_df = pd.DataFrame(list(rmse_scores.items()), columns=["Model", "RMSE"])

# Plot RMSE values
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="RMSE", data=rmse_df, palette="viridis")
plt.xlabel("Model Type")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Comparison of RMSE Across Models")
plt.ylim(min(rmse_scores.values()) - 0.1, max(rmse_scores.values()) + 0.1)  # Keep Y-axis tight
plt.show()



# Select the best model based on RMSE
best_model_name = results_df['RMSE'].idxmin()
best_model = models[best_model_name]

print(f"Best model selected: {best_model_name}")



# Drop 'Price' from training data and one-hot encode features
test_X = test_df.drop(columns=['id'])  # Drop ID but keep all other features
test_X = pd.get_dummies(test_X, drop_first=True)


# Make predictions on test data
test_predictions = best_model.predict(test_X)
print(test_predictions[:10])


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],  # Retain ID column
    'Price': test_predictions
})

# Save to CSV (Kaggle submission format)
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' is ready!")



submission.head()

