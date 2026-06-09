# !pip install wordcloud --quiet


# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import skew, kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from xgboost import XGBRegressor

# Display settings
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)  # Show all columns in DataFrame outputs


# Load the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Display the first few rows
train_df.head()


extra_train_df.head()


train_df = pd.concat([train_df, extra_train_df], ignore_index=True)
train_df


# Check basic information about the dataset
train_df.info()


# Get statistical summary of numerical columns
train_df.describe()


# Check for missing values
train_df.isnull().sum()


# Check missing values
missing_values = train_df.isnull().sum()
missing_values[missing_values > 0]


# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Training Data')
plt.show()


# Fill missing categorical values with mode
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# Fill missing numerical values with median
numerical_cols = ["Compartments", "Weight Capacity (kg)"]
for col in numerical_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)
    test_df[col].fillna(test_df[col].median(), inplace=True)


# Check missing values
missing_values = train_df.isnull().sum()
missing_values[missing_values > 0]


# Check missing values
missing_values = test_df.isnull().sum()
missing_values[missing_values > 0]


# Get statistical summary of numerical columns
train_df.describe()


# Checking skewness
print(f"Skewness of Price: {skew(train_df['Price'])}")
print(f"Kurtosis of Price: {kurtosis(train_df['Price'])}")


plt.figure(figsize=(10, 5))
sns.histplot(train_df["Price"], bins=50, kde=True, color="blue")
plt.title("Distribution of Backpack Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=train_df["Price"], color="red")
plt.title("Boxplot of Price")
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y=train_df["Brand"], order=train_df["Brand"].value_counts().index, palette="coolwarm")
plt.title("Distribution of Brands")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x=train_df["Material"], order=train_df["Material"].value_counts().index, palette="viridis")
plt.xticks(rotation=45)
plt.title("Distribution of Materials Used in Backpacks")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x=train_df["Style"], order=train_df["Style"].value_counts().index, palette="magma")
plt.title("Distribution of Backpack Styles")
plt.xticks(rotation=45)
plt.show()


wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(train_df["Color"].astype(str)))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Colors in Backpacks")
plt.show()


# Compute correlations between numerical features and the target variable
correlation_matrix = train_df[numerical_cols].corr()
correlation_matrix


plt.figure(figsize=(10, 8))
sns.heatmap(train_df[["Compartments", "Weight Capacity (kg)","Price"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# Plot relationships between categorical features and the target variable
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x=col, y='Price', palette='Set2')
    plt.title(f'Price vs {col}')
    plt.xticks(rotation=45)
    plt.show()


# List of numerical columns
numerical_cols = ['Compartments', 'Weight Capacity (kg)']


sns.pairplot(train_df[["Price", "Weight Capacity (kg)", "Compartments"]])
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(x=train_df["Weight Capacity (kg)"], y=train_df["Price"], alpha=0.5)
plt.title("Price vs Weight Capacity")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x=train_df["Compartments"], y=train_df["Price"], palette="Blues")
plt.title("Price vs Number of Compartments")
plt.show()


# Plot distributions of numerical features
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df[col], kde=True, bins=30, color='green')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# Boxplots for numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=train_df[col], color='purple')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()


# Encode categorical features
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le


sns.pairplot(train_df)
plt.show()


# Compute correlations between numerical features and the target variable
correlation_matrix = train_df.corr()
correlation_matrix


# Correlation of numerical features with the target variable
print("Correlation with Price:")
display(correlation_matrix['Price'].sort_values(ascending=False))


# Define features and target
X = train_df.drop(["id", "Price"], axis=1)
y = train_df["Price"]

# Prepare test dataset
X_test = test_df.drop("id", axis=1)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)


# Predictions on validation set
y_val_pred = xgb_model.predict(X_val)

# Compute RMSE
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


# Predict on test data
test_predictions = xgb_model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({"id": test_df["id"], "Price": test_predictions})
submission.to_csv("submission.csv", index=False)

# Display first few rows of submission file
submission.head()


# # Create new features if necessary (example: total compartments)
# train_df['Total_Compartments'] = train_df['Compartments'] + train_df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
# test_df['Total_Compartments'] = test_df['Compartments'] + test_df['Laptop Compartment'].map({'Yes': 1, 'No': 0})

# # Check the new feature
# print("New Feature - Total Compartments:")
# display(train_df[['Compartments', 'Laptop Compartment', 'Total_Compartments']].head())




