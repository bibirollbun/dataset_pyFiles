data_path = "/kaggle/input/breast-cancer-dataset/breast-cancer.csv" #get the data path
print("Dataset loaded.")


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#reading the data using pandas
data_file = pd.read_csv(data_path)
data_file.head()


#display the dataset info
data_file.info()


#statistics of numerical features
data_file.describe()


#missing values
data_file.isnull().sum()


#checking available columns names
data_file.columns


#drop the useless id column (and ignore any errors)
data_file.drop(columns=['id'], inplace=True, errors='ignore')
print("deleted useless columns")


#checking for the unique values in the diagnosis column (should only contain 'M' for malignant or 'B' for benign)
data_file['diagnosis'].unique()


#checking if physical measurements are negative or zero 
(data_file[['radius_mean', 'area_mean', 'perimeter_mean']] <= 0).sum()


# Convert diagnosis to numeric values
data_file["diagnosis"] = data_file["diagnosis"].map({'M': 1, 'B': 0})

# Compute correlation matrix
corr_matrix = data_file.corr()

# Visualize correlation with heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


#dropping columns that are too correlated with each other (>0.9)

data_file.drop(columns=['radius_worst'], inplace=True, errors='ignore')

data_file.drop(columns=['texture_worst'], inplace=True, errors='ignore')

data_file.drop(columns=['perimeter_worst'], inplace=True, errors='ignore')

data_file.drop(columns=['area_worst'], inplace=True, errors='ignore')

data_file.drop(columns=['concave points_worst'], inplace=True, errors='ignore')

print("Useless columns dropped")


#checking for extreme outliers in the mean columns 
columns_to_check = ['smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
data_file[columns_to_check].boxplot(figsize=(8,6))
plt.xticks(rotation=70)
plt.show()


#since some extreme outliers exist, we should remove them 

# Calculate Q1, Q3, and IQR for columns you want to check
Q1 = data_file[columns_to_check].quantile(0.25)
Q3 = data_file[columns_to_check].quantile(0.75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds for each feature
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers by filtering rows that are within the bounds
data_file = data_file[~((data_file[columns_to_check] < lower_bound) | (data_file[columns_to_check] > upper_bound)).any(axis=1)]

#visualizing the cleaned columns to compare before and after
data_file[columns_to_check].boxplot(figsize=(8,6))
plt.xticks(rotation = 70)
plt.show()


# Get absolute correlation with target
correlation_with_target = corr_matrix["diagnosis"].abs().sort_values(ascending=False)

# Select top features with correlation > 0.5
selected_features = correlation_with_target[correlation_with_target > 0.5].index.tolist()
selected_features.remove("diagnosis")  # Exclude the target variable itself

print("Selected Features:", selected_features)


# Keep only selected features + target variable
data_selected = data_file[selected_features + ["diagnosis"]]


#devide the data into numerical features and categorical features
numerical_features = data_file.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data_file.select_dtypes(include=['object']).columns

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# Plot histogram for all numerical columns
data_file[numerical_features].hist(figsize=(12, 10), bins=20)
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply normalization to numerical features
data_file[numerical_features] = scaler.fit_transform(data_file[numerical_features])

print("Numerical features normalized.")


#encode categorical features

for col in categorical_features:
    data_file[col] = LabelEncoder().fit_transform(data_file_cleaned[col])


#graphing every column with effect on diagnosis 

for col in numerical_features:
    if col == 'diagnosis':
        continue
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data_file, x=col, hue='diagnosis', fill=True)
    plt.title(f'Relationship between {col} and Diagnosis')
    plt.show()


correlation_with_diagnosis = data_file_cleaned.corr()['diagnosis'].sort_values(ascending=False)
print(correlation_with_diagnosis)

