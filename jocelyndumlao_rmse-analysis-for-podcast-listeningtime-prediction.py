import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')



# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



train_df.head().style.background_gradient(cmap='plasma')


test_df.head().style.background_gradient(cmap='plasma')


# --- Data Exploration and Visualization ---
# Descriptive Statistics
print("Train data descriptive statistics:")
train_df.describe().style.background_gradient(cmap='tab20c')


print("\nTest data descriptive statistics:")
test_df.describe().style.background_gradient(cmap='tab20c')


# 2. Missing Value Analysis
print("\nMissing values in train data:")
print(train_df.isnull().sum())

print("\nMissing values in test data:")
print(test_df.isnull().sum())



# 3. Target Variable Distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(train_df['Listening_Time_minutes'], kde=True)
plt.title('Distribution of Listening Time (Train)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# 4. Genre vs. Listening Time (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=train_df)
plt.title('Listening Time by Genre')
plt.xlabel('Genre')
plt.ylabel('Listening Time (minutes)')
plt.xticks(rotation=45, ha='right')
plt.show()

# 5. Host Popularity vs. Listening Time (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Host_Popularity_percentage', y='Listening_Time_minutes', data=train_df)
plt.title('Listening Time vs. Host Popularity')
plt.xlabel('Host Popularity (%)')
plt.ylabel('Listening Time (minutes)')
plt.show()

# 6. Publication Day vs. Listening Time (Bar Plot)
plt.figure(figsize=(10, 6))
sns.barplot(x='Publication_Day', y='Listening_Time_minutes', data=train_df)
plt.title('Listening Time by Publication Day')
plt.xlabel('Publication Day')
plt.ylabel('Listening Time (minutes)')
plt.show()

# 7. Episode Length vs. Listening Time (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Episode_Length_minutes', y='Listening_Time_minutes', data=train_df)
plt.title('Listening Time vs. Episode Length')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()


# 8. Correlation Heatmap (Numerical Features)
numerical_features = train_df.select_dtypes(include=np.number).columns
correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# 9. Multiple Line Charts in Subplots: Host & Guest Popularity Over Time (Simulated Time)
# Creating a simulated time component for demonstration
train_df['Time'] = np.arange(len(train_df))
test_df['Time'] = np.arange(len(test_df))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

sns.lineplot(ax=axes[0], x='Time', y='Host_Popularity_percentage', data=train_df, label='Host Popularity')
axes[0].set_title('Host Popularity Over Time')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Popularity (%)')
axes[0].text(0.5, 0.9, f"Max: {train_df['Host_Popularity_percentage'].max():.2f}", transform=axes[0].transAxes, ha='center')  # Max value annotation

sns.lineplot(ax=axes[1], x='Time', y='Guest_Popularity_percentage', data=train_df, label='Guest Popularity', color='orange')
axes[1].set_title('Guest Popularity Over Time')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Popularity (%)')
axes[1].text(0.5, 0.9, f"Max: {train_df['Guest_Popularity_percentage'].max():.2f}", transform=axes[1].transAxes, ha='center')  # Max value annotation

plt.tight_layout()
plt.show()


# 10. Stacked Bar Charts in Subplots: Genre Proportions
genre_counts_train = train_df['Genre'].value_counts(normalize=True) * 100
genre_counts_test = test_df['Genre'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

genre_counts_train.plot(kind='bar', stacked=True, ax=axes[0], color=plt.cm.Paired.colors[:len(genre_counts_train)])
axes[0].set_title('Genre Proportions in Training Data')
axes[0].set_ylabel('Percentage')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(genre_counts_train):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center') # Value Labels

genre_counts_test.plot(kind='bar', stacked=True, ax=axes[1], color=plt.cm.Paired.colors[:len(genre_counts_test)])
axes[1].set_title('Genre Proportions in Test Data')
axes[1].set_ylabel('Percentage')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(genre_counts_test):
    axes[1].text(i, v + 1, f"{v:.1f}%", ha='center') # Value Labels


plt.tight_layout()
plt.show()

# 11. Side-by-Side Histograms: Compare distributions of Host Popularity
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

sns.histplot(train_df['Host_Popularity_percentage'].dropna(), ax=axes[0], color='skyblue', label='Train', kde=True)
axes[0].set_title('Host Popularity Distribution (Train)')
axes[0].set_xlabel('Host Popularity (%)')
axes[0].set_ylabel('Frequency')

sns.histplot(test_df['Host_Popularity_percentage'].dropna(), ax=axes[1], color='salmon', label='Test', kde=True)
axes[1].set_title('Host Popularity Distribution (Test)')
axes[1].set_xlabel('Host Popularity (%)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()




# --- Data Preprocessing ---

#  Handle Missing Values
# Impute numerical features with the mean
numerical_cols_train = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_cols_train.remove('Listening_Time_minutes')  # Remove target variable

numerical_cols_test = test_df.select_dtypes(include=np.number).columns.tolist()


# Ensure that only the common numerical columns are transformed
common_numerical_cols = list(set(numerical_cols_train) & set(numerical_cols_test))

imputer_numerical = SimpleImputer(strategy='mean')
imputer_numerical.fit(train_df[common_numerical_cols])  # Fit on common columns in training data

train_df[common_numerical_cols] = imputer_numerical.transform(train_df[common_numerical_cols])
test_df[common_numerical_cols] = imputer_numerical.transform(test_df[common_numerical_cols])


# Impute categorical features with the most frequent value
categorical_cols = train_df.select_dtypes(exclude=np.number).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = imputer_categorical.fit_transform(train_df[categorical_cols])
test_df[categorical_cols] = imputer_categorical.transform(test_df[categorical_cols])


#  Encode Categorical Features
# Use Label Encoding for simplicity (can be improved with other encoding methods)
for col in train_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  # Use the same encoder fitted on the training data


# Feature Scaling (Standardization)
scaler = StandardScaler()
X = train_df.drop('Listening_Time_minutes', axis=1)
y = train_df['Listening_Time_minutes']
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)



# --- Baseline Model ---
# 1. Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Make Predictions on Validation Set
y_pred_val = model.predict(X_val)

# 4. Evaluate the Model (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')



# --- Prediction and Submission ---

# 1. Make Predictions on the Test Set
predictions = model.predict(test_scaled)

# 2. Create Submission File
submission_df = pd.DataFrame({'id': submission['id'], 'Listening_Time_minutes': predictions})
submission_df.to_csv('submission.csv', index=False)

# 3. Display the Head of the Submission File
print("\nSubmission file head:")
print(pd.read_csv('submission.csv').head())



# --- Results Plot ---
# Visualize Predicted vs. Actual Listening Times (on Validation Set)
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_val, alpha=0.5)
plt.xlabel('Actual Listening Time (minutes)')
plt.ylabel('Predicted Listening Time (minutes)')
plt.title('Actual vs. Predicted Listening Time (Validation Set)')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)  # Diagonal line for reference
plt.show()




