import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


train = pd.read_csv('/kaggle/input/concrete-strength-regression/train.csv')
train.head()


# Display information about the train dataset
train.info()

# Display summary statistics of the train dataset
train.describe()


# Visualizing correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation matrix
corr = train.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


train[train['age'] == 1]


# Remove all data where age=1
train = train[train['age'] != 1]


# Filter data where age is equal to 1
train_age_1 = train[train['age'] == 1]
train_age_1


# Visualizing csMPa with cement
sns.scatterplot(x='cement', y='csMPa', data=train)
plt.title('Concrete Strength vs. Cement')
plt.xlabel('Cement')
plt.ylabel('Concrete Strength (csMPa)')
plt.show()


train_age_7 = train[train['age'] == 7]
train_age_7


# Visualize 'csMPa' and 'cement' for age=7
sns.scatterplot(x='cement', y='csMPa', data=train_age_7)
plt.title('csMPa vs cement for age=7')
plt.xlabel('Cement')
plt.ylabel('csMPa')
plt.show()


train_age_28 = train[train['age'] == 28]
train_age_28


# Visualizing csMPa and cement for age=28
plt.figure(figsize=(12, 6))
sns.scatterplot(x='cement', y='csMPa', data=train[train['age'] == 28])
plt.title('csMPa vs Cement for Age=28')
plt.xlabel('Cement')
plt.ylabel('csMPa')
plt.show()


# Create subplots
plt.figure(figsize=(12, 5))

# Plot for age=7
plt.subplot(1, 2, 1)
sns.scatterplot(y='cement', x='csMPa', data=train_age_7)
plt.title('Age 7')

# Plot for age=28
plt.subplot(1, 2, 2)
sns.scatterplot(y='cement', x='csMPa', data=train_age_28)
plt.title('Age 28')

plt.tight_layout()
plt.show()


# Create a figure to display the charts side by side
plt.figure(figsize=(12, 5))

# Plot for age=7
plt.subplot(1, 2, 1)
sns.scatterplot(x='csMPa', y='water', data=train[train['age'] == 7])
plt.title('Age=7: csMPa vs Water')

# Plot for age=28
plt.subplot(1, 2, 2)
sns.scatterplot(x='csMPa', y='water', data=train[train['age'] == 28])
plt.title('Age=28: csMPa vs Water')

plt.tight_layout()
plt.show()


cols = train.columns[1:8] 
fig, axs = plt.subplots(2, len(cols), figsize=(4*len(cols), 8))

for row, age in enumerate([7, 28]):
    for i, col in enumerate(cols):
        sns.scatterplot(
            x='csMPa', 
            y=col, 
            data=train[train['age'] == age], 
            ax=axs[row, i]
        )
        axs[row, i].set_title(f"{col} (age={age})")

plt.tight_layout()
plt.show()



# Display the first few rows of the train dataset
train.head()


# Get all column names except 'ROW_ID'
columns_to_plot = train.drop('Row ID', axis=1).columns

# Plot csMPA with all columns in different colors
plt.figure(figsize=(12, 6))
for column in columns_to_plot:
    plt.scatter(train['csMPa'], train[column], label=column)
plt.xlabel('csMPa')
plt.ylabel('Values')
plt.legend()
plt.show()


# Drop the 'ROW_ID' column
train.drop('Row ID', axis=1, inplace=True)


# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Splitting the data into features (X) and target variable (y)
X = train.drop('csMPa', axis=1)
y = train['csMPa']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


from sklearn.ensemble import RandomForestRegressor

# Separate features and target variable
X_train = train.drop('csMPa', axis=1)
y_train = train['csMPa']

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor()

# Fit the model
rf_model.fit(X_train, y_train)


test_data = pd.read_csv('/kaggle/input/concrete-strength-regression/test.csv')
test_data.head()


# Drop the 'Row ID' column from the train dataset
test_data.drop('Row ID', axis=1, inplace=True)


# Get feature columns only (exclude the target)
feature_cols = [col for col in train.columns if col != 'csMPa']

# Align test_data to match training features
test_data = test_data[feature_cols]

# Predict
y_pred = model.predict(test_data)



# Combine into one DataFrame
results_df = pd.DataFrame({
    'y_pred': y_pred
}, index=test_data.index)  # keep alignment with test_data rows

# If you also want to include test_data columns:
results_df = pd.concat([test_data, results_df], axis=1)

print(results_df)




