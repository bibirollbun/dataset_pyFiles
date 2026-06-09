# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/mastering-ordinal-regression-with-wine-data/train.csv")
df_test = pd.read_csv("/kaggle/input/mastering-ordinal-regression-with-wine-data/test.csv")



df_train.head(3),df_test.head(3)


df_train.info(),df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_train.describe()


features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']


import seaborn as sns
import matplotlib.pyplot as plt
# Create a boxplot for the features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)  # Arrange plots in a 4x3 grid
    sns.boxplot(y=df_train[feature])
    plt.title(feature)
    plt.tight_layout()  # Adjust layout

# Show the plot
plt.show()


# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each feature
df_cleaned = df_train.copy()
for feature in features:
    df_cleaned = remove_outliers(df_cleaned, feature)

# Create separate boxplots for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)  # Arrange plots in a 4x3 grid
    sns.boxplot(y=df_cleaned[feature])
    plt.title(feature)
    plt.tight_layout()  # Adjust layout

# Show the plot
plt.show()


plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(4, 3, i+1)  # Arrange plots in a 4x3 grid
    sns.histplot(df_cleaned[feature], kde=True)  # 'kde=True' adds the kernel density estimate
    plt.title(feature)
    plt.tight_layout()  # Adjust layout to prevent overlap

# Show the plot
plt.show()


corr_matrix = df_cleaned[features].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


selected_features = ['fixed acidity', 'alcohol', 'citric acid', 'sulphates', 'quality']

sns.pairplot(df_cleaned[selected_features], hue='quality')
plt.show()


df_cleaned['id'] = df_train['id'].copy()


df_cleaned.columns



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to calculate Quadratic Weighted Kappa (QWK)
def quadratic_weighted_kappa(y_true, y_pred):
    # Ensure the values are integers
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    # Find the number of unique classes in y_true and y_pred
    min_class = min(y_true.min(), y_pred.min())
    max_class = max(y_true.max(), y_pred.max())
    
    # Create confusion matrix with size based on the number of unique classes
    hist = np.zeros((max_class - min_class + 1, max_class - min_class + 1), dtype=np.float64)
    
    for i in range(len(y_true)):
        hist[y_true[i] - min_class, y_pred[i] - min_class] += 1

    # Normalize the confusion matrix
    hist = hist / hist.sum()

    # Calculate the quadratic weighted kappa
    # Create an array for the squared differences
    diff_matrix = (np.arange(min_class, max_class + 1)[:, None] - np.arange(min_class, max_class + 1)) ** 2
    num = np.sum(diff_matrix * hist)
    denom = np.sum(diff_matrix * hist.sum(axis=0))

    return 1 - (num / denom)

# Load your data (assuming it's already in the DataFrame 'df')
# Features and target column
features = ['id','fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
X = df_cleaned[features]
y = df_cleaned['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (Standardization)
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the regression model (RandomForestRegressor in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Convert continuous predictions to discrete values (since QWK is for ordinal data)
y_pred_discrete = np.round(y_pred,1).astype(int)

# Calculate Quadratic Weighted Kappa (QWK) metric
qwk = quadratic_weighted_kappa(y_test, y_pred_discrete)
print(f'Quadratic Weighted Kappa (QWK): {qwk}')

# You can also use mean squared error (MSE) for additional evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')



y_test_pred = model.predict(df_test)
y_test_pred=  np.ceil(y_test_pred)


submission_df = pd.read_csv('/kaggle/input/mastering-ordinal-regression-with-wine-data/sample_submission.csv')
submission_df['quality'] = y_test_pred
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

