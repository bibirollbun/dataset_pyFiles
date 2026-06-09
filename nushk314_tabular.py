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


# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up display and plotting
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# 2. Load the data
train = pd.read_csv('/kaggle/input/playground-series-s3e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e8/test.csv')

# 3. Preview the data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain head:\n", train.head())

# 4. Check data types and nulls
print("\nData types:\n", train.dtypes)
print("\nMissing values:\n", train.isnull().sum())

# 5. Summary statistics
print("\nSummary statistics:\n", train.describe())

# 6. Target distribution
target_col = 'Transported' if 'Transported' in train.columns else train.columns[-1]
if train[target_col].dtype == 'bool' or train[target_col].nunique() == 2:
    sns.countplot(x=target_col, data=train)
    plt.title('Target Distribution')
    plt.show()



# 1. Distribution of the target (price)
plt.figure(figsize=(10, 5))
sns.histplot(train['price'], bins=50, kde=True)
plt.title("Distribution of Diamond Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# 2. Count plots for categorical features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(ax=axes[0], x='cut', data=train, order=train['cut'].value_counts().index)
axes[0].set_title('Cut Distribution')

sns.countplot(ax=axes[1], x='color', data=train, order=train['color'].value_counts().index)
axes[1].set_title('Color Distribution')

sns.countplot(ax=axes[2], x='clarity', data=train, order=train['clarity'].value_counts().index)
axes[2].set_title('Clarity Distribution')

plt.tight_layout()
plt.show()

# Select only numeric columns for correlation
numeric_cols = train.select_dtypes(include=np.number)

# 3. Correlation heatmap for numerical features (fixed)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()




from sklearn.preprocessing import OrdinalEncoder

# Drop ID column
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Encode categorical features using OrdinalEncoder
cat_cols = ['cut', 'color', 'clarity']
encoder = OrdinalEncoder()
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# Split target and features
X_train = train.drop(columns=['price'])
y_train = train['price']
X_test = test.copy()

# Confirm preprocessing
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("Encoded categories:", dict(zip(cat_cols, encoder.categories_)))



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Split train data into training and validation sets (80-20 split)
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the model
model_lr = LinearRegression()
model_lr.fit(X_train_part, y_train_part)

# Predict on validation set
y_pred = model_lr.predict(X_valid)

# Evaluate performance
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"Baseline Model Performance:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}")



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Train-test split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the model
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_regressor.fit(X_tr, y_tr)

# Validation predictions
y_val_pred = rf_regressor.predict(X_val)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print(f"Baseline Model - Random Forest Regressor")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get feature importances
importances = rf_regressor.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get feature importances
importances = rf_regressor.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)  # sort ascending for horizontal plot

# Plot
plt.figure(figsize=(10, 6))
bars = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')

# Add importance value labels
for index, row in importance_df.iterrows():
    bars.text(row['Importance'] + 0.001, index, f"{row['Importance']:.4f}", color='black', va='center')

plt.tight_layout()
plt.show()



# Get feature importances
importances = rf_regressor.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for easier reading
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print feature importances
print("Feature Importances:")
for index, row in importance_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")





