# ğŸ“¦ Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler

# ğŸ“� Loading the Data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# ğŸ”� Quick Overview
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print(train.head())
print(train.info())



# ğŸ’¡ EDA - Distribution of Target Variable
plt.figure(figsize=(10,5))
sns.histplot(train['Calories'], bins=50, kde=True, color='orange')
plt.title("Calories Burned Distribution")
plt.show()

# ğŸ’¡ Pairplot for numerical correlation check
sns.pairplot(train.sample(500), diag_kind='kde')
plt.suptitle("Pairplot of Sample Data", y=1.02)
plt.show()

# ğŸ’¡ Fixed: Correlation Heatmap for Numeric Columns Only
plt.figure(figsize=(15,10))
numeric_cols = train.select_dtypes(include=['int64', 'float64'])  # exclude 'Gender'
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()


# ğŸ’¡ Boxplot of Calories vs Gender (if exists)
if 'Gender' in train.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(data=train, x='Gender', y='Calories')
    plt.title("Calories by Gender")
    plt.show()

# ğŸ’¡ Calories vs Age
plt.figure(figsize=(8,5))
sns.scatterplot(data=train, x='Age', y='Calories', hue='Age', palette="viridis")
plt.title("Calories vs Age")
plt.show()

# ğŸ’¡ Calories vs Weight
plt.figure(figsize=(8,5))
sns.scatterplot(data=train, x='Weight', y='Calories', hue='Weight', palette="coolwarm")
plt.title("Calories vs Weight")
plt.show()

# ğŸ’¡ Calories vs Duration
if 'Duration' in train.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=train, x='Duration', y='Calories', hue='Duration')
    plt.title("Calories vs Duration")
    plt.show()

# ğŸ’¡ Count of Gender (if applicable)
if 'Gender' in train.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(x='Gender', data=train)
    plt.title("Count of Gender")
    plt.show()

# ğŸ’¡ Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# ğŸ’¡ Calories vs Height
if 'Height' in train.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=train, x='Height', y='Calories', hue='Height')
    plt.title("Calories vs Height")
    plt.show()

# ğŸ’¡ Calories vs Heart Rate
if 'Heart_Rate' in train.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=train, x='Heart_Rate', y='Calories')
    plt.title("Calories vs Heart Rate")
    plt.show()



# ğŸ§¾ Check column names
print("Train Columns:", train.columns.tolist())
print("Test Columns:", test.columns.tolist())


# ğŸ§ª Preprocessing and Modeling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# âœ… Encode the 'Sex' column: male â†’ 0, female â†’ 1
train['Sex'] = train['Sex'].str.lower().map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].str.lower().map({'male': 0, 'female': 1})

# ğŸ�¯ Split features and target
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)

# ğŸ“� Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ğŸ“¤ Split training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ğŸŒ² Train RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# âœ… Validate model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print("ğŸ“‰ Validation MAE:", round(mae, 2))

# ğŸ§ª Predict on test data
test_preds = model.predict(X_test_scaled)

# ğŸ’¾ Save to submission.csv
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved!")

