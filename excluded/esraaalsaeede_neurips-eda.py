import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train.head()


train.isna().mean().sort_values(ascending=False)


train.isnull().sum()


# Calculate the percentage of missing values in each column
missing = train.isnull().mean().sort_values(ascending=False)

# Plot missing value percentages
plt.figure(figsize=(8, 4))
sns.barplot(x=missing.index, y=missing.values)
plt.title('Missing Value Percentage by Column')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()  # Show the plot



# Plot histogram and KDE (smooth curve) for FFV values
plt.figure(figsize=(6, 4))
sns.histplot(train['FFV'].dropna(), kde=True, bins=30, color='skyblue')
plt.title('FFV Distribution')
plt.xlabel('FFV')
plt.show()



# List of other properties to visualize
numeric_cols = ['Tg', 'Tc', 'Density', 'Rg']

# Loop through each property and plot its distribution
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col].dropna(), kde=True, bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.show()



# Plot a heatmap of correlation between numeric properties
plt.figure(figsize=(6, 5))
sns.heatmap(train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Properties')
plt.show()



# Add basic features
train['smiles_len'] = train['SMILES'].apply(len)
train['capital_count'] = train['SMILES'].str.count(r'[A-Z]')
train['star_count'] = train['SMILES'].str.count(r'\*')
plt.figure(figsize=(6, 4))
sns.scatterplot(data=train, x='smiles_len', y='FFV')
plt.title('SMILES Length vs FFV')
plt.show()



test


from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="EDA")
profile


# ðŸ‘‰ Add this block before trying to access those features
train['smiles_len'] = train['SMILES'].apply(len)
train['atom_count'] = train['SMILES'].str.count(r'[A-Z]')
train['star_count'] = train['SMILES'].str.count(r'\*')


import seaborn as sns
import matplotlib.pyplot as plt

features = ['smiles_len', 'atom_count', 'star_count']

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=train, x=feature, y='FFV')
    plt.title(f'{feature} vs FFV')
    plt.xlabel(feature)
    plt.ylabel('FFV')
    plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Select only rows that have FFV value
df = train[train['FFV'].notna()].copy()

# Features we created earlier
features = ['smiles_len', 'atom_count', 'star_count']
X = df[features]

# Target: FFV
y = df['FFV']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_valid)

# Evaluate
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse:.4f}")


# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")


# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)


sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


# ðŸ“Š 8. Evaluate on validation set
y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Validation MAE (FFV): {mae:.4f}")


# Make sure test set has the same features
test['smiles_len'] = test['SMILES'].apply(len)
test['atom_count'] = test['SMILES'].str.count(r'[A-Z]')
test['star_count'] = test['SMILES'].str.count(r'\*')

# ðŸ”® 9. Predict FFV for test set
X_test = test[features]
ffv_preds = model.predict(X_test)

# ðŸ“¤ 10. Prepare submission
submission = sample_sub.copy()
submission['FFV'] = ffv_preds
submission['Tg'] = 0
submission['Tc'] = 0
submission['Density'] = 0
submission['Rg'] = 0

# ðŸ’¾ 11. Save submission file
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv saved!")





