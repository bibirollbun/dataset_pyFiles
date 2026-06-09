import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.isna().sum()


train.shape


train.head()


# Handling Missing values
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', fontsize=16)
plt.show()


# Create a new column none as if we drop null values it will remove almost 50000 rows which are a lot so to protect it replace categorical values with mode and 
# numerical with mean 
train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)
train['Brand'].fillna(train['Brand'].mode()[0], inplace=True)
train['Material'].fillna(train['Material'].mode()[0], inplace=True)
train['Size'].fillna(train['Size'].mode()[0], inplace=True)
train['Laptop Compartment'].fillna(train['Laptop Compartment'].mode()[0], inplace=True)
train['Waterproof'].fillna(train['Waterproof'].mode()[0], inplace=True)
train['Style'].fillna(train['Style'].mode()[0], inplace=True)
train['Color'].fillna(train['Color'].mode()[0], inplace=True)

test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)
test['Brand'].fillna(test['Brand'].mode()[0], inplace=True)
test['Material'].fillna(test['Material'].mode()[0], inplace=True)
test['Size'].fillna(test['Size'].mode()[0], inplace=True)
test['Laptop Compartment'].fillna(test['Laptop Compartment'].mode()[0], inplace=True)
test['Waterproof'].fillna(test['Waterproof'].mode()[0], inplace=True)
test['Style'].fillna(test['Style'].mode()[0], inplace=True)
test['Color'].fillna(test['Color'].mode()[0], inplace=True)



categorical_features = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color','Size']

# train[categorical_features] = train[categorical_features].fillna('None').astype('string')
# mean_weight = train['Weight Capacity (kg)'].median()
# train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_weight)

# test[categorical_features] = test[categorical_features].fillna('None').astype('string')
# test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight)

X = train.drop(columns=['id','Price'], axis=1)
y = train.Price



# Define the number of rows and columns for subplots
num_features = len(categorical_features)
cols = 3 
rows = (num_features // cols) + (num_features % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    value_counts = train[feature].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f"{feature} Distribution", fontsize=14)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel("Count", fontsize=12)
    if feature == 'Brand':
        axes[i].tick_params(axis='x', rotation=15)

# Hide unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


print("Checking correlations with Price")

fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.boxplot(x=train[feature], y=train['Price'], ax=axes[i])
    axes[i].set_title(f"Price Distribution by {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Price")
    if feature == 'Brand':
        axes[i].tick_params(axis='x', rotation=15)
        
# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import xgboost as xgb
from xgboost import XGBRegressor


numerical_columns = ["Compartments", "Weight Capacity (kg)"]
test=test.drop(columns='id',axis=1)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


column_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ("num", StandardScaler(), numerical_columns)
    ]
)


X_train = column_transformer.fit_transform(X)
X_test = column_transformer.transform(test)


reg = XGBRegressor(n_estimators=150, max_depth=3, eta=0.088, subsample=0.65)
reg.fit(X_train, y)


y_pred = reg.predict(X_test)


y_pred


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission_df['Price'] = y_pred
submission_df.to_csv('submission_changed.csv', index=False)
print(submission_df.head())




