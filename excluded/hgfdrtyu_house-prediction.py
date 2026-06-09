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


# Basic Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Warning suppression
warnings.filterwarnings('ignore')
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# Display settings
pd.set_option('display.max_columns', None)

# Load data
train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)



# Check if 'sale_price' exists
if 'sale_price' in train.columns:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(train['sale_price'], fill=True, color='coral', bw_adjust=0.5)
    plt.axvline(train['sale_price'].mean(), color='blue', linestyle='--', label=f"Mean: {train['sale_price'].mean():,.0f}")
    plt.axvline(train['sale_price'].median(), color='green', linestyle='--', label=f"Median: {train['sale_price'].median():,.0f}")
    plt.title(" Distribution of Sale Price")
    plt.xlabel("Sale Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(" 'sale_price' column not found in training data.")



# Missing value heatmap
import missingno as msno

plt.figure(figsize=(12, 6))
msno.heatmap(train, fontsize=10)
plt.title(" Missing Values Heatmap")
plt.show()



# Distribution of Sale Price (Target Variable)
plt.figure(figsize=(10, 5))
sns.histplot(train['sale_price'], kde=True, bins=50, color='dodgerblue')
plt.title("Distribution of Sale Price", fontsize=14)
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
hb = plt.hexbin(
    x=train['longitude'],
    y=train['latitude'],
    C=train['sale_price'],
    gridsize=50,
    cmap='viridis',
    reduce_C_function=np.mean
)
plt.colorbar(hb, label='Average Sale Price')
plt.title('Geographical Distribution of Sale Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))
sns.scatterplot(x=train['sqft'], y=train['sale_price'], hue=train['grade'], palette='coolwarm', alpha=0.6)
plt.title("Sale Price vs Total Sqft with Grade")
plt.xlabel("Total Sqft")
plt.ylabel("Sale Price")
plt.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



plt.figure(figsize=(16, 12))
numeric_cols = train.select_dtypes(include=np.number).drop(columns=['id']).corr()
sns.heatmap(numeric_cols[['sale_price']].sort_values(by='sale_price', ascending=False), 
            annot=True, cmap='Spectral', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation with Sale Price", fontsize=16)
plt.show()



view_features = ['view_rainier', 'view_olympics', 'view_cascades', 'view_lakewash', 'view_skyline', 'view_sound']
melted_views = train.melt(id_vars='sale_price', value_vars=view_features, var_name='View', value_name='Has_View')
view_price = melted_views[melted_views['Has_View'] == 1].groupby('View')['sale_price'].mean().sort_values()

plt.figure(figsize=(10, 6))
view_price.plot(kind='barh', color='teal')
plt.title("Average Sale Price by View Type")
plt.xlabel("Average Sale Price")
plt.ylabel("View Type")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load train and test data
train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

# Drop rows with missing values (or you can fill them)
train.dropna(inplace=True)

# Features to use (adjust based on your dataset)
features = ['sqft', 'beds', 'bath_full', 'year_built', 'latitude', 'longitude']

# Prepare training data
X_train = train[features]
y_train = train['sale_price']

# Prepare test data
X_test = test[features]

# Train a simple model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict sale_price for test data
preds = model.predict(X_test)  # <--- This defines `preds`

# Create prediction intervals (Â±10% of prediction)
submission = pd.DataFrame({
    'id': test['id'],
    'sale_price_lower': preds * 0.9,
    'sale_price_upper': preds * 1.1
})

# Save to CSV
submission.to_csv('submission.csv', index=False)




