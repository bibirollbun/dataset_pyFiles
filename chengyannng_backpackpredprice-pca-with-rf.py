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


# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import TSNE


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data.info()


# Remove IDs from categorical variables that contain no data
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
missing_categorical = data[categorical_features].isnull().any(axis=1)
data = data[~missing_categorical]
data.info() # Double-check for any missing values
data.head()


data = data.drop(columns=['id'])


# One-hot coding and converting boolean values to 0/1
data_encoded = pd.get_dummies(data, drop_first=True)
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)


# My computer hardware is not very good, randomly select 50,000 samples
data_encoded = data_encoded.sample(n=50000, random_state=42) 


# Resetting index (perception issue)
data_encoded = data_encoded.reset_index(drop=True)
data_encoded


# Standarize
scaler = StandardScaler()
data_encoded_NoPrice = data_encoded.drop(columns='Price')
X_scaled_NoPrice = scaler.fit_transform(data_encoded_NoPrice)


# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled_NoPrice)

# Calculate the proportion of variance and the cumulative proportion of variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Extract Price
y = data_encoded['Price']



# Visualize PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', label="Individual Explained Variance")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='s', linestyle='-', label="Cumulative Explained Variance")
plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Explained")
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid()
plt.show()


pca_17 = PCA(n_components=17)
X_pca_17 = pca_17.fit_transform(X_scaled_NoPrice)
loadings = pd.DataFrame(pca_17.components_, columns=data_encoded_NoPrice.columns, index=[f'PC{i+1}' for i in range(17)])
top_features = loadings.abs().mean().sort_values(ascending=False)
print("Top contributing features:\n", top_features.head(10))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca_17, y, test_size=0.2, random_state=42)

# Build a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("模型評估結果：")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


# Loading test
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Retain Id for the final output
test_ids = test_data['id'] 
# Remove the 'id' column 
test_data.drop(columns=['id'], inplace=True)

# Handle missing values (fill numerical variables with the median and categorical variables with 'None')
for col in test_data.columns:
    if test_data[col].dtype == "object":
        test_data[col].fillna("None", inplace=True)
    else:
        test_data[col].fillna(test_data[col].median(), inplace=True)

# One-hot coding and converting boolean values to 0/1
test_data_encoded = pd.get_dummies(test_data, drop_first=True)
bool_cols = test_data_encoded.select_dtypes(include=['bool']).columns
test_data_encoded[bool_cols] = test_data_encoded[bool_cols].astype(int)

# "Ensure that the test data columns match the training data (fill missing columns with 0 if necessary)
missing_cols = set(data_encoded_NoPrice.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0

# Ensure column order consistency
test_data_encoded = test_data_encoded[data_encoded_NoPrice.columns]



# 對測試資料進行相同的前處理：若測試資料與訓練資料結構相同
X_test_full = test_data_encoded.copy()

# 進行標準化（使用訓練階段 fit 的 scaler）
X_test_scaled = scaler.transform(X_test_full)

# 利用訓練時 fit 好的 PCA 模型進行降維
X_test_pca = pca_17.transform(X_test_scaled)

# 接下來就可以使用 rf_model 進行預測，X_test_pca 的 shape 應為 (樣本數, 17)
test_predictions = model.predict(X_test_pca)



# 建立提交結果
submission = pd.DataFrame({"id": test_ids, "Price": test_predictions})
submission.set_index("id", inplace=True)

print(submission)

submission.to_csv('BackpackPricePred.csv')

