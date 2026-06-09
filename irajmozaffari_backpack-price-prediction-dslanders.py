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


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data  = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_data = pd.concat([train_data, extra_data], ignore_index=True)


train_data.head(6)



train_data.info()
test_data.info()


numeric_features = train_data.select_dtypes(include=['number'])
print(numeric_features.columns.tolist())


categorical_features = train_data.select_dtypes(include=['category'])
print(categorical_features.columns.tolist())



duplicates_per_column = train_data.apply(lambda x: x[x.duplicated()].unique())
print(duplicates_per_column)


duplicate_counts = {}
for col in train_data.columns:
    if train_data[col].dtype == 'object':  # for context column
        duplicate_counts[col] = train_data[col].value_counts()[train_data[col].value_counts() > 1]
    else:  # for numeric columns
        duplicate_counts[col] = train_data[col].astype(str).value_counts()[train_data[col].astype(str).value_counts() > 1]
print(duplicate_counts)



train_data.isna().sum().sum()


train_data.duplicated().sum()


print("Skewness of Weight Capacity:", train_data["Weight Capacity (kg)"].skew())
print("Skewness of Price:", train_data["Price"].skew())

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

sns.histplot(train_data["Weight Capacity (kg)"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Distribution of Weight Capacity (kg)")

sns.histplot(train_data["Price"], bins=20, kde=True, ax=axes[1])
axes[1].set_title("Distribution of Price")

plt.show()



def find_outlier_samples(df, sample_size=5):
    for col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        print(f"{col}: {len(outliers)} outliers ({len(outliers) * 100 / len(df):.2f}%)")
        print(outliers.sample(min(sample_size, len(outliers))))  # نمایش چند نمونه
        print("-" * 40)

numerical_features = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != 'id']

find_outlier_samples(train_data[numerical_features])

rows = 2
cols = min(4, len(numerical_features))  # 4 charts in a row

# size of plot
plt.figure(figsize=(cols * 4, rows * 3))

# small plot
for i, col in enumerate(numerical_features, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(y=train_data[col])
    plt.title(f'{col}', fontsize=10)
    plt.xlabel("")  # delete Y titles to view better
    plt.xticks([])  # delete X titles

# Optimizing chart layout to prevent overlap
plt.tight_layout()
plt.show()



train_data_cat = train_data.copy()

# Using Target Encoding (replacing the average Price value for each category)
for col in train_data_cat.select_dtypes(include=['object']).columns:
    mean_encoding = train_data_cat.groupby(col)['Price'].mean()   # Calculating the average price for each category
    train_data_cat[col] = train_data_cat[col].map(mean_encoding)  # Replacing categorical value with average price
#print(train_data_cat)

# Removing the 'id' column 
train_data_cat = train_data_cat.drop(columns=['id'], errors='ignore')

# Calculating Correlation Matrix
corr_matrix = train_data_cat.corr().round(3)

# Drawing a Heatmap
plt.figure(figsize=(8, 4), dpi=120)

sns.heatmap(corr_matrix, annot=True, fmt=".3f", linewidths=0.5, 
            annot_kws={"size": 6},  # Set the size of the annotations (numbers inside cells)
            square=True,  # Make the cells square-shaped
            mask=(corr_matrix == 1) ,  # Hide cells with correlations less than 0 or greater than 1
            cmap='coolwarm',  # Use a coolwarm color map for correlations between 0 and 1
            cbar_kws={"shrink": 0.8},  # Shrink the color bar size
            linecolor='white',  # Color for the lines separating cells
            xticklabels=corr_matrix.columns,  # Display column names (header) at the top
            yticklabels=corr_matrix.columns)  # Display row names (header) on the sides

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Correlation Heatmap', fontsize=12)
plt.show()



categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_columns = ['Weight Capacity (kg)', 'Compartments']  

# Fill in missing values
numerical_imputer = SimpleImputer(strategy='mean')
train_data[numerical_columns] = numerical_imputer.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = numerical_imputer.transform(test_data[numerical_columns])

categorical_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_columns] = categorical_imputer.fit_transform(train_data[categorical_columns])
test_data[categorical_columns] = categorical_imputer.transform(test_data[categorical_columns])

# Label Encoding for Categorical Features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = test_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    label_encoders[col] = le

print(train_data)
print("___________________")
print(test_data)


# Delete id & Price columns
features = train_data.drop(columns=['id', 'Price'])
target = train_data['Price']
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2, 
                                                    random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Preventing Negative Values
y_pred = np.maximum(y_pred, 0)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f" MAE: {mae:.2f}, MSE: {mse:.2f}")
rmse = np.sqrt(mse)
print("RMSE:", rmse)


X_train = train_data.drop(columns=['id', 'Price'])
y_train = train_data['Price']
X_test = test_data.drop(columns=['id'])  

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

y_pred = np.maximum(y_pred, 0)

# Output
submission_df = pd.DataFrame({'id': test_data['id'], 'Predicted': y_pred})
submission_df.to_csv('submission.csv', index=False)


