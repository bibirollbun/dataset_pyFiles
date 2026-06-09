import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
data.set_index("id", inplace=True)
data.head()


initial_data_count = data.shape[0];
print("Initial number of rows = {}".format(initial_data_count))
print(data.count())


print("Procentage of missing data: \n\n{}".format((data.isnull().sum() / data.count()) * 100 ))


data.dropna(inplace = True)
print(data.count())


clean_data_count = data.shape[0];
print("Number of clean rows = {}".format(clean_data_count))
print("Data lost from dropping missing values = {:.2f}%".format((initial_data_count-clean_data_count)/initial_data_count*100))


data.nunique()


import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # 2 rows, 3 columns

# Histogram (Price Distribution)
sns.histplot(data['Price'], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Price Distribution of Backpacks')
axes[0, 0].set_xlabel('Price')
axes[0, 0].set_ylabel('Count')

# Boxplot (Price vs. Size)
sns.boxplot(y=data['Price'], x=data['Size'], ax=axes[0, 1])
axes[0, 1].set_title('Price Spread of Backpack Sizes')

# Boxplot (Price vs. Brand)
sns.boxplot(y=data['Price'], x=data['Brand'], ax=axes[0, 2])
axes[0, 2].set_title('Price Spread of Backpack Brands')

# Boxplot (Price vs. Material)
sns.boxplot(y=data['Price'], x=data['Material'], ax=axes[1, 0])
axes[1, 0].set_title('Price Spread of Backpack Material')

# Boxplot (Price vs. Color)
sns.boxplot(y=data['Price'], x=data['Color'], ax=axes[1, 1])
axes[1, 1].set_title('Price Spread of Backpack Color')

# Boxplot (Price vs. Laptop Compartment)
sns.boxplot(y=data['Price'], x=data['Laptop Compartment'], ax=axes[1, 1])
axes[1, 1].set_title('Price Spread of Backpack Color')

# Hide the empty subplot (since we have 5 plots but a 2x3 grid)
fig.delaxes(axes[1, 2])

plt.tight_layout()  # Prevent overlapping
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(14, 4))  # 1 row, 3 columns

# Heatmap: Price by Material and Style
pivot_table = data.pivot_table(values="Price", index="Material", columns="Style", aggfunc="mean")
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Avg Price by Material & Style")

# Heatmap: Price by Material and Brand
pivot_table = data.pivot_table(values="Price", index="Material", columns="Brand", aggfunc="mean")
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=axes[1])
axes[1].set_title("Avg Price by Material & Brand")

# Heatmap: Price by Color and Brand
pivot_table = data.pivot_table(values="Price", index="Color", columns="Brand", aggfunc="mean")
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=axes[2])
axes[2].set_title("Avg Price by Color & Brand")

plt.tight_layout()  # Prevent overlap
plt.show()


data.nunique()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data.columns


X = data[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]

y = data['Price']


X.head(2)


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Convert categorical columns to 'category' dtype
for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col])

X = X.apply(pd.to_numeric)


X.head(2)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the XGBoost regressor
XGBoost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20, learning_rate=0.1, max_depth=4, enable_categorical=True)

# Fit the model
XGBoost_model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = XGBoost_model.predict(X_test)


XGBoost_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", XGBoost_rmse)


xgb.plot_importance(XGBoost_model)
plt.show()


from catboost import CatBoostRegressor


#Features and target
X = data[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]

y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X.head(2)


#cat_features
cat_features = ['Brand', 'Material', 'Size', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']


CatBoost_model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=5, verbose=100)
CatBoost_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))


y_pred = CatBoost_model.predict(X_test)

CatBoost_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("CatBoost_RMSE:", CatBoost_rmse)


import shap

# Get SHAP values
explainer = shap.TreeExplainer(CatBoost_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)



print("XGBoost_RMSE:", XGBoost_rmse)
print("CatBoost_RMSE:", CatBoost_rmse)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test_data.set_index("id", inplace=True)
test_data.fillna("nan", inplace=True)
test_data.head(2)


prediction = CatBoost_model.predict(test_data)


output = pd.DataFrame({'id': test_data.index, 'Price': prediction})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

