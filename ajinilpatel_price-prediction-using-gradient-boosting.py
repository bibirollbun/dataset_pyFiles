import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.plotting import lag_plot, autocorrelation_plot
import scipy.stats as stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
# train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# train_data = pd.concat([train_data, train_extra], ignore_index=True)


train_data.head()


train_data.shape


train_data.info()


print(f"Minimum Price: {train_data['Price'].min()}")
print(f"Maximum Price: {train_data['Price'].max()}")



train_data.describe()


plt.figure(figsize=(10, 6))
sns.histplot(train_data['Price'], bins=30, kde=True, color='blue')
plt.title('Original Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=train_data['Price'], color='green')
plt.title('Box Plot of Price')
plt.xlabel('Price')
plt.show()


# import numpy as np

# # log transformation
# train_data['Price_log'] = np.log1p(train_data['Price'])
# train_data[['Price', 'Price_log']].head()



train_data.isnull().sum()


# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5) 
# train_data[['Weight Capacity (kg)']] = imputer.fit_transform(train_data[['Weight Capacity (kg)']])
# test_data[['Weight Capacity (kg)']] = imputer.transform(test_data[['Weight Capacity (kg)']])



# from category_encoders import TargetEncoder
# encoder = TargetEncoder()
# train_data[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']] = encoder.fit_transform(
#     train_data[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']], train_data['Price']
# )
# test_data[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']] = encoder.transform(
#     test_data[['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']]
# )


price = train_data['Price']
plt.figure(figsize=(6, 6))
lag_plot(price)
plt.title("Lag Plot of Price")
plt.show()





# # Function to cap outliers using IQR method
# def cap_outliers(df, col):
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df[col] = df[col].clip(lower_bound, upper_bound)
    
# # Apply this function to the relevant columns
# outlier_cols = ['Size', 'Laptop Compartment', 'Waterproof', 'Style']
# for col in outlier_cols:
#     cap_outliers(train_data, col)
#     cap_outliers(test_data, col)



# num_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']  # Numerical columns

# # Plot boxplots
# for col in num_cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=train_data[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()


# train_data['Brand'].unique()


X = train_data.drop(columns=['Price', 'id'])  # Replace 'Price' with your actual target column name
y = train_data['Price']


numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns





numerical_cols


categorical_cols


knn_imputer = KNNImputer(n_neighbors=5)
X[numerical_cols] = knn_imputer.fit_transform(X[numerical_cols])


categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])


X[numerical_cols].isnull().sum()


X[categorical_cols].isnull().sum()


# Using OneHotEncoder to encode categorical columns
encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid dummy variable trap
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))


X = X.drop(columns=categorical_cols)
X = pd.concat([X, X_encoded], axis=1)


X


scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# rf = RandomForestRegressor()
# rf.fit(X, y)
# importances = rf.feature_importances_

# # Plot the feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(X.columns, importances)
# plt.title("Feature Importances")
# plt.show()


# sfm = SelectFromModel(rf, threshold=0.01)
# sfm.fit(X, y)
# X_selected = X.loc[:, sfm.get_support()]


# pca = PCA(n_components=0.95)  # Keep 95% variance
# X_pca = pca.fit_transform(X_selected)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.shape


# Train a regression model (RandomForestRegressor in this case)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Predict on the validation set
y_pred = model.predict(X_val)


# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R-squared: {r2}")



plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.7)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
plt.title("Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()



# test_ids = test_data['id']

# test_preds = model.predict(X_test)
# submission = pd.DataFrame({'id': test_ids, 'Price': test_preds.flatten()})


import xgboost as xgb


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {mae}')



# Predict on the validation data
y_pred = gb_model.predict(X_val)

# Calculate MAE
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Root Mean Squared Error: {rmse}')


test_data1 = test_data.drop(columns=['id'])
numerical_cols_test = test_data1.select_dtypes(include=['float64', 'int64']).columns
categorical_cols_test = test_data1.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns using the mean strategy
numerical_imputer = SimpleImputer(strategy='mean')
X_numerical_imputed = numerical_imputer.fit_transform(test_data1[numerical_cols_test])

# Impute missing values for categorical columns using the most frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_categorical_imputed = categorical_imputer.fit_transform(test_data1[categorical_cols_test])



X_test_imputed = pd.DataFrame(
    pd.concat([pd.DataFrame(X_numerical_imputed, columns=numerical_cols_test),
               pd.DataFrame(X_categorical_imputed, columns=categorical_cols_test)], axis=1)
)


# Using OneHotEncoder to encode categorical columns
encoder = OneHotEncoder(sparse=False, drop='first')  # Avoid dummy variable trap
X_encoded = pd.DataFrame(encoder.fit_transform(X_test_imputed[categorical_cols_test]), columns=encoder.get_feature_names_out(categorical_cols_test))





X_test_transformed = X_test_imputed.drop(columns=categorical_cols_test)
X_test_transformed = pd.concat([X_test_transformed, X_encoded], axis=1)


scaler = MinMaxScaler()
X_test_transformed[numerical_cols_test] = scaler.fit_transform(X_test_transformed[numerical_cols_test])


X_test_transformed


y_test_pred = gb_model.predict(X_test_transformed)





submission = pd.DataFrame({
    'id': test_data['id'],  # Replace with the correct 'id' column name
    'Price': y_test_pred  # Predicted prices
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)




