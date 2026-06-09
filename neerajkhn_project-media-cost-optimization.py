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


#Load Library 

import pandas as pd

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb



#load dataset
train_data=pd.read_csv("/kaggle/input/playground-series-s3e11/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s3e11/test.csv")


train_data.head(1)


train_data.info()


train_data.columns


#Checking missing value

train_data.isnull().sum()



# List of features to check for outliers
features = ['store_sales(in millions)', 'unit_sales(in millions)', 'total_children',
            'num_children_at_home', 'avg_cars_at home(approx).1',
            'gross_weight', 'recyclable_package', 'low_fat', 'units_per_case', 'store_sqft', 'coffee_bar',
            'video_store', 'salad_bar', 'prepared_food', 'florist']

# Outlier detection using IQR method
def remove_outliers(df, features):
    # Calculate the IQR for each feature
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df






# Remove outliers
train_data_cleaned = remove_outliers(train_data.copy(), features)


train_data.shape


train_data_cleaned.shape



from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns

# Step 1: Calculate correlation matrix
corr_matrix = train_data_cleaned[features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 2: Calculate VIF for each feature
# Add constant to the data (for VIF calculation)
X = train_data_cleaned[features]
X_with_const = add_constant(X)  # Adds a constant column for VIF calculation

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

# Display VIF data
print(vif_data)

# Step 3: Feature selection based on VIF and correlation
# Thresholds for VIF and correlation
VIF_THRESHOLD = 10  # Features with VIF above 10 are considered to have high multicollinearity
CORR_THRESHOLD = 0.8  # Correlation greater than 0.8 means features are highly correlated

# Remove features with high VIF
high_vif_features = vif_data[vif_data['VIF'] > VIF_THRESHOLD]['Feature'].tolist()
print(f"Features with high VIF: {high_vif_features}")

# Remove features with high correlation
correlated_features = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > CORR_THRESHOLD:
            colname = corr_matrix.columns[i]
            correlated_features.append(colname)

print(f"Highly correlated features: {set(correlated_features)}")

# Combine features to drop: high VIF and high correlation
features_to_drop = set(high_vif_features + correlated_features)
print(f"Features to drop: {features_to_drop}")

# Final feature set after dropping highly correlated and high-VIF features
final_features = [f for f in features if f not in features_to_drop]
print(f"Selected features: {final_features}")

# Now, we can continue with feature selection
X_train_selected = train_data[final_features]

# Continue with your machine learning model training using X_train_selected



# After outlier removal, you can proceed with model training:
# Separate features and target variable again
X_train = train_data_cleaned.drop(columns=['id','cost','salad_bar', 'prepared_food'])  # Assuming 'MediaCost' is the target
y_train = train_data_cleaned['cost']

# Now you can continue with the training process as shown in the earlier response


X_train.head(1)


X_test=test_data.drop(columns=['id','salad_bar', 'prepared_food']) 
X_test.head(2)


# For the test data, we need the same features, but without the target
X_test = test_data  # Test data doesn't have 'MediaCost'

# Step 3: Split train data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



X_train_split.head(1)


X_val_split.head(2)


X_train_split.head(1)


y_train_split.head(1)


X_test.head(1)


# Step 4: Train a machine learning model (RandomForest or XGBoost)
# Here, we will use XGBoost Regressor for demonstration

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=5)

# Train the model on the training data
model.fit(X_train_split, y_train_split, early_stopping_rounds=50, eval_set=[(X_val_split, y_val_split)], verbose=False)




X_test1=test_data.drop(columns=['id','salad_bar', 'prepared_food']) 

X_test1.head(1)


# Step 5: Make predictions on the test set
y_pred = model.predict(X_test1)








y_pred 


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on the validation set
y_pred = model.predict(X_val_split)

# Calculate MAE, RMSE, and R^2
mae = mean_absolute_error(y_val_split, y_pred)
rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
r2 = r2_score(y_val_split, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")



# Step 6: Export the submission file
# Prepare the submission dataframe (Kaggle typically expects 'Id' and 'MediaCost' columns in the submission)
submission = pd.DataFrame({
      # Ensure you have 'Id' in your test dataset
    'cost': y_pred
})

# Save the submission file as a CSV
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' has been saved!")


submission.head(10)




