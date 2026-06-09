# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train


test


"""Shape of the train and test data - to check whether enogh data is available for model"""
print(train.shape)
print(test.shape)

print("*"*100)

"""Check the Data Type of overall data"""
print(train.info())

print("*"*100)

"""To check for unique data in each column"""
print(train.nunique())

print("*"*100)

"""Check for null values in the Data"""
print(train.isna().sum())

print("*"*100)


# Summary Statistics
train.describe()


numerical_features = train.select_dtypes(include='number').drop(columns='id')

# Layout setup
cols = 2
rows = (len(numerical_features.columns) + 1) // cols

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
axes = axes.flatten()

# Plot boxplots
for i, col in enumerate(numerical_features.columns):
    sns.boxplot(y=train[col],color='lightseagreen', ax=axes[i])
    axes[i].set_title(f'Box Plot of {col} (Outlier Detection)')
    axes[i].set_ylabel(col)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Outlier Removal Process
train_clean = train.copy()

# Loop through each numerical column and apply IQR filtering
for col in numerical_features.columns:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep only rows within bounds
    train_clean = train_clean[(train_clean[col] >= lower_bound) & (train_clean[col] <= upper_bound)]

# Result
print(f"Original shape: {train.shape}")
print(f"Cleaned shape: {train_clean.shape}")


numerical_features = train_clean.select_dtypes(include='number').drop(columns='id')

# Basic setup
cols = 2
rows = (len(numerical_features.columns) + 1) // cols

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(numerical_features.columns):
    sns.histplot(train_clean[col], kde=True, bins=10, color='lightseagreen', edgecolor='black',element="bars", ax=axes[i])
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any extra axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

""" If you look at the distribution of age and calories which exhibits a positive skewness (or right skewness) 
wherelse the temp is negatively skewed - which suggest that age is directly propotional to the calorie burnt and 
inversely proportional to the temp of the body - hypothesis right now - will further analyse to confirm """


# Only one categorical column is present
categorical_features = train.select_dtypes(include='object').columns

for col in categorical_features:
    print(f"Counts for {col}:")
    print(train[col].value_counts().reset_index())
    print("\n")

"""the data based on sex are equally distributed"""



# am creating age bins, since the data is right skewed , would like to see the counts to see whether imbalance in the dataset 
bins = [20, 30, 40, 50, 60, 70, 80] # np.inf handles all values greater than or equal to 80
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
train_clean['Age_Group'] = pd.cut(train_clean['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

print(train_clean.Age_Group.value_counts().sort_index())


# Univariate Analysis 
print(train_clean.groupby(["Sex"])[["Height"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex"])[["Weight"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex"])[["Duration"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex"])[["Heart_Rate"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex"])[["Body_Temp"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex"])[["Calories"]].mean().reset_index())


print(train_clean.groupby(["Age_Group"])[["Height"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Age_Group"])[["Weight"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Age_Group"])[["Duration"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Age_Group"])[["Heart_Rate"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Age_Group"])[["Body_Temp"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Age_Group"])[["Calories"]].mean().reset_index())


# Biivariate Analysis 
print(train_clean.groupby(["Sex","Age_Group"])[["Height"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex","Age_Group"])[["Weight"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex","Age_Group"])[["Duration"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex","Age_Group"])[["Heart_Rate"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex","Age_Group"])[["Body_Temp"]].mean().reset_index())
print("*"*100)
print(train_clean.groupby(["Sex","Age_Group"])[["Calories"]].mean().reset_index())


corr_data = train_clean.drop(columns={"id", "Sex", "Age_Group"})

sns.heatmap(data=corr_data.corr(), annot=True, cmap="cividis", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Since the correlation is high , am creating a new feature BMI 

train_clean['Height'] = train_clean['Height'] / 100  
train_clean['BMI'] = train_clean['Weight'] / (train_clean['Height'] ** 2)
train_clean.drop(columns={"Height","Weight"},inplace=True)
train_clean

# assumption creating a new feature workout intensity
train_clean['Workout_Intensity'] = (train_clean['Duration'] * train_clean['Heart_Rate'])/train_clean['Body_Temp']
train_clean.drop(columns={"Duration","Heart_Rate","Body_Temp"},inplace=True)
train_clean


numerical_cols = train_clean.select_dtypes(include='number')

# Compute correlation matrix
correlation_matrix = numerical_cols.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='cividis', fmt=".2f", square=True, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Numerical Features", fontsize=14)
plt.show()


train_clean['Calories'] = train_clean.pop('Calories')
#train_clean.drop(columns="id",inplace=True)
train_clean


#For training test
# Label encoding for Categorical Variables
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Separate features and target
X = train_clean.drop(columns="Calories", axis=1)
y = train_clean["Calories"]

# Label encode categorical variables
LE = LabelEncoder()
for col in ("Sex", "Age_Group"):
    X[col] = LE.fit_transform(X[col])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Select only numeric columns for scaling
cols_to_scale = ["Age", "BMI", "Workout_Intensity"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[cols_to_scale])
X_val_scaled = scaler.transform(X_val[cols_to_scale])

# Replace the original columns with the scaled ones
X_train_scaled_df = X_train.copy()
X_val_scaled_df = X_val.copy()

X_train_scaled_df[cols_to_scale] = X_train_scaled
X_val_scaled_df[cols_to_scale] = X_val_scaled

# Output for inspection
print("*" * 100)
X_train_Scaled=X_train_scaled_df.copy()
X_val_Scaled=X_val_scaled_df.copy()

print(X_train_Scaled)
print(X_val_Scaled)



#for test set 
bins = [20, 30, 40, 50, 60, 70, 80] # np.inf handles all values greater than or equal to 80
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
test['Age_Group'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False, include_lowest=True)


test['Height'] = test['Height'] / 100  
test['BMI'] = test['Weight'] / (test['Height'] ** 2)
test.drop(columns={"Height","Weight"},inplace=True)
test

# assumption creating a new feature workout intensity
test['Workout_Intensity'] = (test['Duration'] * test['Heart_Rate'])/test['Body_Temp']
test.drop(columns={"Duration","Heart_Rate","Body_Temp"},inplace=True)
test

# label encoding test set 
for i in ("Sex","Age_Group"):
    test[i]=LE.fit_transform(test[i])

#test.drop(columns="id",inplace=True)

cols_to_scale = ["Age", "BMI", "Workout_Intensity"]

# Make a copy to avoid modifying original test
X_test_scaled_df = test.copy()

# Apply the scaler only to relevant columns
scaled_test_values = scaler.transform(X_test_scaled_df[cols_to_scale])

# Replace the columns in the copied test set
X_test_scaled_df[cols_to_scale] = scaled_test_values

# This is your final scaled test set
X_test_Scaled = X_test_scaled_df.copy()

# Preview
print(X_test_Scaled.head())


#Linear Regression Model 

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_log_error

lr=LinearRegression()
lr.fit(X_train_Scaled,y_train)

y_train_pred = lr.predict(X_train_Scaled)
y_val_pred = lr.predict(X_val_Scaled)

y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)

train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

print("Train RMSLE:", train_rmsle)
print("Validation RMSLE:", val_rmsle)


#LAsso Regerssion
lasso = Lasso(alpha=0.01)  # You can tune alpha
lasso.fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = lasso.predict(X_train_Scaled)
y_val_pred = lasso.predict(X_val_Scaled)

# Clip predictions to ensure non-negative values
y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

# Display results
print("Train RMSLE (Lasso):", train_rmsle)
print("Validation RMSLE (Lasso):", val_rmsle)


#Ridge Regerssion
ridge = Ridge(alpha=0.01)  # You can tune alpha
ridge.fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = ridge.predict(X_train_Scaled)
y_val_pred = ridge.predict(X_val_Scaled)

# Clip predictions to ensure non-negative values
y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

# Display results
print("Train RMSLE (Ridge):", train_rmsle)
print("Validation RMSLE (Ridge):", val_rmsle)


#Ridge Regerssion
elastic = ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42) 
elastic .fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = elastic .predict(X_train_Scaled)
y_val_pred = elastic .predict(X_val_Scaled)

# Clip predictions to ensure non-negative values
y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

# Display results
print("Train RMSLE (elastic ):", train_rmsle)
print("Validation RMSLE (elastic ):", val_rmsle)


# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor( n_estimators=300,max_depth=15,min_samples_split=15,min_samples_leaf=5,max_features='sqrt',random_state=42)

# Fit the model
rf.fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = rf.predict(X_train_Scaled)
y_val_pred = rf.predict(X_val_Scaled)

# Clip predictions to avoid negative values (RMSLE requires non-negative)
y_train_pred = np.maximum(0, y_train_pred)
y_val_pred = np.maximum(0, y_val_pred)

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

print("Train RMSLE (Random Forest):", train_rmsle)
print("Validation RMSLE (Random Forest):", val_rmsle)


from sklearn.ensemble import GradientBoostingRegressor

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.1,max_depth=5,min_samples_split=15,min_samples_leaf=5,max_features='sqrt',random_state=42)

# Fit the model
gbr.fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = np.maximum(0, gbr.predict(X_train_Scaled))
y_val_pred = np.maximum(0, gbr.predict(X_val_Scaled))

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

print("Train RMSLE (GB):", train_rmsle)
print("Validation RMSLE (GB):", val_rmsle)



from lightgbm import LGBMRegressor

# Initialize LightGBM Regressor
lgbm = LGBMRegressor(
  n_estimators=5000,learning_rate=0.05,num_leaves=31,colsample_bytree=0.8,subsample=0.9,
  max_depth=14,random_state=42,verbose=-1)


# Fit the model
lgbm.fit(X_train_Scaled, y_train)

# Predict on training and validation sets
y_train_pred = np.maximum(0, lgbm.predict(X_train_Scaled))
y_val_pred = np.maximum(0, lgbm.predict(X_val_Scaled))

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

print("Train RMSLE (LightGBM):", train_rmsle)
print("Validation RMSLE (LightGBM):", val_rmsle)



from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_log_error

xgb_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=14,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    verbosity=1
)

# Fit the model with early stopping
xgb_model.fit(
    X_train_Scaled, y_train,
    eval_set=[(X_val_Scaled, y_val)],
    early_stopping_rounds=200,
    verbose=100
)

# Predict
y_train_pred = np.maximum(0, xgb_model.predict(X_train_Scaled))
y_val_pred = np.maximum(0, xgb_model.predict(X_val_Scaled))

# Compute RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, y_train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))

print("Train RMSLE (XGBoost):", train_rmsle)
print("Validation RMSLE (XGBoost):", val_rmsle)



kaggle_predictions = xgb_model.predict(X_test_Scaled)
submission = pd.DataFrame({
    "id": X_test_Scaled["id"],
    "Calories": kaggle_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission)

