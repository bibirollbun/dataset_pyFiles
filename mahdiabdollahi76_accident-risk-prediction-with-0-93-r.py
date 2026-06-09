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


# Core libraries for numerical operations and data manipulation
import numpy as np
import pandas as pd

# Utility for splitting data into training and validation sets
from sklearn.model_selection import train_test_split

# Visualization libraries for exploratory data analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing tools for encoding and scaling features
from sklearn.preprocessing import OneHotEncoder, StandardScaler, _function_transformer, OrdinalEncoder, MinMaxScaler

# Pipeline and column-wise transformation utilities for clean preprocessing workflows
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature selection using sequential search (forward or backward)
from sklearn.feature_selection import SequentialFeatureSelector as sfs

# Gradient boosting model from XGBoost â€” powerful for tabular regression tasks
from xgboost import XGBRegressor

# Evaluation metric: Mean Absolute Error (MAE) for regression performance
from sklearn.metrics import mean_absolute_error , r2_score

# Ensemble model: Random Forest Regressor for baseline or comparison
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression


# Load training data from CSV file
data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

# Drop the 'id' column as it's not informative for modeling
data = data.drop("id", axis=1)

# Display the dataset to verify successful loading and structure
data


# Display dataset structure, data types, and non-null counts
print("Data Info:\n")
print(data.info())

# Show the number of missing values per feature
print("\nMissing Values per Feature:\n")
print(data.isna().sum())


# Compute the correlation matrix for all numerical features
# Visualize feature relationships using a heatmap to identify potential multicollinearity or predictive patterns

corr = data.select_dtypes(["number"]).corr()
sns.heatmap(corr , cmap='coolwarm')


# Visualize the relationship between accident risk and road curvature
plt.figure()
sns.lineplot(x='accident_risk', y='curvature', data=data)

# Explore how the number of reported accidents correlates with accident risk
plt.figure()
sns.barplot(x='num_reported_accidents', y='accident_risk', data=data)

# Examine the impact of weather conditions on the number of reported accidents
plt.figure()
sns.barplot(x='weather', y='num_reported_accidents', data=data)


# Analyze how lighting conditions influence accident risk
sns.barplot(x='lighting', y='accident_risk', data=data)


sns.barplot(x = 'lighting' , y='num_reported_accidents' , data = data)


data_target = data['accident_risk']

# Split data into training and validation sets (80/20 split)
train_data, validate_data, y_train, y_validate = train_test_split(
    data.drop('accident_risk', axis=1),
    data_target,
    train_size=0.8,
    random_state=42,
    stratify=data_target
)



def feature_importance( X  , X_val , y  , y_val ,model = 'xgboost' ):
  # Identify unique data types in the dataset (e.g., object, bool, numeric)
  type_ = (X.dtypes).unique()
  feature_impact = []
  # Helper function to preprocess, train, and evaluate a single feature
  def train_valid(transformer ,train , valid_data , feature_ , type_data , y_val_):
    # Apply preprocessing pipeline to training and validation data
      train = transformer.fit_transform(train)
      valid = transformer.transform(valid_data)
      # Train XGBoost model on the transformed feature
      feature_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
      feature_model.fit(train , y)
      # Predict and evaluate using Mean Absolute Error
      predict = feature_model.predict(valid)
      mse = mean_absolute_error( predict, y_val_)
       # Store feature name and its corresponding MAE
      feature_impact.append((feature_ , mse , type_data))
      return feature_impact
  # Helper function to build preprocessing pipeline based on feature type
  def data_piplines(data_type_ = 'int64'):
    if data_type_ == 'O':
      # For categorical features: encode using Ordinal + OneHot
      pip = Pipeline([
          ("ordinal" , OrdinalEncoder()),
          ("ohe" , OneHotEncoder())
      ])

    elif data_type_ == 'bool' :
      pip = Pipeline([
          ("ordinal" , OrdinalEncoder())
      ])
    # For boolean features: encode using OrdinalEncoder
    elif data_type_ in ['int64' , 'float64'] :
      # For numeric features: scale using StandardScaler
      pip = Pipeline([
          ("scaler" , StandardScaler())
      ])
    return pip
  # Loop through each data type and evaluate feature importance
  for data_type in type_:
    # Select features of the current data type
    data_train = X.select_dtypes([data_type])
    data_validate = X_val.select_dtypes([data_type])
    data_col = (data_train.columns).tolist()

    if model == 'xgboost':
      for feature in data_col:
        # Build appropriate pipeline and transformer for the feature
        if data_type == 'O':
          pip = data_piplines(data_type)
          # Train and evaluate the feature
          transformer = ColumnTransformer([('onehot', pip, [feature])], remainder='drop')
          train_valid(transformer , train = data_train ,valid_data= data_validate , feature_=feature , type_data=data_type , y_val_=y_val)

        elif data_type == 'bool' :
          pip = data_piplines(data_type)
          transformer = ColumnTransformer([
              ('onehot', pip, [feature])
              ], remainder='drop')

          train_valid(transformer , train = data_train ,valid_data= data_validate , feature_=feature , type_data=data_type , y_val_=y_val)

        elif data_type == 'int64' or data_type == 'float64' :
          pip = data_piplines(data_type)
          transformer = ColumnTransformer([
              ('onehot', pip, [feature])
              ], remainder='drop')
          train_valid(transformer , train = data_train ,valid_data= data_validate , feature_=feature , type_data=data_type , y_val_=y_val)


    elif model == 'random_forest':
      for feature in data_col:
        feature_model = RandomForestRegressor(n_estimators=1000, random_state=42)
        feature_model.fit(data_train[feature] , y)
        feature_impact.append((feature , mean_absolute_error(feature_model.predict(valid) , y_val)))
  feature_importance = pd.DataFrame(feature_impact , columns=['col' , 'mse' , 'type'])
  return feature_importance.sort_values(by='mse')





# Evaluate feature importance using custom function
# This function trains a model on each feature individually and computes its MAE on validation data
# Lower MAE indicates higher predictive value for the target variable (accident_risk)
f_impact = feature_importance(X = train_data, X_val = validate_data, y_val = y_validate, y = y_train)

# Display the resulting DataFrame showing each feature's name, MAE score, and data type
f_impact



# Visualize feature importance scores using a bar plot
# Each bar represents a feature, and its height corresponds to the MAE (lower is better)
sns.barplot(x = f_impact['col'], y = f_impact['mse'])



# Drop selected features from the training dataset
# These features may be redundant, low-impact, or removed based on prior analysis
train_data = train_data.drop(['road_signs_present', 'time_of_day', 'num_lanes'], axis=1)



f_impact = feature_importance(X = train_data, X_val = validate_data, y_val = y_validate, y = y_train)
f_impact


# Build a ColumnTransformer to apply appropriate preprocessing to each feature type
transformer = ColumnTransformer([
    # Apply OneHotEncoder to categorical features (type: object)
    ("ohe", OneHotEncoder(), f_impact[f_impact['type'] == 'object']['col'].tolist()),

    # Apply MinMaxScaler to numerical features (type: int64 or float64)
    ("scaler", MinMaxScaler(), f_impact[(f_impact['type'] == 'int64') | (f_impact['type'] == 'float64')]['col'].tolist()),

    # Apply OrdinalEncoder to boolean features (type: bool)
    ("ordinal", OrdinalEncoder(), f_impact[f_impact['type'] == 'bool']['col'].tolist())
], remainder="passthrough")  # Leave all other columns unchanged


# Fit the transformer on training data and apply transformations
train = transformer.fit_transform(train_data)

# Convert the transformed training data to a DataFrame with proper column names
train = pd.DataFrame(train, columns=transformer.get_feature_names_out())

# Apply the same transformation to validation data
validate = transformer.transform(validate_data)

# Convert the transformed validation data to a DataFrame with matching column names
validate = pd.DataFrame(validate, columns=transformer.get_feature_names_out())


from sklearn.base import BaseEstimator, TransformerMixin

# Custom stacking model that learns from prediction errors across multiple folds
# Designed for regression tasks with support for CatBoost, XGBoost, and Linear Regression
class StackErorModel(BaseEstimator, TransformerMixin):

    def __init__(self, data, target, test, test_target, model='catboost', cv=3) -> None:
        # Store training and test data
        self.data = data
        self.target = target
        self.test = test
        self.test_target = test_target
        self.cv = cv  # Number of cross-validation folds

        # Initialize containers for cross-validation splits and final training data
        self.cv_data = []
        self.fainal_data = pd.DataFrame()
        self.fainal_target = pd.DataFrame()
        self.fainal_test_data = pd.DataFrame()
        self.fainal_test_target = pd.DataFrame()

        # Select base model based on user input
        if model == 'LinearRegression':
            self.model = LinearRegression()
        elif model == 'XGBoost':
            self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        elif model == 'catboost':
            # Base CatBoost model for fold-level training
            self.model = CatBoostRegressor(
                
                iterations=1500,
                learning_rate=0.07,
                depth=10,
                verbose=100,
                loss_function='RMSE',
                random_strength=5,
                l2_leaf_reg=10
            )
            # Final CatBoost model trained on error-based features
            self.fainal_model = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.8,
                depth=9,
                verbose=100,
                loss_function='RMSE',
                random_strength=5,
                l2_leaf_reg=10
            )

    def fit(self, x, y=None):
        # Perform cross-validation and collect prediction errors
        for i in range(self.cv):
            train, valid, y_train, y_valid = train_test_split(
                self.data, self.target, test_size=0.2, random_state=i
            )
            valid = valid.reset_index(drop=True)
            y_valid = y_valid.reset_index(drop=True)
            self.cv_data.append((train, valid, y_train, y_valid))

        count = 0
        for train, valid, y_train, y_valid in self.cv_data:
            # Train base model and predict on validation fold
            self.model.fit(train, y_train)
            prediction = self.model.predict(valid)

            # Compute absolute error and store predictions
            absolute_error = np.abs(np.array(prediction) - np.array(y_valid))
            self.fainal_data["absolute_error"] = absolute_error
            self.fainal_data["prediction"] = prediction
            self.fainal_target["target"] = y_valid

            # On first fold, also evaluate on test set
            if count == 0:
                self.test = self.test.reset_index(drop=True)
                self.target = self.target.reset_index(drop=True)
                test_pred = self.model.predict(self.test)
                test_absolute_error = np.abs(np.array(test_pred) - np.array(self.test_target))
                self.fainal_test_data["absolute_error"] = test_absolute_error
                self.fainal_test_data["prediction"] = test_pred
                self.fainal_test_target["target"] = self.test_target

                # Train final model on error-based features
                self.fainal_model.fit(self.fainal_data, self.fainal_target)
                self.fainal_model.save_model(f'model{count}.cbm')
            else:
                # Continue training final model using previous iteration as init_model
                self.fainal_model.fit(
                    self.fainal_data, self.fainal_target, init_model=f'model{count-1}.cbm'
                )
                self.fainal_model.save_model(f'model{count}.cbm')

            count += 1

        return self

    def transform(self, x, y=None):
        # Return final model and test predictions with error features
        return self.fainal_model, self.fainal_test_data, self.fainal_test_target



stack_model = StackErorModel(data=train , target=y_train , test = validate , test_target=y_validate , model='catboost' , cv=3)
model , test , target= stack_model.fit_transform(train)


# Generate predictions on the test set using the final model
pered = model.predict(test)

# Evaluate model performance using Mean Absolute Error (MAE)
print(mean_absolute_error(target, pered))  # Lower MAE indicates better accuracy

# Compute RÂ² score to assess goodness of fit
print(r2_score(target, pered))  # Closer to 1 means better explanatory power


