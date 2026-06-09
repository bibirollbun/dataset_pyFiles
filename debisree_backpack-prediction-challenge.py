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


#import libraries:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.metrics import mean_squared_error

import optuna
import warnings
warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  


#Reading Data:

test_filepath = "/kaggle/input/playground-series-s5e2/test.csv"
train_filepath = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(train_filepath, index_col="id")

print(df.shape)


df_test = pd.read_csv(test_filepath, index_col="id")

print(df_test.shape)

extra_filepath = "/kaggle/input/playground-series-s5e2/training_extra.csv"
df_extra = pd.read_csv(extra_filepath, index_col="id")
print(df_extra.shape)


#Exploring Training Data:

df.head()


#df_extra:

df_extra.head()


#Let's concat df and df_extra to have more in the training data:

df_train = pd.concat([df, df_extra], axis =0, ignore_index = True)
df_train.shape


# Missing Vlaue?

df_train.isnull().sum()


#Any missing value in test set?

df_test.isnull().sum()


### Columns:

df_train.columns


df_train.dtypes


### Price - Target variable:

df_train['Price'].describe()


df_train.head()


#Histogram:

# Set figure size for better visibility
plt.figure(figsize=(10, 6))

# Create histogram with Seaborn for a clean look
sns.histplot(df_train['Price'], bins=20, kde=True, color="skyblue", edgecolor="black")

# Calculate and plot the average price as a red dotted line
avg_price = df_train['Price'].mean()
plt.axvline(avg_price, color='red', linestyle='dotted', linewidth=2, label=f'Avg Price: {avg_price:.2f}')

# Add labels and title for better clarity
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Price Distribution with Average Price Line", fontsize=14)
plt.legend()

# Show the plot
plt.show()


### Brand:

x=df_train['Brand'].value_counts()


### Plot:
plt.figure(figsize=(8, 5))
plt.bar(x.index, x.values, color='skyblue')
plt.xlabel('Brands')
plt.ylabel('Counts')
plt.show()


# Create the boxplot
# plt.figure(figsize=(8,5))
# sns.boxplot(x='Brand', y='Price', data=df_train_, palette="pastel")

# # Labels and Title
# plt.xlabel("Brand")
# plt.ylabel("Price ($)")
# plt.title("Price Distribution by Brand")

# # Show plot
# plt.show()

# Set figure size
plt.figure(figsize=(10, 6))

# Box Plot: Shows the price distribution of materials per brand
sns.boxplot(x="Brand", y="Price", hue="Material", data=df_train, palette="Set2")

# Add title
plt.title("Brand-wise Price Distribution Across Materials")
plt.xlabel("Brand")
plt.ylabel("Price ($)")
plt.legend(title="Material")
plt.show()



df_train['Material'].value_counts()



df_train['Compartments'].value_counts()

# Box Plot: Shows the price distribution of materials per brand
sns.boxplot(x="Compartments", y="Price", data=df_train, )

# Add title
plt.title("Price Distribution Across compartments")
plt.xlabel("Compartments")
plt.ylabel("Price ($)")
plt.legend(title="Compartments")
plt.show()


df_train['Size'].value_counts()


# Box Plot: Shows the price distribution of materials per brand
sns.boxplot(x="Size", y="Price", data=df_train, )

# Add title
plt.title("Price Distribution Across sizes")
plt.xlabel("Size")
plt.ylabel("Price ($)")
plt.legend(title="Size")
plt.show()


df_train['Laptop Compartment'].value_counts()


# Box Plot: Shows the price distribution of materials per brand
sns.boxplot(x="Laptop Compartment", y="Price", data=df_train, )

# Add title
plt.title("Price Distribution Across Laptop Compartments")
plt.xlabel("Laptop Compartment")
plt.ylabel("Price ($)")

plt.show()


df_train['Waterproof'].value_counts()


df_train['Style'].value_counts()


df_train['Color'].value_counts()



# Create histogram 
sns.histplot(df_train['Weight Capacity (kg)'], bins=20, kde=True, color="skyblue", edgecolor="black")
plt.show()


df_train.columns


xx= df_train[df_train['Laptop Compartment'] == 'No']
xx['Compartments'].hist().plot()
plt.show()


yy= df_train[df_train['Laptop Compartment'] == 'Yes']
yy['Compartments'].hist().plot()
plt.show()


### Data Cleaning:
### Missing Vlaue Imputation - 


def fill_missing_values(df):
    
    df_filled = df.copy()  # Avoid modifying original DataFrame
    
    for col in df_filled.columns:
        if df_filled[col].dtype == 'object':  # Categorical columns
            df_filled[col] = df_filled[col].fillna('Missing')  # Assign after fill
        elif df_filled[col].dtype in ['int64', 'float64']:  # Numeric columns
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())  # Assign after fill
        
    
    return df_filled


# def fill_missing_values(df):
#     df_filled = df.copy()  # Avoid modifying the original DataFrame
#     grouping_cols = ['Brand', 'Material', 'Size','Laptop Compartment','Waterproof','Style','Color']
    
#     for col in df_filled.columns:
        
#         if df_filled[col].dtype == 'object':  # Categorical columns
#             df_filled[col] = df_filled[col].fillna('Missing')
#         elif df_filled[col].dtype in ['int64', 'float64']:  # Numeric columns
            
#             df_filled[col] = df_filled.groupby(grouping_cols)[col].transform(lambda x: x.fillna(x.mean()))
    
#     return df_filled




#Implement the function on Test set:

train = fill_missing_values(df_train)
print(train.shape)
train.isnull().sum()



test = fill_missing_values(df_test)
print(test.shape)
test.isnull().sum()


cols_to_convert = ['Brand', 'Material', 'Size','Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
train[cols_to_convert] = train[cols_to_convert].apply(lambda x: x.astype('category'))
test[cols_to_convert] = test[cols_to_convert].apply(lambda x: x.astype('category'))


# Size - Label encoding & others - frequency encoding


# def encode_columns(df, cols):
#     """
#     Apply Frequency Encoding to all columns except 'Size', 
#     which uses Label Encoding.
    
#     """
#     encoders = {}  # To store the LabelEncoder for 'Size'
    
#     for col in cols:
#         if col in df.columns:
#             if col == "Size":
#                 # Apply Label Encoding only for 'Size'
#                 le = LabelEncoder()
#                 df[col] = le.fit_transform(df[col].astype(str))  
#                 encoders[col] = le  # Store LabelEncoder for 'Size'
#             else:
#                 # Apply Frequency Encoding for all other categorical columns
#                 df[col] = df[col].map(df[col].value_counts() / len(df))
    
#     return df



# def encode_columns(df):
#     df_encoded = df.copy()
    
#     # Label Encoding for 'Size'
#     le = LabelEncoder()
#     df_encoded['Size'] = le.fit_transform(df_encoded['Size'].astype(str))
    
#     # Frequency Encoding for 'Brand', 'Material', and 'Color'
#     for col in ['Brand', 'Material', 'Color']:
#         df_encoded[col] = df_encoded[col].map(df_encoded[col].value_counts())
    
#     # One-Hot Encoding for 'Laptop Compartment', 'Waterproof', and 'Style'
#     one_hot_cols = ['Laptop Compartment', 'Waterproof', 'Style']
#     df_encoded = pd.get_dummies(df_encoded, columns=one_hot_cols, drop_first=True)
    
#     return df_encoded


# def encode_columns(df, cols):
#     """
#     Encode specified columns in the DataFrame.

#     """
#     df_encoded = df.copy()
#     encoders = {}
    
#     # Define which columns to label encode
#     label_cols = ['Size', 'Laptop Compartment', 'Waterproof']
    
#     for col in cols:
#         if col in df_encoded.columns:
#             if col in label_cols:
#                 # Apply Label Encoding for specified columns
#                 le = LabelEncoder()
#                 df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#                 encoders[col] = le
#             else:
#                 # Apply Frequency Encoding for all other columns
#                 df_encoded[col] = df_encoded[col].map(df_encoded[col].value_counts() / len(df_encoded))
    
#     return df_encoded





def encode_columns(df, cols):
    """
    Label encodes a list of categorical columns in a DataFrame.
    
   
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        
    return df_encoded




col = ['Brand', 'Material', 'Size',  'Laptop Compartment',
       'Waterproof', 'Style', 'Color']


df_train_encoded = encode_columns(train, col)


#df_train_encoded = train


# Apply label encoding function
df_test_encoded = encode_columns(test, col)


#df_test_encoded = test


df_train_encoded['brand_w_mean'] = df_train_encoded.groupby('Brand')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['mat_w_mean'] = df_train_encoded.groupby('Material')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['size_w_mean'] = df_train_encoded.groupby('Size')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['comp_w_mean'] = df_train_encoded.groupby('Laptop Compartment')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['wf_w_mean'] = df_train_encoded.groupby('Waterproof')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['style_w_mean'] = df_train_encoded.groupby('Style')['Weight Capacity (kg)'].transform('mean')
df_train_encoded['color_w_mean'] = df_train_encoded.groupby('Color')['Weight Capacity (kg)'].transform('mean')
#df_train_encoded['wt_mean'] = df_train_encoded.groupby(['Brand','Material','Size'])['Weight Capacity (kg)'].transform('mean')

# df_train_encoded['col_mode'] = df_train_encoded.groupby(['Brand', 'Material', 'Size'])['Color'] \
#     .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Missing')

# df_train_encoded['wf_mode'] = df_train_encoded.groupby(['Brand', 'Material', 'Size'])['Waterproof'] \
#     .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Missing')

df_train_encoded.head()


df_test_encoded['brand_w_mean'] = df_test_encoded.groupby('Brand')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['mat_w_mean'] = df_test_encoded.groupby('Material')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['size_w_mean'] = df_test_encoded.groupby('Size')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['comp_w_mean'] = df_test_encoded.groupby('Laptop Compartment')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['wf_w_mean'] = df_test_encoded.groupby('Waterproof')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['style_w_mean'] = df_test_encoded.groupby('Style')['Weight Capacity (kg)'].transform('mean')
df_test_encoded['color_w_mean'] = df_test_encoded.groupby('Color')['Weight Capacity (kg)'].transform('mean')



#df_test_encoded['compartment_mean'] = df_test_encoded.groupby(['Brand','Material','Size'])['Compartments'].transform('mean')
#df_test_encoded['wt_mean'] = df_test_encoded.groupby(['Brand','Material','Size'])['Weight Capacity (kg)'].transform('mean')

# df_test_encoded['col_mode'] = df_test_encoded.groupby(['Brand', 'Material', 'Size'])['Color'] \
#     .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Missing')

# df_test_encoded['wf_mode'] = df_test_encoded.groupby(['Brand', 'Material', 'Size'])['Waterproof'] \
#     .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Missing')

df_test_encoded.head()


df_train_encoded.columns


### Data Prep for Modeling:
### Data Splitting 

target = "Price"
X = df_train_encoded.drop(columns= target)


y = df_train_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,          # Increased number of trees
    learning_rate=0.02,        # Lower learning rate for gradual learning
    max_depth=4,               # Limits the depth of each tree
    subsample=0.8,             # Row subsampling to reduce overfitting
    colsample_bytree=0.5,      # Feature subsampling to reduce overfitting
    random_state=42,
    enable_categorical=True,
)

# xgb_model = XGBRegressor(
#     objective='reg:squarederror',  # For regression
#     n_estimators=1000,              # High number of trees with early stopping in practice
#     learning_rate=0.01,             # Low learning rate for gradual learning
#     max_depth=6,                    # Moderate tree depth
#     min_child_weight=10,            # Higher value to avoid splits on noise
#     subsample=0.7,                  # Row subsampling to reduce variance
#     colsample_bytree=0.5,           # Column subsampling to reduce overfitting
#     gamma=0.1,                      # Minimum loss reduction required to make a split
#     reg_alpha=0.1,                  # L1 regularization
#     reg_lambda=1.0,                 # L2 regularization
#     tree_method='hist',             # Fast histogram-based algorithm (use 'gpu_hist' if using GPU)
#     n_jobs=-1,                      # Use all CPU cores
#     random_state=42
# )

# xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# param_distributions = {
#     'n_estimators': [300, 500, 700],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 4, 5, 7],
#     'gamma': [0, 0.1, 0.2],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0],
#     'reg_alpha': [0, 0.01, 0.1],
#     'reg_lambda': [1.0, 1.5, 2.0]
# }


# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_distributions,
# #     n_iter=50,
# #     scoring='neg_mean_squared_error',
# #     cv=5,
# #     verbose=1,
# #     random_state=42,
# #     n_jobs=-1
# # )

# # random_search.fit(X_train, y_train)
# # print("Best parameters:", random_search.best_params_)
# # best_model = random_search.best_estimator_


# xgb_model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     early_stopping_rounds=10,
#     verbose=True
# )

# # Make predictions
# y_pred = xgb_model.predict(X_test)

# # Compute RMSE
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Root Mean Squared Error (RMSE): {rmse}")



# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")



#Feature Importance - Get values and sort them
feature_importance = xgb_model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]  # Sort indices in descending order

# Sort feature names and importance values accordingly
sorted_features = np.array(X_train.columns)[sorted_indices]
sorted_importance = feature_importance[sorted_indices]

# Plot Feature Importance (Sorted)
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Invert y-axis for highest importance on top
plt.show()



# Here, n_features_to_select=5 is arbitrary; adjust it based on your needs.



# # Set up RFECV with a 5-fold cross-validation and use negative mean squared error as the scoring metric
# rfecv = RFECV(estimator=xgb_model, step=1, cv=KFold(5, shuffle=True, random_state=42),
#               scoring='neg_mean_squared_error')

# # Fit RFECV on the training data
# rfecv.fit(X_train, y_train)

# # The optimal number of features
# optimal_n_features = rfecv.n_features_
# print("Optimal number of features:", optimal_n_features)

# # List the selected features
# selected_features = X_train.columns[rfecv.support_]
# print("Selected features:")
# print(selected_features)




# Evaluate performance with the selected features
# xgb_model.fit(X_train[selected_features], y_train)
# preds = xgb_model.predict(X_test[selected_features])
# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE with selected features:", rmse)


# Subset the test data to only the features used for training
# df_test_final = df_test_encoded[selected_features]


# Handle the prediction and submission file creeation:

test_pred = xgb_model.predict(df_test_encoded)
test_pred

submission = pd.DataFrame({'id': test.index, 'Price': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)



#Test Prediction values 


plt.figure(figsize=(6,4))
plt.hist(test_pred, bins=100)
plt.title("Test Predictions")
plt.show()


test_pred.mean()


# Initialize the SHAP explainer
# explainer = shap.Explainer(xgb_model)
# shap_values = explainer(X_test)

# # Visualize the SHAP summary plot
# shap.summary_plot(shap_values, X_test)




# Define the objective function for Optuna:

# # def objective(trial):
# #     # Suggest hyperparameters to optimize
# #     params = {
# #         "n_estimators": trial.suggest_int("n_estimators", 50, 500),
# #         "max_depth": trial.suggest_int("max_depth", 3, 15),
# #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
# #         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
# #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
# #         "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
# #         "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
# #         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
# #         "objective": "reg:squarederror",
# #         "random_state": 42
# #     }

# #     # Initialize and train the model
# #     model = XGBRegressor(**params)
# #     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Compute RMSE
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#     return rmse  # Optuna will minimize this

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)  # Run 50 trials

# # Get the best parameters
# best_params = study.best_params
# print("Best Hyperparameters:", best_params)

# # Train XGBoost with best parameters
# best_xgb_model = XGBRegressor(**best_params)
# best_xgb_model.fit(X_train, y_train)




# Make final predictions
# final_predictions = best_xgb_model.predict(X_test)

# # Compute RMSE
# final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
# print(f"Optimized RMSE: {final_rmse}")


# Feature Importance - Get values and sort them
# feature_importance = best_xgb_model.feature_importances_
# sorted_indices = np.argsort(feature_importance)[::-1]  # Sort indices in descending order

# # Sort feature names and importance values accordingly
# sorted_features = np.array(X_train.columns)[sorted_indices]
# sorted_importance = feature_importance[sorted_indices]

# # Plot Feature Importance (Sorted)
# plt.figure(figsize=(10, 6))
# plt.barh(sorted_features, sorted_importance)
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("XGBoost Feature Importance (Sorted)")
# plt.gca().invert_yaxis()  # Invert y-axis for highest importance on top
# plt.show()




# Handle the prediction and submission file creeation:

# test_pred = best_xgb_model.predict(df_test_encoded)
# test_pred

# submission = pd.DataFrame({'id': test.index, 'Price': test_pred})
# submission.to_csv('/kaggle/working/submission.csv', index=False)
# display(submission)


y_train.std()




