# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, validation_curve, cross_val_score
from catboost import CatBoostRegressor, Pool
from category_encoders import TargetEncoder, cat_boost
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import shap
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

display(test)

# Combine train and train extra data sets into one
df = pd.concat([train, train_extra], axis=0, ignore_index=True)

# Rows and Cols
display(df.shape)
display(test.shape)


# Descriptive stats
df.describe()

# Columns and 
df.info()
test.info()

# Data tyoes
display(df.dtypes)

#Summarize
df.head()

# Check for null/missing values
display(df.dtypes, df.isnull().sum())


# Plot price distributions
plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=50, kde=True, color="green")
plt.title("Price Distribution")
plt.show()

# Compare distributions between train and test
sns.kdeplot(df['Weight Capacity (kg)'], label="Train", shade=True)
sns.kdeplot(test['Weight Capacity (kg)'], label="Test", shade=True)


categorical_cols = ['Size', 'Brand', 'Material', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']

# Log transform the weight capacity and compartments
#df['Weight Capacity (kg)'] = np.log1p(df['Weight Capacity (kg)']) 
#df['Compartments'] = np.log1p(df['Compartments'])
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# Repeat for the test data set
#test['Weight Capacity (kg)'] = np.log1p(test['Weight Capacity (kg)'])
#test['Compartments'] = np.log1p(test['Compartments']) 
test[categorical_cols] = test[categorical_cols].fillna('Unknown')

# Impute missing medians using Test columns
for col in test.select_dtypes(include=['number']).columns:
    medianVal = df[col].median()
    df[col].fillna(medianVal, inplace=True)
    test[col].fillna(medianVal, inplace=True)


# Drop the target Price column
X = df.drop(columns=['Price'])
y = df['Price']
display(X) # Before encoding

# Fit encoder and transform the features 
cbe = cat_boost.CatBoostEncoder() 
X = cbe.fit(X, y).transform(X) 
display(X) # After encoding
    
# Split the training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {'iterations': 1000, 'depth': 4, 'learning_rate': 0.21, 'l2_leaf_reg': 5}

# Try CatBoost 
cbr = CatBoostRegressor(**params, random_seed=42, early_stopping_rounds=50, loss_function='RMSE')

# Create train and test pools
train_pool = Pool(X_train, label=y_train)
test_pool = Pool(X_test, label=y_test)

# Train the model
cbr.fit(train_pool, eval_set=test_pool, verbose=200, early_stopping_rounds=50, use_best_model=True, plot=True)

# Predict
y_pred = cbr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"CatBoost Root Mean Squared Error (RMSE): {rmse}")



# Grid Search for hyperparameter tuning. 
#model = CatBoostRegressor()

#params = {'learning_rate': [0.05, 0.015, 0.025],
#        'depth': [4, 5, 6],
#        'l2_leaf_reg': [3, 5, 7]}

#grid_search_result = model.grid_search(params,
#                                       X=X_train,
#                                       y=y_train,
#                                       plot=True,
#                                       cv=3,
#                                       partition_random_seed=42,
#                                       calc_cv_statistics=True,
#                                       search_by_train_test_split=True,
#                                       refit=True,
#                                       shuffle=True,
#                                      train_size=0.8,
#                                       verbose=True)


feature_importance = cbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')


explainer = shap.Explainer(cbr)
shap_values = explainer(X_test)
shap_importance = shap_values.abs.mean(0).values
shap.plots.bar(shap_values, max_display=X_test.shape[0])


# Handle the prediction using the best iteration and submission file creeation
test = cbe.transform(test)

test_pred = cbr.predict(test)
test_pred

submission = pd.DataFrame({'id': test.index, 'Price': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)


# Remove old file(s)
#import os
#os.remove('/kaggle/working/submission.csv')

