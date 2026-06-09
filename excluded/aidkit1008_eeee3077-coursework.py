#Imports the NumPy library and assigns it the alias np; it's commonly used for numerical and matrix operations.
import numpy as np # linear algebra
# Imports the pandas library as pd, which is widely used for data manipulation and reading/writing CSV files.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Imports the built-in os module to interact with the operating system (e.g., file paths).
import os

# Recursively walks through all directories and files under the specified input path.
for dirname, _, filenames in os.walk(r"/kaggle/input"):
    #Iterates through each file name in the current directory.
    for filename in filenames:
        #Prints the full path of each file found in the input directory.
        print(os.path.join(dirname, filename))


#Loads the training dataset from a CSV file into a DataFrame named df.
df=pd.read_csv(r"/kaggle/input/california-house-prices/train.csv")
#Displays summary information about the DataFrame, including column names, data types, and non-null counts.
df.info()
#Randomly displays 3 rows from the DataFrame for a quick glance at the data content.
df.sample(3)


# Creates a deep copy of the df DataFrame and stores it in df_train for further manipulation.
df_train=df.copy()
#Loads the test dataset into a new DataFrame named df_test.
df_test=pd.read_csv(r"/kaggle/input/california-house-prices/test.csv")
# Iterates over a list of column names that represent dates in the dataset.
for field in ['Listed On', 'Last Sold On']:
    # Converts the specified columns in df_train to datetime objects for proper date/time handling.
    df_train[field]=pd.to_datetime(df[field])
    # Similarly converts the same columns in df_test to datetime format.
    df_test[field]=pd.to_datetime(df_test[field])


cate_cols = []   # Initialize an empty list to store categorical column names
num_cols = []    # Initialize an empty list to store numerical column names
date_cols = []   # Initialize an empty list to store date/time column names

dtypes = df_train.dtypes   # Get the data types of each column in the DataFrame

for col, dtype in dtypes.items():   # Iterate over each column name and its data type
    if dtype == 'object':           # If the column type is 'object' (typically strings or categories)
        cate_cols.append(col)       # Add it to the list of categorical columns
    elif dtype.name.startswith('datetime'):   # If the column type is datetime
        date_cols.append(col)       # Add it to the list of datetime columns
    else:                           # Otherwise, it is treated as a numerical column
        num_cols.append(col)        # Add it to the list of numerical columns



id_col = 'Id'                      # Define the name of the ID column
target_col = 'Sold Price'         # Define the name of the target variable column (what we want to predict)

for col in [id_col, target_col]:  # Iterate over both the ID and target column names
    num_cols.remove(col)          # Remove them from the list of numerical columns (we don't want to process them as features)

print(cate_cols, num_cols, date_cols)  # Print out the lists of categorical, numerical, and datetime columns



from sklearn.base import BaseEstimator, TransformerMixin  # Import base classes to create custom transformers for sklearn pipelines
from pandas.api.types import is_string_dtype, is_numeric_dtype  # Import utilities to check column data types in pandas

class Num_Features(BaseEstimator, TransformerMixin):  # Define a custom class for handling numerical feature transformation
    def __init__(self, cols = [], fillna = False, addna = False):  # Constructor to initialize options and tracking variables
        self.fillna = fillna       # Whether to fill missing values
        self.cols = cols           # Columns to process
        self.addna = addna         # Whether to add new indicator columns for NaNs
        self.na_cols = []          # Columns with missing values (to track for _na suffix)
        self.imputers = {}         # Dictionary to store median values used for filling NaNs
    def fit(self, X, y=None):     # Fit method is called during training to compute statistics
        for col in self.cols:     # Loop over each selected column
            if self.fillna:       # If missing value imputation is enabled
                self.imputers[col] = float(X[col].median())  # Store the median value as a float for later imputation
            if self.addna and X[col].isnull().sum():  # If flag is set and column has missing values
                self.na_cols.append(col)  # Record the column to later add a _na missing-indicator column
        print(self.na_cols, self.imputers)  # Print the missing value columns and imputation values (for debugging)
        return self  # Return self to allow chaining
        
    def transform(self, X, y=None):  # Transform method applies changes to the data
        df = X.loc[:, self.cols]     # Create a new DataFrame with only the selected columns
        
        # Fill missing values using the median computed during fit
        for col in self.imputers:    
            df[col] = df[col].fillna(self.imputers[col])  # Avoid inplace operations to prevent chained assignment warning
        
        # Add boolean indicator columns for missing values if required
        for col in self.na_cols:
            df[col + '_na'] = df[col].isnull()  # Create a new column indicating whether the value was originally NaN
        return df  # Return the transformed DataFrame



class Imputer(BaseEstimator, TransformerMixin):  # Define a custom imputer class compatible with scikit-learn's pipeline
    def __init__(self, strategy, fill_value):     # Constructor to initialize strategy and fill value
        self.strategy = strategy                  # Store the imputation strategy (not used directly here, but reserved for extension)
        self.fill_value = fill_value              # Store the value used to fill missing entries

    def fit(self, X, y=None):       # fit method (does nothing in this case since value is predefined)
        return self                 # Return self to allow use in scikit-learn pipelines

    def transform(self, X, y=None):        # transform method applies the imputation
        for col, content in X.items():     # Iterate over each column in the DataFrame
            X[col].fillna(self.fill_value, inplace=True)  # Fill missing values with the specified fill_value
        return X                           # Return the modified DataFrame


from sklearn.pipeline import Pipeline  # Import Pipeline to create sequential data processing steps
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler  # Import common preprocessing tools
from sklearn.impute import SimpleImputer  # Import SimpleImputer for filling in missing values

# Define a numerical feature processing pipeline
num_pipeline = Pipeline([
    ('select_num', Num_Features(cols=num_cols, fillna='median', addna=True)),  
    # Step 1: Apply custom Num_Features transformer to selected numerical columns
    # - 'fillna' is incorrectly set to 'median' (string), should be True or False
    # - 'addna=True' will add binary columns indicating missing values
])

num_pipeline.fit(df_train)  # Fit the pipeline to the training data (calculates medians, finds missing values)

X_num = num_pipeline.fit_transform(df_train)  # Transform the training data using the pipeline and store the result in X_num



class CatEncoder(BaseEstimator, TransformerMixin):  # Define a custom categorical encoder compatible with sklearn pipeline
    def __init__(self, cols, max_n_cat=7, onehot_cols=[], orders={}):
        self.cols = cols                # List of categorical column names to process
        self.onehot_cols = onehot_cols # Columns explicitly marked to be one-hot encoded
        self.cats = {}                 # Dictionary to store category values for label encoding
        self.max_n_cat = max_n_cat     # Threshold for the max number of categories to one-hot encode
        self.orders = orders           # Custom category orders for specific columns

    def fit(self, X, y=None):  # Fit method learns categories and encoding strategy
        df_cat = X.loc[:, self.cols]  # Extract only the categorical columns
        
        for n, c in df_cat.items():   # Loop over each categorical column
            df_cat[n].fillna('NAN', inplace=True)  # Fill missing values with string 'NAN'
            df_cat[n] = c.astype('category').cat.as_ordered()  # Convert to ordered categorical type
            
            if n in self.orders:  # If user provided custom category order
                df_cat[n].cat.set_categories(self.orders[n], ordered=True, inplace=True)  # Apply the specified order
            
            cats_count = len(df_cat[n].cat.categories)  # Count the number of unique categories

            # If the column has only 2 categories or too many categories, use label encoding
            if cats_count <= 2 or cats_count > self.max_n_cat:
                self.cats[n] = df_cat[n].cat.categories  # Save categories for future transformation
                if n in self.onehot_cols:
                    self.onehot_cols.remove(n)  # Remove from one-hot list if it's now label-encoded
            
            # If suitable for one-hot encoding and not already marked, add to one-hot list
            elif n not in self.onehot_cols:
                self.onehot_cols.append(n)
        
        print(self.onehot_cols)  # Output final one-hot column list for verification
        return self

    
    def transform(self, df, y=None):  # Transform method applies encoding to the data
        X = df.loc[:, self.cols].copy()  # Work on a copy of the input categorical columns

        for col in self.cats:  # Apply label encoding to columns marked for it
            X[col] = X[col].fillna('NAN')  # Fill missing values again with 'NAN'
            X[col] = pd.Categorical(X[col], categories=self.cats[col], ordered=True)  # Convert to ordered category using stored list

            # Safety check: ensure the dtype is still categorical
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype('category')
            
            X[col] = X[col].cat.codes  # Convert category to integer codes (label encoding)

        if len(self.onehot_cols):  # If any columns are marked for one-hot encoding
            df_1h = pd.get_dummies(X[self.onehot_cols], dummy_na=True)  # Create one-hot encoded DataFrame (include NaN as category)
            df_drop = X.drop(self.onehot_cols, axis=1)  # Drop one-hot encoded columns from original
            return pd.concat([df_drop, df_1h], axis=1)  # Concatenate label-encoded and one-hot encoded parts

        return X  # If no one-hot encoding needed, return the label-encoded DataFrame



cat_pipeline = Pipeline([
    ('cat_encoder', CatEncoder(cols=cate_cols))  # Define a pipeline step that applies the custom CatEncoder to categorical columns
])

cat_pipeline.fit(df_train)  # Fit the pipeline to the training data — this will learn encoding strategies based on category counts

cat_pipeline.fit(df_train)  # (Redundant) This line repeats the fitting process — it can be removed without affecting functionality

X_cate = cat_pipeline.fit_transform(df_train)  # Fit and transform the training data, producing the encoded categorical features



def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    field = df[field_name]  # Extract the datetime column from the DataFrame

    if prefix is None:
        prefix = re.sub('[Dd]ate$', '', field_name)  # Remove the word "date" from the field name to use as prefix
    # Define standard date attributes to extract
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
            'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']  # Optionally add time-related features
    # Pandas removed `dt.week` in v1.1.10
    # Handle deprecated `dt.week` by using isocalendar if available (for pandas ≥ 1.1)
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    # Loop through each attribute and add it to the DataFrame
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
        # Use getattr for dynamic attribute access; handle Week separately
    mask = ~field.isna()  # Create a mask to identify non-null entries
    # Add a Unix timestamp (seconds since epoch) for the datetime field
    df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, np.nan)
    if drop:
        df.drop(field_name, axis=1, inplace=True)  # Drop the original date column if requested
    return df  # Return the modified DataFrame



import re  # Import regular expressions library for string operations (used in add_datepart)

class Datepart(BaseEstimator, TransformerMixin):  # Define a custom sklearn-compatible transformer for extracting datetime features
    def __init__(self, cols, time=False):  # Constructor with columns to process and whether to extract time components
        self.cols = cols                  # List of datetime columns to transform
        self.time = time                  # Boolean flag to include time-based features (Hour, Minute, Second)

    def fit(self, X, y=None):  # Fit does nothing here since transformation doesn't require learning from data
        return self            # Return self to comply with sklearn pipeline API

    def transform(self, X, y=None):  # Applies transformation logic to the input DataFrame
        df_dates = X.loc[:, self.cols]  # Select only the datetime columns for processing
        for col in self.cols:
            add_datepart(df_dates, col, time=False)  # Apply the earlier defined function to expand date features (time is off here)
        return df_dates  # Return the DataFrame with expanded datetime features
# Define a pipeline to process datetime columns
date_pipeline = Pipeline([
    ('datepart', Datepart(cols=date_cols)),  # Step 1: Extract temporal features from datetime columns
    ('imputer', Imputer(strategy="constant", fill_value=-1)),  # Step 2: Fill any missing values with -1
])



X_date = date_pipeline.fit_transform(df_train)  
# Fit the date_pipeline to the training data and transform it in one step
# - This applies the Datepart transformer to extract features from datetime columns
# - Then applies the Imputer to fill missing values with -1
# - The resulting DataFrame X_date contains engineered date features ready for model input



y_train = np.log(df_train[target_col])  
# Apply natural logarithm transformation to the target variable ('Sold Price')
# - This helps normalize the distribution and stabilize variance
# - Useful for regression when target has a long-tailed or skewed distribution
X_train = pd.concat([X_num, X_cate, X_date], axis=1)  
# Concatenate the transformed numerical, categorical, and date features into one feature matrix
# - axis=1 means horizontal concatenation (column-wise)
# - Resulting X_train will be used as input for model training
X_train.shape, y_train.shape  
# Display the shapes of the final feature matrix and target vector
# - Useful to check that the number of training samples match


from sklearn.model_selection import ParameterGrid  # Import tool to iterate over all combinations of parameter values
from sklearn.ensemble import RandomForestRegressor  # Import the RandomForestRegressor model from sklearn
model = RandomForestRegressor(oob_score=True, random_state=3, n_jobs=-1)  
# Initialize a Random Forest model with:
# - oob_score=True: enables Out-Of-Bag (OOB) evaluation for internal cross-validation
# - random_state=3: ensures reproducibility
# - n_jobs=-1: uses all available CPU cores for training
params = {
    'n_estimators': [200],             # Number of trees in the forest (try 200 trees)
    'min_samples_leaf': [2],           # Minimum number of samples required at a leaf node
    'max_features': [0.5],             # Fraction of features to consider when looking for the best split
    'max_depth': [13],                 # Maximum depth of the trees
    'min_samples_split': [2]           # Minimum number of samples required to split an internal node
}
# These are hyperparameters to tune. You can add more values to each list to perform grid search over multiple configurations.
best_score = 0  # Initialize the best OOB score seen so far
for g in ParameterGrid(params):  # Loop over each combination of hyperparameters generated by ParameterGrid
    model.set_params(**g)        # Set the model with the current hyperparameter combination
    model.fit(X_train, y_train)  # Train the model on the training data
    if model.oob_score_ > best_score:  # If current model's OOB score is better than previous best
        best_score = model.oob_score_  # Update the best OOB score
        best_grid = g                  # Store the best parameter set
        print('oob:', best_score, best_grid)  # Print out the best score and corresponding parameters



from sklearn.ensemble import RandomForestRegressor  
# Import the RandomForestRegressor model from scikit-learn's ensemble module
m = RandomForestRegressor(
    n_jobs=-1,               # Use all available CPU cores for parallel training
    n_estimators=200,        # Number of decision trees in the forest
    oob_score=True,          # Enable out-of-bag samples to estimate the generalization performance
    max_depth=17,            # Maximum depth of each tree
    min_samples_leaf=4,      # Minimum number of samples required at each leaf node
    min_samples_split=2,     # Minimum number of samples required to split an internal node
    max_features=0.5         # Fraction of features to consider when looking for the best split
)
m.fit(X_train, y_train)  
# Train the Random Forest model on the preprocessed training data (X_train) and target values (y_train)
m.oob_score_  
# Return the Out-Of-Bag (OOB) R² score, which estimates the model’s performance on unseen data


def rf_feat_importance(m, df):
    # Create a DataFrame showing feature importance
    return pd.DataFrame({
        'cols': df.columns,              # Column names from the input feature DataFrame
        'imp': m.feature_importances_    # Corresponding importance scores from the trained Random Forest model
    }).sort_values('imp', ascending=False)  # Sort the features by importance in descending order

fi = rf_feat_importance(m, X_train)  
# Call the function to compute feature importances for the trained model `m` using the training data `X_train`
# Store the result in `fi`, which is a DataFrame with columns ['cols', 'imp']

fi[:50]  
# Display the top 50 most important features ranked by the Random Forest model



del_cols = []  
# List of columns to explicitly delete (currently empty).
# Originally included redundant or low-value columns like 'Address', 'Summary', or columns with data leakage.
# Can be uncommented or extended as needed.
keep_cols = ['Listed Price', 'Tax assessed value', 'Annual tax amount', 
             'Last Sold Price', 'Total interior livable area', 'Zip']
# A manually curated list of essential columns that you may want to keep regardless of their importance score.
# These may have domain knowledge or known predictive power.

Threshold = 0.0009  
# Set a minimum feature importance threshold.
# Features with importance below this value will be dropped.

to_keep = fi[fi.imp > Threshold].cols  
# Select only those features whose importance score is greater than the defined threshold.
# Returns a pandas Series of column names.

to_keep = [col for _, col in to_keep.items()]  
# Convert the Series to a standard Python list using list comprehension.
# `items()` returns (index, value), and we only collect the value (i.e., column name).



for col in del_cols:
    if col in to_keep:
        to_keep.remove(col)  
# Loop through each column in del_cols.
# If the column is currently in the to_keep list, remove it.
# This ensures explicitly unwanted features are excluded, even if they passed the importance threshold.

for col in keep_cols:
    if col not in to_keep:
        to_keep.append(col)  
# Loop through manually specified essential features.
# If any of them are not already in the to_keep list, add them.
# This ensures critical features (based on domain knowledge) are always included.

print(to_keep)  
# Print the final list of features to be kept for model input.

df_keep = X_train[to_keep].copy()  
# Create a new DataFrame containing only the selected features.
# .copy() is used to ensure it’s an independent copy and not just a view.



m1 = RandomForestRegressor(
    n_jobs=-1,              # Use all available CPU cores to speed up training
    random_state=3,         # Set random seed for reproducibility
    n_estimators=300,       # Use 300 decision trees in the forest
    oob_score=True,         # Enable Out-Of-Bag score for validation without a separate validation set
    max_depth=13,           # Limit the maximum depth of each tree to avoid overfitting
    min_samples_leaf=2,     # Require at least 2 samples at each leaf node
    min_samples_split=2,    # A node must have at least 2 samples to be split
    max_features=0.5        # Use 50% of features when looking for the best split at each node
)
# Initialize a new Random Forest model with the specified hyperparameters
m1.fit(df_keep, y_train)  
# Train the model using the refined set of features (`df_keep`) and the log-transformed target (`y_train`)

print(m1.oob_score_)  
# Print the Out-Of-Bag R² score, an internal estimate of model performance on unseen data



m1.oob_score_  
# This returns the Out-Of-Bag (OOB) R² score of the trained Random Forest model `m1`
# It estimates how well the model generalizes to unseen data, using the data not seen by each tree during training.



cols = to_keep  
# List of all features currently selected for model training

scores = []  # Store OOB scores when each feature is removed
feats = []   # Store corresponding feature names
for col in cols:  
    tmp = to_keep.copy()  # Make a fresh copy of the current feature list

    if col in keep_cols:
        continue  # Skip features that are manually specified as "must-keep"
    tmp.remove(col)  # Remove the current feature under test
    df_tmp = X_train[tmp].copy()  # Create a temporary training DataFrame without this feature
    m1 = RandomForestRegressor(
        n_jobs=-1,              # Use all CPU cores
        random_state=3,         # Seed for reproducibility
        n_estimators=30,        # Use fewer trees to speed up evaluation
        oob_score=True,         # Use Out-of-Bag score for validation
        max_depth=13, 
        min_samples_leaf=2, 
        min_samples_split=2, 
        max_features=0.5
    )
    m1.fit(df_tmp, y_train)  # Train the model without the current feature
    scores.append(m1.oob_score_)  # Save the resulting OOB score
    feats.append(col)             # Track which feature was removed
#     print(col, m1.oob_score_)
# Combine scores and feature names into tuples and sort by score in descending order
to_del = sorted(zip(scores, feats), reverse=True)

to_del


# 最好提交的特征，18个
to_keep_final = [
    'Listed Price',                    # Asking price of the property
    'Tax assessed value',              # Government-assessed property value
    'Last Sold Price',                 # Price at which the property was last sold
    'Zip',                             # Zip code (region indicator)
    'Total interior livable area',     # Interior size of the home
    'Elementary School Score',         # Quality rating of nearby elementary school
    'Listed OnElapsed',                # Time since listing, in seconds since epoch
    'Last Sold OnElapsed',             # Time since last sale, in seconds since epoch
    'Full bathrooms',                  # Number of full bathrooms
    'Year built',                      # Year the house was constructed
    'Listed OnYear',                   # Year of listing
    'Lot',                             # Lot size
    'Parking',                         # Parking availability
    'Type',                            # Property type (e.g., house, condo)
    'Middle School Score',             # Quality rating of nearby middle school
    'High School Distance',            # Distance to the nearest high school
    'Elementary School Distance',      # Distance to the nearest elementary school
    'Bedrooms'                         # Number of bedrooms
]
# This list contains the top 18 features empirically selected for optimal submission performance
# to_keep_final=['Listed Price', 'Tax assessed value', 'Last Sold Price', 'Zip', 'Total interior livable area', 'Listed OnElapsed', 'Elementary School Score', 'Last Sold OnElapsed', 'Year built', 'Listed OnYear', 'High School Distance', 'Lot', 'Parking', 'Middle School Score', 'Elementary School Distance', 'Region', 'Bedrooms', 'High School Score', 'Heating', 'Appliances included', 'Flooring', 'Middle School Distance']
X_train_final = X_train[to_keep_final].copy()  
# Create a final training set by selecting only the best-performing features
# .copy() ensures it's a standalone DataFrame (not a view)



# 2nd pass grid search to determine the final parameters
from sklearn.model_selection import ParameterGrid  # Import tool for exhaustive grid search over hyperparameter combinations
model = RandomForestRegressor(
    oob_score=True,      # Enable out-of-bag evaluation
    random_state=3,      # Set seed for reproducibility
    n_jobs=-1,           # Use all available CPU cores
    max_features=0.5     # Initial setting for the number of features to consider at each split (will be overwritten in the loop)
)
# Initialize the base RandomForestRegressor model; parameters will be overwritten in each loop iteration
params = {
    'n_estimators': [500],        # Number of trees to use in the forest
    'min_samples_leaf': [2],      # Minimum number of samples required at a leaf node
    'max_features': [0.5],        # Proportion of features to consider at each split
    'max_depth': [10],            # Maximum depth of each tree
    'min_samples_split': [2]      # Minimum number of samples required to split an internal node
}
# You can extend each list to perform a broader grid search
best_score = 0  # Variable to keep track of the best OOB score observed
for g in ParameterGrid(params):        # Iterate over all combinations of parameters
    model.set_params(**g)              # Apply current parameter combination to the model
    model.fit(X_train_final, y_train)  # Train the model on the final selected feature set
    
    if model.oob_score_ > best_score:  # If current model’s OOB score is better than previous best
        best_score = model.oob_score_  # Update best score
        best_grid = g                  # Save the best-performing parameter configuration
        print('best oob:', best_score, best_grid)  # Print current best score and associated parameters



# 最好成绩的超参数
model_final = RandomForestRegressor(
    n_jobs=-1,             # Use all available CPU cores for parallel training
    n_estimators=550,      # Use 550 decision trees in the forest (found from tuning)
    max_depth=17,          # Limit tree depth to 17 to reduce overfitting
    min_samples_leaf=4,    # Require at least 4 samples at each leaf node (controls complexity)
    min_samples_split=2,   # Minimum number of samples required to split an internal node
    max_features=0.45      # Use 45% of features when choosing best split at each node (helps generalization)
)
# This model is configured with the best hyperparameters found from previous grid search or empirical tuning
model_final.fit(X_train_final, y_train)  
# Train the final model using the best parameters and the reduced feature set (X_train_final)
# y_train is the log-transformed sale price target



X_test_num = num_pipeline.transform(df_test)  
# Apply the numerical preprocessing pipeline to the test data
# This includes filling missing values and generating any additional indicator columns
X_test_cate = cat_pipeline.transform(df_test)  
# Apply the categorical encoding pipeline to the test data
# This performs label encoding or one-hot encoding as learned from the training set
X_test_date = date_pipeline.transform(df_test)  
# Apply the date processing pipeline to extract date-related features (e.g., year, month, elapsed time)
# Any missing values are also filled (e.g., with -1)
df_t = pd.concat([X_test_num, X_test_cate, X_test_date], axis=1)  
# Concatenate all processed numerical, categorical, and date features column-wise
# This produces the full feature matrix for the test set
df_t = df_t[to_keep_final]  
# Keep only the 18 best-performing features (as selected earlier) in the same order as used during training
# This ensures the model receives the exact same input structure as it was trained on



pred = model_final.predict(df_t)  
# Use the trained final model to make predictions on the test dataset `df_t`
# `df_t` should be the final test feature matrix with the same preprocessing and column order as X_train_final
# The predictions are in log scale because the model was trained on log-transformed prices

df_pred = pd.DataFrame({
    'Id': df_test['Id'],              # Retrieve the corresponding 'Id' column from the original test dataset
    'Sold Price': np.exp(pred)        # Apply inverse of log (i.e., exponential) to get original price scale
})
# Create a DataFrame for submission — each row is a prediction with its matching ID
print(df_pred.head())  
# Display the first few rows of the prediction DataFrame to verify formatting and values
df_pred.to_csv('submission.csv', index=False)  
# Export the final predictions to a CSV file named 'submission.csv'
# `index=False` prevents pandas from writing row numbers into the CSV
# This file is ready for submission to a competition like Kaggle


