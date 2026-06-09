# Core libraries for data manipulation, numerical operations, and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Machine learning models
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Model selection and validation tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.exceptions import ConvergenceWarning


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Suppress specific, non-critical warnings to keep the output clean
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")

# Configure pandas to display all columns and use full screen width for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# Format floating-point numbers to three decimal places
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the training dataset from a CSV file
train = pd.read_csv("/kaggle/input/wine-qualit-dataset/train.csv")
# Load the test dataset from a CSV file
test = pd.read_csv("/kaggle/input/wine-qualit-dataset/test.csv")


# Concatenate the train and test dataframes for unified preprocessing
# ignore_index=True resets the index of the combined dataframe
df = pd.concat([train, test], ignore_index=True)

# Display the shape (rows, columns) of the combined dataframe
df.shape


# Calculate and display the total number of null (missing) values for each column.
# This helps to identify which columns need data imputation or special handling.
df.isnull().sum()
# These 10,000 missing values are expected and correspond to the rows from the test set.


# --- IDENTIFY COLUMN TYPES PROGRAMMATICALLY ---

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Analyzes a dataframe and returns lists of column names based on their data type and cardinality.
    This helps to automate the process of separating features into categorical and numerical types.

    Args:
        dataframe (pd.DataFrame): The dataframe to be analyzed.
        cat_th (int): Threshold for numerical columns that should be treated as categorical.
                      A column with fewer unique values than cat_th is considered 'num_but_cat'.
        car_th (int): Threshold for categorical columns with high cardinality (too many unique values).
                      A column with more unique values than car_th is considered 'cat_but_car'.

    Returns:
        tuple: A tuple containing lists of column names: (cat_cols, cat_but_car, num_cols).
    """

    # Identify categorical columns (dtype 'O' for object/string)
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Identify numerical columns that have a low number of unique values (behaving like categories)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # Identify categorical columns that have a very high number of unique values (high cardinality)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Finalize the list of categorical columns by adding num_but_cat and removing cat_but_car
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Finalize the list of numerical columns by excluding num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Print a summary of the findings
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

# Apply the function to our dataframe to automatically get the lists of different column types
cat_cols, cat_but_car, num_cols = grab_col_names(df)

# EXPECTED OUTPUT & INTERPRETATION:
# Observations: 25000
# Variables: 13
# cat_cols: 1
# num_cols: 12
# cat_but_car: 0
# num_but_cat: 1
# The function correctly identifies that 'quality' is a numerical column that behaves like a category (num_but_cat).
# The remaining 12 columns are classified as numerical.


# --- ANALYZE FEATURE CORRELATION ---

def high_correlated_cols(dataframe, plot=False, corr_th=0.85):
    """
    Identifies and optionally visualizes highly correlated features in a dataframe.
    This helps in understanding multicollinearity and deciding if any redundant features should be removed.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the numerical features to analyze.
        plot (bool): If True, a heatmap of the correlations will be displayed.
        corr_th (float): The correlation absolute value threshold. Pairs of features with a
                         correlation higher than this will be identified.

    Returns:
        list: A list of column names that are highly correlated with at least one other column.
    """
    # Calculate the pairwise correlation matrix for the dataframe
    corr = dataframe.corr()

    # Get the absolute values of the correlations to handle both positive and negative high correlations
    cor_matrix = corr.abs()

    # Create a boolean mask for the upper triangle of the matrix to avoid duplicate pairs (corr(A,B) == corr(B,A))
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

    # Identify columns that have a correlation value greater than the threshold in the upper triangle
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # If plotting is requested by the user, display a heatmap
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (13, 9)})

        # Generate the heatmap with annotations for clarity
        # annot=True: displays the correlation values on the cells.
        # fmt='.2f': formats the numbers to two decimal places.
        sns.heatmap(corr, cmap="RdBu", annot=True, fmt='.2f')
        plt.show()

    return drop_list

# Call the function on the numerical columns of our dataframe to see the correlation heatmap
# and get a list of features with a correlation higher than 0.85 (if any).
high_correlated_cols(df[num_cols], plot=True)

# EXPECTED OUTPUT & INTERPRETATION:
# The function will likely return an empty list, as no features in this dataset
# have a correlation coefficient greater than 0.85 with each other.
# The heatmap provides a valuable visualization of all feature relationships.


# --- DEFINE OUTLIER HANDLING FUNCTIONS ---
# This section defines a set of helper functions to identify and handle outliers in the dataset.

def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    """
    Calculates the lower and upper outlier thresholds for a given variable using the Interquartile Range (IQR) method.
    This implementation uses user-defined quantiles (e.g., 1% and 99%) for a more flexible outlier definition.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        variable (str): The name of the column (variable) to calculate thresholds for.
        low_quantile (float): The lower quantile to use for the IQR calculation.
        up_quantile (float): The upper quantile to use for the IQR calculation.

    Returns:
        tuple: A tuple containing the calculated lower and upper limits for outliers.
    """
    # Calculate the specified lower and upper quantile values
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)

    # Calculate the interquantile range based on the specified quantiles
    interquantile_range = quantile_three - quantile_one

    # Calculate the upper and lower outlier limits using the 1.5 * IQR rule
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Checks if a given column contains any outliers by using the thresholds from the outlier_thresholds() function.

    Args:
        dataframe (pd.DataFrame): The dataframe to check.
        col_name (str): The name of the column to check for outliers.

    Returns:
        bool: Returns True if outliers are present, otherwise returns False.
    """
    # Get the outlier thresholds for the specified column
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    # Check if any value in the column is outside the calculated limits
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    """
    Replaces outliers in a specified column with the calculated outlier thresholds.
    This process is also known as "capping" or "winsorizing".
    The operation is performed in-place on the provided dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        variable (str): The name of the column in which to cap the outliers.
    """
    # Get the outlier thresholds for the specified variable
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    # Replace values below the lower limit with the lower limit itself
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

    # Replace values above the upper limit with the upper limit itself
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    


# Use a dictionary comprehension to check each numerical column for outliers.
# It iterates through the list of numerical columns (num_cols), excluding the target variable 'quality',
# and stores the boolean result from the check_outlier() function for each.
outlier_summary = {col: check_outlier(df, col) for col in num_cols if col != "quality"}

# Use a list comprehension to filter the dictionary created above.
# It creates and prints a new list containing only the names of the columns
# for which the check_outlier() function returned True.
print([col for col, has_outlier in outlier_summary.items() if has_outlier])


# EXPECTED OUTPUT & INTERPRETATION:
# ['residual sugar', 'chlorides', 'sulphates']
#
# This output identifies the specific columns that contain outlier values
# based on our previously defined quantile method. These are the columns
# that we will target for outlier treatment in the next step.


outlier_cols = ['residual sugar', 'chlorides', 'sulphates']

for col in outlier_cols:
    replace_with_thresholds(df, col)# --- 10. HANDLE OUTLIERS BY CAPPING ---

# Manually define a list containing the names of the columns that were identified as having outliers in the previous step.
outlier_cols = ['residual sugar', 'chlorides', 'sulphates']

# Iterate through each column in the outlier_cols list.
for col in outlier_cols:
    # Apply the capping function to the current column.
    # This function replaces outlier values with the calculated lower and upper thresholds in-place.
    replace_with_thresholds(df, col)
    
    # Print a confirmation message to indicate that the outlier treatment has been applied to the column.
    print(f"'{col}' applied replace_with_thresholds")

# EXPECTED OUTPUT:
# 'residual sugar' applied replace_with_thresholds
# 'chlorides' applied replace_with_thresholds
# 'sulphates' applied replace_with_thresholds


# --- DEFINE FUNCTION FOR DETAILED NUMERICAL SUMMARY ---

def num_summary(dataframe, numerical_col, plot=False):
    """
    Displays a detailed statistical summary and an optional histogram for a specified numerical column.
    This is used for in-depth Exploratory Data Analysis (EDA).

    Args:
        dataframe (pd.DataFrame): The dataframe containing the column.
        numerical_col (str): The name of the numerical column to summarize.
        plot (bool): If True, a histogram of the column's distribution will be displayed.
    """
    # Define a specific list of quantiles for a more detailed statistical summary
    quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    
    # Print the descriptive statistics for the column, including the custom quantiles
    print(dataframe[numerical_col].describe(quantiles).T)

    # If the plot argument is set to True, generate and display a histogram
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    # Print a separator for better readability when summarizing multiple columns in a loop
    print("#####################################\n\n\n")


# Get an updated list of all numerical columns. This is done again to include any
# new features that might have been created during feature engineering.
num_cols = [col for col in df.columns if df[col].dtypes != "O"]

# Iterate through each of the numerical columns.
for col in num_cols:
    # Call the num_summary function for each column with plot=True.
    # This will display both the detailed statistical summary and a histogram,
    # allowing for a final visual inspection of the feature distributions before modeling.
    num_summary(df, col, True)
    


# --- FEATURE ENGINEERING ---
# Create new features from existing ones to capture more complex relationships and potentially improve model performance.

# Print the initial number of columns for comparison.
print("Number of columns before feature engineering:", df.shape[1])


# 1. Acidity-related Features
# Combine fixed and volatile acidity to create a 'total_acidity' feature.
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']

# Create a ratio of total acidity to pH, which might reflect the overall acid strength.
# A small epsilon (1e-6) is added to the denominator to prevent division-by-zero errors.
df['acidity_to_ph_ratio'] = df['total_acidity'] / (df['pH'] + 1e-6)


# 2. Sulfur-related Features
# The ratio of free SO2 to total SO2 can provide insights into the wine's chemical stability and preservation.
df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)


# 3. Alcohol and Other Interactions
# The ratio of alcohol to density might represent the "body" of the wine.
df['alcohol_to_density_ratio'] = df['alcohol'] / (df['density'] + 1e-6)

# Create an interaction term between alcohol and sulphates, two features known to be positively correlated with quality.
df['alcohol_sulphates_interaction'] = df['alcohol'] * df['sulphates']


# 4. Ratio of Negatively Impacting Features
# Create a ratio of volatile acidity to chlorides, both of which can negatively impact wine quality.
df['volatile_acidity_to_chlorides'] = df['volatile acidity'] / (df['chlorides'] + 1e-6)


# --- VERIFY THE NEW FEATURES ---

# Print the final number of columns to confirm that new features have been added.
print("Number of columns after feature engineering:", df.shape[1])

# Display the first few rows of the newly created columns to inspect the results.
print("\nNewly created columns:")
print(df[['total_acidity', 'acidity_to_ph_ratio', 'sulfur_ratio', 'alcohol_to_density_ratio', 'alcohol_sulphates_interaction', 'volatile_acidity_to_chlorides']].head())



cat_cols, cat_but_car, num_cols = grab_col_names(df)


# --- ANALYZE FEATURE SKEWNESS ---

# Create a list of all numerical columns that will be used as features.
# This excludes the 'id' column (an identifier) and 'quality' (the target variable).
numeric_features = [col for col in num_cols if col not in ['id', 'quality']]

# Calculate the skewness for each of the selected numerical features.
# Skewness measures the asymmetry of the feature's distribution.
skewness = df[numeric_features].skew()

# Print the header for the output.
print("Skewness of Numerical Features:")
# Print the calculated skewness values, sorted in descending order to easily see the most skewed features.
print(skewness.sort_values(ascending=False))



# This output helps identify which features have a skewed distribution.
# Features with a high absolute skewness value (e.g., > 1.0) are strong candidates
# for a logarithmic transformation to make their distribution more symmetrical,
# which can improve the performance of some machine learning models.


# --- APPLY LOGARITHMIC TRANSFORMATION TO SKEWED FEATURES ---

# First, ensure the skewness series is calculated based on the current state of the features.
skewness = df[numeric_features].skew()

# Create a list of column names where the skewness value is greater than 1.0.
# These are the columns we will apply the transformation to.
skewed_cols = skewness[skewness > 1].index.tolist()

# Iterate through the list of highly skewed columns.
for col in skewed_cols:
    # Apply the log1p transformation to each column.
    # log1p(x) calculates log(1+x), which is a useful and stable transformation
    # for handling features that may have zero values. This helps to reduce positive skewness.
    df[col] = np.log1p(df[col])

# Print a confirmation message followed by the list of transformed columns for verification.
print("Logarithmic transformation applied to the following columns:")
print(skewed_cols)



# --- PREPARE DATA FOR MODELING: SPLITTING AND PIPELINE CREATION ---

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# 1. Split the master dataframe back into training and testing sets
# The training set consists of all rows where the 'quality' target variable is not null.
train_df = df[df['quality'].notna()]
# The test set consists of all rows where 'quality' is null.
test_df = df[df['quality'].isna()]


# 2. Prepare the final test set for prediction
# Store the 'id' column from the test set, which is required for the submission file.
test_ids = test_df['id']
# Drop the 'id' and the empty 'quality' columns to create the final test feature set.
test_df = test_df.drop(['id', 'quality'], axis=1)


# 3. Define the feature matrix (X) and target vector (y) from the training set
# The target vector 'y' is the 'quality' column.
y = train_df['quality']
# The feature matrix 'X' contains all columns from the training set except for 'id' and 'quality'.
X = train_df.drop(['id', 'quality'], axis=1)

# A crucial step: Ensure the column order in the training set (X) matches the test set.
# This prevents potential errors during the model's prediction phase.
X = X[test_df.columns]


# 4. Create a basic preprocessing pipeline
# This pipeline will chain all preprocessing steps. For now, it only contains the RobustScaler.
# The machine learning model will be added to this pipeline in the next steps.
pipeline = Pipeline([
    ('scaler', RobustScaler())
])

# Print the final shapes of the datasets to verify the split was successful.
print("\nData splitting and pipeline creation complete.")
print(f"Training set shape (X): {X.shape}")
print(f"Test set shape (test_df): {test_df.shape}")



# --- EVALUATE BASELINE MODELS WITH QUADRATIC WEIGHTED KAPPA ---

from sklearn.metrics import make_scorer, cohen_kappa_score

# Define a list of machine learning models to be evaluated.
# Each item is a tuple containing a short name and an instance of the model class.
# Linear models were excluded based on initial poor performance.
models = [('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor(random_state=42)),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor(random_state=42)),
          ("XGBoost", XGBRegressor(objective='reg:squarederror', random_state=42)),
          ("LightGBM", LGBMRegressor(verbose=-1, random_state=42)),
          ("CatBoost", CatBoostRegressor(verbose=False, random_state=42))]

# 1. Define a custom scoring function for Quadratic Weighted Kappa (QWK)
def qwk_scorer(y_true, y_pred):
    """
    A custom scorer that first converts continuous regression predictions into integer classes
    and then calculates the QWK score.
    """
    # Round the continuous predictions to the nearest integer.
    y_pred_rounded = y_pred.round().astype(int)
    # Clip the predictions to ensure they are within the valid range of the target variable (3 to 8).
    y_pred_clipped = np.clip(y_pred_rounded, 3, 8)
    # Calculate and return the QWK score.
    return cohen_kappa_score(y_true, y_pred_clipped, weights='quadratic')

# Create a scorer object from our custom function that can be used by scikit-learn's evaluation tools.
# 'greater_is_better=True' indicates that higher scores are better.
kappa_scorer = make_scorer(qwk_scorer, greater_is_better=True)


# 2. Loop through the models to evaluate each one
for name, regressor in models:
    # For each model, create a pipeline that first scales the data and then fits the model.
    # This ensures that data scaling is properly handled within each fold of the cross-validation.
    model_pipeline = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('model', regressor)
    ])

    # Perform 5-fold cross-validation using the pipeline and our custom kappa scorer.
    # 'n_jobs=-1' uses all available CPU cores to speed up the process.
    scores = cross_val_score(model_pipeline, X, y, cv=5, scoring=kappa_scorer, n_jobs=-1)

    # Print the mean and standard deviation of the QWK scores from the cross-validation.
    print(f"QWK Score: {round(scores.mean(), 4)} (+/- {round(scores.std(), 4)})  [{name}]")


# A list of models and their corresponding mean QWK scores from 5-fold cross-validation.
# This provides a robust baseline performance for each model, helping us select the best candidates
# for further tuning.


# The following tuning steps for LightGBM have been commented out
# to streamline the final notebook and reduce execution time.
# Based on the initial baseline evaluation, Gradient Boosting (GBM) was selected
# as the champion model to focus on for final tuning and submission.
# The results from these searches can be found in the project's earlier exploratory versions.
# Best QWK Score (LGBM): 0.331602883825449
"""
# --- HYPERPARAMETER TUNING FOR LIGHTGBM ---

# Define the pipeline, including the scaler and the LightGBM model.
# random_state ensures reproducibility, verbose=-1 suppresses unnecessary output from LightGBM.
lgbm_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('model', LGBMRegressor(random_state=42, verbose=-1))
])

# Configure the pipeline to output pandas DataFrames after transformations.
# This preserves feature names and prevents a common UserWarning.
lgbm_pipeline.set_output(transform="pandas")


# Define the parameter grid to search.
# GridSearchCV will test all possible combinations of these hyperparameter values.
# The 'model__' prefix is used to specify that the parameter belongs to the 'model' step of the pipeline.
param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__num_leaves': [20, 31, 40]
}

# Initialize the GridSearchCV object.
# estimator: The model or pipeline to tune.
# param_grid: The dictionary of parameters to test.
# scoring: The evaluation metric to optimize (our custom kappa_scorer).
# cv: The number of cross-validation folds.
# n_jobs: Number of CPU cores to use (-1 means use all available cores).
# verbose: Controls the amount of logging output during the search (2 is detailed).
grid_search = GridSearchCV(estimator=lgbm_pipeline,
                           param_grid=param_grid,
                           scoring=kappa_scorer,
                           cv=5,
                           n_jobs=-1,
                           verbose=2)

# Start the grid search process on the training data.
# This can be computationally intensive and may take some time.
grid_search.fit(X, y)

# After the search is complete, print the best results.
print("\n--- Search Complete ---")
# Print the best QWK score found during the search.
print(f"Best QWK Score: {grid_search.best_score_}")
# Print the combination of hyperparameters that resulted in the best score.
print(f"Best Parameters: {grid_search.best_params_}")
"""


# The following tuning steps for SVR have been commented out
# to streamline the final notebook and reduce execution time.
# Based on the initial baseline evaluation, Gradient Boosting (GBM) was selected
# as the champion model to focus on for final tuning and submission.
# The results from these searches can be found in the project's earlier exploratory versions.
# Best QWK Score (SVR): 0.3323257909898154
"""
# --- HYPERPARAMETER TUNING FOR SUPPORT VECTOR REGRESSOR (SVR) ---

# Define the pipeline for the SVR model, including the scaler.
svr_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('model', SVR())
])

# Define the parameter grid for SVR.
# C: Regularization parameter. It trades off correct classification of training examples against maximization of the decision function's margin.
# gamma: Defines how far the influence of a single training example reaches.
# kernel: Specifies the kernel type to be used in the algorithm. 'rbf' (Radial Basis Function) is a powerful and flexible default.
param_grid_svr = {
    'model__C': [1, 10, 50],
    'model__kernel': ['rbf'],
    'model__gamma': ['scale', 'auto']
}

# Initialize the GridSearchCV object for SVR.
# We reuse the same kappa_scorer from the previous steps.
grid_search_svr = GridSearchCV(estimator=svr_pipeline,
                               param_grid=param_grid_svr,
                               scoring=kappa_scorer,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)

# Start the hyperparameter search for the SVR model.
print("Starting hyperparameter search for SVR...")
grid_search_svr.fit(X, y)

# After the search is complete, print the best results found.
print("\n--- SVR Search Complete ---")
print(f"Best QWK Score (SVR): {grid_search_svr.best_score_}")
print(f"Best Parameters (SVR): {grid_search_svr.best_params_}")
"""


# The optimal parameters discovered from that search are now hard-coded and used
# directly in the next step to train the final model. This saves significant
# computation time and makes the final version of the notebook run much faster.
# Best QWK Score (GBM): 0.3328044347281005
"""
# --- HYPERPARAMETER TUNING FOR GRADIENT BOOSTING REGRESSOR (GBM) ---

# Define the pipeline for the GBM model, including the scaler and the model itself.
# random_state is set for consistent, reproducible results.
gbm_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Define the parameter grid for GBM.
# n_estimators: The number of boosting stages (trees) to perform.
# learning_rate: Shrinks the contribution of each tree.
# max_depth: The maximum depth of the individual regression estimators.
param_grid_gbm = {
    'model__n_estimators': [50, 100, 200, 300, 400, 500],
    'model__learning_rate': [0.05, 0.1, 0.03, 0.07],
    'model__max_depth': [3, 4, 5, 6]
}

# Initialize the GridSearchCV object for GBM.
# We reuse the same kappa_scorer from the previous steps.
grid_search_gbm = GridSearchCV(estimator=gbm_pipeline,
                               param_grid=param_grid_gbm,
                               scoring=kappa_scorer,
                               cv=5,
                               n_jobs=-1,
                               verbose=0)

# Start the hyperparameter search for the GBM model.
print("Starting hyperparameter search for GBM...")
grid_search_gbm.fit(X, y)

# After the search is complete, print the best results found.
print("\n--- GBM Search Complete ---")
print(f"Best QWK Score (GBM): {grid_search_gbm.best_score_}")
print(f"Best Parameters (GBM): {grid_search_gbm.best_params_}")
"""


# --- TRAIN THE FINAL CHAMPION MODEL ---
# Now that we've selected the best model (GBM) and its optimal parameters,
# we will train it on the entire training dataset.

# 1. Instantiate the final model using the best hyperparameters found during tuning.
final_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)

# 2. Create the final pipeline that chains the scaler and the chosen model.
final_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('model', final_model)
])

# 3. Fit the final pipeline on the ENTIRE training dataset (X and y).
# This trains the model on all available data before making predictions on the test set.
final_pipeline.fit(X, y)
print("The final model has been trained on the full training data.")

# 4. Extract the trained model object from the pipeline.
# This is necessary to access model-specific attributes, such as feature_importances_, for further analysis.
trained_gbm_model = final_pipeline.named_steps['model']



# --- VISUALIZE FEATURE IMPORTANCE OF THE FINAL MODEL ---

def plot_importance(model, features, num=len(X), save=False):
    """
    Creates and displays a horizontal bar plot of feature importances from a trained model.

    Args:
        model: A trained model object that has the 'feature_importances_' attribute (e.g., RandomForest, GradientBoosting).
        features (pd.DataFrame): The dataframe of features (X) used to get the column names.
        num (int): The number of top features to display in the plot.
        save (bool): If True, the plot will be saved to a file named 'importances.png'.
    """
    # Create a dataframe from the feature importances and feature names.
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    
    # Set up the plot figure size and font scale.
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    
    # Create the bar plot, sorted by importance value in descending order.
    # [0:num] slices the dataframe to show only the top 'num' features.
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    # Set the title and layout for the plot.
    plt.title('Feature Importance')
    plt.tight_layout()
    # Display the plot.
    plt.show()
    
    # If the save flag is set to True, save the figure.
    if save:
        plt.savefig('importances.png')

# Call the function to plot the feature importances from our trained GBM model.
# 'trained_gbm_model' is the model object we extracted from the pipeline in the previous step.
# 'X' is passed to get the feature names for the y-axis labels.
plot_importance(trained_gbm_model, X)



# --- VISUALIZE FINAL MODEL PERFORMANCE (TRUE VS. PREDICTED) ---

def plot_true_vs_predicted(results_df, true_col_jitter, pred_col_jitter, true_col_raw, model_name):
    """
    Generates a scatter plot of predicted values versus true values, with vertical lines indicating
    the boundaries for each true quality score. This helps to visualize the model's prediction distribution
    for each class.

    Args:
        results_df (pd.DataFrame): Dataframe containing the true and predicted values.
        true_col_jitter (str): Column name for the jittered true values (for the x-axis).
        pred_col_jitter (str): Column name for the jittered predicted values (for the y-axis).
        true_col_raw (str): Column name for the original, non-jittered true values (for placing vertical lines).
        model_name (str): The name of the model, to be used in the plot title.
    """
    # Set up the plot figure size.
    plt.figure(figsize=(10, 8))

    # Create a scatter plot using the jittered data to prevent overplotting and show density.
    sns.scatterplot(x=true_col_jitter, y=pred_col_jitter, data=results_df, alpha=0.3, s=50)

    # Get the unique, sorted true quality scores.
    unique_qualities = sorted(results_df[true_col_raw].unique())
    # Loop through the unique scores and draw a vertical line for each.
    for i, quality in enumerate(unique_qualities):
        # Add a label only to the first line to keep the legend clean.
        label = 'True Quality Scores' if i == 0 else ""
        plt.axvline(x=quality, color='red', linestyle='--', linewidth=1.5, label=label)

    # Set the labels, title, and other plot properties for clarity.
    plt.xlabel("True Quality (with Jitter)")
    plt.ylabel("Predicted Quality (with Jitter)")
    plt.title(f"{model_name}: Prediction Distribution vs. True Quality Scores")
    plt.legend()
    plt.grid(True)
    plt.show()


# 1. Prepare the data for plotting
# Get predictions from the final trained pipeline on the training data.
train_predictions = final_pipeline.predict(X)

# Create a new dataframe to hold the results.
results_df = pd.DataFrame({
    'true_quality_raw': y,
    'predicted_quality': train_predictions
})

# Add a small amount of random noise ("jitter") to the data.
# This is a visualization technique to better see the density of points when dealing with discrete values.
jitter_strength = 0.1
results_df['true_quality_jitter'] = results_df['true_quality_raw'] + np.random.normal(0, jitter_strength, size=len(results_df))
results_df['predicted_quality_jitter'] = results_df['predicted_quality'] + np.random.normal(0, jitter_strength, size=len(results_df))


# 2. Call the plotting function
# Generate the plot for our final GBM model.
plot_true_vs_predicted(results_df, 'true_quality_jitter', 'predicted_quality_jitter', 'true_quality_raw', 'GBM')



# --- CREATE FINAL SUBMISSION FILE ---

# 1. Instantiate the final model with the best parameters found during hyperparameter tuning.
final_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)

# 2. Create the final pipeline, chaining the scaler and the model.
final_pipeline = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('model', final_model)
])

# 3. Fit the final pipeline on the ENTIRE training dataset.
# This ensures the model learns from all available data before making final predictions.
final_pipeline.fit(X, y)
print("The final model has been trained on the full training data.")

# 4. Make predictions on the unseen test set.
# The output will be continuous floating-point values.
predictions_float = final_pipeline.predict(test_df)

# 5. Convert the predictions to the required integer format for the competition.
# First, round the float predictions to the nearest integer.
predictions_int = np.round(predictions_float).astype(int)
# Then, clip the values to ensure they are within the valid range of quality scores (3 to 8).
final_predictions = np.clip(predictions_int, 3, 8)
print("Predictions have been generated and formatted for the test set.")

# 6. Create the submission DataFrame.
# It must contain two columns: 'Id' and the predicted 'quality'.
submission_df = pd.DataFrame({'id': test_ids.astype(int), 'quality': final_predictions})

# 7. Save the DataFrame to a CSV file.
# 'index=False' is crucial to prevent pandas from writing the DataFrame index to the file.
submission_df.to_csv('submission.csv', index=False)

# Print a confirmation message and a preview of the submission file.
print("\n'submission.csv' has been created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())




