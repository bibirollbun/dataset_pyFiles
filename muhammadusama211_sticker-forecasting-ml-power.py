import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# ===========================
# Linear Regression Models
# ===========================
from sklearn.linear_model import (
    LinearRegression,      # Basic Linear Regression
    Ridge,                 # Linear regression with L2 regularization
    Lasso,                 # Linear regression with L1 regularization
    ElasticNet,            # Combination of L1 and L2 regularization
    BayesianRidge,         # Bayesian version of Ridge Regression
    HuberRegressor,        # Robust regression for outlier handling
    TheilSenRegressor,     # Robust regression using median slopes
    RANSACRegressor        # Robust regression using random sampling
)

# ===========================
# Tree-Based Regression Models
# ===========================
from sklearn.tree import DecisionTreeRegressor  # Simple decision tree for regression
from sklearn.ensemble import (
    RandomForestRegressor,                # Bagging ensemble of decision trees
    GradientBoostingRegressor,            # Sequential boosting with decision trees
    HistGradientBoostingRegressor,        # Histogram-based boosting (faster for large datasets)
    BaggingRegressor,                     # Bagging with customizable base regressors
    StackingRegressor,                    # Ensemble of models stacked together
    VotingRegressor                       # Combines predictions from multiple regressors
)

# ===========================
# Boosting Frameworks
# ===========================
import lightgbm as lgb                        # LightGBM for gradient boosting
from xgboost import XGBRegressor             # XGBoost for optimized gradient boosting
from catboost import CatBoostRegressor       # CatBoost for categorical feature handling

# ===========================
# Kernel-Based Models
# ===========================
from sklearn.svm import SVR                        # Support Vector Regression (linear and nonlinear)
from sklearn.gaussian_process import GaussianProcessRegressor  # Probabilistic regression with Gaussian processes

# ===========================
# Neural Network-Based Models
# ===========================
from sklearn.neural_network import MLPRegressor    # Multi-Layer Perceptron for regression

# ===========================
# Specialized Regression Models
# ===========================
from sklearn.cross_decomposition import PLSRegression  # Partial Least Squares regression
from sklearn.isotonic import IsotonicRegression        # Non-parametric isotonic regression

# ===========================
# Probabilistic Regression Models
# ===========================
from sklearn.linear_model import QuantileRegressor     # Quantile regression for conditional quantiles

# ===========================
# Utilities for Model Training and Evaluation
# ===========================
from sklearn.model_selection import (
    train_test_split,       # Split data into training and test sets
    GridSearchCV,           # Grid search for hyperparameter tuning
    RandomizedSearchCV      # Random search for hyperparameter tuning
)

from sklearn.metrics import (
    mean_squared_error,          # Evaluate model with Mean Squared Error
    mean_absolute_error,         # Evaluate model with Mean Absolute Error
    r2_score,                    # Evaluate model with R-squared score
    mean_absolute_percentage_error  # Evaluate model with Mean Absolute Percentage Error
)


df_tr = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_ts = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


df_tr.head()


import pandas as pd
from IPython.display import display, HTML

# Function to style the heading
def styled_heading(text, background_color="black", text_color="skyblue"):
    return f"""
    <p style="
        background-color: {background_color}; 
        font-family: 'Allura', cursive;
        color: {text_color}; 
        font-size: 140%; 
        text-align: center; 
        border: 2px solid black; 
        border-radius: 5px; 
        padding: 10px; 
        box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.5); 
        font-weight: bold; 
        letter-spacing: 2px;">
        {text}
    </p>
    """

# Helper function for error messages
def print_error(message):
    display(HTML(styled_heading("Error", background_color="red", text_color="white")))
    print(f"An error occurred: {message}")

# Helper function to generate a colored horizontal line
def colored_line(color="skyblue"):
    return f"<hr style='border: none; height: 3px; background-color: {color}; margin: 15px 0;' />"

# Main function to analyze datasets
def print_dataset_analysis(train_dataset, test_dataset, n_top=5, heading_color="black", line_color="skyblue"):
    try:
        # Display top rows
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading(f"ğŸ”� Top {n_top} Rows of Training Dataset", background_color=heading_color)))
        display(HTML(train_dataset.head(n_top).to_html(index=False)))
        
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading(f"ğŸ”� Top {n_top} Rows of Test Dataset", background_color=heading_color)))
        display(HTML(test_dataset.head(n_top).to_html(index=False)))
        
        # Dataset summary
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading("ğŸ“Š Summary of Dataset", background_color=heading_color)))
        display(HTML(train_dataset.describe().to_html()))
        
        # Null values and percentages
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading("â�Œ Null Values in Datasets", background_color=heading_color)))
        
        train_null_count = train_dataset.isnull().sum()
        train_null_percentage = (train_null_count / len(train_dataset)) * 100
        train_null_summary = pd.DataFrame({
            "Null Count": train_null_count[train_null_count > 0],
            "Null Percentage (%)": train_null_percentage[train_null_percentage > 0]
        })
        
        test_null_count = test_dataset.isnull().sum()
        test_null_percentage = (test_null_count / len(test_dataset)) * 100
        test_null_summary = pd.DataFrame({
            "Null Count": test_null_count[test_null_count > 0],
            "Null Percentage (%)": test_null_percentage[test_null_percentage > 0]
        })
        
        # Display training dataset nulls
        display(HTML("<h3>Training Dataset:</h3>"))
        if train_null_count.sum() == 0:
            display(HTML("<p>No null values in the training dataset.</p>"))
        else:
            display(HTML(train_null_summary.to_html()))
        
        # Display test dataset nulls
        display(HTML("<h3>Test Dataset:</h3>"))
        if test_null_count.sum() == 0:
            display(HTML("<p>No null values in the test dataset.</p>"))
        else:
            display(HTML(test_null_summary.to_html()))
        
        # Duplicate rows
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading("â™»ï¸� Duplicate Values in Datasets", background_color=heading_color)))
        
        train_duplicates = train_dataset.duplicated().sum()
        test_duplicates = test_dataset.duplicated().sum()
        
        display(HTML(f"<p><strong>Training Dataset:</strong> {train_duplicates} duplicate rows</p>"))
        display(HTML(f"<p><strong>Test Dataset:</strong> {test_duplicates} duplicate rows</p>"))
        
        # Shape of datasets
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading("ğŸ“� Dataset Shape", background_color=heading_color)))
        display(HTML(f"<p><strong>Training Dataset:</strong> {train_dataset.shape[0]} rows, {train_dataset.shape[1]} columns</p>"))
        display(HTML(f"<p><strong>Test Dataset:</strong> {test_dataset.shape[0]} rows, {test_dataset.shape[1]} columns</p>"))
    
    except Exception as e:
        print_error(str(e))

# Function to display unique values in columns
def print_unique_values(dataset, line_color="skyblue"):
    try:
        display(HTML(colored_line(line_color)))
        display(HTML(styled_heading("ğŸ”¢ Unique Values in Dataset", background_color="black")))
        
        unique_values_table = "<table style='border-collapse: collapse; width: 100%; text-align: left;'>"
        unique_values_table += "<tr style='background-color: black; color: skyblue ;'><th>Column Name</th><th>Data Type</th><th>Unique Values</th></tr>"
        
        for column in dataset.columns:
            unique_values = dataset[column].unique()[:7]
            unique_values_str = ', '.join(map(str, unique_values))
            data_type = dataset[column].dtype
            unique_values_table += f"<tr><td>{column}</td><td>{data_type}</td><td>{unique_values_str}</td></tr>"
        
        unique_values_table += "</table>"
        display(HTML(unique_values_table))
    
    except Exception as e:
        print_error(str(e))



print_dataset_analysis(df_tr, df_ts)
print_unique_values(df_tr)


# we have only 3 percent of null values instead of importing thsoe i will prefer to remove them 
df_tr = df_tr.dropna(subset=['num_sold'])


df_tr['date'].max()


df_tr['num_sold'] = np.log1p(df_tr['num_sold'])


import pandas as pd
import numpy as np

cat_c = ['country', 'store', 'product','month_name','day_of_week']
 
def date(df, cat_c, min_year=2010, max_year=2016):
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year, day, month, and other time-based features
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    
    # Apply sinusoidal encoding to cyclical features (month, day)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # Drop original 'date' column
    df.drop('date', axis=1, inplace=True)

    # Create a group feature (time-based grouping, adjusting for year range)
    df['group'] = (df['year'] - min_year) * 48 + df['month'] * 4 + df['day'] // 7
    
    # Apply cyclical encoding for the 'year' feature
    # Adjust the cyclical encoding scaling factor based on the year range
    year_range = max_year - min_year
    df['cos_year'] = np.cos(df['year'] * (2 * np.pi) / year_range)
    df['sin_year'] = np.sin(df['year'] * (2 * np.pi) / year_range)
    
    # Convert categorical features to 'category' dtype
    for c in cat_c:
        df[c] = df[c].astype('category')

    return df

# Example usage:
# Assuming `train` and `test` are your dataframes and `cat_c` is a list of categorical columns

df_tr = date(df_tr, cat_c, min_year=2010, max_year=2016)
df_ts = date(df_ts, cat_c, min_year=2010, max_year=2016)


df_tr.head()


df_tr.dtypes


df_tr['country'].value_counts()


import pandas as pd

# Define the add_gdp_data function
def add_gdp_data(df, gdp_data, country_col='country'):
    """
    Adds GDP data to a DataFrame for the past ten years.

    Parameters:
    - df: pd.DataFrame, the original DataFrame containing the countries.
    - gdp_data: dict, a dictionary where keys are country names and values
      are lists of GDP data for the past ten years.
    - country_col: str, the name of the column containing country names.

    Returns:
    - pd.DataFrame, the updated DataFrame with GDP columns added.
    """
    for year_offset in range(10):
        year = 2024 - year_offset
        df[f'GDP_{year}'] = df[country_col].map(lambda country: gdp_data.get(country, [None]*10)[year_offset])
    return df

# Sample GDP data for the past 10 years
gdp_data = {
    'Finland': [53883, 53883, 53883, 48783, 48783, 48783, 45703, 42878, 43788, 43788],
    'Italy': [35476, 35476, 35476, 34318, 34318, 34318, 34318, 35476, 36016, 36016],
    'Singapore': [82794, 82794, 82794, 59798, 65233, 64582, 59797, 57666, 56571, 56571],
    'Norway': [89154, 89154, 89154, 67176, 75295, 82773, 75504, 70911, 74318, 74318],
    'Canada': [52051, 52051, 52051, 43258, 46213, 46213, 45032, 42210, 43249, 43249],
    'Kenya': [1377, 1377, 1377, 1377, 1377, 1377, 1377, 1377, 1377, 1377]
}

# Example DataFrame (replace this with df_tr in your case)

df_tr = pd.DataFrame(df_tr)

# Adding GDP data to the DataFrame
df_tr_updated = add_gdp_data(df_tr, gdp_data, country_col='country')
df_ts_updated = add_gdp_data(df_ts, gdp_data, country_col='country')


# Print the updated DataFrame
print(df_tr_updated.head())



df_tr_updated.columns


df_ts_updated.columns


df_ts_updated.head()


import pandas as pd

def one_hot_encode(df):
    # Identify the categorical columns
    categorical_cols = df.select_dtypes(include=['category']).columns
    
    # Perform one-hot encoding on categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

# Example usage:
# Assuming df is your DataFrame
df_tr_en = one_hot_encode(df_tr_updated)
df_ts_en = one_hot_encode(df_ts_updated)


df_tr_en.info()


df_ts_en.info()


import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

# Load and prepare data
X = df_tr_en.drop('num_sold', axis=1)
y = df_tr_en['num_sold']

# MAPE function
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Stable hyperparameters for LightGBMRegressor
# params = {
#     'n_estimators': 100,
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'min_child_samples': 20,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.1
# }

params = {
          'n_estimators': 680,
          'max_depth': 3, 
          'num_leaves': 343,
          'learning_rate': 0.2444178793191865,
          'min_child_samples': 10, 
          'subsample': 0.7, 
          'colsample_bytree': 0.9,
          'reg_alpha': 0.661406343212664, 
          'reg_lambda': 5.175656526671005
}


# params = {
#           'n_estimators': 923,
#           'max_depth': 20,
#           'num_leaves': 53, 
#           'learning_rate': 0.03584519780714426,
#           'min_child_samples': 14,
#           'subsample': 1.0,
#           'colsample_bytree': 0.8, 
#           'reg_alpha': 1.5275755749466307, 
#           'reg_lambda': 3.683655682780799
# }
# Models list
models = [
    (LGBMRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        min_child_samples=params['min_child_samples'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda']
    ), "LightGBM")
]

# KFold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
results = []

for model, name in models:
    fold_scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        score = mape(y_val, y_pred)
        fold_scores.append(score)
    
    # Average score across folds
    avg_score = np.mean(fold_scores)
    results.append((name, avg_score))

# Create a DataFrame of results
results_df = pd.DataFrame(results, columns=["Model", "MAPE"])

# Filter and display models with MAPE < 10
good_models = results_df[results_df['MAPE'] < 10].sort_values(by="MAPE")
print("Good Models (MAPE < 10):")
print(good_models)

# Display all models sorted by MAPE
print("\nAll Models Sorted by MAPE:")
print(results_df.sort_values(by="MAPE"))


df_ts.columns


df_tr.columns


# df_ts_updated = df_ts_updated.drop(columns=['num_sold'])


predictions = model.predict(df_ts_en)


predictions_inverse_log = np.expm1(predictions)


df_s = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


df_s.head()


# Align indexes between df_s and predictions (if predictions is a Series)
df_s['num_sold'] = predictions_inverse_log[df_s.index]



# df_s['num_sold'] = predictions
df_s.to_csv('0_876171_lightgbm_GPD.csv',index=False)

