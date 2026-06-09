import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy.stats import normaltest
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
from scipy.stats import skew
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_csv = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_submission_csv = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')


train_csv.columns


# Checks for duplicate rows
duplicates=train_csv.duplicated()
train_csv[duplicates]
#no duplicates so go ahead


pd.set_option('display.max_columns', None)
train_csv.head(3)


train_csv.info()


#Imputation
na_counts=train_csv.isna().sum()
na_counts


num_cols=train_csv.select_dtypes(include=[np.number])

# Calculate skewness for each column
skewness_results = num_cols.apply(lambda x: x.skew()).to_frame(name="Skewness")

# Classify Skewness Type
skewness_results["Skewness Type"] = skewness_results["Skewness"].apply(
    lambda x: "Symmetric (Normal)" if -0.5 <= x <= 0.5 else 
              "Moderate Skew" if -1 <= x < -0.5 or 0.5 < x <= 1 else 
              "Highly Skewed"
)

# Display results
print(skewness_results)


# Categorize columns based on skewness for outliers
normal=list(skewness_results[skewness_results['Skewness Type']=="Symmetric (Normal)"].index)
skewed=list(skewness_results[skewness_results['Skewness Type']!="Symmetric (Normal)"].index)


def impute_based_on_skewness(data):
    for col in num_cols.columns:
        if data[col].isnull().sum() > 0:  # Apply imputation only if there are missing values
            col_skewness = skew(data[col].dropna())  # Compute skewness ignoring NaNs
            
            # Normal Distribution (Mean Imputation)
            if -0.5 <= col_skewness <= 0.5:
                imputer = SimpleImputer(strategy="mean")
                data.loc[:, col] = imputer.fit_transform(data[[col]])                

            # Skewed Distribution (Median Imputation)
            else:
                imputer = SimpleImputer(strategy="median")
                data.loc[:, col] = imputer.fit_transform(data[[col]])  # Use 2D array
    return data
    


train_csv=pd.DataFrame(impute_based_on_skewness(train_csv.copy()))


# Impute missing categorical values with mode (most frequent value)
categorical_cols = train_csv.select_dtypes(include=["object"]).columns

# Apply imputation
imputer = SimpleImputer(strategy="most_frequent")
train_csv[categorical_cols] = imputer.fit_transform(train_csv[categorical_cols])

print("Categorical values imputed using most frequent strategy!")


#Imputation check
na_counts=train_csv.isna().sum()
na_counts


pd.options.display.float_format = '{:.2f}'.format
train_csv.describe()



# Function to detect outliers using Z-score
def detect_outliers_zscore(data, threshold=3):
    outlier_summary = {}
    
    for col in data.columns:
        z_scores = np.abs(zscore(data[col].dropna()))  # Compute absolute Z-scores
        outlier_count = (z_scores > threshold).sum()  # Count values above threshold

        outlier_summary[col] = {
            "Total Outliers": outlier_count,
            "Percentage of Outliers": round((outlier_count / len(data)) * 100, 2)
        }

    return pd.DataFrame(outlier_summary).T

# Run the function
outlier_results_z = detect_outliers_zscore(train_csv[normal])

# Display results
print(outlier_results_z)




# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    outlier_summary = {}
    
    for col in data.columns:
        Q1 = data[col].quantile(0.25)  # 25th percentile
        Q3 = data[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outlier_count = outliers.count()
        
        outlier_summary[col] = {
            "Total Outliers": outlier_count,
            "Percentage of Outliers": round((outlier_count / len(data)) * 100, 2)
        }

    return pd.DataFrame(outlier_summary).T

# Run the function
outlier_results = detect_outliers_iqr(train_csv[skewed])

# Display the result
print(outlier_results)





# Apply log transformation only on skewed columns
skewed_cols = ["Annual Income", "Previous Claims","Premium Amount"]  # Modify based on data distribution

transformed_data = train_csv.copy()

for col in skewed_cols:
    if col in transformed_data.columns:  # Ensure the column exists
        transformed_data[col] = np.log1p(transformed_data[col])  # log1p avoids log(0) issues


train_csv=transformed_data.copy()


train_csv_=train_csv.copy()


train_csv['Policy Start Date']=pd.to_datetime(train_csv['Policy Start Date'])
train_csv['Year']=train_csv['Policy Start Date'].dt.year
train_csv['Day']=train_csv['Policy Start Date'].dt.day
train_csv['Month']=train_csv['Policy Start Date'].dt.month
train_csv.drop(columns=['id','Policy Start Date'],inplace=True)


X = train_csv.drop(columns=["Premium Amount"])  # Replace "target" with actual target column name
y = train_csv["Premium Amount"]

# Identify categorical columns (CatBoost expects string names or indices)
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Initialize CatBoost model
model = CatBoostRegressor(
    iterations=500,        # Number of boosting rounds
    learning_rate=0.05,    # Step size for learning
    depth=6,               # Tree depth
    cat_features=categorical_cols,  # Let CatBoost handle categorical data
    loss_function="RMSE",  # Root Mean Squared Error (good for regression)
    eval_metric="MAE",     # Mean Absolute Error for evaluation
    verbose=100
)
# Train model (CatBoost handles encoding internally)
model.fit(X, y)


from sklearn.metrics import accuracy_score


y_pred = model.predict(X)

# Evaluation Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


mae = mean_absolute_error(y, y_pred)
regression_accuracy = 1 - (mae / np.mean(y))


regression_accuracy


submit = pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")
submit["Premium Amount"] = np.exp( y )-1
submit.to_csv("submission.csv",index=False)

