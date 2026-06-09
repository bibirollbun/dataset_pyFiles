import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


print("Loading the Dataset ---- ")
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
print("Dataset Loaded")


df_train.head()


df_train.info()


# Lets Distinguish firstly the Categorical columms and Numerical columns 
print("Dividing the numerical_columns and categorical_columns into different lists")

numerical_columns = []
categorical_columns = []

for column in df_train.columns:
    if df_train[column].dtype != 'O':  # Check if column is numerical
        numerical_columns.append(column)
    else:
        categorical_columns.append(column)

# Print the results
print(f"Numerical Columns: {numerical_columns}")
print(f"Categorical Columns: {categorical_columns}")



# Lets Firstly look into the number of different categorical values in the perticular column
print("Printing the count of distinct values in the categorical columns : ")
for column in categorical_columns:
    print(f'{column} : {len(df_train[column].unique())}')


# lets check the correlation of categorical columns to the target columns 
# we can use different correlation techniques on different columnns based on the distinct values 

import scipy.stats as stats
target_column = 'Listening_Time_minutes'

# 1. Apply ANOVA for High-Cardinality Columns (Podcast_Name, Episode_Title)

for col in ['Podcast_Name', 'Episode_Title']:
    anova_result = stats.f_oneway(*[df_train[df_train[col] == cat][target_column] for cat in df_train[col].unique()])
    print(f"ANOVA p-value for {col}: {anova_result.pvalue}")

# 2. Apply Cramer’s V for Nominal Categorical Column (Genre)
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape)-1)))

cramers_v_score = cramers_v(df_train['Genre'], pd.qcut(df_train[target_column], q=4, duplicates='drop'))
print(f"Cramer’s V for Genre: {cramers_v_score}")

# 3 Apply Spearman’s Correlation for Ordered Categorical Columns (Publication_Day, Publication_Time, Episode_Sentiment)
for col in ['Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
    spearman_corr, p_value = stats.spearmanr(df_train[col], df_train[target_column])
    print(f"Spearman Correlation for {col}: {spearman_corr}, p-value: {p_value}")


# for Numerical data only
correlation_matrix = df_train[numerical_columns].corr()


correlation_matrix


# Lets check for the Missing values 
print(df_train.isnull().sum())


# since episode_length_minutes have missing values and we have decided to use it so will only handle that 
# firsly check for any outliers 

sns.boxplot(df_train["Episode_Length_minutes"])


# There is not much outlier lets count them 
z_scores = np.abs(stats.zscore(df_train["Episode_Length_minutes"]))  

# Count outliers (Z-score > 3)
outlier_counts_z = (z_scores >= 3).sum()
print("Outliers Count Using Z-score:\n", outlier_counts_z)

# There is one outlier 
# so lets replace the missing values with the median value 


def count_outliers_iqr(df, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return outlier_counts

# Count outliers using IQR
outlier_counts_iqr = count_outliers_iqr(df_train, numerical_columns)
print("Outliers Count Using IQR:\n", outlier_counts_iqr)


# lets remove the outliers 
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Apply IQR-based outlier removal
df_no_outliers = remove_outliers_iqr(df_train, numerical_columns)

print(f"Original dataset shape: {df_train.shape}")
print(f"After IQR outlier removal: {df_no_outliers.shape}")



# so will replace the missing values with median value
df_no_outliers['Episode_Length_minutes'].fillna(df_no_outliers["Episode_Length_minutes"].median(), inplace=True)


df_no_outliers.columns


# lets encode the categorical colummns
# i will use different techniques for different columns based on the Cardinality
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder

# Make a copy to avoid modifying original data
df_encoded = df_no_outliers.copy()

# Ordinal Encoding for ordered categorical columns
ordinal_features = ["Publication_Day", "Publication_Time"]
ordinal_enc = OrdinalEncoder()
df_encoded[ordinal_features] = ordinal_enc.fit_transform(df_encoded[ordinal_features])

# One-Hot Encoding for 'Episode_Sentiment' (Low Cardinality)
df_encoded = pd.get_dummies(df_encoded, columns=['Episode_Sentiment'], drop_first=True)

# Target Encoding for 'Podcast_Name' (High Cardinality)
target_enc_1 = TargetEncoder()
target_enc_2 = TargetEncoder()
df_encoded["Podcast_Name"] = target_enc_1.fit_transform(df_encoded["Podcast_Name"], df_encoded["Listening_Time_minutes"])

# Optional: Handle Episode_Title (100 categories - can be dropped or target-encoded)
# df_encoded = df_encoded.drop(columns=['Episode_Title'])  # Drop if not impactful
df_encoded["Episode_Title"] = target_enc_2.fit_transform(df_encoded["Episode_Title"], df_encoded["Listening_Time_minutes"])  # If keeping

# Check encoded dataset
print(df_encoded.head())



df_encoded.head()


# Printing the count of distinct values in the categorical columns : 
# Podcast_Name : 48
# Episode_Title : 100
# Genre : 10
# Publication_Day : 7
# Publication_Time : 4
# Episode_Sentiment : 3


# lets scale the numerical data for the linear model or distance based 
# while we can easily train using tree based algo as it can handle row data no need for scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded_scaled = df_encoded.copy()
numerical_columns_independent = numerical_columns[1:]
print(numerical_columns_independent)
df_encoded_scaled[numerical_columns_independent] = scaler.fit_transform(df_encoded_scaled[numerical_columns_independent])

print("Scaling is Done")


df_encoded.columns
columns_used = ["Podcast_Name","Episode_Title","Publication_Day","Publication_Time","Episode_Sentiment_Neutral","Episode_Sentiment_Positive", "Episode_Length_minutes"]
# Printing the count of distinct values in the categorical columns : 
# Podcast_Name : 48
# Episode_Title : 100
# Publication_Day : 7
# Publication_Time : 4
# Episode_Sentiment : 3



df_encoded[columns_used].head()


df_encoded_scaled[columns_used].head()


from sklearn.model_selection import train_test_split

# target column 
target = "Listening_Time_minutes"
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    df_encoded[columns_used], df_no_outliers[target], test_size=0.2, random_state=42
)

# Splitting for linear/distance-based models (Scaled)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    df_encoded_scaled[columns_used], df_no_outliers[target], test_size=0.2, random_state=42
)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import XGBoost, LightGBM, CatBoost if available
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    # from catboost import CatBoostRegressor
    xgb_available, lgb_available = True, True
except ImportError:
    xgb_available, lgb_available = False, False

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN Regressor": KNeighborsRegressor(),
    # "SVR": SVR(),
    # "MLP Regressor": MLPRegressor(max_iter=10)
}

# Add tree-based models that require installation/
if xgb_available:
    models["XGBoost"] = XGBRegressor()
if lgb_available:
    models["LightGBM"] = LGBMRegressor()
# if catb_available:
#     models["CatBoost"] = CatBoostRegressor(verbose=0)

# Store results
results = []

for name, model in models.items():
    print(f"Training {name}...")
    
    # Select appropriate dataset
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "KNN Regressor", "SVR", "MLP Regressor"]:
        model.fit(X_train_scaled, y_train_scaled)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_tree, y_train_tree)
        y_pred = model.predict(X_test_tree)
    
    # Evaluate model
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "KNN Regressor", "SVR", "MLP Regressor"]:
        mae = mean_absolute_error(y_test_scaled, y_pred)
        mse = mean_squared_error(y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_scaled, y_pred)
    else: 
        mae = mean_absolute_error(y_test_tree, y_pred)
        mse = mean_squared_error(y_test_tree, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_tree, y_pred)
        
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R² Score": r2})

# Convert results to DataFrame  
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="R² Score", ascending=False))



# lets go with random forest then 
# no need to scale the data then just simply use it
trained_model = models["Random Forest"]


## lets get the test data 
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_test.head()


df_test.info()


## lets look for the missing values 
print(df_test.isnull().sum())


# Lets remove the missing values from episode_length_minutes column
df_test['Episode_Length_minutes'].fillna(df_test["Episode_Length_minutes"].median(), inplace=True)


## lets look for the missing values 
print(df_test.isnull().sum())


# i will be using and changing only the required columns 
df_test_encoded = df_test.copy()

# Apply Ordinal Encoding (Use already fitted encoder)
df_test_encoded[ordinal_features] = ordinal_enc.transform(df_test_encoded[ordinal_features])

# Apply One-Hot Encoding (Ensure same structure as df_encoded)
df_test_encoded = pd.get_dummies(df_test_encoded, columns=['Episode_Sentiment'], drop_first=True)

# Apply Target Encoding using previously fitted encoders
df_test_encoded["Podcast_Name"] = target_enc_1.transform(df_test_encoded["Podcast_Name"])
df_test_encoded["Episode_Title"] = target_enc_2.transform(df_test_encoded["Episode_Title"])

# Ensure df_test_encoded has the same columns as df_encoded (excluding target)
common_columns = df_encoded.columns.difference(['Listening_Time_minutes'])  # Exclude target column
df_test_encoded = df_test_encoded.reindex(columns=common_columns, fill_value=0)  # Fill missing columns with 0

# Check transformed test data
print(df_test_encoded.head())




# lets predict the values 
df_pred = trained_model.predict(df_test_encoded[columns_used])


# lets Save them to csv
df_result = df_test[['id']].copy()
df_result["Listening_Time_minutes"] = df_pred 

# Display the final result
print(df_result.head())

df_result.to_csv("submission.csv", index=False)





