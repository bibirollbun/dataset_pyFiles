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


# Importing neccasery libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px

# run '!pip install fancyimpute' First
from fancyimpute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load datasets
train_df = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv").set_index("id")
extra_df = pd.read_csv(r"/kaggle/input/playground-series-s5e2/training_extra.csv").set_index("id")

# Shuffle the extra data to randomize selection
extra_df = extra_df.sample(frac=1, random_state=42)  # Ensures reproducibility

# Split extra_df into two separate samples
extra_train = extra_df.iloc[:int(1e6)]  # First 1M rows for training
extra_test = extra_df.iloc[int(1e6):int(1e6) + int(100000)]  # Next 100K rows for testing

# Concatenate with train_df
train_df = pd.concat([train_df, extra_train])
test_df = extra_test  # Ensuring no overlap

# Show dataset shape
print(f"Training Dataset Shape: {train_df.shape}")
print(f"Testing Dataset Shape: {test_df.shape}")

train_df.head()


# loading the training dataset
train_df = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv").set_index("id")
extra_df = pd.read_csv(r"/kaggle/input/playground-series-s5e2/training_extra.csv").set_index("id")
train_df = pd.concat([train_df, extra_df.sample(int(1e6))])
test_df = pd.concat([train_df, extra_df.sample(int(100000))])


# Showing the shaoe and rows of the dataset
print(f"Dataset Shape: {train_df.shape}")
train_df.head()


# The structure of the dataset 
train_df.info()


# Inistiaging missing values in the dataset
train_df.isnull().sum()


 # Visualizes missing values count and percentage per column
msno.bar(train_df) 
plt.show()


# Inverstigating the details of each column in the dataset
for col in train_df.columns:
    if train_df[col].dtype == "object":
        print(train_df[col].value_counts())
        print("=" * 40)
    else:
        print(col)
        print(train_df[col].describe())
        print("=" * 40)


# Using the 'IterativeImputer' Deeplearning model to impute the missing data
def impute_missing(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Initialize Ordinal Encoder
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    # Store the original categories for later decoding
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    # Apply Iterative Imputer for missing values
    imputer = IterativeImputer()
    df.iloc[:, :] = imputer.fit_transform(df)
    
    # Convert numeric values back to categorical values
    df[categorical_cols] = encoder.inverse_transform(df[categorical_cols])

    return df

# Check final dataset
train_df = impute_missing(train_df)
# extra_df = impute_missing(extra_df)

train_df.isnull().sum()


def remove_outliers(df, columns, threshold=0.45):
    df_cleaned = df.copy()
    
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25) 
        Q3 = df_cleaned[col].quantile(0.75) 
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Remove rows where values are outside the IQR range
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned

# Define numerical columns to check for outliers
num_cols = train_df.select_dtypes("number").columns.tolist()
print(num_cols)
# Remove outliers
train_df = remove_outliers(train_df, num_cols)

# Check the difference
print(f"Cleaned dataset size: {train_df.shape[0]}")


fig = px.histogram(train_df, x="Compartments", nbins=20, title="Distribution of Compartments")
fig.show()


fig = px.histogram(train_df, x="Weight Capacity (kg)", title="Distribution of Weight Capacity (kg)")
fig.show()


# Count values for Waterproof feature
waterproof_counts = train_df['Waterproof'].value_counts()

# Create an enhanced pie chart
fig = px.pie(
    names=waterproof_counts.index,
    values=waterproof_counts.values,
    title="Waterproof Backbag Percantage (%)",
    hole=0.4, 
    color=waterproof_counts.index,
    color_discrete_map={"Yes": "#1f77b4", "No": "#ff7f0e"}
)

# Update layout for better aesthetics
fig.update_traces(
    textinfo="percent+label", 
    pull=[0.05, 0], 
    marker=dict(line=dict(color="#000000", width=1))  
)

fig.show()


compartments_price = train_df.groupby("Material")['Price'].sum().sort_values(ascending = False)
fig = px.bar(x = compartments_price.index, y = compartments_price)
fig.show()


# Grouping each brand by its average price
brand_price = train_df.groupby("Brand")[['Price', 'Weight Capacity (kg)']].mean().sort_values(by = "Price", ascending = False)
brand_price


# Reset the index and rename the index column properly
brand_price = brand_price.reset_index()  

# Melt the DataFrame for Plotly (long format)
brand_price_melted = brand_price.melt(id_vars="Brand",  # Use the correct column name
                                      value_vars=["Price", "Weight Capacity (kg)"], 
                                      var_name="Metric", 
                                      value_name="Value")

# Create grouped bar plot
fig = px.bar(brand_price_melted, 
             x="Brand", 
             y="Value", 
             color="Metric", 
             barmode="group",  # Grouped bars
             title="Comparison of Average Price and Weight Capacity by Brand",
             labels={"Brand": "Brand", "Value": "Value", "Metric": "Feature"},
             color_discrete_map={"Price": "#FF6F61", "Weight Capacity (kg)": "#1F77B4"})

# Customize layout for a cleaner look
fig.update_layout(
    xaxis_title="Brand",
    yaxis_title="Value",
    legend_title="Feature",
    template="plotly_white",  # Clean background
    font=dict(size=14),
    bargap=0.2  # Reduce gap between bars
)

fig.show()



# Splitting the features and target into train and test data
X = train_df.drop(columns = ['Price'])
y = train_df['Price']

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

enc = OneHotEncoder(sparse_output=False) 
X_train_encoded = enc.fit_transform(X_train[categorical_cols])
X_test_encoded = enc.transform(X_test[categorical_cols])

# Convert encoded features to DataFrame with proper column names
X_train_encoded = pd.DataFrame(X_train_encoded, columns=enc.get_feature_names_out(categorical_cols))
X_test_encoded = pd.DataFrame(X_test_encoded, columns=enc.get_feature_names_out(categorical_cols))

# Reset index to align concatenation
X_train_encoded.reset_index(drop=True, inplace=True)
X_test_encoded.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# Drop original categorical columns and replace with encoded data
X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

# final processed train data
X_train.head()


# Applying Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


xgb_regressor = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42
)
param_grid = {
    'n_estimators': [100, 300, 500], 
    'learning_rate': [0.01, 0.05, 0.1],  
    'max_depth': [3, 5, 10], 
    'subsample': [0.3, 0.5, 0.7, 0.9],  # Row sampling ratio
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9]  # Feature sampling ratio
}

grid_search = RandomizedSearchCV(
    estimator=xgb_regressor, 
    param_distributions=param_grid,
    scoring='neg_mean_squared_error',
    cv=5, 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


xgb_regressor = grid_search.best_estimator_


# Make predictions
y_pred_train = xgb_regressor.predict(X_train)
y_pred_test = xgb_regressor.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Print evaluation metrics
print(f"ðŸ”¹ MAE: {mae:.2f}")
print(f"ðŸ”¹ RMSE: {rmse:.2f}")
print(f"ðŸ”¹ RÂ² Score: {r2:.4f}")


import pickle as pkl

# Save encoder
with open("/kaggle/working/encoder.pkl", "wb") as f:
    pkl.dump(enc, f)

# Save scaler
with open("/kaggle/working/scaler.pkl", "wb") as f:
    pkl.dump(scaler, f)



def process_data(df: pd.DataFrame,
                 enc_path: str = "/kaggle/working/encoder",
                 stand_path: str = "/kaggle/working/scaler"):
    
    # Removing missing values
    df = impute_missing(df)
    
    # Remove outliers
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = remove_outliers(df, numerical_cols)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_enc = enc.transform(df.loc[:, categorical_cols])
    
    # Convert encoded features to DataFrame with original index
    df_enc = pd.DataFrame(df_enc, columns=enc.get_feature_names_out(categorical_cols), index=df.index)
    
    # Drop original categorical columns and replace with encoded data
    df = df.drop(columns=categorical_cols).join(df_enc)

    # Standardizing the dataset (excluding 'Price')
    df_scaled = scaler.transform(df.drop(columns=['Price']))
    
    # Convert scaled data to DataFrame with original index
    df_scaled = pd.DataFrame(df_scaled, columns=df.drop(columns=['Price']).columns, index=df.index)  

    # Concatenate Price column with scaled data
    df = df[['Price']].join(df_scaled)

    return df

# Process test_df without resetting index
test_df = process_data(test_df)

# Display first few rows
test_df.head()


test_pred = xgb_regressor.predict(test_df.drop(columns = ['Price']))
mse = mean_squared_error(test_df['Price'], test_pred)
rmse = np.sqrt(mse)
rmse


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv").set_index("id")
id = test.index
test['Price'] = [0] * len(test)
test = process_data(test).drop(columns = ['Price'])
test.head()


y_sub = xgb_regressor.predict(test)

sub = pd.DataFrame({"id": test.index, "Price": y_sub})
sub.head()


sub.to_csv('/kaggle/working/submission.csv')

