import os
print(os.listdir("/kaggle/input"))

import warnings
# Suppress specific FutureWarnings related to inf handling in Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

import optuna
import scipy.stats as stats


# Adjust the filename based on the competition dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Display dataframe
train_df


test_df


# See Features
print(train_df.columns)
print("Number of useful features:", len(train_df.columns)-2) # Exclude id and target


print("\nUnique Values for Categorical Features:")
for column in train_df.columns:
    if train_df[column].dtype == 'object':  # Check if the column is categorical
        unique_values = train_df[column].unique()
        print(f"{column}: {unique_values}")

print("\nSmallest and Largest Values for Numerical Features:")
for column in train_df.columns:
    if train_df[column].dtype != 'object':  # Check if the column is numerical
        min_value = train_df[column].min()
        max_value = train_df[column].max()
        print(f"{column}: Min = {min_value}, Max = {max_value}")


# Check for NaN or infinite values in the DataFrame
print(train_df.isna().sum())  # Count of NaN values per column


# Check for NaN or infinite values in the DataFrame
print(test_df.isna().sum())  # Count of NaN values per column


train_df.describe()


train_df.describe(include='object')


print(train_df[train_df['Number_of_Ads'] > 6]['Number_of_Ads'])


print(train_df[train_df['Host_Popularity_percentage'] > 100]['Host_Popularity_percentage'])


print(train_df[train_df['Guest_Popularity_percentage'] > 100]['Guest_Popularity_percentage'])


train_df['Number_of_Ads'] = train_df['Number_of_Ads'].clip(upper=6)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].clip(upper=6)

train_df['Host_Popularity_percentage'] = train_df['Host_Popularity_percentage'].clip(upper=100)
test_df['Host_Popularity_percentage'] = test_df['Host_Popularity_percentage'].clip(upper=100)

train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].clip(upper=100)
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].clip(upper=100)


# Separate features and target
X_train = train_df.drop(columns=['Listening_Time_minutes', 'id'])  # Drop 'id' and target column
y_train = train_df['Listening_Time_minutes']

# Drop 'id' from test data
train_df = train_df.drop('id', axis=1)
X_test = test_df.drop(columns=['id'])

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
            ('scaler', StandardScaler())  # Standardize numerical data
        ]), numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Encode categorical data as numbers
        ]), categorical_cols)
    ]
)

# Create a preprocessor with ColumnTransformer using KNNImputer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', Pipeline([
#             ('imputer', KNNImputer(n_neighbors=5)),  # Use KNN imputer for numerical features
#             ('scaler', StandardScaler())
#         ]), numerical_cols),
        
#         ('cat', Pipeline([
#             ('imputer', KNNImputer(missing_values=np.nan, n_neighbors=5)),  # This step is not ideal for categorical features
#             ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
#         ]), categorical_cols)
#     ]
# )

# Apply transformations to the training and test data (features only)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Combine the column names
all_columns = numerical_cols + categorical_cols

# Debugging: Check the shape of the transformed data
print(f"Shape of X_train_transformed: {X_train_transformed.shape}")
print(f"Shape of X_test_transformed: {X_test_transformed.shape}")


# Create DataFrames from the transformed arrays
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=all_columns)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=all_columns)

# Debugging: Check the column names and shapes of the transformed DataFrames
print(f"Shape of X_train_transformed_df: {X_train_transformed_df.shape}")
print(f"Shape of X_test_transformed_df: {X_test_transformed_df.shape}")

X_train_transformed_df.head()


# Histograms for Distribution of Numerical Features
train_df.hist(figsize=(12, 6), bins=150)
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()


# Get categorical columns
categorical_columns = train_df.select_dtypes(include=['object', 'category']).columns
categorical_columns


# Get categorical columns
categorical_columns = ['Genre', 'Publication_Day','Publication_Time', 'Episode_Sentiment']

# Define the number of rows and columns for the grid
rows, cols = 2, 2

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

# Flatten axes array to make indexing easier
axes = axes.flatten()
plt.suptitle('Distribution of Categorical Features')

# Loop through each categorical column and plot
for i, col in enumerate(categorical_columns):
    axes[i].bar(train_df[col].value_counts().index, train_df[col].value_counts().values)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    
    # Rotate the x-axis labels for better readability
    axes[i].tick_params(axis='x', rotation=45)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Scatterplot Matrix - Visualize the pairwise relationships between features to understand any correlation
features = train_df.columns
sns.pairplot(train_df[features])
plt.suptitle('Pairplot for Numerical Features', y=1.02)
plt.show()


# Correlation Heatmap to understand the linear relationships between numerical variables
df_preproc = pd.DataFrame(X_train_transformed, columns=X_train.columns)  # Create a DataFrame for the scaled features
df_preproc['Listening_Time_minutes'] = y_train.values

plt.figure(figsize=(10, 6))
corr = df_preproc.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Correlation Heatmap to understand the linear relationships between numerical variables
df_preproc = pd.DataFrame(X_train, columns=X_train.columns)  # Create a DataFrame for the scaled features
numerical_columns = train_df.select_dtypes(include=['number']).columns
numerical_columns = numerical_columns[numerical_columns != 'Listening_Time_minutes']
df_preproc = df_preproc[numerical_columns]

df_preproc['Listening_Time_minutes'] = y_train

plt.figure(figsize=(10, 6))
corr = df_preproc.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Define the pairs of variables you want to plot
pairs = [
    ('Listening_Time_minutes', 'Genre'),
    ('Listening_Time_minutes', 'Publication_Day'),
    ('Listening_Time_minutes', 'Publication_Time'),
    ('Listening_Time_minutes', 'Episode_Sentiment')
]

# Define the number of rows and columns for the grid
rows, cols = 2, 2  # Adjust according to the number of plots you need

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

# Flatten axes array to make indexing easier
axes = axes.flatten()

# Loop through each pair and create the appropriate plot
for i, (x_col, y_col) in enumerate(pairs):
    if train_df[y_col].dtype == 'object':  # Categorical variable
        sns.barplot(data=train_df, x=y_col, y=x_col, ax=axes[i])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Define the pairs of variables you want to plot
pairs = [
    ('Listening_Time_minutes', 'Genre'),
    ('Listening_Time_minutes', 'Publication_Day'),
    ('Listening_Time_minutes', 'Publication_Time'),
    ('Listening_Time_minutes', 'Episode_Sentiment')
]

# Define the number of rows and columns for the grid
rows, cols = 2, 2  # Adjust according to the number of plots you need

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

# Flatten axes array to make indexing easier
axes = axes.flatten()

# Loop through each pair and create the appropriate plot
for i, (x_col, y_col) in enumerate(pairs):
    if train_df[y_col].dtype == 'object':  # Categorical variable
        sns.boxplot(data=train_df, x=y_col, y=x_col, ax=axes[i])
    else:  # Continuous variable
        sns.scatterplot(data=train_df, x=x_col, y=y_col, ax=axes[i])

    axes[i].set_title(f'{x_col} vs {y_col}')
    axes[i].set_xlabel(x_col)
    axes[i].set_ylabel(y_col)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Get categorical features
numerical_features = train_df.select_dtypes(include=['number']).columns
numerical_features = numerical_features.drop('Listening_Time_minutes')

# Create subplots
num_plots = len(numerical_features)
fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=2, figsize=(12, 3 * (num_plots // 3 + 1)))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each numerical feature
for i, feature in enumerate(numerical_features):
    # Dynamically determine the number of bins based on the feature's range or standard deviation
    range_feature = train_df[feature].max() - train_df[feature].min()
    num_bins = max(5, int(range_feature // 10))  # Adjust the number of bins based on the range (you can change this logic)
    
    # Create bins for the feature values
    bins = np.linspace(train_df[feature].min(), train_df[feature].max(), num_bins)
    
    # Digitize the feature values into bins
    bin_indices = np.digitize(train_df[feature], bins)
    
    # Calculate the mean of y_train for each bin
    bin_means = [y_train[bin_indices == i].mean() for i in range(1, len(bins))]
    
    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot on the respective subplot
    ax = axes[i]
    ax.plot(bin_centers, bin_means, marker='o', linestyle='-', color='b')
    ax.set_title(f"y_train vs {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Mean_Listening_Time_minutes")
    ax.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Get categorical features
categorical_features = train_df.select_dtypes(include=[object]).columns

# Create subplots
num_plots = len(categorical_features)
fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(12, 3 * (num_plots // 3 + 1)))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Loop through each categorical feature
for i, feature in enumerate(categorical_features):
    # Group by the feature and calculate the mean of y_train for each category
    category_means = train_df.groupby(feature)[y_train.name].mean()

    # Sort the category means to ensure the line goes up
    sorted_category_means = category_means.sort_values()

    # Plot on the respective subplot
    ax = axes[i]
    sorted_category_means.plot(marker='o', kind='line', color='b', ax=ax)
    ax.set_title(f"y_train vs {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Mean_Listening_Time_minutes")
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


def evaluate_models(X_train, y_train, random_state=26):
    # Define KFold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Initialize the models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Bayesian Ridge': BayesianRidge(),

        'XGBoost': xgb.XGBRegressor(),
        'CatBoost': CatBoostRegressor(silent=True),
        'LightGBM': lgb.LGBMRegressor(verbose=-1),

        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Extra Trees': ExtraTreesRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        
        'K-Nearest Neighbors': KNeighborsRegressor(),
        # 'SVM': SVR(),
        # 'Gaussian Process': GaussianProcessRegressor()
    }

    # Dictionary to store results
    model_scores = {}
    
    # Loop through each model
    for model_name, model in models.items():
        # Perform cross-validation and compute the negative RMSE
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        
        # Convert to RMSE
        rmse_scores = np.sqrt(-scores)
        
        # Convert MSE to positive RMSE
        mean_rmse = np.mean(rmse_scores)  # Compute the mean RMSE across folds
        
        # Print the mean RMSE for this model
        print(f'{model_name} - Mean RMSE: {mean_rmse:.4f}')

        # Store the result
        model_scores[model_name] = mean_rmse

    # Sort models by RMSE in ascending order (lower RMSE is better)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

    # Print results in descending order (best models first)
    print("\nModels sorted by RMSE:")
    for model, score in sorted_models:
        print(f"{model}: Mean RMSE = {score:.4f}")


evaluate_models(X_train_transformed, y_train)


def train_and_plot_model(model, X, y, test_size=0.2, random_state=26):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred  # Calculate residuals

    # Compute performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    print(f"Model: {model.__class__.__name__}")
    print(f"Test RMSE: {rmse:.4f}")
    # print(f"Test R² Score: {r2:.4f}")

    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)

    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    min_val = min(min(y_test), min(y_pred))  # Ensure the line spans both y_test and y_pred ranges
    max_val = max(max(y_test), max(y_pred))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
    axes[0, 0].set_title("Predicted vs Actual Values")
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    
    # 2. Residuals Plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_title("Residuals vs Predicted")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")

    # 3. Residual Histogram & KDE
    sns.histplot(residuals, kde=True, ax=axes[1, 0], bins=25, color="blue")
    axes[1, 0].set_title("Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals")

    # 4. QQ Plot (Normality Check)
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot of Residuals")

    plt.show()


model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
train_and_plot_model(model, X_train_transformed, y_train)


def generate_predictions_and_save(model, X_test, output_filename='predictions.csv'):   
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Prepare the DataFrame with 'id' and 'Listening_Time_minutes'
    output_df = pd.DataFrame({
        'id': test_df['id'],  # Assuming 'id' starts from 0, adjust if needed
        'Listening_Time_minutes': predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_filename, index=False)
    print(f'Predictions saved to {output_filename}')


# Define and train the model
best_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
best_model.fit(X_train_transformed, y_train)


# Generate predictions
generate_predictions_and_save(best_model, X_test_transformed, output_filename='rf_estimators_100.csv')

