# Importing necessary libraries for data manipulation, visualization, and modeling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Importing additional libraries for preprocessing, model evaluation, and hyperparameter tuning
from itertools import product
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set inline display for matplotlib
%matplotlib inline


# Read datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

# Display the first few rows
train_df.head()


test_df.head()


sample_submission_df.head()


train_df.columns


# Checking general information about the dataset (column types, non-null counts, memory usage)
print("\n=== General Information ===")
print(train_df.info())

# Check for missing values
print("\n=== Missing Values ===")
missing_values = train_df.isnull().sum()
print(missing_values)

# Plot missing values if any
if (missing_values > 0).any():
    plt.figure(figsize=(8, 6))
    missing_values[missing_values > 0].plot(kind='bar', title='Missing Values Count')
    plt.ylabel('Count')
    plt.show()
else:
    print("No missing values found.")


# Summary statistics for numerical features
print("\n=== Summary Statistics ===")
train_df.describe()


# Displaying the unique count of values in each column for understanding data distribution
for col in train_df:
    print(col, train_df[col].nunique())


# Plot distributions for numerical features
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# Boxplots for numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=train_df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()


# Correlation heatmap for numerical features
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


# Calculate skewness and kurtosis
print("\n=== Skewness and Kurtosis ===")
for col in numerical_cols:
    skewness = train_df[col].skew()
    kurtosis = train_df[col].kurt()
    print(f"{col}: Skewness = {skewness:.2f}, Kurtosis = {kurtosis:.2f}")


# Analyze categorical features
categorical_cols = train_df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


# Grouped analysis by categorical variables
if 'num_sold' in train_df.columns:
    for col in categorical_cols:
        grouped_sales = train_df.groupby(col)['num_sold'].mean()
        plt.figure(figsize=(10, 6))
        grouped_sales.sort_values().plot(kind='bar', title=f'Average Sales by {col}')
        plt.ylabel('Average Sales')
        plt.show()


# Interaction analysis: Heatmap for sales grouped by pairs of categorical features
if len(categorical_cols) > 1:
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            interaction_df = train_df.groupby([categorical_cols[i], categorical_cols[j]])['num_sold'].mean().unstack()
            if interaction_df is not None:
                plt.figure(figsize=(12, 8))
                sns.heatmap(interaction_df, annot=True, fmt=".1f", cmap="viridis")
                plt.title(f'Heatmap of Sales by {categorical_cols[i]} and {categorical_cols[j]}')
                plt.show()


# Analyze the target variable 'num_sold'
if 'num_sold' in train_df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_df['num_sold'], kde=True, bins=30)
    plt.title("Distribution of 'num_sold'")
    plt.xlabel('num_sold')
    plt.ylabel('Frequency')
    plt.show()
    
    # Log-transformed distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(np.log1p(train_df['num_sold']), kde=True, bins=30)
    plt.title("Log-Transformed Distribution of 'num_sold'")
    plt.xlabel('log(num_sold)')
    plt.ylabel('Frequency')
    plt.show()


# Time-series analysis if 'date' column exists
if 'date' in train_df.columns:
    # Create a temporary datetime index for time-series analysis
    train_df['date'] = pd.to_datetime(train_df['date'])
    temp_df = train_df.set_index('date', inplace=False)

    # Yearly sales
    if 'num_sold' in temp_df.columns:
        plt.figure(figsize=(10, 6))
        temp_df['num_sold'].resample('Y').sum().plot(title='Yearly Sales')
        plt.ylabel('Total Sales')
        plt.show()
        
        # Monthly sales
        plt.figure(figsize=(10, 6))
        temp_df['num_sold'].resample('M').mean().plot(title='Monthly Average Sales')
        plt.ylabel('Average Sales')
        plt.show()
        
        # Weekly trends
        temp_df['weekday'] = temp_df.index.weekday
        weekday_sales = temp_df.groupby('weekday')['num_sold'].mean()
        plt.figure(figsize=(8, 6))
        weekday_sales.plot(kind='bar', title='Average Sales by Day of the Week')
        plt.xlabel('Weekday (0 = Monday, 6 = Sunday)')
        plt.ylabel('Average Sales')
        plt.show()
    
    # Remove the temporary datetime index to revert train_df to its original state
    del temp_df


def preprocess_and_evaluate_with_hyperparameter_tuning(train, test):
    """
    Preprocess data using various strategies, perform hyperparameter tuning with XGBoost,
    and select the best preprocessing pipeline based on MAPE score on validation data.

    Parameters:
    - train (pd.DataFrame): The training dataset with features and target variable.
    - test (pd.DataFrame): The test dataset for which predictions are made.

    Returns:
    - best_X_train: Processed training features of the best combination.
    - best_X_val: Processed validation features of the best combination.
    - best_y_train: Training labels.
    - best_y_val: Validation labels.
    - best_X_test: Processed test features.
    - best_model: The best tuned XGBoost model.
    - best_preprocessing: The best preprocessing strategy combination.
    """
    
    # Options for preprocessing
    imputation_strategies = ['mean', 'median', None]  # Added None for dropping missing rows
    encoding_strategies = ['label', 'onehot']
    scaling_strategies = ['standard', 'minmax', None]

    # Hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Keep track of the best combination
    best_mape = float('inf')
    best_preprocessing = None
    best_X_train, best_X_val, best_y_train, best_y_val = None, None, None, None
    best_X_test = None
    best_model = None

    # Split train into training and validation sets
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Drop 'id' column from all datasets
    train = train.drop(columns=['id'])
    val = val.drop(columns=['id'])
    test = test.drop(columns=['id'])

    # Iterate over preprocessing options
    for imputation in imputation_strategies:
        for encoding in encoding_strategies:
            for scaling in scaling_strategies:
                print(f"Trying combination: Imputation={imputation}, Encoding={encoding}, Scaling={scaling}")

                # Create copies of the datasets
                train_copy = train.copy()
                val_copy = val.copy()
                test_copy = test.copy()

                # Handle missing values in `num_sold`
                if imputation is None:
                    # Drop rows with missing target values
                    train_copy = train_copy.dropna(subset=['num_sold'])
                    val_copy = val_copy.dropna(subset=['num_sold'])
                else:
                    # Impute missing values
                    imputer = SimpleImputer(strategy=imputation)
                    train_copy['num_sold'] = imputer.fit_transform(train_copy[['num_sold']])
                    val_copy['num_sold'] = imputer.transform(val_copy[['num_sold']])

                # Convert date to datetime and extract features
                for df in [train_copy, val_copy, test_copy]:
                    df['date'] = pd.to_datetime(df['date'])
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['day'] = df['date'].dt.day
                    df['weekday'] = df['date'].dt.weekday
                    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
                    df.drop(columns=['date'], inplace=True)

                # Separate features and target
                feature_cols = train_copy.columns.difference(['num_sold'])
                X_train = train_copy[feature_cols]
                y_train = train_copy['num_sold']
                X_val = val_copy[feature_cols]
                y_val = val_copy['num_sold']
                X_test = test_copy[feature_cols]

                # Encoding categorical variables
                cat_cols = ['country', 'store', 'product']
                if encoding == 'label':
                    encoder = LabelEncoder()
                    for col in cat_cols:
                        X_train[col] = encoder.fit_transform(X_train[col])
                        X_val[col] = encoder.transform(X_val[col])
                        X_test[col] = encoder.transform(X_test[col])
                elif encoding == 'onehot':
                    transformer = ColumnTransformer(
                        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                        remainder='passthrough'
                    )
                    X_train = pd.DataFrame(
                        transformer.fit_transform(X_train),
                        columns=transformer.get_feature_names_out()
                    )
                    X_val = pd.DataFrame(
                        transformer.transform(X_val),
                        columns=transformer.get_feature_names_out()
                    )
                    X_test = pd.DataFrame(
                        transformer.transform(X_test),
                        columns=transformer.get_feature_names_out()
                    )

                # Feature scaling
                if scaling == 'standard':
                    scaler = StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
                    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
                elif scaling == 'minmax':
                    scaler = MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
                    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

                # Hyperparameter tuning using RandomizedSearchCV
                xgb_model = XGBRegressor(
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0,
                    random_state=42
                )

                random_search = RandomizedSearchCV(
                    estimator=xgb_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    scoring='neg_mean_absolute_percentage_error',
                    cv=3,
                    verbose=2,
                    random_state=42
                )

                random_search.fit(X_train, y_train)
                tuned_model = random_search.best_estimator_

                # Evaluate the model on the validation set
                y_pred = tuned_model.predict(X_val)
                mape = mean_absolute_percentage_error(y_val, y_pred)
                print(f"Combination: {mape:.4f} MAPE")

                # Check if this is the best preprocessing combination
                if mape < best_mape:
                    best_mape = mape
                    best_preprocessing = (imputation, encoding, scaling)
                    best_X_train, best_X_val, best_y_train, best_y_val = X_train, X_val, y_train, y_val
                    best_X_test = X_test
                    best_model = tuned_model

    print(f"\nBest preprocessing: {best_preprocessing} with MAPE = {best_mape:.4f}")
    return best_X_train, best_X_val, best_y_train, best_y_val, best_X_test, best_model, best_preprocessing


# Perform preprocessing optimization and hyperparameter tuning
best_X_train, best_X_val, best_y_train, best_y_val, best_X_test, best_model, best_preprocessing = preprocess_and_evaluate_with_hyperparameter_tuning(train_df, test_df)


# Evaluate the final tuned model on the validation set
y_pred = best_model.predict(best_X_val)
final_mape = mean_absolute_percentage_error(best_y_val, y_pred)
print(f"Final Model MAPE: {final_mape}")


# Predict on the test set
test_df['num_sold'] = best_model.predict(best_X_test)


# Create submission file
submission = test_df[['id', 'num_sold']]
submission.to_csv('submission.csv', index=False)


# Plot feature importance
xgb.plot_importance(best_model, importance_type='weight', max_num_features=10)
plt.title("Top 10 Features")
plt.show()




