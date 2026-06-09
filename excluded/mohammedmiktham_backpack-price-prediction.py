# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



# 1. Basic Dataset Overview
def comprehensive_eda(df, target_variable):
    """Performs a comprehensive EDA with missing value handling and key visualizations."""
    print("--- EDA ---")

    # 1. Data Overview
    print("\n1. Data Overview:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nData types:\n", df.dtypes)

    # 2. Missing Value Analysis
    print("\n2. Missing Value Analysis:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Visualize missing values (optional, but helpful)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    # Impute missing values (before further EDA that relies on complete data)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':  # Categorical
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill with mode
            else:  # Numerical
                df[col] = df[col].fillna(df[col].median())  # Fill with median
    print("\nMissing values after imputation:")
    print(df.isnull().sum())  # Verify imputation

    # 3. Descriptive Statistics
    print("\n3. Descriptive Statistics:")
    print(df.describe())

    # 4. Target Variable Distribution
    print("\n4. Target Variable Distribution:")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_variable], kde=True)
    plt.title(f'Distribution of {target_variable}')
    plt.show()

    # 5. Numerical Feature Distributions
    print("\n5. Numerical Feature Distributions:")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove(target_variable)  # remove the target
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # 6. Categorical Feature Distributions
    print("\n6. Categorical Feature Distributions:")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')  # Rotate labels if needed
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()


    # 7. Correlation Heatmap (for numerical features)
    print("\n7. Correlation Heatmap:")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    correlation_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


# Separate features and target
X_train = train_data.drop(columns=['Price', 'id'])
y_train = train_data['Price']
X_test = test_data.drop(columns=['id'])


# Perform EDA on the training data
comprehensive_eda(train_data.copy(), 'Price')


# 8. Scatter Plots (Numerical Features)
print("\n8. Scatter Plots (Numerical Features):")
numerical_cols = train_data.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('Price')
if len(numerical_cols) > 1: # Make sure there are at least 2 numerical features
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=train_data[numerical_cols[i]], y=train_data[numerical_cols[j]], hue=train_data['Price'])
            plt.title(f'{numerical_cols[i]} vs. {numerical_cols[j]}')
            plt.show()


# Define categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']


# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  #missing value fil for median
    ('scaler', StandardScaler())  #values for no outsite range (normalization)
])


#categorical features missing value fil
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) #categorical features values convert to numerical values
])


# Combine used transformers method
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Splitting data AFTER imputation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)   #train - price and id 80% and test-price 20% devided


X_train_processed = preprocessor.fit_transform(X_train_split)
X_val_processed = preprocessor.transform(X_val_split)
X_test_processed = preprocessor.transform(X_test)


# Initialize models (reduced n_estimators for speed)
rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=20, random_state=42)  # Reduced estimators


# Train and evaluate Random Forest
rf_model.fit(X_train_processed, y_train_split)
rf_pred = rf_model.predict(X_val_processed)
rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))
rf_r2 = r2_score(y_val_split, rf_pred)
print(f"Random Forest - RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")


# Train and evaluate Gradient Boosting
gb_model.fit(X_train_processed, y_train_split)
gb_pred = gb_model.predict(X_val_processed)
gb_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred))
gb_r2 = r2_score(y_val_split, gb_pred)
print(f"Gradient Boosting - RMSE: {gb_rmse:.4f}, R2: {gb_r2:.4f}")


ensemble_predictions = (rf_pred + gb_pred) / 2  # Simple average of the two models
ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_predictions))
ensemble_r2 = r2_score(y_val_split, ensemble_predictions)
print(f"Ensemble - RMSE: {ensemble_rmse:.4f}, R2: {ensemble_r2:.4f}")


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error


# Splitting data AFTER imputation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train_split)
X_val_processed = preprocessor.transform(X_val_split)
X_test_processed = preprocessor.transform(X_test)

# Initialize models (reduced n_estimators for speed)
rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=20, random_state=42)  # Reduced estimators

# Train and evaluate Random Forest
rf_model.fit(X_train_processed, y_train_split)
rf_pred = rf_model.predict(X_val_processed)
rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))
rf_r2 = r2_score(y_val_split, rf_pred)
rf_mae = mean_absolute_error(y_val_split, rf_pred)
rf_mape = mean_absolute_percentage_error(y_val_split, rf_pred)
rf_medae = median_absolute_error(y_val_split, rf_pred)
print("Random Forest Metrics:")
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  R2: {rf_r2:.4f}")
print(f"  MAE: {rf_mae:.4f}")
print(f"  MAPE: {rf_mape:.4f}")
print(f"  MedAE: {rf_medae:.4f}")

# Train and evaluate Gradient Boosting
gb_model.fit(X_train_processed, y_train_split)
gb_pred = gb_model.predict(X_val_processed)
gb_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred))
gb_r2 = r2_score(y_val_split, gb_pred)
gb_mae = mean_absolute_error(y_val_split, gb_pred)
gb_mape = mean_absolute_percentage_error(y_val_split, gb_pred)
gb_medae = median_absolute_error(y_val_split, gb_pred)
print("Gradient Boosting Metrics:")
print(f"  RMSE: {gb_rmse:.4f}")
print(f"  R2: {gb_r2:.4f}")
print(f"  MAE: {gb_mae:.4f}")
print(f"  MAPE: {gb_mape:.4f}")
print(f"  MedAE: {gb_medae:.4f}")

ensemble_predictions = (rf_pred + gb_pred) / 2  # Simple average of the two models
ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_predictions))
ensemble_r2 = r2_score(y_val_split, ensemble_predictions)
ensemble_mae = mean_absolute_error(y_val_split, ensemble_predictions)
ensemble_mape = mean_absolute_percentage_error(y_val_split, ensemble_predictions)
ensemble_medae = median_absolute_error(y_val_split, ensemble_predictions)
print("Ensemble Metrics:")
print(f"  RMSE: {ensemble_rmse:.4f}")
print(f"  R2: {ensemble_r2:.4f}")
print(f"  MAE: {ensemble_mae:.4f}")
print(f"  MAPE: {ensemble_mape:.4f}")
print(f"  MedAE: {ensemble_medae:.4f}")


# Predict on test data
rf_test_pred = rf_model.predict(X_test_processed)
gb_test_pred = gb_model.predict(X_test_processed)


# Ensemble predictions on test data
ensemble_test_predictions = (rf_test_pred + gb_test_pred) / 2


# Remove outliers from Price 
Q1 = train_data['Price'].quantile(0.25)
Q3 = train_data['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_data = train_data[(train_data['Price'] >= lower_bound) & (train_data['Price'] <= upper_bound)]


numerical_cols = ['Compartments', 'Weight Capacity (kg)', 'Price']
correlation_matrix = train_data[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train) # fit AND transform train data
X_test_processed = preprocessor.transform(X_test)  # transform test data

rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=20, random_state=42)  # Reduced estimators

# Train Random Forest
rf_model.fit(X_train_processed, y_train) # Train on ALL training data
# Train Gradient Boosting
gb_model.fit(X_train_processed, y_train)  # Train on ALL training data


# Predict on test data
rf_test_pred = rf_model.predict(X_test_processed)
gb_test_pred = gb_model.predict(X_test_processed)

# Ensemble predictions on test data
ensemble_test_predictions = (rf_test_pred + gb_test_pred) / 2


rf_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred))
rf_r2 = r2_score(y_val_split, gb_pred)
print(f"Random Forest Validation - RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")

gb_rmse = np.sqrt(mean_squared_error(y_val_split, gb_pred))
gb_r2 = r2_score(y_val_split, gb_pred)
print(f"Gradient Boosting Validation - RMSE: {gb_rmse:.4f}, R2: {gb_r2:.4f}")


ensemble_predictions = (rf_pred + gb_pred) / 2  # Simple average of the two models
ensemble_rmse = np.sqrt(mean_squared_error(y_val_split, ensemble_predictions))
ensemble_r2 = r2_score(y_val_split, ensemble_predictions)
print(f"Ensemble - RMSE: {ensemble_rmse:.4f}, R2: {ensemble_r2:.4f}")


submission = pd.DataFrame({'id': test_data['id'],'Price': ensemble_test_predictions})


submission.to_csv('submission.csv', index=False)

