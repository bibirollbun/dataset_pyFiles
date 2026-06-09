import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df = df.set_index("RefId")
df.info()


y = df["IsBadBuy"]  # Target variable
X = df.drop(columns=["IsBadBuy"])  # Features


from sklearn.model_selection import train_test_split

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
X_train.shape,X_test.shape


columns = X_train.columns
categorical = [
    "Auction",
    "Make",
    "Color",
    "Transmission",
    "WheelType",
    "Nationality",
    "Size",
    "TopThreeAmericanName",
    "IsOnlineSale"  # Add this if it's categorical
]
continuous = [col for col in X_train.columns if col not in categorical]

# Convert 'IsOnlineSale' to object type
X_train['IsOnlineSale'] = X_train['IsOnlineSale'].astype('object')
X_test['IsOnlineSale'] = X_test['IsOnlineSale'].astype('object')  # Apply the same change to X_test





import numpy as np
import pandas as pd

def initial_preproc(df):
    """
    Preprocess dataset by applying logical ranges, handling categorical inconsistencies, grouping rare categories, and dropping unwanted columns.

    Parameters:
    - df (pd.DataFrame): The input dataset.

    Returns:
    - df (pd.DataFrame): The preprocessed dataset.
    """
    # Define logical ranges for continuous features
    logical_ranges = {
        'VehicleAge': (0, 30),
        'VehOdo': (0, 120000),
        'MMRAcquisitionAuctionAveragePrice': (800, 46000),
        'MMRAcquisitionAuctionCleanPrice': (1000, 46000),
        'MMRAcquisitionRetailAveragePrice': (1000, 46000),
        'MMRAcquisitonRetailCleanPrice': (1000, 46000),
        'MMRCurrentAuctionAveragePrice': (300, 46000),
        'MMRCurrentAuctionCleanPrice': (400, 46000),
        'MMRCurrentRetailAveragePrice': (800, 46000),
        'MMRCurrentRetailCleanPrice': (1000, 46000),
        'VehBCost': (1000, 46000),
        'WarrantyCost': (400, 8000)
    }

    # Apply logical range filtering for continuous variables
    for col, (min_val, max_val) in logical_ranges.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if min_val <= x <= max_val else np.nan)

    # Convert 'NOT AVAIL' in 'color' column to NaN
    if "Color" in df.columns:
        df["Color"] = df["Color"].replace('NOT AVAIL', np.nan)

    # Replace 'MANUAL' in the 'Transmission' column with 'Manual'
    if "Transmission" in df.columns:
        df["Transmission"] = df["Transmission"].replace('MANUAL', 'Manual')

    # Function to group rare categories
    def group_rare_categories(df, col, threshold=0.01):
        """
        Group rare categories in a column into an 'OTHER' category.

        Parameters:
        - df (pd.DataFrame): The input dataset.
        - col (str): The column to process.
        - threshold (float): The frequency threshold for grouping (default: 0.01).

        Returns:
        - df (pd.DataFrame): The dataset with rare categories grouped.
        """
        freq = df[col].value_counts(normalize=True)  # Get frequency distribution
        rare_classes = freq[freq < threshold].index  # Find categories with frequency < threshold
        df[col] = df[col].apply(lambda x: 'OTHER' if x in rare_classes else x)
        return df

    # Apply grouping for 'color' and 'make' if they exist
    for cat_col in ["Color","Make"]:
        if cat_col in df.columns:
            df = group_rare_categories(df, cat_col)

    # Drop unwanted columns if they exist
    drop_columns = [
        "PurchDate", "VehYear", "Model", "Trim", "SubModel", "WheelTypeID", "BYRNO", "VNZIP1", "VNST", 'PRIMEUNIT', 'AUCGUART'
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

    return df


# Apply preprocessing to X_train and X_test
X_train = initial_preproc(X_train)
X_test = initial_preproc(X_test)

X_train.shape, X_test.shape






# Check 'Transmission' column
print("Unique values in 'Transmission' (X_train):", X_train['Transmission'].unique())
print("Unique values in 'Transmission' (X_test):", X_test['Transmission'].unique())

# Check 'Color' column
print("Unique values in 'Color' (X_train):", X_train['Color'].unique())
print("Unique values in 'Color' (X_test):", X_test['Color'].unique())

# Check 'Make' column
print("Unique values in 'Make' (X_train):", X_train['Make'].unique())
print("Unique values in 'Make' (X_test):", X_test['Make'].unique())





import numpy as np

def feature_screening(data, min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    processed_data = data.copy()

    # Identify categorical and continuous columns
    categorical = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
    continuous = processed_data.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Step 1: Remove continuous features with a coefficient of variation (CV) < min_cv
    cv_values = processed_data[continuous].std() / processed_data[continuous].mean()
    screen_cv = cv_values[cv_values < min_cv].index.tolist()

    # Step 2: Remove categorical features where the mode category percentage is > mode_threshold%
    mode_category = processed_data[categorical].apply(lambda x: x.value_counts(normalize=True).max() * 100)
    screen_mode = mode_category[mode_category > mode_threshold].index.tolist()

    # Step 3: Remove categorical features where > distinct_threshold% of values are unique
    distinct_percentage = processed_data[categorical].apply(lambda x: x.nunique() / len(x) * 100)
    screen_distinct = distinct_percentage[distinct_percentage > distinct_threshold].index.tolist()

    # Combine all screened features and drop them
    screened_features = screen_cv + screen_mode + screen_distinct
    processed_data = processed_data.drop(columns=screened_features, errors='ignore')

    return processed_data, screened_features




# Apply feature screening to X_train and X_test
X_train, dropped_train_features = feature_screening(X_train)
X_test, dropped_test_features = feature_screening(X_test)

X_train.shape, X_test.shape


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def outlier_handling(data, y=None, contamination=0.01):
    """
    Detect and remove outliers from the dataset.

    Parameters:
    - data (pd.DataFrame): The input dataset (features only).
    - y (pd.Series, optional): The target values. Not used in this function but kept for compatibility.
    - contamination (float): The proportion of outliers in the dataset (default: 0.01).

    Returns:
    - outlier_index (pd.Index): Indices of the outliers.
    """
    # Create a copy of the data
    data_iso = data.copy()

    # Handle missing values: drop rows with NaN values (if necessary)
    data_iso = data_iso.dropna()

    # Separate continuous and categorical columns
    continuous_columns = data_iso.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Apply Z-score scaling to continuous columns only
    scaler = StandardScaler()
    data_iso[continuous_columns] = scaler.fit_transform(data_iso[continuous_columns])

    # Fit Isolation Forest model
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(data_iso[continuous_columns])

    # Predict outliers (1 for inliers, -1 for outliers)
    outliers = clf.predict(data_iso[continuous_columns])

    # Get the index of outliers
    outlier_index = data_iso.index[outliers == -1]

    return outlier_index


# Call the function on X_train and y_train
outlier_index = outlier_handling(X_train, y_train, contamination=0.01)

# Drop the outliers from both X_train and y_train
X_train = X_train.drop(outlier_index)
y_train = y_train.drop(outlier_index)

# Check the shapes of X_train and y_train after removing outliers
print(X_train.shape, y_train.shape)



import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(train, test):
    """
    Handle missing values in the dataset by:
    1. Discarding rows with 4 or more null values in price-related columns.
    2. Discarding rows with 50% or more null values across all fields.
    3. Imputing remaining missing values using median for continuous fields and mode for categorical fields.

    Parameters:
    - train (pd.DataFrame): Training data.
    - test (pd.DataFrame): Testing data.

    Returns:
    - train (pd.DataFrame): Processed training data.
    - test (pd.DataFrame): Processed testing data.
    """
    # Price-related columns
    price_columns = [
        'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
        'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
        'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
        'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'
    ]

    # Step 1: Discard rows with 4 or more null values in price-related columns
    train = train[train[price_columns].isnull().sum(axis=1) < 4]
    test = test[test[price_columns].isnull().sum(axis=1) < 4]

    # Step 2: Discard rows with 50% or more null values across all fields
    train = train[train.isnull().sum(axis=1) / train.shape[1] < 0.5]
    test = test[test.isnull().sum(axis=1) / test.shape[1] < 0.5]

    # Step 3: Impute remaining missing values
    # Separate continuous and categorical columns
    continuous_columns = train.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = train.select_dtypes(include=['object', 'category']).columns

    # Impute continuous columns with median
    if len(continuous_columns) > 0:
        numerical_imputer = SimpleImputer(strategy='median')
        train[continuous_columns] = numerical_imputer.fit_transform(train[continuous_columns])
        test[continuous_columns] = numerical_imputer.transform(test[continuous_columns])

    # Impute categorical columns with mode
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        train[categorical_columns] = categorical_imputer.fit_transform(train[categorical_columns])
        test[categorical_columns] = categorical_imputer.transform(test[categorical_columns])

    return train, test


# Apply the function to handle missing values
X_train, X_test = handle_missing_values(X_train, X_test)
X_train.shape, X_test.shape


# Drop rows with missing values in X_train and y_train
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]  # Align y_train with X_train

# Drop rows with missing values in X_test and y_test
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]  # Align y_test with X_test

# Verify shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



X_train.info()
X_test.info()


pip install scorecardbundle


import numpy as np
import pandas as pd
from scorecardbundle.feature_discretization import ChiMerge as cm

chi_merge_list = ["VehBCost", "WarrantyCost"]
X_train[chi_merge_list] = X_train[chi_merge_list].astype(float)

def discretizer(train, test, y, chi_list):
    # Drop missing values in relevant columns
    train = train.dropna(subset=chi_list)
    y = y.loc[train.index]  # Align y_train with train

    # Ensure numeric type
    train[chi_list] = train[chi_list].astype(float)

    # Initialize ChiMerge
    trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3, output_dataframe=True)
    trans_cm.fit(train[chi_list], y.astype(int)) 

    # Ensure boundaries exist
    if not trans_cm.boundaries_:
        raise ValueError("ChiMerge did not generate boundaries. Check input data!")

    # Add -inf to boundaries
    boundaries_dict = {key: np.insert(boundaries, 0, -np.inf) for key, boundaries in trans_cm.boundaries_.items()}

    # Apply discretization and remove original features
    for key, boundaries in boundaries_dict.items():
        column_name = f"{key}_cat_cm"
        train[column_name] = pd.cut(train[key], bins=boundaries, labels=False, right=False)
        test[column_name] = pd.cut(test[key], bins=boundaries, labels=False, right=False)

    # Drop original features after transformation
    train = train.drop(columns=chi_list)
    test = test.drop(columns=chi_list)

    return train, test

# Apply discretization
X_train, X_test = discretizer(X_train, X_test, y_train, chi_merge_list)

# Check shapes
print(X_train.shape, X_test.shape)




from sklearn.preprocessing import PowerTransformer
import numpy as np

# Define the list of features to transform globally
transform_list = [
    'VehicleAge', 'VehOdo',
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'
]

def transform(train, test, transform_list):
    """
    Transform selected features using Yeo-Johnson or Box-Cox transformation.

    Parameters:
    - train (pd.DataFrame): Training data.
    - test (pd.DataFrame): Testing data.
    - transform_list (list): List of features to transform.

    Returns:
    - train (pd.DataFrame): Transformed training data.
    - test (pd.DataFrame): Transformed testing data.
    """
    # Initialize the transformer for Yeo-Johnson (handles both positive and negative values)
    yeo_johnson_transformer = PowerTransformer(method='yeo-johnson', standardize=True)

    # Initialize the transformer for Box-Cox (handles only positive values)
    box_cox_transformer = PowerTransformer(method='box-cox', standardize=True)

    # Iterate through selected features
    for feature in transform_list:
        # Check if the feature contains non-positive values
        if (train[feature] <= 0).any():
            # Apply Yeo-Johnson transformation
            train[feature] = yeo_johnson_transformer.fit_transform(train[[feature]]).flatten()
            test[feature] = yeo_johnson_transformer.transform(test[[feature]]).flatten()
        else:
            # Apply Box-Cox transformation
            train[feature] = box_cox_transformer.fit_transform(train[[feature]]).flatten()
            test[feature] = box_cox_transformer.transform(test[[feature]]).flatten()

    return train, test



# Apply the function to transform features
X_train, X_test = transform(X_train, X_test, transform_list)

# Check the shapes of the transformed data
X_train.shape, X_test.shape



import pandas as pd

def categorize_features(df):
    """
    Categorize features into continuous, categorical, nominal, and ordinal.

    Parameters:
    - df (pd.DataFrame): The dataset.

    Returns:
    - continuous (list): List of continuous features.
    - categorical (list): List of categorical features.
    - nominal (list): List of nominal features.
    - ordinal (list): List of ordinal features.
    """
    # Identify continuous features (numerical)
    continuous = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Identify categorical features (non-numerical)
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Classify categorical features into nominal and ordinal
    nominal = []
    ordinal = []

    for feature in categorical:
        # Check if the feature has an inherent order (ordinal)
        if is_ordinal(df[feature]):
            ordinal.append(feature)
        else:
            nominal.append(feature)

    return continuous, categorical, nominal, ordinal

def is_ordinal(series):
    """
    Determine if a categorical feature is ordinal based on its unique values.

    Parameters:
    - series (pd.Series): The feature column.

    Returns:
    - bool: True if the feature is ordinal, False otherwise.
    """
    # Example: Check if the feature has an inherent order
    unique_values = series.unique()
    if sorted(unique_values) == list(unique_values):
        return True
    return False

# Example usage
continuous, categorical, nominal, ordinal = categorize_features(X_train)

# Get lengths of each category
len_continuous = len(continuous)
len_categorical = len(categorical)
len_nominal = len(nominal)
len_ordinal = len(ordinal)

# Print the results
print("Number of Continuous Features:", len_continuous)
print("Number of Categorical Features:", len_categorical)
print("Number of Nominal Features:", len_nominal)
print("Number of Ordinal Features:", len_ordinal)


discretized_features = [ 'Auction', 'VehicleAge', 'Make','Color', 'Transmission', 'WheelType',
                         'VehOdo', 'Nationality','Size','TopThreeAmericanName','MMRAcquisitionAuctionAveragePrice',
                       'MMRAcquisitionAuctionCleanPrice','MMRAcquisitionRetailAveragePrice',
                       'MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                       'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice',
                       'IsOnlineSale','VehBCost_cat_cm','WarrantyCost_cat_cm']

def scenario(data, scen_list):
    return data[scen_list]


X_train_discretized = scenario(X_train, discretized_features)

X_test_discretized = scenario(X_test, discretized_features)





X_data = pd.concat((X_train_discretized,X_test_discretized), keys=['train','test'])
y_data = pd.concat((y_train, y_test), keys=['train','test'])

X_data.to_csv('X_data_discretized_no_scaling',index=True)
y_data.to_csv('y_data_discretized_no_scaling',index=True)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the preprocessing steps for numerical and categorical features separately
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
min_max = MinMaxScaler()

# Initialize RFECV for feature selection
wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=29),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

# Define the preprocessing steps for numerical and nominal features
numerical_preprocessing_1 = Pipeline(steps=[
    ('scaler', min_max)  # Apply MinMax scaling
])

nominal_preprocessing_1 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  # Apply one-hot encoding
    ('scaler', min_max)  # Apply MinMax scaling
])

# Define the ColumnTransformer for numerical and categorical features
preprocessor_1 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_1, continuous),  # Apply numerical preprocessing
    ('nom', nominal_preprocessing_1, categorical)  # Apply nominal preprocessing
])

# Define the pipeline
pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor_1),  # Apply preprocessing
    ('wrapper', wrapper),  # Apply feature selection
    ('model', DecisionTreeClassifier(random_state=17))  # Train the model
])

# Train the pipeline
pipeline_1.fit(X_train, y_train)

# Print the optimal number of features selected by RFECV
optimal_features = pipeline_1.named_steps['wrapper'].n_features_
print(f"Optimal number of features: {optimal_features}")

# Get feature names after preprocessing
try:
    feature_names = pipeline_1.named_steps['preprocessor'].get_feature_names_out()
    print("Feature names after preprocessing:", feature_names)
except AttributeError:
    print("Feature names could not be retrieved. Ensure your transformers support `get_feature_names_out`.")

# Use the pipeline for prediction
predictions_1 = pipeline_1.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_1)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions_1))



from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the preprocessing steps for numerical and categorical features separately
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
z_score = StandardScaler()  # StandardScaler for z-score normalization
pca = PCA(n_components=2,  random_state=717)  # PCA to retain 95% of the variance

# Define the preprocessing steps for numerical and nominal features
numerical_preprocessing_2 = Pipeline(steps=[
    ('scaler', z_score),  # Apply z-score normalization
    ('pca', pca)  # Apply PCA
])

nominal_preprocessing_2 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  # Apply one-hot encoding
    ('scaler', z_score)  # Apply z-score normalization
])

# Define the ColumnTransformer for numerical and categorical features
preprocessor_2 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_2, continuous),  # Apply numerical preprocessing
    ('nom', nominal_preprocessing_2, categorical)  # Apply nominal preprocessing
])

# Initialize RFECV for feature selection
wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=29),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

# Define the pipeline
pipeline_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_2),  # Apply preprocessing
    ('wrapper', wrapper),  # Apply feature selection
    ('model', DecisionTreeClassifier(random_state=17))  # Train the model
])

# Train the pipeline
pipeline_2.fit(X_train, y_train)

# Print the optimal number of features selected by RFECV
optimal_features = pipeline_2.named_steps['wrapper'].n_features_
print(f"Optimal number of features: {optimal_features}")

# Use the pipeline for prediction
predictions_2 = pipeline_2.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_2)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions_2))

# Get feature names after preprocessing
try:
    feature_names = pipeline_2.named_steps['preprocessor'].get_feature_names_out()
    print("Feature names after preprocessing:", feature_names)
except AttributeError:
    print("Feature names could not be retrieved. Ensure your transformers support `get_feature_names_out`.")


# pca_model = pipeline_2.named_steps['preprocessor'].named_transformers_['num'].named_steps['pca']
# print("PCA Components:")
# print(pca_model.components_)  # This prints the loadings (weights of each original feature in the PCA components)



from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the preprocessing steps for numerical and categorical features separately
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
z_score = StandardScaler()  # StandardScaler for z-score normalization
lda = LDA(n_components=None)  # LDA for dimensionality reduction

# Define the preprocessing steps for numerical and nominal features
numerical_preprocessing_3 = Pipeline(steps=[
    ('scaler', z_score),  # Apply z-score normalization
    ('lda', lda)  # Apply LDA
])

nominal_preprocessing_3 = Pipeline(steps=[
    ('nominal', one_hot_encoder),  # Apply one-hot encoding
    ('scaler', z_score)  # Apply z-score normalization
])

# Define the ColumnTransformer for numerical and categorical features
preprocessor_3 = ColumnTransformer(transformers=[
    ('num', numerical_preprocessing_3, continuous),  # Apply numerical preprocessing
    ('nom', nominal_preprocessing_3, categorical)  # Apply nominal preprocessing
])

# Initialize RFECV for feature selection
wrapper = RFECV(
    estimator=DecisionTreeClassifier(random_state=29),
    step=1,
    min_features_to_select=10,
    cv=5,
    n_jobs=-1
)

# Define the pipeline
pipeline_3 = Pipeline(steps=[
    ('preprocessor', preprocessor_3),  # Apply preprocessing
    ('wrapper', wrapper),  # Apply feature selection
    ('model', DecisionTreeClassifier(random_state=17))  # Train the model
])

# Train the pipeline
pipeline_3.fit(X_train, y_train)

# Print the optimal number of features selected by RFECV
optimal_features = pipeline_3.named_steps['wrapper'].n_features_
print(f"Optimal number of features: {optimal_features}")

# Use the pipeline for prediction
predictions_3 = pipeline_3.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_3)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, predictions_3))

# Get feature names after preprocessing
try:
    feature_names = pipeline_3.named_steps['preprocessor'].get_feature_names_out()
    print("Feature names after preprocessing:", feature_names)
except AttributeError:
    print("Feature names could not be retrieved. Ensure your transformers support `get_feature_names_out`.")




import pandas as pd
# Load the test dataset
test = pd.read_csv('/kaggle/input/DontGetKicked/test.csv')

# Set 'RefId' as the index
test.set_index('RefId', inplace=True)

# Check the shape of the test dataset
test.shape


# print("Type of test:", type(test))
# print("Test data:", test)


# test = initial_preproc(test)

test = initial_preproc(test)

test.shape


# Detect outliers in the test data
outlier_indices = outlier_handling(test)  # Pass only the test features

# Drop outliers from the test data
test = test.drop(outlier_indices)

# Check the shape of the test data after removing outliers
print("Shape of test after outlier handling:", test.shape)


# # Assuming you have the original columns stored elsewhere (e.g., `X_train_original`)
# X_train['VehBCost'] = X_train['VehBCost']
# X_train['WarrantyCost'] = X_train['WarrantyCost']

# # Now apply discretizer
# test = discretizer(X_train, test, y_train, chi_merge_list)
# test.shape


# # X_train, X_test = discretizer(X_train, X_test, y_train, chi_merge_list)
# chi_merge_list = ["VehBCost", "WarrantyCost"]
# test = discretizer(X_train, test, y_train, chi_merge_list)
# test.shape


# X_train, test = handle_missing_values(X_train, test)
# test.shape


# Call transform() with both train and test datasets
X_train_transformed, test_transformed = transform(X_train, test, transform_list)

# Now you can access the transformed test dataset
print(test_transformed.shape)



test.info()


predictions_1 = pipeline_1.predict(test)


# Create the submission DataFrame
submission_df = pd.DataFrame(data={'RefId': test.index, 'IsBadBuy': predictions_1})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")


submission_df.head()


