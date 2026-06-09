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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl





# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt

# Load each CSV file into a DataFrame
sample_submission_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
data_dictionary_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Display the first few rows of each DataFrame
print("Sample Submission DataFrame:")
print(sample_submission_df.head())


# Filter out target columns from the data dictionary
data_dictionary_df_test = data_dictionary_df[(data_dictionary_df['variable'] != 'efs_time') & (data_dictionary_df['variable'] != 'efs')]

numerical_cols_train = data_dictionary_df[data_dictionary_df['type'] == 'Numerical']['variable']
numerical_cols_test = data_dictionary_df_test[data_dictionary_df_test['type'] == 'Numerical']['variable']

# Get categorical and numerical columns
categorical_cols_train = data_dictionary_df[data_dictionary_df['type'] == 'Categorical']['variable']
categorical_cols_test = data_dictionary_df_test[data_dictionary_df_test['type'] == 'Categorical']['variable']

# Update numerical and categorical column lists after dropping columns
numerical_cols_train = [col for col in numerical_cols_train if col in train_df.columns]
categorical_cols_train = [col for col in categorical_cols_train if col in train_df.columns]

# Update numerical and categorical column lists after dropping columns
numerical_cols_test = [col for col in numerical_cols_test if col in test_df.columns]
categorical_cols_test = [col for col in categorical_cols_test if col in test_df.columns]




# Preprocessing function
def preprocessing(df, numerical_columns, categorical_columns):
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_columns:
        df[col] = df[col].fillna('unknown')

    return df

# Preprocess training and test data
train = preprocessing(train_df, numerical_cols_train, categorical_cols_train)
test = preprocessing(test_df, numerical_cols_test, categorical_cols_test)

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')



# Update numerical and categorical column lists to exclude 'efs_time' and 'efs'
numerical_cols_train = [col for col in numerical_cols_train if col not in ['efs_time', 'efs']]
categorical_cols_train = [col for col in categorical_cols_train if col not in ['efs_time', 'efs']]

# Separate features and target in the training data
X_train = train.drop(columns=["efs_time", "efs"])  # Features
y_train = train[["efs_time", "efs"]]  # Target

# Print the filtered columns to verify
print("Numerical Columns in X_train:", numerical_cols_train)
print("Categorical Columns in X_train:", categorical_cols_train)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_train),
        ('cat', categorical_transformer, categorical_cols_train)
    ])


# Fit and transform the training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Transform the test data using the same preprocessor
X_test_transformed = preprocessor.transform(test)

# Convert transformed data to DataFrame
X_train_df = pd.DataFrame(
    X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
    columns=preprocessor.get_feature_names_out()
)

X_test_df = pd.DataFrame(
    X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
    columns=preprocessor.get_feature_names_out()
)


# Standardize the features (excluding 'efs_time' and 'efs')
features = X_train_df
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Create a new DataFrame with standardized features
X_train_df = pd.DataFrame(standardized_features, columns=features.columns)

# Add back target columns to the training DataFrame
X_train_df["efs_time"] = y_train["efs_time"].values
X_train_df["efs"] = y_train["efs"].values


# Fit the Cox Proportional Hazards model with L2 regularization (Ridge)
baseline_model = CoxPHFitter(penalizer=0.001)  # Adjust the penalizer value as needed
baseline_model.fit(X_train_df, duration_col="efs_time", event_col="efs")

# Get risk scores for test data
risk_scores = baseline_model.predict_partial_hazard(X_test_df)

# Predict survival function for the first patient in the test data
survival_function = baseline_model.predict_survival_function(X_test_df.iloc[0:1])

# Plot the estimated survival function
plt.step(survival_function.index, survival_function.values.flatten(), where="post")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.title("Predicted Survival Function")
plt.show()


# Evaluate C-index on training data (since test data does not have target columns)
c_index = concordance_index(X_train_df["efs_time"], -baseline_model.predict_partial_hazard(X_train_df), X_train_df["efs"])
print("C-index on training data:", c_index)

# Generate the submission file
sub = sample_submission_df.copy()
sub["prediction"] = risk_scores
sub.to_csv("submission.csv", index=False)
print("Sub shape:", sub.shape)
sub.head()


# # Calculate the score metric
# from metric import score

# y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
# y_pred = train[["ID"]].copy()
# y_pred["prediction"] = baseline_model.predict_partial_hazard(X_train_df)

# m = score(y_true.copy(), y_pred.copy(), "ID")
# print(f"\nOverall CV for Baseline Model =", m)















































# sample_submission_df.head(50)


# train_df['efs']


# train_df['efs_time'].head(5)





# print("\nData Dictionary DataFrame:")
# print(data_dictionary_df.head())


# print(train_df.isna().sum())
# train_df.shape[0]


# print(len(train_df.columns), len(test_df.columns))

# train_columns = set(train_df.columns)
# test_columns = set(test_df.columns)

# for col in train_columns:
#     if col not in test_columns:
#         print(col)


# plt.hist(train_df.loc[train_df.efs==1,"efs_time"],bins=100,label="efs=1, Did Not Survive")
# plt.hist(train_df.loc[train_df.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Survived")
# plt.xlabel("Time of Observation, efs_time")
# plt.ylabel("Density")
# plt.title("Times of Observation. Either time to death, or time observed alive.")
# plt.legend()
# plt.show()


# print(train_df['efs'].value_counts())  # Check distribution of event occurrences
# print(train_df['efs'].nunique())  # Check how many unique survival times exist


# print(train_df['efs_time'].nunique())  # Count unique survival times
# print(train_df['efs_time'].describe())  # Check distribution


# # The error occurs because train_1 is a sparse matrix (csr_matrix) returned by ColumnTransformer.fit_transform(), but lifelines.CoxPHFitter expects a Pandas DataFrame.

# # Convert train_1 to DataFrame
# train_1_df = pd.DataFrame(train_1.toarray() if hasattr(train_1, "toarray") else train_1, 
#                           columns=preprocessor.get_feature_names_out())

# # Add back duration and event columns
# train_1_df["efs_time"] = train["efs_time"].values
# train_1_df["efs"] = train["efs"].values

# # Convert test_1 to DataFrame
# test_1_df = pd.DataFrame(test_1.toarray() if hasattr(test_1, "toarray") else test_1, 
#                          columns=preprocessor.get_feature_names_out())



# IMPLEMENTING THE COX PROPORTIONAL HAZARDS MODEL 


# !pip install lifelines


# import numpy as np
# import pandas as pd
# import os
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from lifelines import CoxPHFitter
# from lifelines.utils import concordance_index
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Load each CSV file into a DataFrame
# sample_submission_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
# data_dictionary_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
# train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
# test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# # Display the first few rows of each DataFrame
# print("Sample Submission DataFrame:")
# print(sample_submission_df.head())


# print(train_df.columns)


# print(test_df.columns)


# # Filter out target columns from the data dictionary
# data_dictionary_df_test = data_dictionary_df[(data_dictionary_df['variable'] != 'efs_time') & (data_dictionary_df['variable'] != 'efs')]

# numerical_cols_train = data_dictionary_df[data_dictionary_df['type'] == 'Numerical']['variable']
# numerical_cols_test = data_dictionary_df_test[data_dictionary_df_test['type'] == 'Numerical']['variable']


# # Get categorical and numerical columns
# categorical_cols_train = data_dictionary_df[data_dictionary_df['type'] == 'Categorical']['variable']
# categorical_cols_test = data_dictionary_df_test[data_dictionary_df_test['type'] == 'Categorical']['variable']


# # Step 1: Calculate correlation matrix for numerical columns
# corr_matrix = train_df[numerical_cols_train].corr().abs()


# # Identify highly correlated numerical columns (e.g., correlation > 0.9)
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# collinear_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
# print("Collinear numerical columns to remove:", collinear_cols)


# # Drop collinear numerical columns
# train_df = train_df.drop(columns=collinear_cols)
# test_df = test_df.drop(columns=collinear_cols)


# # Update numerical and categorical column lists after dropping columns
# numerical_cols_train = [col for col in numerical_cols_train if col in train_df.columns]
# categorical_cols_train = [col for col in categorical_cols_train if col in train_df.columns]

# # Update numerical and categorical column lists after dropping columns
# numerical_cols_test = [col for col in numerical_cols_test if col in test_df.columns]
# categorical_cols_test = [col for col in categorical_cols_test if col in test_df.columns]


# # Preprocessing function
# def preprocessing(df, numerical_columns, categorical_columns):
#     for col in numerical_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#         df[col] = df[col].fillna(df[col].median())
    
#     for col in categorical_columns:
#         df[col] = df[col].fillna('unknown')

#     return df

# # Preprocess training and test data
# train = preprocessing(train_df, numerical_cols_train, categorical_cols_train)
# test = preprocessing(test_df, numerical_cols_test, categorical_cols_test)


# # Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Preprocessing for numerical data
# numerical_transformer = SimpleImputer(strategy='constant')


# # Separate features and target in the training data
# X_train = train.drop(columns=["efs_time", "efs"])  # Features
# y_train = train[["efs_time", "efs"]]  # Target


# # Print the filtered columns to verify
# numerical_cols_train = [col for col in numerical_cols_train if col in X_train.columns]
# categorical_cols_train = [col for col in categorical_cols_train if col in X_train.columns]


# print("Numerical Columns in X_train:", numerical_cols_train)
# print("Categorical Columns in X_train:", categorical_cols_train)


# # Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols_train),
#         ('cat', categorical_transformer, categorical_cols_train)
#     ])

# # Fit and transform the training data
# X_train_transformed = preprocessor.fit_transform(X_train)

# # Transform the test data using the same preprocessor
# X_test_transformed = preprocessor.transform(test)


# # Convert transformed data to DataFrame
# X_train_df = pd.DataFrame(
#     X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
#     columns=preprocessor.get_feature_names_out()
# )

# X_test_df = pd.DataFrame(
#     X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
#     columns=preprocessor.get_feature_names_out()
# )


# # Standardize the features (excluding 'efs_time' and 'efs')
# features = X_train_df
# scaler = StandardScaler()
# standardized_features = scaler.fit_transform(features)


# # Create a new DataFrame with standardized features
# X_train_df = pd.DataFrame(standardized_features, columns = features.columns)


# # Add back target columns to the training DataFrame
# X_train_df["efs_time"] = y_train["efs_time"].values
# X_train_df["efs"] = y_train["efs"].values


# # Fit the Cox Proportional Hazards model with L2 regularization (Ridge)
# baseline_model = CoxPHFitter(penalizer=0.001)  # Adjust the penalizer value as needed
# baseline_model.fit(X_train_df, duration_col="efs_time", event_col="efs")


# # Print the summary of the fitted model
# print(baseline_model.summary)


# # Get risk scores for test data
# risk_scores = baseline_model.predict_partial_hazard(X_test_df)

# # Predict survival function for the first patient in the test data
# survival_function = baseline_model.predict_survival_function(X_test_df.iloc[0:1])


# # Plot the estimated survival function
# plt.step(survival_function.index, survival_function.values.flatten(), where="post")
# plt.xlabel("Days")
# plt.ylabel("Survival Probability")
# plt.title("Predicted Survival Function")
# plt.show()

# # Evaluate C-index on training data (since test data does not have target columns)
# c_index = concordance_index(X_train_df["efs_time"], -baseline_model.predict_partial_hazard(X_train_df), X_train_df["efs"])
# print("C-index on training data:", c_index)


# c-index without dropping collinear columns 0.677
# c-index with the median imputer values 0.6768
# c-index with the mean imputer values 0.6767
# c-index with the mode imputer values 0.6763
# c-index after using the pipeline only instead of the preprocessing function 0.6699
























































