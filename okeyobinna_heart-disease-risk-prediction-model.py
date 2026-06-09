# Importing the libraries needed for Data exploration, cleaning, visualising, preprocessing, model building and evaluation
import pandas as pd  # for Data exploration, cleaning and manipulation
import numpy as np
import seaborn as sns  # for Data visualisation
import matplotlib.pyplot as plt # for Data visualisation
from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from category_encoders import BinaryEncoder


# Reading and converting the 'heart_train' CSV file to a pandas DataFrame
# This will ensure and ease the Data cleaning, manipulation,visualization, preprocessing and model building
heart_disease_df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv')


# Accessing the number of rows and columns in the DataFrame
# This will provide an overview of the number of rows and columns in the DataFrame
heart_disease_df.shape


# Displaying the first 5 rows of the DataFrame 
heart_disease_df.head(5)


# Displaying the Datatypes of each column in the DataFrame
# This will provide information about the number of numerical and categorical columns in the DataFrame
# Categorical columns will be encoded, and Numerical columns will be standardised..  
heart_disease_df.info()


# Checking for duplicate rows within the DataFrame
# The presence of duplicate rows will greatly affect the data quality and integrity.
# As a result, duplicate rows will be deleted to preserve the data quality and ensure that a credible model is built.
heart_disease_df.duplicated().sum()


# Checking for missing values in the DataFrame
# The presence of missing values will affect the data quality and integrity of the DataFrame
# Missing values will be addressed by filling them with generated or deleting them.
heart_disease_df.isna().sum()


# Creating a copy of heart_disease
heart_disease_cleaning_df = heart_disease_df.copy()


# Displaying the statistical summary of all numerical columns in the DataFrame. This summary is used to detect anomalies or abnormal values, which can guide the data cleaning process.
# A value of 0 in the 'Cholesterol' column is considered invalid, as it's medically impossible to have 0 mg/dL cholesterol.
# A value of -2.6 in the 'Oldpeak' column is also abnormal; valid Oldpeak values in clinical datasets typically range from 0 to around 6.
# Based on this review, both 'Cholesterol' and 'Oldpeak' will undergo data cleaning and correction.
heart_disease_cleaning_df.describe()


# This code plots the distribution of the 'Cholesterol' and 'Oldpeak' columns to visually inspect their value distributions, check for skewness, and detect potential outliers.
# The warnings module is used to suppress any FutureWarnings related to plotting.

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)

    insight_columns = ['Cholesterol', 'Oldpeak']

    fig, ax = plt.subplots(1, 2, figsize=(9, 5))
    ax = ax.flatten()

    for idx, col in enumerate(insight_columns):
        sns.histplot(data=heart_disease_cleaning_df, x=col, ax=ax[idx], kde=True)
        ax[idx].set_title(f'{col} distribution plot')
    
    plt.tight_layout()
    plt.show()


def clean_cholesterol_and_oldpeak_features(df):
    """
    Cleans the 'Cholesterol' and 'Oldpeak' columns in a DataFrame.
    
    - Replaces 'Cholesterol' values of 0 (which are invalid) with the median of non-zero values.
    - Converts negative 'Oldpeak' values to positive (by taking absolute value), assuming they are entry errors.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Cholesterol' and 'Oldpeak' columns.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df.copy()

    # Replace Cholesterol values of 0 with the median of valid (non-zero) values
    if 'Cholesterol' in df.columns:
        valid_chol = df.loc[df['Cholesterol'] > 0, 'Cholesterol']
        median_chol = valid_chol.median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_chol

    # Convert negative Oldpeak values to positive
    if 'Oldpeak' in df.columns:
        df.loc[df['Oldpeak'] < 0, 'Oldpeak'] = df.loc[df['Oldpeak'] < 0, 'Oldpeak'].abs()

    return df

heart_disease_cleaning_df = clean_cholesterol_and_oldpeak_features(heart_disease_cleaning_df) 


# This code plots the distribution of numerical columns with continuous data, to visually inspect their value distributions, check for skewness, and detect potential outliers.
# The warnings module is used to suppress any FutureWarnings related to plotting.
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    
    insight_columns = ['Cholesterol', 'Oldpeak', 'Age', 'RestingBP', 'MaxHR']

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax = ax.flatten()

    for idx, col in enumerate(insight_columns):
        sns.histplot(data=heart_disease_cleaning_df, x=col, ax=ax[idx], kde=True)
        ax[idx].set_title(f'{col} distribution plot')

    for j in range(len(insight_columns), len(ax)):
        fig.delaxes(ax[j])
    
    plt.tight_layout()
    plt.show()


# Applying log transformation to the 'Oldpeak' column.
# The 'Oldpeak' values are right-skewed, meaning that most values are low with a few high outliers.
# Log transformation is applied to:
# - Reduce skewness and approximate a more symmetric (normal-like) distribution.
# - Improve the performance of machine learning models that assume normally distributed features.
# Adding 1 before taking the log ensures that zero values do not result in undefined (log(0)) outputs.
heart_disease_cleaning_df['Oldpeak'] = np.log(heart_disease_cleaning_df['Oldpeak'] + 1)


# Plotting the distribution of the transformed Oldpeak feature
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)

    plt.figure(figsize=(7, 5))
    sns.histplot(data=heart_disease_cleaning_df, x='Oldpeak', kde=True)
    plt.title('Transformed Oldpeak Distribution plot')
    plt.tight_layout()
    plt.show()


# Visualizing the frequency distribution of categorical features including 'HeartDisease' and 'FastingBS' using count plots.
# This provides an intuitive visual overview of the class distribution across categorical columns.
# It also aids in detecting class imbalance or unusual patterns in feature values that may influence preprocessing or modeling decisions.
# All object-type columns are considered categorical.
# 'HeartDisease' and 'FastingBS', though numeric, are treated as categorical for visualization.

# Create a list of all object-type (categorical) columns
count_columns = heart_disease_cleaning_df.select_dtypes(include=['object']).columns.tolist()

# Manually include additional relevant categorical columns
count_columns.extend(['HeartDisease', 'FastingBS'])

# Extract only the selected categorical columns
count_df = heart_disease_cleaning_df.loc[:, count_columns]

# Convert 'HeartDisease' and 'FastingBS' from int to string for consistent plotting
count_df = count_df.astype({'HeartDisease': 'str', 'FastingBS': 'str'})

# Prepare a 2x4 subplot layout to accommodate up to 8 plots
fig, ax = plt.subplots(2, 4, figsize=(14, 10))
ax = ax.flatten()

# Iterate through each column and plot its value counts using Seaborn's countplot
for idx, col in enumerate(count_columns):
    sns.countplot(data=count_df, x=col, ax=ax[idx])
    ax[idx].set_title(f'{col} Count Plot')          
    ax[idx].set_ylabel('Frequency')                

    for container in ax[idx].containers:
        ax[idx].bar_label(container, fontsize=9)

for j in range(len(count_columns), len(ax)):
    fig.delaxes(ax[j])

# Adjust subplot spacing for clarity
plt.tight_layout()
plt.show()






# Creating a copy of the cleaning DataFrame for KNNClassifier model building
model_df = heart_disease_cleaning_df.copy()


# This function encodes categorical features in the heart_disease_cleaning_df DataFrame.
# Binary categorical features ('Sex' and 'ExerciseAngina') are label-encoded using predefined mappings.
# All remaining object-type categorical features are one-hot encoded.
# The result is a fully numeric DataFrame, which is required for computing correlations.
# correlation analysis is carried out on the encoded dataframe and presented on a heatmap

def encode_features(df):
    # Create a copy of the input DataFrame to avoid modifying the original data
    df = df.copy()

    # Define dictionaries to encode binary categorical variables
    sex_dict = {'M': 0, 'F': 1}
    exerciseangina_dict = {'Y': 0, 'N': 1}

    # Get a list of object-type columns excluding binary-encoded columns
    # This will be used for one-hot encoding
    onehot_columns = df.select_dtypes(include='object').drop(columns=['Sex', 'ExerciseAngina']).columns.tolist()

    # Apply label encoding using the mapping dictionaries
    df['Sex'] = df['Sex'].map(sex_dict)
    df['ExerciseAngina'] = df['ExerciseAngina'].map(exerciseangina_dict)

    # Apply one-hot encoding to the remaining categorical columns
    df = pd.get_dummies(df, columns=onehot_columns)

    return df

# Apply the encoding function to the cleaned dataset
encoded_df = encode_features(heart_disease_cleaning_df)

# Compute the correlation of each feature with the target variable 'HeartDisease'
# This helps to assess the strength and direction of each featureâ€™s linear relationship with heart disease
corr_df = encoded_df.corrwith(encoded_df['HeartDisease']).to_frame()

# Rename the resulting single-column DataFrame for clarity
corr_df = corr_df.rename(columns={0: 'Correlation values'})

# Visualise the correlation values using a heatmap
# This provides a clear view of how each feature correlates with the target variable
plt.figure(figsize=(6, 7))
sns.heatmap(corr_df, annot=True)
plt.title('Correlation Heatmap Between Heart Disease and Other Features')
plt.tight_layout()
plt.show()


model_df.sample(n=9)


# Creating a list of features for binary encoding
binary_columns = ['Sex', 'ExerciseAngina']              

# Creating a list of feature for one-hot encoding
onehot_columns = model_df.select_dtypes(include='object').drop(columns=binary_columns).columns.tolist()

# Creating a list of feature for scaling/standardization
scaler_columns = model_df.select_dtypes(exclude='object').drop(columns=['HeartDisease']).columns.tolist()

# Creating the feature (X) and Target (y) variable
X = model_df.drop(columns=['HeartDisease'])
y = model_df['HeartDisease']

# Splitting the X and y variables into the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Create a preprocessing pipeline
data_preprocessor = ColumnTransformer(
    transformers = [
        ('binary_encode', BinaryEncoder(), binary_columns),
        ('onehot_encode', OneHotEncoder(), onehot_columns),
        ('Scaler', MinMaxScaler(), scaler_columns)
    ] 
)

score_func = partial(mutual_info_classif, random_state=42)            # This ensures reproductivity in feature selection.
pipeline = Pipeline([
    ('data_preprocessor', data_preprocessor),
    ('feature_selector', SelectKBest(score_func=score_func)),
    ('classification_model', KNeighborsClassifier())
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'feature_selector__k': [9, 12, 11, 10],
    'classification_model__n_neighbors': [10, 11, 15, 12, 13],
    'classification_model__metric': ['euclidean', 'manhattan'],
    'classification_model__weights': ['uniform', 'distance']
},
    


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_builder = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='accuracy',
    n_jobs=-1, 
    verbose=1
)

# Fit GridSearchCV
grid_builder.fit(X_train, y_train)

# Best model and parameters
print("\nBest Parameters:", grid_builder.best_params_)
print("Best Cross-Validation Accuracy:", grid_builder.best_score_)

# Evaluate on the test set
heart_risk_model = grid_builder.best_estimator_
y_pred = heart_risk_model.predict(X_test)

# Print the accuracy score of the model
print(f'------------------The Accuracy Score of the Model------------')
print(accuracy_score(y_test, y_pred))


# Cleaning Cholesterol and Oldpeak
def clean_cholesterol_and_oldpeak_features(df):
    """
    Cleans the 'Cholesterol' and 'Oldpeak' columns:
    - Replaces 0 in 'Cholesterol' with the median of non-zero values.
    - Converts negative 'Oldpeak' values to positive.
    """
    df = df.copy()

    if 'Cholesterol' in df.columns:
        valid_chol = df[df['Cholesterol'] > 0]['Cholesterol']
        median_chol = valid_chol.median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_chol

    if 'Oldpeak' in df.columns:
        df.loc[df['Oldpeak'] < 0, 'Oldpeak'] = df['Oldpeak'].abs()

    return df

# Transform Oldpeak using log
def clean_oldpeak(df):
    """
    Applies log transformation to the 'Oldpeak' column to reduce skewness.
    """
    df = df.copy()
    if 'Oldpeak' in df.columns:
        df['Oldpeak'] = np.log(df['Oldpeak'] + 1)
    return df

# Predict and return submission-style DataFrame
def model_prediction_df(df, model):
    """
    A trained model is used to predict target values for a DataFrame.
    Returns a new DataFrame with predictions, suitable for Kaggle submission.
    """
    prediction_values = model.predict(df)

    prediction_df = pd.DataFrame({
        'id': df.index,
        'HeartDisease': prediction_values
    })

    return prediction_df

if __name__ == '__main__':
    # Load or define your test DataFrame
    df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')
    knn_model = heart_risk_model

    # Clean and preprocess the data
    df_cleaned = clean_cholesterol_and_oldpeak_features(df)
    df_cleaned = clean_oldpeak(df_cleaned)
    
    # Generate a prediction DataFrame using the already trained model
    submission_df = model_prediction_df(df_cleaned, knn_model)

    # Save the predictions to CSV
    submission_df.to_csv("submission.csv", index=False)


submission_df

