import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import warnings


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #, TargetEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



pd.set_option('display.max_columns',90)
# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Path to the directory
dir_path = '/kaggle/input/equity-post-HCT-survival-predictions/'

# Reading the Training data
train_df = pd.read_csv(dir_path+'train.csv', low_memory = False)
# Reading the Testing data
test_df = pd.read_csv(dir_path+'test.csv')


# Viewing the information of the data frame
train_df.info()


# The Nulls exist in the dataframe column wise
train_df.isna().sum()


train_df.duplicated().sum()


train_df.drop('ID', axis='columns',inplace=True)


# Counting the Numeric and Non-Numeric columns in the dataFrame

column_types = {'Numeric': 0, 'Non_Numeric':0} # Defined a function to store the counts

# Creating an Empty list to store the numeric columns
numeric_columns = [] 

# Loop through the columns of a dataframe
for col in train_df.columns:

    if pd.api.types.is_numeric_dtype(train_df[col]):
        column_types['Numeric'] += 1 # Applying the incriment for the Numeric column counts
        numeric_columns.append(col)
    else:
        column_types['Non_Numeric'] += 1 # Applying the incriment for the Non-Numeric column counts


# Collecting the list of Non_numeric columns
non_numeric_columns = [item for item in train_df.columns if item not in numeric_columns]

# Removing 'efs' and 'efs_time' columns from the list as they are TARGET variables
items_to_exclude = ['efs','efs_time']
numeric_columns = [item for item in numeric_columns if item not in items_to_exclude]


numeric_columns, non_numeric_columns


# Creating a bar chart
plt.bar(column_types.keys(), column_types.values(), color=['blue', 'orange'])
plt.xlabel('Column Type') # Adding label to the X-axis
plt.ylabel('Count')       # Adding label to the Y-axis
plt.title('Count of Numeric and Non-Numeric Columns')  # Title of the Visual


# Display value counts on top of each bar
for i, (key, value) in enumerate(column_types.items()):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()


# Creating a function to fill NaNs in the dataframe 

def fill_missing_values(train_df, test_df, num_cols=numeric_columns):
    """
    Fill missing values in the training and test DataFrames.
    
    Categorical columns in the test DataFrame will have missing values filled 
    with the mode from the training DataFrame, and numerical columns will be filled 
    with the median from the training DataFrame.

    Parameters:
    train_df (pd.DataFrame): Input training DataFrame with missing values.
    test_df (pd.DataFrame): Input test DataFrame with missing values.

    Returns:
    tuple: (train_df, test_df) - DataFrames with missing values filled.
    """
    
    # Fill missing values in categorical columns in train_df with mode
    for col in train_df.select_dtypes(include=['object', 'category']).columns:
        mode_value = train_df[col].mode()[0]
        train_df[col].fillna(mode_value, inplace=True)
        # Fill missing values in test_df with mode from train_df
        test_df[col].fillna(mode_value, inplace=True)

    # Fill NaNs in numerical columns in train_df with median
    numerical_cols = num_cols
    imputer = SimpleImputer(strategy='median')
    
    # Fit imputer on train_df and transform both train and test dataframes
    train_df[numerical_cols] = imputer.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = imputer.transform(test_df[numerical_cols])
    
    return train_df, test_df



# Filling the NaNs in the training and testing dataframes
train_df, test_df = fill_missing_values(train_df, test_df)


# As it is a binary classification task we are seeing the distribution of class
sns.countplot(x = 'efs', data = train_df)
plt.title('Distribution of "efs" (Target) among all patients')
plt.show()


# Plotting the histogram for all the numeric columns in the DF
train_df.hist(figsize=(15, 15), bins=30)
plt.suptitle('Histograms of Numeric Features')
plt.show()


# Create a figure and axis array for subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# List of columns to plot
cols = ['donor_age', 'age_at_hct', 'year_hct', 'karnofsky_score']

# Iterate over the columns and axes to plot each column in a separate subplot
for i, col in enumerate(cols):
    row = i // 2
    col_idx = i % 2
    sns.kdeplot(data=train_df, x=col, hue='efs', fill=True, ax=axs[row, col_idx])
    axs[row, col_idx].set_title(f'Distribution of {col} by EFS Outcome')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# Initial Correlation matrix

initial_cor = train_df[numeric_columns].corr()

plt.figure(figsize=(15, 15))
sns.heatmap(initial_cor, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


def apply_pca(train_df, test_df, regex_pattern='^hla', n_components=3):
    """
    Apply PCA to the columns that match the given regex pattern.

    Parameters:
    train_df (DataFrame): The input training DataFrame.
    test_df (DataFrame): The input test DataFrame.
    regex_pattern (str): The regex pattern to filter columns.
    n_components (int): The number of PCA components to keep.

    Returns:
    tuple: A tuple containing the transformed train_df and test_df.
    """
    # Filter columns based on the regex pattern
    train_hla_cols = train_df.filter(regex=regex_pattern)
    train_hla_features = train_hla_cols.columns

    test_hla_cols = test_df.filter(regex=regex_pattern)
    test_hla_features = test_hla_cols.columns

    # Ensure both dataframes have the same HLA columns
    common_hla_features = list(set(train_hla_features) & set(test_hla_features))

    # Handle NaN values (impute with 0)
    X_train_hla = train_df[common_hla_features].fillna(0)
    X_test_hla = test_df[common_hla_features].fillna(0)

    # Apply PCA on training data
    pca = PCA(n_components=n_components)
    X_train_hla_pca = pca.fit_transform(X_train_hla)

    # Transform test data using the fitted PCA
    X_test_hla_pca = pca.transform(X_test_hla)

    # Create DataFrames for the transformed features
    X_train_hla_pca_df = pd.DataFrame(X_train_hla_pca,
                                      columns=[f'PCA_{i+1}' for i in range(n_components)],
                                      index=train_df.index)  # Preserve index
    X_test_hla_pca_df = pd.DataFrame(X_test_hla_pca,
                                     columns=[f'PCA_{i+1}' for i in range(n_components)],
                                     index=test_df.index)  # Preserve index

    # Check explained variance
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Add transformed features back to the original datasets
    train_df_pca = train_df.drop(columns=common_hla_features).join(X_train_hla_pca_df)
    test_df_pca = test_df.drop(columns=common_hla_features).join(X_test_hla_pca_df)

    return train_df_pca, test_df_pca


df_pca_train, df_pca_test = apply_pca(train_df, test_df)


for col in ['donor_age', 'age_at_hct', 'year_hct', 'karnofsky_score']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, x='efs', y=col)
    plt.title(f'Boxplot of {col} by EFS')
    plt.show()


for col in ['donor_age', 'age_at_hct']:
    sns.violinplot(data=train_df, x='efs', y=col)
    plt.title(f'Violin Plot of {col} by EFS')
    plt.show()


for col in ['dri_score', 'psych_disturb', 'diabetes', 'prim_disease_hct', 'gvhd_proph']:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train_df, x=col, hue='efs')
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col} by EFS')
    plt.show()


crosstab = pd.crosstab(train_df['sex_match'], train_df['efs'], normalize='index')
crosstab.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(6, 4))
plt.title('Stacked Bar Chart for Sex Match vs EFS')
plt.ylabel('Proportion')
plt.legend(title='EFS')
plt.show()


! pip install lifelines


from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 5))

for event in [0, 1]:
    mask = train_df['efs'] == event
    kmf.fit(train_df.loc[mask, 'efs_time'], event_observed=train_df.loc[mask, 'efs'])
    kmf.plot(label=f'EFS={event}')

plt.title('Kaplan-Meier Survival Curves')
plt.ylabel('Survival Probability')
plt.xlabel('Time')
plt.show()


from lifelines import CoxPHFitter

cph = CoxPHFitter()
train_df_cox = train_df[['efs_time', 'efs'] + ['donor_age', 'age_at_hct', 'karnofsky_score']]
cph.fit(train_df_cox, duration_col='efs_time', event_col='efs')
cph.print_summary()
cph.plot()


# With the help of the dictionary calculating the unique number of categories in each column
unq_values = {}

for col in train_df.columns:
    if train_df[col].dtype == object:
        unq_values[col] = len(train_df[col].unique())


# Filtering the columns based on category values

binary_cols = [col for col, val in unq_values.items() if val <= 2]  # Apply binary encoding

ordinal_cols = [col for col, val in unq_values.items() if val > 2 and val < 4 ]  # Apply one hot encoding

target_cols = [col for col, val in unq_values.items() if val >= 4]  # Apply target encoding

binary_cols, ordinal_cols, target_cols


def preprocess_data(train_df, test_df, ordinal_cols, target_cols):
    """
    Preprocesses train and test data by applying categorical encoding and handling missing values.
    
    Parameters:
    train_df (pd.DataFrame): Training dataset
    test_df (pd.DataFrame): Test dataset
    ordinal_cols (list): List of ordinal columns to one-hot encode
    target_cols (list): List of columns for target encoding
    
    Returns:
    pd.DataFrame, pd.DataFrame: Processed train and test datasets
    """
    # Define mapping dictionaries
    mappings = {
        'prod_type': {'BM': 0, 'PB': 1},
        'graft_type': {'Bone marrow': 0, 'Peripheral blood': 1},
        'vent_hist': {'No': 0, 'Yes': 1},
        'rituximab': {'No': 0, 'Yes': 1},
        'mrd_hct': {'Negative': 0, 'Positive': 1},
        'in_vivo_tcd': {'No': 0, 'Yes': 1},
        'melphalan_dose': {'N/A, Mel not given': 0, 'MEL': 1}
    }
    
    # Apply mapping to both train and test datasets
    for col, mapping in mappings.items():
        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping)
    
    # Handle missing values
    train_df[list(mappings.keys())] = train_df[list(mappings.keys())].fillna(-1)
    test_df[list(mappings.keys())] = test_df[list(mappings.keys())].fillna(-1)
    
    # One-Hot Encoding for ordinal columns
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    encoded_train = pd.DataFrame(one_hot_encoder.fit_transform(train_df[ordinal_cols]),
                                 columns=one_hot_encoder.get_feature_names_out(ordinal_cols))
    train_df = train_df.drop(ordinal_cols, axis=1).join(encoded_train)
    
    encoded_test = pd.DataFrame(one_hot_encoder.transform(test_df[ordinal_cols]),
                                columns=one_hot_encoder.get_feature_names_out(ordinal_cols))
    test_df = test_df.drop(ordinal_cols, axis=1).join(encoded_test)
    
    # Target Encoding
    target_encoder = ce.TargetEncoder(cols=target_cols)
    
    #target_encoder = TargetEncoder()
    train_df[target_cols] = target_encoder.fit_transform(train_df[target_cols], train_df['efs'])
    test_df[target_cols] = target_encoder.transform(test_df[target_cols])
    
    return train_df, test_df


train_df , test_df = preprocess_data(df_pca_train, df_pca_test, ordinal_cols, target_cols)


# Correlation matrix after transforming the data
initial_cor = train_df.corr()

plt.figure(figsize=(50, 50))
sns.heatmap(initial_cor, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('The final Correlation Matrix')
plt.show()


train_df.describe()


# Converting all numerics to float data type for smoother calculation
X = train_df.drop(columns=['efs', 'efs_time']).astype(float)  # Drop target columns
y = train_df['efs']  # Target variable

# Compute mutual information
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

# Store and sort the results
mi_results = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_results = mi_results.sort_values(by='MI Score', ascending=False)

# Display results
print(mi_results)



mi_results = mi_results.sort_values(by='MI Score', ascending=True)

# Create a bar chart using Matplotlib
plt.figure(figsize=(12, 12))
plt.barh(mi_results['Feature'], mi_results['MI Score'], color='skyblue')

# Add labels and title
plt.xlabel('Mutual Information Score')
plt.ylabel('Categorical Features')
plt.title('Mutual Information Scores for all Features')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=-45)

# Adjust layout to add more space for labels if necessary
plt.tight_layout()

# Show the plot
plt.show()



# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy:.2f}')
print('The classification report is')
print(classification_report)


# Extract feature importance
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(12, 12))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Random Forest Model')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.tight_layout()
plt.show()




