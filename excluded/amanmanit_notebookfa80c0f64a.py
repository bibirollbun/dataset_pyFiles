import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')



train_df.head()


print(f"\033[1mShape of the training data is:\033[0m {train_df.shape}")



# Examine data types
train_df.info ()


def analyze_object_columns(df):
    """
    Analyze all object/string columns in the given DataFrame.

    For each column:
    - Prints the column name
    - Shows number of missing (NaN) values
    - Displays the count of unique values
    - Lists the first 12 unique values (if more, adds ellipsis)
    """

    # Select all columns with data type 'object' or 'string'
    obj_cols = df.select_dtypes(include=['object', 'string'])

    # Loop through each object/string column
    for col in obj_cols.columns:
        print(f"\n--- Column: {col} ---")

        # Print number of missing values
        print(f"Missing values      : {df[col].isna().sum()}")

        # Print number of unique values
        print(f"Unique values count: {df[col].nunique()}")

        # Print first 10 unique values (if available)
        unique_vals = df[col].dropna().unique()
        preview = unique_vals[:12]
        suffix = ' ...' if len(unique_vals) > 10 else ''
        print(f"Sample unique values: {preview}{suffix}")

analyze_object_columns(train_df)


def analyze_object_columns(df):
    """
    Analyze all object/string columns in the given DataFrame.

    For each column:
    - Prints the column name
    - Shows number of missing (NaN) values
    - Displays the count of unique values
    - Lists the first 10 unique values (if more, adds ellipsis)
    """

    # Select all columns with data type 'object' or 'string'
    obj_cols = df.select_dtypes(include=['number'])

    # Loop through each object/string column
    for col in obj_cols.columns:
        print(f"\n--- Column: {col} ---")

        # Print number of missing values
        print(f"Missing values      : {df[col].isna().sum()}")

        # Print number of unique values
        print(f"Unique values count: {df[col].nunique()}")

        # Print first 10 unique values (if available)
        unique_vals = df[col].dropna().unique()
        preview = unique_vals[:10]
        suffix = ' ...' if len(unique_vals) > 10 else ''
        print(f"Sample unique values: {preview}{suffix}")

analyze_object_columns(train_df)


# The 'pdays' column has a value of -1 for clients not previously contacted.
# Let's create a new feature to capture this information.
train_df['contacted_before'] = (train_df['pdays'] != -1).astype(int)
test_df['contacted_before'] = (test_df['pdays'] != -1).astype(int)



train_df.drop('id', axis = 1, inplace = True)
# test_df.drop('id', axis = 1, inplace = True)


X = train_df.drop('y', axis = 1)
y = train_df['y']


numerical_features = X.select_dtypes(include = 'number').columns
categorical_features = X.select_dtypes(exclude = 'number').columns


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2, random_state=41)


numerical_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('scaler', MinMaxScaler())
    # ('pca', PCA(n_components = 5))

])

categorical_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')
preprocessor



# Create a pipeline that first preprocesses the data and then trains the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        n_estimators=6000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=41,
        n_jobs=-1,
        verbose=-1  # Suppress verbose output
    ))
])
model_pipeline



model_pipeline.fit(X, y)


# Get predicted probabilities for the positive class
y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_val, y_pred_proba)

print("ROC-AUC Score:", roc_auc)


# 0.9684848352489523


y_pred = model_pipeline.predict_proba(test_df)[:, 1]

submission=pd.DataFrame({"id": test_df["id"],
                         "y":y_pred
})

submission.to_csv('submission.csv',index=False)

