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


import pandas as pd
import numpy as np
from math import ceil
import warnings
warnings.filterwarnings("ignore")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def read_csv(file_path):
    """
    Read data from CSV file and return a pandas DataFrame.

    Parameters:
    - file_path: str, the path to the CSV file.

    Returns:
    - pd.Dataframe, the loaded DataFrame
    """
    return pd.read_csv(file_path)

def dataset_info_statistics(data):
    """
    Display information and basics statistics about the dataset

    Parameters:
    - data: Dataframe, input data
    """

    #Display general information about the dataset 
    print("Dataset Information")
    print(data.info())
    print("\n")

    #Displat basics statistics for numerical columns
    print("Basic Statistics for numerical columns:")
    print(data.describe())
    print("\n")

def check_null(data):
    """
    Check for null values in the dataset

    Parameters:
    - data: pandas DataFrame, input data

    Returns:
    - pd.series, the count of null values for each column
    """
    print("Null Values in the Dataset:")
    return data.isnull().sum()

def check_duplicated(data):
    """
    Checking for any duplicated rows in the dataset.

    Parameters:

    - data: pandas Dataframe, input data

    Returns:
    - bool, True if duplicated rows exist, False otherwise.
    """

    return data.duplicated().any()

def plot_graph(data, figsize=(15, 10)):
    """
    Simple EDA visualization - clean histograms in grid layout

    Parameters: 
    - Pandas Datafram, input data

    Returns:
    - None
    """
    # Get all columns (numerical and categorical)
    columns = data.columns.tolist()
    
    # Calculate grid dimensions
    n_cols = len(columns)
    cols = 3  # 3 columns per row
    rows = (n_cols + cols - 1) // cols
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten() if rows > 1 else axes
    
    for i, column in enumerate(columns):
        ax = axes_flat[i]
        
        # Check if column is numerical or categorical
        if data[column].dtype in ['object', 'category']:
            # Categorical - use countplot
            data[column].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(column)
            ax.tick_params(axis='x', rotation=45)
        else:
            # Numerical - use histogram with KDE
            sns.histplot(data[column].dropna(), bins=30, kde=True, ax=ax, 
                        color='steelblue', alpha=0.7, edgecolor='white')
            ax.set_title(column)
            ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide empty subplots
    for i in range(n_cols, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def sep_feature_target(data, target_column):
    """
    Seperates features and target variables

    Parameters:
    - data: Pandas DataFrame, features
    - target_column: str, the column representing the target variable.

    Returns:
    - X : pandas Dataframe, features
    - y : pandas Series, target variable
    """

    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]

    return X, y

def perform_train_test_split(X,y, test_size=0.20, random_state=42):
    """
    Perform train test split

    Parameters:
    -X: pandas DataFrame, features.
    -y: pandas series, target variable
    -test_size: float, optional, the portion of the dataset to incluse in the test split (default is 0.2)
    -random_state: int, optional, seed for random number generation (default is 42)

    Returns:
    - X_train: pandas Dataframe, features for training.
    - X_test: pandas DataFrame, features for testing.
    - y_train: pandas Series, target variable for training.
    - y_test: pandas Series, target variable for training
    """

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test



train = read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = read_csv('/kaggle/input/playground-series-s5e5/test.csv')

data = train.copy()


plot_graph(data)


data.columns


X, y = sep_feature_target(data, 'Calories')


X = X.drop(columns=['id'])


X_train, X_test, y_train, y_test = perform_train_test_split(X, y)


X.shape


X_train.shape


X_test.shape


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


preprocessor = ColumnTransformer(transformers=[
    ('Cat' , OrdinalEncoder(), ['Sex']),
    ('num' , StandardScaler(), ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']),
],remainder='passthrough')


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


from sklearn import set_config


set_config(display='diagram')


pipeline


print(X_train.columns)



pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


r2_score(y_test,y_pred)


from sklearn.metrics import mean_absolute_error


mean_absolute_error(y_test,y_pred)


kfold = KFold(n_splits=5,shuffle=True, random_state=42)


cv_results = cross_val_score(pipeline,X,y,cv = kfold,scoring='r2')
cv_results


cv_results.mean()


def model_scorer(model_name,model):
    
    output=[]
   
    
    output.append(model_name)
    
    pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(r2_score(y_test,y_pred))
    output.append(mean_absolute_error(y_test,y_pred))
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
    output.append(cv_results.mean())
    
    return output


model_dict={
    'log':LinearRegression(),
    # 'RF':RandomForestRegressor(),
    'XGBR':XGBRegressor()
}


model_output=[]
for model_name,model in model_dict.items():
    model_output.append(model_scorer(model_name,model))


model_output


preprocessor = ColumnTransformer(transformers=[
    ('ordinal',OrdinalEncoder(),['Sex']),
    ('num',StandardScaler(),['Age',
                            'Height',
                            'Weight',
                            'Duration',
                            'Heart_Rate',
                            'Body_Temp']),
    
],remainder='passthrough')


pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',XGBRegressor())
    
])


pipeline.fit(X,y)


submission_df = test.copy()
submission_df.head(5)


# Ensure no negative predictions
preds = pipeline.predict(submission_df.drop(columns=['id']))
preds = preds.clip(min=0)  # Replace negative values with 0

# Create submission
final_submission = pd.DataFrame({
    'id': submission_df['id'],
    'Calories': preds
})
final_submission.to_csv("submission.csv", index=False)





