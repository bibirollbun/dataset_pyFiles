# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install psynlig


import numpy as np
import pandas as pd

# Scikit-learn modules
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, mean_squared_error, 
    r2_score, precision_score, recall_score, f1_score
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
import plotly.express as px
from psynlig import plot_correlation_heatmap
# Bokeh for interactive plots
from bokeh.plotting import figure, show, output_notebook
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Spectral6, Viridis256
from bokeh.models import ColumnDataSource
from sklearn.model_selection import cross_val_score

# Suppress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Enable Bokeh plots in notebooks
output_notebook()

# Print confirmation
print("Libraries successfully imported and updated!")


train_data=pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
submission=pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")


train_data.info()


from sklearn.impute import SimpleImputer

def impute_by_categories(train_data,test_data, category_columns=['gender', 'education_level', 'exercise_frequency']):
    """
    Impute missing values for numeric columns based on the median value within each
    combination of specified categorical variables.
    
    Parameters:
    -----------
    df_train : pandas.DataFrame
        Training dataset
    df_test : pandas.DataFrame
        Test dataset
    category_columns : list
        List of categorical columns to group by for imputation
    
    Returns:
    --------
    df_train, df_test : tuple of pandas.DataFrame
        Processed datasets with imputed values
    """
    # Ensure numeric columns are selected, excluding 'premium_amount'
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns.drop('premium_amount')

    # Combine train and test for consistent category handling
    df_train['is_train'] = True
    df_test['is_train'] = False
    combined = pd.concat([df_train, df_test], ignore_index=True)

    # Impute missing values group by group
    imputer = SimpleImputer(strategy='median')
    for group_values, group_df in combined.groupby(category_columns):
        mask = (combined[category_columns] == pd.Series(group_values, index=category_columns)).all(axis=1)
        if mask.sum() > 0:
            combined.loc[mask, numeric_columns] = imputer.fit_transform(group_df[numeric_columns])

    # Split the combined dataset back into train and test
    df_train = combined[combined['is_train']].drop(columns='is_train')
    df_test = combined[~combined['is_train']].drop(columns='is_train')

    return train_data,test_data



train_data


test_data


train_data.isna().sum()


train_data.dropna(inplace=True)


train_data.info()


# Summary statistics
summary = train_data.describe()

# Create the heatmap
fig = px.imshow(
    summary,
    color_continuous_scale="RdYlGn",
    title="Heatmap for Summary Statistics",
    labels={"x": "Columns", "y": "Statistics"}  # Axis labels
)

fig.update_layout(
    title_font_size=20,
    xaxis_title="Data Columns",
    yaxis_title="Summary Metrics",
    xaxis_tickangle=45
)

# Show the plot
fig.show()


neumirical_data = train_data.select_dtypes(include=["float64", "int64"])
sns.pairplot(data=neumirical_data, diag_kind='kde', markers='+')
plt.show()


p = figure(
    title="Annual Income  and Premium Amount",
    x_axis_label="Annual Income",
    y_axis_label="Premium Amount",
    width=800, height=600
)
p.scatter(train_data["Annual Income"], train_data["Premium Amount"], size=8, color="navy", alpha=0.6)
output_notebook()
show(p)


sns.pairplot(
    train_data[["Annual Income", "Credit Score", "Premium Amount"]],
    hue="Premium Amount",
    palette="viridis"
)
plt.show()


numerical_columns = train_data.select_dtypes(include=['int', 'float']).columns

for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_data[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()


datetime_columns = train_data.select_dtypes(include=['object']).columns

for col in datetime_columns:
    try:
        # Convert the column to datetime format
        train[col] = pd.to_datetime(train[col], errors='raise')
        test[col] = pd.to_datetime(test[col], errors='raise')
        
        # Convert datetime to epoch time
        train[col] = train[col].astype(np.int64) / 10**9
        test[col] = test[col].astype(np.int64) / 10**9

        print(f"Converted '{col}' to epoch time.")
    except Exception:
        continue


datetime_columns = test_data.select_dtypes(include=['object']).columns

for col in datetime_columns:
    try:
        # Convert the column to datetime format
        train[col] = pd.to_datetime(train[col], errors='raise')
        test[col] = pd.to_datetime(test[col], errors='raise')
        
        # Convert datetime to epoch time
        train[col] = train[col].astype(np.int64) / 10**9
        test[col] = test[col].astype(np.int64) / 10**9

        print(f"Converted '{col}' to epoch time.")
    except Exception:
        continue


le = LabelEncoder()

for col in train_data.columns:
    if train_data[col].dtype == 'object':  
        train_data[col] = le.fit_transform(train_data[col])  


le = LabelEncoder()

for col in test_data.columns:
    if test_data[col].dtype == 'object':  
        test_data[col] = le.fit_transform(test_data[col])  


from psynlig import plot_correlation_heatmap
plt.style.use('seaborn-talk')
kwargs = {
    'heatmap': {
        'vmin': -1,
        'vmax': 1,
        'cmap': 'viridis',
    },
    'figure': {
        'figsize': (10, 8),
    },
}

plot_correlation_heatmap(train_data, bubble=True, annotate=False, **kwargs)
plt.show()


 
X_train = train_data.drop(columns=["Premium Amount"])
y_train = train_data["Premium Amount"]



imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

if test_data.isna().sum().sum() > 0:
    print("Test data still contains NaNs after imputation.")
else:
    print("Imputation completed successfully. No NaNs in test data.")


imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
test = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test_data.columns)


# Instantiate the model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)
# model.fit(X_train, y_train)

# Fit the model
model.fit(X_train, y_train)



# Predictions
test_predictions = model.predict(test)

# Model evaluation
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")


# Create submission
submission = pd.DataFrame({
    'id': test_data['id'],  # Ensure this is the correct ID column
    'Premium Amount': test_predictions
})

# Save to CSV
submission.to_csv('submission_final.csv', index=False)
print("Submission file 'submission_final.csv' created successfully!")



submission


