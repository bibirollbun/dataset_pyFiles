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
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor

# Suppress warnings (optional)
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col=["id"])
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


test.head()


train.describe(include = 'all')


test.describe(include = 'all')


sns.countplot(data=train, x='Sex')


sns.countplot(data=test, x='Sex')


def plot_hist(df1, df2, num_cols):
    for col in num_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left plot (train)
        sns.histplot(df1[col], kde=True, ax=ax1)
        ax1.set_title(f'Train - {col}')

        # Right plot (test)
        sns.histplot(df2[col], kde=True, ax=ax2)
        ax2.set_title(f'Test - {col}')

        plt.tight_layout()
        plt.show()


num_col = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

plot_hist(train, test, num_col)


train.columns


plt.figure()
sns.histplot(train['Calories'], kde=True, label='Calories')


plt.title("Calories hist")

plt.show()


for col in  ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']:
    plt.figure()
    sns.boxplot(x=train[col])
    plt.title(f'boxplot - {col}' )
    plt.show()


def remove_outliers_iqr(df, columns, threshold=1.5):
    """
    Removes outliers from specified columns using the IQR method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to process.
        threshold (float): Multiplier for IQR (default: 1.5). Higher = stricter outlier removal.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed (rows dropped).
    """
    df_clean = df.copy()
    for col in columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filter rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean.reset_index(drop=True)


train_clean = remove_outliers_iqr(train, ['Height', 'Weight', 'Heart_Rate', 'Body_Temp', 'Calories'], threshold=1)


for col in  ['Height', 'Weight', 'Heart_Rate', 'Body_Temp', 'Calories']:
    plt.figure()
    sns.boxplot(x=train_clean[col])
    plt.title(f'boxplot - {col}' )
    plt.show()


# Correlation heatmap
corr = train[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Calories']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Top correlations with target

print("Top Correlations with Listening_Time_minutes:")
print(corr['Calories'].sort_values(ascending=False))


train_clean.info()


train_df = train_clean.copy()

X = train_df.drop(columns=['Calories'])
y = train_df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#numeric_features = list(X.select_dtypes(include=['float64']).columns).drop(['time_cos','time_sin', 'day_cos', 'day_sin'])

numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns



numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
    )


'''model = Pipeline(steps=[
    ('preprocessor', preprocessor),
  ('regressor',  GradientBoostingRegressor(min_samples_leaf=5))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

#model.fit(X_train, np.log1p(y_train))
#y_pred = np.expm1(model.predict(X_test)) 

rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"Baseline RMSE: {rmse:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RÂ²: {r2_score(y_test, y_pred):.2f}")
print(f"RMSLE: {np.sqrt(mean_squared_log_error(y_test, y_pred)):.4f}")'''


'''model = Pipeline(steps=[
    ('preprocessor', preprocessor),
  ('regressor',  GradientBoostingRegressor(min_samples_leaf=5))
])

#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#y_pred = np.maximum(y_pred, 0)

model.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(model.predict(X_test)) 

rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"Baseline RMSE: {rmse:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RÂ²: {r2_score(y_test, y_pred):.2f}")
print(f"RMSLE: {np.sqrt(mean_squared_log_error(y_test, y_pred)):.4f}")'''


base_models = [
    ('gb', GradientBoostingRegressor(min_samples_leaf=5)),
    ('rf', RandomForestRegressor(max_samples=100000, n_estimators=50, min_samples_leaf=100))
]


meta_model = LinearRegression()


stack = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5, 
    n_jobs=-1
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
  ('regressor',  stack)
])


#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#y_pred = np.maximum(y_pred, 0)

model.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(model.predict(X_test)) 

rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print(f"Baseline RMSE: {rmse:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RÂ²: {r2_score(y_test, y_pred):.2f}")
print(f"RMSLE: {np.sqrt(mean_squared_log_error(y_test, y_pred)):.4f}")


plt.scatter(y_test, y_pred)


def create_kaggle_submission(
    pipeline,  # Your sklearn Pipeline (includes preprocessor + model)
    test_data,  # test data
    sample_sub,  # Kaggle's sample submission
    output_file="submission.csv",  # Output filename
    id_column="id",  # Name of ID column (e.g., 'PassengerId', 'Id')
    target_column="SalePrice"  # Name of target column (e.g., 'Survived')
):
    """
    Creates a Kaggle submission file using a pre-fitted sklearn Pipeline.
    
    Args:
        pipeline: Fitted sklearn Pipeline (preprocessor + model).
        test_file_path (str): Path to test data CSV.
        sample_submission_path (str): Path to sample submission CSV.
        output_file (str): Output submission filename.
        id_column (str): Name of the ID column in test data.
        target_column (str): Name of the target column in sample submission.
    """

    
    # Ensure ID column is preserved
    ids = test_data[id_column]
    
    # Drop ID column (if it exists in features) to avoid preprocessing issues
    if id_column in test_data.columns:
        test_features = test_data.drop(columns=[id_column])
    else:
        test_features = test_data
    
    # Predict using the pipeline (auto-applies preprocessing)
    predictions = pipeline.predict(test_features)
    
    # Format predictions to match sample submission
    
    #sample_sub[target_column] = predictions
    sample_sub[target_column] = np.expm1(predictions)
    #sample_sub[target_column] = np.maximum(predictions, 0)
    
    # Save to CSV
    sample_sub.to_csv(output_file, index=False)
    print(f"âœ… Submission file saved as '{output_file}'")
    print(f"ðŸ“Š Sample submission head:\n{sample_sub.head()}")


create_kaggle_submission(
    pipeline=model,
    test_data=test,
    sample_sub=submission,
    output_file='/kaggle/working/submission.csv',
    id_column="id",  # Adjust for your competition
    target_column="Calories"  # Adjust for your competition
)










