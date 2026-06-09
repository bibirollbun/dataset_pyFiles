import numpy as np # linear algebra
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", 100)
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')

df.sample(6)


print(df.columns,'\n',test_df.columns)


df.isna().sum()


test_df.isna().sum()


from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="YData Profiling Report")
#profile.to_file("rapor.html")  # Raporu Saving html
profile


categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

print('Categorcla :',categorical_columns,'\n','Numerical  :',numerical_columns)


df[numerical_columns].info()


df[categorical_columns].info()


# Uniq Values of Categorical
for col in categorical_columns:
    print(col,'\n',df[col].unique(),'\n',"-" * 30)  


# Uniq Values of 
for col in numerical_columns:
    print(col,'\n',df[col].unique(),'\n',"-" * 30)  


df['Premium Amount'].describe().T


def Categorical_Encode(df):
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # If no categorical columns are found, print a warning and return the DataFrame
    if len(categorical_columns) == 0:
        print("Warning: No categorical columns found in the DataFrame.")
        return df
    
    # Fill missing values with the most frequent value using SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    
    # Apply Label Encoding to each categorical column
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df


def fix_dates(df, date_column='Policy Start Date'):
    # Check if the specified date column exists in the DataFrame
    if date_column not in df.columns:
        print(f"Warning: '{date_column}' column not found in the DataFrame.")
        return df
    
    # Convert the date column to datetime format, coercing errors to NaT (Not a Time)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Check for missing or invalid dates in the date column
    if df[date_column].isnull().any():
        print(f"Warning: Missing or invalid dates found in the '{date_column}' column.")
    
    # Extract year, month, and day from the date column
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    
    # Drop the original date column
    df.drop(date_column, axis=1, inplace=True)
    
    return df



def fill_missing_with_iqr(df):
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    if len(numeric_columns) == 0:
        print("UyarÄ±: DataFrame'de numeric kolon bulunamadÄ±.")
        return df
    
    for col in numeric_columns:
        if df[col].isnull().any():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            random_values = np.random.uniform(lower_bound, upper_bound, size=df[col].isnull().sum())
            df.loc[df[col].isnull(), col] = random_values
    
    return df


#date
clean_train = fix_dates(df)
clean_test = fix_dates(test_df)


# categorical
clean_train = Categorical_Encode(clean_train)
clean_test  =Categorical_Encode(clean_test)

# numeric
clean_train = fill_missing_with_iqr(clean_train) 
clean_test  = fill_missing_with_iqr(clean_test)


df.head()


clean_train.head()


clean_train.info()


clean_train.isna().sum()


important_columns = ['Annual Income', 'Health Score', 'Credit Score', 'Premium Amount']

for col in important_columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    sns.histplot(clean_train[col], kde=True, color='blue', ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequency')
    
    # Boxplot
    sns.boxplot(x=clean_train[col], color='red', ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    axes[1].set_xlabel(col)
    
    # Violin Plot
    sns.violinplot(x=clean_train[col], color='green', ax=axes[2])
    axes[2].set_title(f'Violin Plot of {col}')
    axes[2].set_xlabel(col)
    
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(clean_train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Corelasyon Heatmap')
plt.show()


from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import time

def algo_test_large(x, y, sample_size=None):
   
    # Use a sample of the data if sample_size is provided, otherwise use the full dataset
    if sample_size:
        sample_data = x.sample(n=sample_size, random_state=42)
        sample_target = y.loc[sample_data.index]
    else:
        sample_data = x
        sample_target = y

    # Define models to test
    models = {
        'Linear': LinearRegression(n_jobs=-1),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(n_jobs=-1),
        'LightGBM': LGBMRegressor(n_jobs=-1),
        'SGD': SGDRegressor(),
        'Random Forest': RandomForestRegressor(n_jobs=-1)
    }

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(sample_data, sample_target, test_size=0.1, random_state=42)
    results = []

    # Train and evaluate each model
    for name, model in models.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        training_time = time.time() - start_time
        results.append((name, r2, rmse, mae, training_time))

    # Create a DataFrame with the results
    result_df = pd.DataFrame(results, columns=['Model', 'R_Squared', 'RMSE', 'MAE', 'Training Time (s)'])
    return result_df.sort_values('R_Squared', ascending=False)


x = clean_train.drop(columns=['id', 'Premium Amount'])
y = clean_train['Premium Amount']

test = clean_test.drop(columns=['id'])


print(x.shape,test.shape)
print(x.columns.tolist(),'\n',test.columns.tolist()) # same  , its good


#  If the columns are different, you can follow this step to make them the same:
test = test.reindex(columns=x.columns, fill_value=0)


results = algo_test_large(x, y)
print(results)


from lightgbm import LGBMRegressor
model = LGBMRegressor(n_jobs=-1, random_state=42)
model.fit(x, y)

prediction = model.predict(test)


prediction


submission = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')

submission['Premium Amount'] = prediction

submission.to_csv('result.csv', index=False)

print("Submission dosyasÄ± kaydedildi: 'submission.csv'")


import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = model.feature_importances_
feature_names = x.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    min_data_in_leaf=20,
    lambda_l1=0.0,
    lambda_l2=0.0,
    random_state=42,
    n_jobs=-1
)

model.fit(x, y)

new_pred = model.predict(test)


submission = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')

submission['Premium Amount'] = new_pred

submission.to_csv('result_New.csv', index=False)

