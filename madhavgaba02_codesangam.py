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


# some basic libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



# loading the dataset

train_dataset=pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
test_dataset=pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/test.csv')

print("âœ… Data Loaded Successfully!")


train_dataset.head()


print('trainig data shape: ',train_dataset.shape)
print('test data shape: ',test_dataset.shape)


#training dataset null values

train_dataset.isnull().sum() 


#test dataset null values

test_dataset.isnull().sum() 


X=train_dataset.iloc[:,1:-1] #all features except 'id' and 'premium amount'
y=train_dataset.iloc[:,-1] # target feature
ids=test_dataset.iloc[:,0] # 'id' column
X_test=test_dataset.iloc[:,1:] # all columns except 'id'

columns=X.columns
columns


# identifying categorical and numerical columns for preprocessing

num_cols=[col for col in columns if X[col].dtypes in ('int64','float64')]
cat_cols=[col for col in columns if col not in num_cols]
print(f'numercial columns: {num_cols}')
print()
print(f'categorical_columns: {cat_cols}')


#calculating missing data percentage for training dataset

missing_values=X.isnull().sum()
missing_percentage= (missing_values/len(X))*100
missing_percentage.round(2)


# calculating missing data percentage for test dataset

missing_values_test=X_test.isnull().sum()
missing_percentage_test= (missing_values_test/len(X))*100
missing_percentage_test.round(2)


# Categorize columns by missing percentage

low_missing=[] # <=5%
moderate_missing=[] # >5% and <=15%
high_missing=[] # >15%

for i in range(18):
    if(missing_percentage[i]<=5):
        low_missing.append(columns[i])
    elif(missing_percentage[i]>5 and missing_percentage[i]<=15):
        moderate_missing.append(columns[i])
    else:
        high_missing.append(columns[i])

print("\nLow missing:", low_missing)
print("\nModerate missing:", moderate_missing)
print("\nHigh missing:", high_missing)


# code for handeling numerical missing values

def handle_numeric_missing(train_df,test_df):

    #low missing -> mean
    for i in low_missing:
        if ( i in num_cols):
            temp_mean=train_df[i].mean()
            train_df[i].fillna(temp_mean,inplace =True)
            test_df[i].fillna(temp_mean,inplace=True)

    # moderate missing -> median
    for i in moderate_missing:
        if ( i in num_cols):
            temp_median=train_df[i].median()
            train_df[i].fillna(temp_median,inplace=True)
            test_df[i].fillna(temp_median,inplace=True)

    #high missing -> drop
    train_df.drop(columns=high_missing,inplace=True,errors='ignore')
    test_df.drop(columns=high_missing,inplace=True,errors='ignore')
    
    return train_df,test_df


#code for handeling categorical missing values

def handle_categorical_missing(train_df,test_df):

    for i in low_missing + moderate_missing:
        if i in cat_cols:
            temp_mode=train_df[i].mode()[0]
            train_df[i].fillna(temp_mode, inplace= True)
            test_df[i].fillna(temp_mode, inplace= True)
    
    #train_df[col].fillna('Unknown', inplace=True)
    #test_df[col].fillna('Unknown', inplace=True)
    return train_df,test_df


def feature_engineering_numeric(train_df, test_df):
    """
    Adds numeric interaction features to both train and test datasets.
    """
    for df in [train_df, test_df]:
        if 'Annual Income' in df.columns and 'Age' in df.columns:
            df['Income_per_Age'] = df['Annual Income'] / df['Age']
        if 'Age' in df.columns and 'Health_score' in df.columns:
            df['Age_Health'] = df['Age'] * df['Health_score']

    return train_df, test_df




def process_datetime(train_df,test_df):
    # Convert to datetime
    col = 'Policy Start Date'
    train_df[col] = pd.to_datetime(train_df[col])
    test_df[col] = pd.to_datetime(test_df[col])

    for df in [train_df, test_df]:
        df[f'{col}_Year'] = df[col].dt.year
        df[f'{col}_Month'] = df[col].dt.month
        df[f'{col}_Day'] = df[col].dt.day
        df[f'{col}_Weekday'] = df[col].dt.weekday
        df[f'{col}_Hour'] = df[col].dt.hour
        df[f'{col}_Minute'] = df[col].dt.minute
        df[f'{col}_Second'] = df[col].dt.second

    # Drop original datetime column
    train_df.drop(columns=[col], inplace=True)
    test_df.drop(columns=[col], inplace=True)

    return train_df, test_df


# Skewness of numerical columns

skew_values= train_dataset[num_cols].skew()
print('skewness of numerical columns: ')
print(skew_values)


plt.figure(figsize=(18,10))

for i,col in enumerate(num_cols):
    plt.subplot(2,4,i+1)
    sns.boxplot(y=train_dataset[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()

plt.show()


def compute_outlier_bounds(df,lower_percentile=0.01,upper_percentile=0.99):

    lower_bounds=df.quantile(lower_percentile)
    upper_bounds= df.quantile(upper_percentile)
    
    return lower_bounds, upper_bounds


def cap_outliers(train_df, test_df, lower_bounds, upper_bounds):
    """
    Caps outliers in numeric columns for both train and test datasets using provided bounds.
    """
    train_df_capped = train_df.copy()
    test_df_capped = test_df.copy()

    for col in lower_bounds.index:  # assuming lower_bounds and upper_bounds are pandas Series
        if col in train_df_capped.columns:
            train_df_capped[col] = np.clip(train_df_capped[col], lower_bounds[col], upper_bounds[col])
        if col in test_df_capped.columns:
            test_df_capped[col] = np.clip(test_df_capped[col], lower_bounds[col], upper_bounds[col])

    return train_df_capped, test_df_capped



def preprocess(X,X_test):
    
    X,X_test=handle_numeric_missing(X,X_test)
    X,X_test=handle_categorical_missing(X,X_test)
    X,X_test=feature_engineering_numeric(X,X_test)
    X,X_test= process_datetime(X,X_test)

    global columns,num_cols,car_cols
    columns=X.columns
    
    num_cols=[col for col in columns if X[col].dtypes in ('int64','float64')]
    cat_cols=[col for col in columns if col not in num_cols]

    lower_bounds,upper_bounds= compute_outlier_bounds(X[num_cols])
    X,X_test=cap_outliers(X,X_test,lower_bounds,upper_bounds)

    return X,X_test, cat_cols,num_cols,lower_bounds,upper_bounds

X,X_test,cat_cols,num_cols,lower_bounds,upper_bounds=preprocess(X,X_test)


# Libraries for Preprocessing

from sklearn.pipeline import Pipeline       
from sklearn.compose import ColumnTransformer           
from sklearn.preprocessing import StandardScaler, PowerTransformer ,OrdinalEncoder


# Numeric Columns Pipeline
from sklearn.impute import SimpleImputer
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))      # gonna fill with mean if any nan value remains
])


# Categorical Columns Pipeline

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),      # # gonna fill with 'unknown' if any nan value remains
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))        # ordinal encode categorical features
])



# Combine Numeric & Categorical Pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),    # Apply numeric pipeline to numeric columns
        ('cat', cat_transformer, cat_cols)     # Apply categorical pipeline to categorical columns
    ]
)



from xgboost import XGBRegressor 


# Pipeline : preprocessing + Random Forest Regressor

xgb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('xgb', XGBRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,))
])


from sklearn.model_selection import RandomizedSearchCV, KFold 
from scipy.stats import randint, uniform

# --- Define the Hyperparameter Search Space ---
param_dist = {
    'xgb__max_depth': randint(5, 8),
    'xgb__subsample': uniform(0.6, 0.4),
    'xgb__colsample_bytree': uniform(0.6, 0.4),
    'xgb__reg_lambda': uniform(0.5, 2.5),
    'xgb__learning_rate': [0.01, 0.02, 0.05],
}

# --- Prepare KFold for the Search ---
kf = KFold(n_splits=3, shuffle=True, random_state=42) 

# --- Set up and Run a Much Faster RandomizedSearchCV ---
rs_cv = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=10,         \
    cv=kf,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("Starting FAST Randomized Search for best hyperparameters...")

rs_cv.fit(X, y)

# --- Print the Best Results ---
print("\n===== Randomized Search Complete =====")
print(f"Best parameters found: {rs_cv.best_params_}")
best_rmse = -rs_cv.best_score_
print(f"Best cross-validated RMSE: {best_rmse:.4f}")


final_params = {
    'colsample_bytree': 0.7801997007878172,
    'learning_rate': 0.02,
    'max_depth': 5,
    'reg_lambda': 2.9140800826863984,
    'subsample': 0.9233589392465844,
    'n_estimators': 500,  # A reasonable number, since we don't have early stopping here
    'random_state': 42,
    'n_jobs': -1
}

final_xgb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('xgb', XGBRegressor(**final_params))
])


#Train the final model on ALL your training data

print("Training the final model on all data...")
final_xgb_pipeline.fit(X, y)
print("Final model is ready!")



#predictions on test dataset

predictions= final_xgb_pipeline.predict(X_test)


# Prepare submission DataFrame and save CSV

submission = pd.DataFrame({
    'id': ids, 
    'Premium Amount': predictions   
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

