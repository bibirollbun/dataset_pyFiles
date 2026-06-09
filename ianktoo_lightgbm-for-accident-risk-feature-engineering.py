#Lets install libraries we need
! pip install pandas numpy lightgbm


!pwd
!ls -la


# preprocessing
# check the data

# Lets define the file names
TRAIN_DATA='/kaggle/input/playground-series-s5e10/train.csv'
TEST_DATA = '/kaggle/input/playground-series-s5e10/test.csv'



# Utilities
import pandas as pd
import numpy as np
from typing import Optional, List

# utility functions
def load_data_frame_from_csv(file_name : str)->Optional[pd.DataFrame]:
    """
    Loads data from a csv file to a dataframe

    Arg:
        file_name: Name of the file
    
    Returns:
        pd.DataFrame
    
    Exceptions:
        FileNotFound
        Exception
    
    Returns:
        pd.DataFrame | None
    
    Examples:
    >>> load_data_from_csv(file_name="/kaggle/input/playground-series-s5e10/train.csv")

    """
    try:
        df = pd.read_csv(file_name)
        print(f"File '{file_name}' loaded!✅")
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found! Please check your path. ❌")
        return None
    except Exception as e:
        print(f"An unexpected error occured while loading the file. {e} ⚠️")
        return None

def encode_cyclical_features(df, column, max_val):
    """
    Description:
        Encodes a cyclical feature using sin and cosine transformations.
        We basically want features that are cylindrical in nature to wrap around, for example: 24 hrs

    Args:
        df: pd.DataFrame
        column: str
        max_value: int
    
    Returns:
        df
    """
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    df = df.drop(column, axis=1)
    return df

def category_feature_engineer(df: pd.DataFrame , category_list: List['str'])->pd.DataFrame :
    """
    Description:
        Converts pd columns to category columns given a category list.
    Args:
        df: Dataframe
        category_list: category lust
    Returns:
        df
    """
    for col in category_list:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


import pandas as pd
import numpy as np
from typing import Optional
 
df = load_data_frame_from_csv(TRAIN_DATA)
df.head()


# what is the column that we are targeting?
# in our case, it is the 'accident_risk' column
TARGET_COLUMN = 'accident_risk'
ID_COLUMN = 'id'

# lets now separate our data so that we can handle them separately using df
# Features will be saved as (X) and target will be saved as (y)
X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
y = df[TARGET_COLUMN]

X, y


! pip install -U scikit-learn


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# split the data into test and train using sklearn.model_selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42 )


print(X_train.head(2))
print(X_train.columns)


# Feature engineering.
categorical_features = [
    'road_type', 'lighting','weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season'
]


# we feature engineer both the training and validation data
X_train = category_feature_engineer(df=X_train, category_list=categorical_features)
X_val = category_feature_engineer(df=X_val, category_list=categorical_features)

"""
for col in categorical_features:
    # check if the columns exist
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        # do the same for validation data
        X_val[col] = X_val[col].astype('category')
        # do the same for validation data
"""
X_train.dtypes, X_val.dtypes


# forgot to check for duplicates
print(f"Total number of duplicate rows (after first occurrence): {df.duplicated().sum()}")


import lightgbm as lgb

lgb_params = {
    'objective' : 'regression',
    'metric' : 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,   # Randomly select 80% of features for each tree
    'bagging_fraction': 0.8,   # Randomly select 80% of data for each tree
    'bagging_freq': 1,         # Perform bagging every iteration
    'verbose': -1,             # Suppress verbose output
    'n_jobs': -1,              # Use all CPU cores
    'seed': 42

}

# Initialize and train the LightGBM model
model = lgb.LGBMRegressor(**lgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    # Use early stopping to prevent overfitting
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

print("Model training complete. ✅")


from sklearn.metrics import mean_squared_error

# Make predictions on the validation set
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration_)

# LightGBM can sometimes predict values slightly outside this range.
y_pred_val_clipped = np.clip(y_pred_val, 0, 1)

# Calculate MSE first
mse = mean_squared_error(y_val, y_pred_val_clipped)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Validation
print(f"\nValidation RMSE: {rmse:.4f}")


# load the test files
df_test = load_data_frame_from_csv('/kaggle/input/playground-series-s5e10/test.csv')

test_ids = df_test[ID_COLUMN]
X_test = df_test.drop(columns=[ID_COLUMN]) # Drop the ID column from features
X_test = category_feature_engineer(df=X_test, category_list=categorical_features)
X_test.dtypes



# Generate predictions on the preprocessed test features
raw_predictions = model.predict(X_test, num_iteration=model.best_iteration_)
print("Predictions generated.")

final_predictions_clipped = np.clip(raw_predictions, 0, 1)

# --- Create the Submission DataFrame ---
# Ensure the columns match the required format: 'id', 'accident_risk'
submission = pd.DataFrame({
    ID_COLUMN: test_ids,
    TARGET_COLUMN: final_predictions_clipped.round(8)
})

# --- Save the Submission File ---
submission_file_name = 'final_submission.csv'
submission.to_csv(submission_file_name, index=False)

print("\n--- Submission File Ready ---")
print(f"File saved as: **{submission_file_name}**")
print("First 5 rows of the submission file:")
print(submission.head())

