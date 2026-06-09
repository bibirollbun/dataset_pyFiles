import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


import os

# Base directory for the competition data
data_dir = '/kaggle/input/playground-series-s5e10'

# Initialize variables to store the full paths
train_file_path = None
test_file_path = None
submission_file_path = None

# os.walk yields (root_directory, list_of_directories, list_of_files)
for root, dirs, files_in_dir in os.walk(data_dir):
    # Iterate through the files found in the current directory (root)
    for filename in files_in_dir:
        # Construct the full path
        full_path = os.path.join(root, filename)

        # Check the file name and assign the full path
        if filename.startswith('train'):
            train_file_path = full_path
        elif filename.startswith('test'):
            test_file_path = full_path
        elif filename.startswith('sample_submission'): # Often 'sample_submission' or similar
            submission_file_path = full_path

# Display the results
print(f"Train File Path: {train_file_path}")
print(f"Test File Path: {test_file_path}")
print(f"Submission File Path: {submission_file_path}")

# Now you can use these paths to load your data with pandas:
# import pandas as pd
# df_train = pd.read_csv(train_file_path)


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
submission_df = pd.read_csv(submission_file_path)


train_df.head()


id_var = train_df['id']
train_df.drop(columns=['id'],inplace=True)
train_df.head()


train_df.info()


train_df = train_df.replace(' ', np.nan)
train_df.isna().sum()


train_df.describe()


numerical_cols = train_df.select_dtypes(include=['int64','float64']).columns
categorical_cols = train_df.select_dtypes(exclude=['int64','float64','bool']).columns
boolean_cols = train_df.select_dtypes(include=['bool']).columns


## Checking the density plot of numerical _features
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(10,8))
count = 0
for i in range(2):
    for j in range(3):
        if count == len(numerical_cols):
            break
        sns.histplot(data=train_df,x=numerical_cols[count],bins=50,kde=True,ax=ax[i][j])
        count+=1
    
plt.title('Density_plot') 


## checking for outlier 
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
count = 0
for i in range(2):
    for j in range(3):
        if count == len(numerical_cols):
            break
        sns.boxplot(data=train_df,x=numerical_cols[count],ax=ax[i][j])
        count+=1
fig.suptitle('Box_plot')


## checling the violin plot 
### To ensure that the num_reported_accident outlier are actually outliers or not 
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
count = 0
for i in range(2):
    for j in range(3):
        if count == len(numerical_cols):
            break
        sns.violinplot(data=train_df,x=numerical_cols[count],ax=ax[i][j])
        count+=1
fig.suptitle('violinplot')


len(train_df) - len(train_df[train_df['num_reported_accidents'] > 3]) , len(train_df)


train_df = (train_df[train_df['num_reported_accidents'] < 3])
train_df.head()


def eda(train_df):
    train_df = (train_df[train_df['num_reported_accidents'] < 3])
    return train_df


### lets look the countplt/frequency table of categorical features
train_df['road_type'].value_counts().plot.bar()


train_df['weather'].value_counts().plot.bar()


## as this is the regression problem and also evaluated on "root_mean_squared_error" so lets build 


numerical_cols


train_df['Density_Index'] = train_df['speed_limit'] / train_df['num_lanes']
train_df['Complex_Risk'] = train_df['curvature'] * train_df['speed_limit']
train_df['Accident_Rate'] = train_df['num_reported_accidents'] / train_df['num_lanes']


# Binning speed_limit
bins = [0, 30, 60, 100] # Define bins for speed limits
labels = ['Low_Speed', 'Medium_Speed', 'High_Speed']
train_df['Speed_Category'] = pd.cut(
    train_df['speed_limit'], 
    bins=bins, 
    labels=labels, 
    right=False
)


# Log transform 
train_df['log_curvature'] = np.log1p(train_df['curvature'])
# Square root transform
train_df['sqrt_num_lanes'] = np.sqrt(train_df['num_lanes'])


train_df.head()


## fuction later to perform on test 
def feature_engineering(train_df):
    # retio and product
    train_df['Density_Index'] = train_df['speed_limit'] / train_df['num_lanes']
    train_df['Complex_Risk'] = train_df['curvature'] * train_df['speed_limit']
    train_df['Accident_Rate'] = train_df['num_reported_accidents'] / train_df['num_lanes']
    # Binning speed_limit
    bins = [0, 30, 60, 100] # Define bins for speed limits
    labels = ['Low_Speed', 'Medium_Speed', 'High_Speed']
    train_df['Speed_Category'] = pd.cut(
        train_df['speed_limit'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    # Log transform 
    train_df['log_curvature'] = np.log1p(train_df['curvature'])
    # Square root transform
    train_df['sqrt_num_lanes'] = np.sqrt(train_df['num_lanes'])
    return train_df


unique_vals = {}
for col in categorical_cols:
    vals = train_df[col].unique()
    unique_vals[col] = vals

## print
for key,value in unique_vals.items():
    print(f"columns {key}")
    print(f"unique_values : {value}")
    print('--'*30)


def encode(train_df):
    categorical_cols = train_df.select_dtypes(include=['object','category']).columns
    temp_df = train_df.copy(deep=True)
    encoded_df = pd.get_dummies(train_df,columns=categorical_cols,drop_first=True)
    return encoded_df


train_df = encode(train_df)


from sklearn.model_selection import train_test_split
X = train_df.drop(columns=['accident_risk'])
y = train_df['accident_risk']
## train test split
X_train,X_test , y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
## validation split 
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,random_state=42,test_size=0.2)


X_train


X_numerical_columns = X.select_dtypes(include=['int64','float64']).columns
X_categorical_columns = X.select_dtypes(include=['object','category']).columns
X_boolean_columns = X.select_dtypes(include=['bool']).columns


from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import StandardScaler
def preprocessing(X_train_df, X_test_df, X_val_df):
    """
    Applies StandardScaler to numerical columns across training, test, and validation sets.
    The scaler is fit only on the training data.
    """
    # Create copies to avoid SettingWithCopyWarning
    X_train = X_train_df.copy()
    X_test = X_test_df.copy()
    X_val = X_val_df.copy()
    
    # Identify numerical columns based on the training set
    # NOTE: Assuming all three sets have the same columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # 1. Initialize and Fit Scaler ONLY on X_train
    scalar = StandardScaler()
    scalar.fit(X_train[numerical_cols])
    
    # 2. Transform all three sets using .loc for safe assignment
    
    # Training Set
    X_train.loc[:, numerical_cols] = scalar.transform(X_train[numerical_cols])
    
    # Test Set
    X_test.loc[:, numerical_cols] = scalar.transform(X_test[numerical_cols])
    
    # Validation Set
    X_val.loc[:, numerical_cols] = scalar.transform(X_val[numerical_cols])
    
    # NOTE: You would perform encoding (like OneHotEncoder) for
    # categorical/boolean columns here, also fitting only on X_train.
    
    return X_train, X_test, X_val,scalar

# Example of how to call it:
# X_train_scaled, X_test_scaled, X_val_scaled = preprocessing_fixed(X_train, X_test, X_val)


X_train,X_test,X_val,scalar = preprocessing(X_train,X_test,X_val)



X_val


from xgboost import XGBRegressor


X_train.dtypes


xgb_reg = XGBRegressor(
        n_estimators = 500,
        objective= "reg:squarederror",
        eval_metric= "rmse",
    early_stopping_rounds = 100
)
xgb_reg.fit(
    X_train,y_train,
    eval_set = [(X_train,y_train),(X_val,y_val)],
    verbose = True
)


from sklearn.metrics import mean_squared_error
y_pred = xgb_reg.predict(X_train)
train_error = mean_squared_error(y_train,y_pred)
y_pred = xgb_reg.predict(X_test)
test_error = mean_squared_error(y_test,y_pred)
print(f"Training error :{train_error}")
print(f'Testing error :{test_error}')


##debuging 
X_train.shape


train_loss = xgb_reg.evals_result()['validation_0']['rmse']
val_loss = xgb_reg.evals_result()['validation_1']['rmse']
plt.figure(figsize=(8,6))
plt.plot(train_loss,label=['training loss'])
plt.plot(val_loss,label=['validation loss'])
plt.xlabel('n_estimators')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()


test_df.head()


submission_df.head()


def _test_(df):
    id_var = df['id']
    df.drop(columns=['id'],inplace=True)
    df = feature_engineering(df)
    df = encode(df)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df.loc[:,numerical_cols] = scalar.transform(df[numerical_cols])
    y_pred = xgb_reg.predict(df)
    return y_pred,id_var


y_pred ,id_var = _test_(test_df)


test_df.shape , submission_df.shape


y_pred.shape


# Prepare submission
sub = pd.DataFrame({
    "id": id_var,
    "accident_risk": y_pred
})

# Save submission file
sub.to_csv("submission.csv", index=False)




