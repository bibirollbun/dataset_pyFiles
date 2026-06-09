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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_df.head()


train_df.columns


train_df.describe()


train_df.info()


## check how many null values are there
train_df.isnull().sum()


## check how many null values are there
test_df.isnull().sum()


## sample_submission
sample_submission


train_df.columns


## function to plot some graphs and print the dtype of feature
def show_details_and_graphs(df,df_col):
    col_dtype=df[df_col].dtype
    print("Feature:",df_col)
    print("Data type:",col_dtype)
    print("Total missing/null values:",df[df_col].isnull().sum())

    print()
    if(col_dtype=='O'):
        print("Total Unique Values:",df[df_col].nunique())
        plt.figure(figsize=(15,6))

        plt.subplot(1,2,1)
        plt.title("Bar Plot for {}".format(df_col))
        plt.ylabel("Count")
        df[df_col].value_counts().plot(kind='bar')

        plt.subplot(1,2,2)
        plt.title("Pie Chart for {}".format(df_col))
        df[df_col].value_counts().plot(kind='pie', autopct='%.2f%%')
        plt.show()


    elif(col_dtype!='O'):
        print("Mean:",np.round(df[df_col].mean(),2))
        print("Median:",np.round(df[df_col].median(),2))
        print("Minimum:",df[df_col].min())
        print("Maximum:",df[df_col].max())
        print("Std:",np.round(df[df_col].std(),2))
        print("Skew:",df[df_col].skew())
        
        plt.figure(figsize=(18,15))

        plt.subplot(2,2,1)
        plt.title("Histogram for '{}'".format(df_col))
        df[df_col].plot(kind='hist')

        plt.subplot(2,2,2)
        plt.title("KDE plot for '{}'".format(df_col))
        df[df_col].plot(kind='kde')

        plt.subplot(2,2,3)
        plt.title("Box Plot for '{}'".format(df_col))
        df[df_col].plot(kind='box')

        plt.subplot(2,2,4)
        plt.title("Distplot for '{}'".format(df_col))
        sns.distplot(df[df_col])

        plt.show()


    else:
        print("Datatype of feature is neither numeric not categorical...")


## function to find and print all the rows where outlier is present
def check_outlier(df,df_col):
    if df[df_col].dtype!='O':
        print("Feature Name : {}".format(df_col))
        df_col_mean = df[df_col].mean()
        df_col_std = df[df_col].std()

        df_col_lower_limit = df_col_mean - 3*df_col_std 
        df_col_upper_limit = df_col_mean + 3*df_col_std 

        print("Based on Z-Score test :")
        print()
        return df[(df[df_col]<df_col_lower_limit) | (df[df_col]>df_col_upper_limit)]
    else:
        print("This is a categorical Feature...")


from scipy.stats import zscore

def get_zscore_outlier_limits(df, feature, threshold=3):
    mean = df[feature].mean()
    std = df[feature].std()
    lower_limit = mean - threshold * std
    upper_limit = mean + threshold * std
    return lower_limit, upper_limit


train_df = train_df[(train_df['Height']>=get_zscore_outlier_limits(train_df,'Height')[0]) & (train_df['Height']<=get_zscore_outlier_limits(train_df,'Height')[1])]


train_df = train_df[(train_df['Weight']>get_zscore_outlier_limits(train_df,'Weight')[0]) & (train_df['Weight']<get_zscore_outlier_limits(train_df,'Weight')[1])]


train_df = train_df[(train_df['Heart_Rate']>get_zscore_outlier_limits(train_df,'Heart_Rate')[0]) & (train_df['Heart_Rate']<get_zscore_outlier_limits(train_df,'Heart_Rate')[1])]


train_df = train_df[(train_df['Body_Temp']>get_zscore_outlier_limits(train_df,'Body_Temp')[0]) & (train_df['Body_Temp']<get_zscore_outlier_limits(train_df,'Body_Temp')[1])]


train_df = train_df[(train_df['Calories']>get_zscore_outlier_limits(train_df,'Calories')[0]) & (train_df['Calories']<get_zscore_outlier_limits(train_df,'Calories')[1])]


num_col_list = [feature for feature in train_df.columns if train_df[feature].dtype!='O']
cat_col_list = [feature for feature in train_df.columns if train_df[feature].dtype=='O']


plt.figure(figsize=(9,9))
sns.heatmap(train_df[num_col_list].corr(),annot=True)


train_df.head()


X = train_df.drop(columns=['id','Calories']) ## Independent features
y = train_df['Calories']


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


print("Shape of X_train:",X_train.shape)
print("Shape of X_test:",X_test.shape)


X_train_copy = X_train.copy()
X_test_copy = X_test.copy()


def feature_engineering2(df):
    # ## Body Mass Index feature 
    # df['BMI'] = df['Weight'] / ((df['Height']/10) ** 2) ## here divided Height by 10 to get it in meter

    ## deviation in body temp accoding to (37) average normal body temperature for humans.
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0

    # ## Heart Rate per minute
    # df['Heart_Rate_per_Minute'] = df['Heart_Rate'] / df['Duration']

    # Workout Intensity Index
    df['Workout_Intensity'] = (df['Heart_Rate'] * df['Duration']) / 1000

    # ## Ratio of Height to Weight
    # df['Height_to_Weight_Ratio'] = df['Height'] / df['Weight']

    # # Height x Weight 
    # df['Height_Weight_Product'] = df['Height'] * df['Weight']

    # Body Temp × Heart Rate
    df['Temp_HR_Product'] = df['Body_Temp'] * df['Heart_Rate']


    # Calories Burned Estimate (assuming MET = 8 for moderate activity)
    MET = 8
    df['Calories_Burned'] = df['Duration'] * MET * df['Weight'] * 0.0175


    ### ceating only ['Temp_Deviation','Workout_Intensity','Temp_HR_Product','Calories_Burned'] these 4 features

    return df


train_df_fe = feature_engineering2(train_df)


num_col_list_in_train_df_fe = [feature for feature in train_df_fe.columns if train_df_fe[feature].dtype!='O']


num_col_list_in_train_df_fe


plt.figure(figsize=(12,12))
sns.heatmap(train_df_fe[num_col_list_in_train_df_fe].corr(),annot=True)





def feature_engineering(df):
    # ## Body Mass Index feature 
    # df['BMI'] = df['Weight'] / ((df['Height']/10) ** 2) ## here divided Height by 10 to get it in meter

    ## deviation in body temp accoding to (37) average normal body temperature for humans.
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0

    # Workout Intensity Index
    df['Workout_Intensity'] = (df['Heart_Rate'] * df['Duration']) / 1000

    # Body Temp × Heart Rate
    df['Temp_HR_Product'] = df['Body_Temp'] * df['Heart_Rate']


    # Calories Burned Estimate (assuming MET = 8 for moderate activity)
    MET = 8
    df['Calories_Burned'] = df['Duration'] * MET * df['Weight'] * 0.0175


    df.drop('Height',axis=1,inplace=True)

    return df


### applying feature_engineering function on X_train and X_test 
X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,OneHotEncoder


numerical_features = [feature for feature in X_train.columns if X_train[feature].dtype!='O']
categorical_features = [feature for feature in X_train.columns if X_train[feature].dtype=='O']


from sklearn.compose import ColumnTransformer


## preprocessor -->
preprocessor = ColumnTransformer(
    transformers=[
        ("StandardScaler",StandardScaler(),numerical_features),
        ("OneHotEncoder",OneHotEncoder(sparse_output=False,dtype=int,handle_unknown='ignore'),categorical_features)
    ]
,remainder='passthrough')


X_train_trf = preprocessor.fit_transform(X_train)
X_test_trf = preprocessor.transform(X_test)


X_train.shape


X_train.head(5)


## doing same process with test df
test_df.drop(columns=['id'],inplace=True)

test_df_copy =test_df.copy() ## creating copy of test_df

test_df = feature_engineering(test_df)

test_df_trf = pd.DataFrame(preprocessor.transform(test_df))


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    rmsle = np.sqrt(mean_squared_log_error(true,predicted))
    r2=r2_score(true,predicted)
    print("R2 Score:{:.4f}".format(r2))
    print("MAE:{:.4f}".format(mae))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    print("RMSLE:{:.4f}".format(rmsle))
    
    # ---------


id_column = sample_submission["id"]


## Model Training and Model Selection

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor


# ## Model training
# models={
#     "CatBoostRegressor":CatBoostRegressor(
#     iterations= 3500,
#     depth= 12,
#     loss_function= 'RMSE',
#     l2_leaf_reg= 3,
#     random_seed= 42,
#     eval_metric= 'RMSE',
#     silent=True
#     ) 
#     }

# model_name_list = []
# r2score_list = []
# rmse_list = []
# mae_list = []
# mse_list = []
# rmsle_list = []

# for i in range(len(list(models))):

#     model_name = list(models.keys())[i]
    
#     model=list(models.values())[i]
#     model.fit(X_train_trf,y_train) ## Train Model on X_train_trf (encoded)

#     ## Make Predictions
#     y_train_pred=model.predict(X_train_trf)
#     y_train_pred = np.clip(y_train_pred, 1, 314)
    
#     y_test_pred=model.predict(X_test_trf)
#     y_test_pred = np.clip(y_test_pred, 1, 314)
    
#     print(model_name,"=============>")
#     print()
#     print("Evaluating Train Dataset")
#     evaluate_model(y_train,y_train_pred)

#     print(f"\n{'-'*50}\n")
    
#     print("Evaluating Test Dataset")

#     ### for performance on test_df
#     evaluate_model(y_test,y_test_pred)

#     ### appending the vlaues in list 
#     model_name_list.append(model_name)
#     r2score_list.append(r2_score(y_test,y_test_pred))
#     rmse_list.append(np.sqrt(mean_squared_error(y_test,y_test_pred)))
#     mae_list.append(mean_absolute_error(y_test,y_test_pred))
#     mse_list.append(mean_squared_error(y_test,y_test_pred))
#     rmsle_list.append(np.sqrt(mean_squared_log_error(y_test,y_test_pred)))
    
#     print("="*50)
#     print("\n")

#     ## prediction
#     prediction = model.predict(test_df_trf)
#     prediction = np.clip(prediction, 1, 314)

#     result = pd.DataFrame(
#     {
#         'id':id_column,
#         'Calories':prediction
#     }
#     )

#     result.to_csv('{}_prediction.csv'.format(model_name),index=False)
#     print("File saved as '{}_prediction.csv'....".format(model_name))
#     print()



# performance_df = pd.DataFrame({
#     'ML Algo Name': model_name_list,
#     'R2 Score': r2score_list,
#     'RMSE': rmse_list,
#     'MAE': mae_list,
#     'MSE': mse_list,
#     'RMSLE': rmsle_list
# })


pip install tensorflow


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define the model
def create_deep_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Initialize model
input_dim = X_train_trf.shape[1]
dl_model = create_deep_model(input_dim)

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = dl_model.fit(
    X_train_trf, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_test_pred = dl_model.predict(X_test_trf).flatten()
y_test_pred = np.clip(y_test_pred, 1, 314)

y_train_pred = dl_model.predict(X_train_trf).flatten()
y_train_pred = np.clip(y_train_pred, 1, 314)

## Evaluation
print("Deep Learning Model =============>")
print("Evaluating Train Dataset")
evaluate_model(y_train, y_train_pred)

print("\nEvaluating Test Dataset")
evaluate_model(y_test, y_test_pred)


# ## Predictions 
# prediction = dl_model.predict(test_df_trf).flatten()
# prediction = np.clip(prediction, 1, 314)

# result = pd.DataFrame({
#     'id': id_column,
#     'Calories': prediction
# })
# result.to_csv('DeepLearningRegressor_prediction.csv', index=False)
# print("File saved as 'DeepLearningRegressor_prediction.csv'")



## Predictions 
prediction = dl_model.predict(test_df_trf).flatten()
prediction = np.clip(prediction, 1, 314)

result = pd.DataFrame({
    'id': id_column,
    'Calories': prediction
})
result.to_csv('DeepLearningRegressor_prediction.csv', index=False)
print("File saved as 'DeepLearningRegressor_prediction.csv'")














