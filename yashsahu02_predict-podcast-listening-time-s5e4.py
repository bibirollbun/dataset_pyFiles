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


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


train.columns


test.columns


train.info()


X = train.drop("Listening_Time_minutes", axis=1) ## Independent Features 
y = train["Listening_Time_minutes"]  ## Target Feature


X


y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


X_train.head()


## check is there any null values
X_train.isnull().sum()


X_test.isnull().sum()


## importing libraries for visualization
import matplotlib.pyplot as plt 
import seaborn as sns


## function to plot some graphs and print the dtype of feature
def show_details_and_graphs(df,df_col):
    col_dtype=df[df_col].dtype
    print("Feature:",df_col)
    print("Data type:",col_dtype)
    print("Total null values:",df[df_col].isnull().sum())

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


X_train.columns


show_details_and_graphs(X_train,"id")


X_train["id"].head()


### Here let's drop 'id' feature


cols_to_drop = []


X_train.drop("id",axis=1,inplace=True)
X_test.drop("id",axis=1,inplace=True)

cols_to_drop.append("id")


X_train.shape


X_train[X_train.isnull().any(axis=1)]


null_value_indexes=X_train[X_train.isnull().any(axis=1)].index


X_train = X_train.dropna()


X_train.shape


y_train.shape ## before dropping null values


## dropping same rows in y_train also -->
y_train = y_train.drop(null_value_indexes)


y_train.shape ## after dropping null values


# show_details_and_graphs(X_train,"Podcast_Name")


X_train["Podcast_Name"].value_counts()


podcast_name_dict = train['Podcast_Name'].value_counts().to_dict()


X_train["Podcast_Name"].value_counts().head(10).plot(kind='bar')

### bar chart for top 10 Podcast in "Podcast_Name" feature


X_train["Podcast_Name"].value_counts().head(10).plot(kind='pie',autopct="%0.2f%%")

### bar chart for top 10 Podcast in "Podcast_Name" feature


X_train['Podcast_Name'] = X_train['Podcast_Name'].map(podcast_name_dict)
X_test['Podcast_Name'] = X_test['Podcast_Name'].map(podcast_name_dict)


X_train.head()


# show_details_and_graphs(X_train,"Episode_Title")


X_train["Episode_Title"].value_counts()


X_train["Episode_Title"].value_counts().head(10).plot(kind='bar')


X_train["Episode_Title"].value_counts().head(10).plot(kind='pie',autopct="%0.2f%%")


episode_title_dict = train['Episode_Title'].value_counts().to_dict()


X_train['Episode_Title'] = X_train['Episode_Title'].map(episode_title_dict)
X_test['Episode_Title'] = X_test['Episode_Title'].map(episode_title_dict)


X_train.head(5)


show_details_and_graphs(X_train,"Episode_Length_minutes")


X_train['Episode_Length_minutes'].plot(kind='kde')
X_train['Episode_Length_minutes'].fillna(X_train['Episode_Length_minutes'].mean()).plot(kind='kde',color='red')  ## filling null values with mean
X_train['Episode_Length_minutes'].fillna(X_train['Episode_Length_minutes'].median()).plot(kind='kde',color='green')  ## filling null values with median

plt.show()


# show_details_and_graphs(X_train,"Genre")


X_train.columns


X_train['Genre']


train['Genre'].value_counts()


# show_details_and_graphs(X_train,"Host_Popularity_percentage")


show_details_and_graphs(X_train,"Publication_Day")


days_dict = {
    "Monday":0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday":6
}


X_train['Publication_Day'] = X_train['Publication_Day'].map(days_dict)
X_test['Publication_Day'] = X_test['Publication_Day'].map(days_dict)


# show_details_and_graphs(X_train,"Publication_Time")


X_train['Publication_Time']


# show_details_and_graphs(X_train,"Guest_Popularity_percentage")


X_train['Guest_Popularity_percentage']


# show_details_and_graphs(X_train,"Number_of_Ads")


X_train["Number_of_Ads"].describe()


X_train["Number_of_Ads"]


# show_details_and_graphs(X_train,"Episode_Sentiment")


X_train['Episode_Sentiment']


## checking if there any null value in target feature


y.isnull().sum() ## There is no any null values in Target Feature


check_outlier(X_train,"Episode_Length_minutes")


check_outlier(X_train,"Host_Popularity_percentage")


X_train.columns


check_outlier(X_train,"Guest_Popularity_percentage")


check_outlier(X_train,"Number_of_Ads")


X_train.columns


num_col_list = [feature for feature in X_train.columns if X_train[feature].dtype!="O"]
cat_col_list = [feature for feature in X_train.columns if X_train[feature].dtype=="O"]


num_col_list


cat_col_list


X_train["Publication_Time"]


X_train["Genre"]


for feature in cat_col_list:
    print(f"Feature Name : {feature} =======>")
    print(X_train[feature].value_counts())
    print("Total Unique Values : {}".format(X_train[feature].nunique()))
    print()


test.isnull().sum()


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler,OneHotEncoder


# One-Hot Encoding Pipeline
cat_ohe_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', dtype=int))
])

# Numerical Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])


# Column Transformer
transformer = ColumnTransformer(transformers=[
    ('cat_ohe', cat_ohe_pipeline, cat_col_list),
    ('num_pipeline', num_pipeline, num_col_list)
], remainder='passthrough')  # Keeps other columns as they are


X_train_trf = transformer.fit_transform(X_train)
X_test_trf = transformer.transform(X_test)


X_train_trf


cols_to_drop


train.columns


## doing same process with test df
test.drop(columns=cols_to_drop,inplace=True)

test['Podcast_Name'] = test['Podcast_Name'].map(podcast_name_dict)
test['Episode_Title'] = test['Episode_Title'].map(episode_title_dict)


test['Publication_Day'] = test['Publication_Day'].map(days_dict)


test_trf = transformer.transform(test)


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)
    print("R2 Score:{:.4f}".format(r2))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    print("MAE:{:.4f}".format(mae))
    # ---------
    return 0


sample_submission.head()


id_column = sample_submission["id"]


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

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


from sklearn.model_selection import KFold
kfold_validation=KFold(5)
from sklearn.model_selection import cross_val_score


## Model training
models={
    # "Linear_Regression":LinearRegression(),
    # "Lasso":Lasso(),
    # "Ridge":Ridge(),
    # "ElasticNet":ElasticNet(),
    # "KNeighborsRegressor":KNeighborsRegressor(),
    # "SVR":SVR(),
    # "DecisionTreeRegressor":DecisionTreeRegressor(),
    "RandomForest":RandomForestRegressor(),
    # "AdaBoost":AdaBoostRegressor(),
    # "GradientBoost":GradientBoostingRegressor(),
    # "XGBRegressor":XGBRegressor(),
    # "LGBMRegressor":LGBMRegressor(),
    # "CatBoostRegressor":CatBoostRegressor()
}

for i in range(len(list(models))):

    model_name = list(models.keys())[i]
    model=list(models.values())[i]
    model.fit(X_train_trf,y_train) ## Train Model on X_train_trf (encoded)

    ## Make Predictions
    y_train_pred=model.predict(X_train_trf)
    y_test_pred=model.predict(X_test_trf)

    print(model_name,"=============>")
    print()
    print("Evaluating Train Dataset")
    evaluate_model(y_train,y_train_pred)

    cv_scores=cross_val_score(model,X_train_trf,y_train,cv=kfold_validation)
    print("CV Scores:",cv_scores)
    print("Average CV Score:",np.mean(cv_scores))
    
    print(f"\n{'-'*50}\n")
    
    print("Evaluating Test Dataset")
    evaluate_model(y_test,y_test_pred)
    print("="*50)
    print("\n")

    ## prediction
    prediction = model.predict(test_trf)

    result = pd.DataFrame(
    {
        'id':id_column,
        'Listening_Time_minutes':prediction
    }
    )

    result.to_csv('{}_prediction.csv'.format(model_name),index=False)
    print("File saved as '{}_prediction.csv'....".format(model_name))
    print()




