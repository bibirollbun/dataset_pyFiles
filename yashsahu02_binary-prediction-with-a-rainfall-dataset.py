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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


test.head()


sample_submission


## shape of train 
train.shape


## check if any null value
train.isnull().sum()


## There is no any null values


train.columns


train.dtypes


train['rainfall'].value_counts()


## rainfall is out target feature


## importing libraries for visualization
import matplotlib.pyplot as plt 
import seaborn as sns


## function to plot some graphs and print the dtype of feature
def show_details_and_graphs(df,df_col):
    col_dtype=df[df_col].dtype
    print("Data type:",col_dtype)
    print("Total null values:",df[df_col].isnull().sum())

    print()
    if(col_dtype=='O'):
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
    


train.columns


## id feature -> This holds unique value for every row which not make any sense in prediction so let's drop it
train.drop('id',axis=1,inplace=True)


X = train.drop(columns=['rainfall'])
y = train['rainfall']


# from sklearn.model_selection import train_test_split


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# X_train.head(3)


# show_details_and_graphs(X_train,'day')


# check_outlier(X_train,'day')


# show_details_and_graphs(X_train,'pressure')


# check_outlier(X_train,'pressure')


# X_train.head(3)


# show_details_and_graphs(X_train,'maxtemp')


# check_outlier(X_train,'maxtemp')


# show_details_and_graphs(X_train,'temparature')


# check_outlier(X_train,'temparature')


# show_details_and_graphs(X_train,'mintemp')


# check_outlier(X_train,'mintemp')


# show_details_and_graphs(X_train,'dewpoint')


# check_outlier(X_train,'dewpoint')


X.head(3)


# X_train['sqr_dewpoint'] = X_train['dewpoint']*X_train['dewpoint']
X['sqr_dewpoint'] = X['dewpoint']*X['dewpoint']


# show_details_and_graphs(X_train,'sqr_dewpoint')


# X_train.head(2)


# show_details_and_graphs(X_train,'humidity')


X['cube_humidity'] = X['humidity']*X['humidity']*X['humidity']


# show_details_and_graphs(X_train,'cube_humidity')


# show_details_and_graphs(X_train,'cloud')


# X_train.head(3)


X['cube_cloud'] = X['cloud']*X['cloud']*X['cloud']


# show_details_and_graphs(X_train,'cube_cloud')


X.head(3)


# show_details_and_graphs(X_train,'sunshine')


# X_train['Log_sunshine'] = X_train['sunshine'].apply(np.log1p)
X['Log_sunshine'] = X['sunshine'].apply(np.log1p)


# show_details_and_graphs(X_train,'Log_sunshine')


# show_details_and_graphs(X_train,'winddirection')


# show_details_and_graphs(X_train,'windspeed')


# X_train['Log_windspeed'] = X_train['windspeed'].apply(np.log1p)
X['Log_windspeed'] = X['windspeed'].apply(np.log1p)


# show_details_and_graphs(X_train,'Log_windspeed')


# X_train.corr()
X.corr()


# X_train.head(2)


# X_train.drop(columns=['dewpoint','humidity','cloud','sunshine','windspeed'], inplace=True)
X.drop(columns=['dewpoint','humidity','cloud','sunshine','windspeed'], inplace=True)


# X_test['sqr_dewpoint'] = X_test['dewpoint']*X_test['dewpoint']
# X_test['cube_humidity'] = X_test['humidity']*X_test['humidity']*X_test['humidity']
# X_test['cube_cloud'] = X_test['cloud']*X_test['cloud']*X_test['cloud']
# X_test['Log_sunshine'] = X_test['sunshine'].apply(np.log1p)
# X_test['Log_windspeed'] = X_test['windspeed'].apply(np.log1p)


# X_test.drop(columns=['dewpoint','humidity','cloud','sunshine','windspeed'], inplace=True)


from sklearn.impute import SimpleImputer


from sklearn.compose import ColumnTransformer


train.corr()


### creating imputer for every column
mean_imputer = SimpleImputer(strategy='mean')


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X.shape



## Column Transformer to imputer NaN
trf1 = ColumnTransformer([
    ('Imputer', mean_imputer, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Column indices
], remainder='passthrough')

## Column Transformer to apply StandardScaler
trf2 = ColumnTransformer([
    ('Scaler', scaler, [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Column indices
], remainder='passthrough')


from sklearn.pipeline import Pipeline


# **Pipeline to first impute missing values, then scale**
pipeline = Pipeline([
    ('imputer', trf1),
    ('scaler', trf2)
])


## applying preprocessor
# X_train_trf = pipeline.fit_transform(X_train)
# X_test_trf = pipeline.transform(X_test)
X_trf = pipeline.fit_transform(X)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score


# Creating a function to evaluate the model
def evaluate_model(true, predicted):
    print("- Accuracy Score: {:.4f}".format(accuracy_score(true,predicted)))
    print("- F1 Score: {:.4f}".format(f1_score(true,predicted)))
    print("- Precision Score: {:.4f}".format(precision_score(true,predicted)))
    print("- Recall Score: {:.4f}".format(recall_score(true,predicted)))
    print("- Roc Auc Score: {:.4f}".format(roc_auc_score(true,predicted)))
    print("- Confusion Matrix: \n",confusion_matrix(true,predicted))


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from lightgbm import LGBMClassifier 

from catboost import CatBoostClassifier


id_column = sample_submission['id']


id_column


test


test.drop(columns=['id'],inplace=True)


test['sqr_dewpoint'] = test['dewpoint']*test['dewpoint']
test['cube_humidity'] = test['humidity']*test['humidity']*test['humidity']
test['cube_cloud'] = test['cloud']*test['cloud']*test['cloud']
test['Log_sunshine'] = test['sunshine'].apply(np.log1p)
test['Log_windspeed'] = test['windspeed'].apply(np.log1p)


test.drop(columns=['dewpoint','humidity','cloud','sunshine','windspeed'], inplace=True)


test.head(3)


test.isnull().sum()


test


test_trf = pipeline.transform(test)


pd.DataFrame(test_trf).isnull().sum()


## Model Training -->
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    # "DecisionTreeClassifier": DecisionTreeClassifier(),
    # "AdaBoostClassifier": AdaBoostClassifier(),
    # "GradientBoostingClassifier": GradientBoostingClassifier(),
    # "XGBClassifier": XGBClassifier(),
    # "SVC": SVC(probability=True),
    # "GaussianNB": GaussianNB(),
    # # "MultinomialNB": MultinomialNB(), ### This may raise error due to presence of negative values (resulting due to StandardScaler)
    # "BernoulliNB": BernoulliNB(),
    # "LGBMClassifier": LGBMClassifier(),
    # "CatBoostClassifier":CatBoostClassifier()
}

for i in range(len(list(models))):
    model=list(models.values())[i]
    model_name = list(models.keys())[i]
    model.fit(X_trf,y)
    
    # y_train_pred=model.predict(X_train_trf)
    # y_test_pred=model.predict(X_test_trf)
    
    # print(model_name,"============>")
    # print()
    # print("--- Model Performance for Training Dataset ---")
    # evaluate_model(y_train,y_train_pred)
    # print()
    # print("--- Model Performance for Test Dataset ---")
    # evaluate_model(y_test,y_test_pred)
    # print()
    # print(f"\n{'-'*50}\n")
    prediction_probability = model.predict_proba(test_trf)
    # prediction_probability = np.round(prediction_probability,2)
    prediction_probability = np.round(pd.Series(prediction_probability[:,1]),2) ### Probability of 1 upto 2 point digits

    result = pd.DataFrame({
        'id':id_column,
        'rainfall':prediction_probability
    })  

    result.to_csv("{}_prediction.csv".format(model_name),index=False)
    print("File Saved as {}_prediction.csv".format(model_name))
    print(f"\n{'-'*50}\n")
    print()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


log_reg = LogisticRegression()

# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 4, 20),  # Regularization strength
    'penalty': ['l1', 'l2']  # Regularization type
}

# Perform GridSearchCV
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_trf, y)

# Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final model using best parameters
best_model = LogisticRegression(**best_params, solver='liblinear')
best_model.fit(X_trf, y)

prediction_probability = best_model.predict_proba(test_trf)
# prediction_probability = np.round(prediction_probability,2)
prediction_probability = np.round(pd.Series(prediction_probability[:,1]),2) ### Probability of 1 upto 2 point digits

result = pd.DataFrame({
    'id':id_column,
    'rainfall':prediction_probability
})  

result.to_csv("LogisticRegression_Tuned_prediction.csv",index=False)
print("File Saved as LogisticRegression_Tuned_prediction.csv")
print(f"\n{'-'*50}\n")
print()


# prediction_probability = best_model.predict_proba(test_trf)
# # prediction_probability = np.round(prediction_probability,2)
# prediction_probability = np.round(pd.Series(prediction_probability[:,1]),2) ### Probability of 1 upto 2 point digits

# result = pd.DataFrame({
#     'id':id_column,
#     'rainfall':prediction_probability
# })  

# result.to_csv("LogisticRegression_Tuned_prediction.csv",index=False)
# print("File Saved as LogisticRegression_Tuned_prediction.csv")
# print(f"\n{'-'*50}\n")
# print()


from sklearn.model_selection import train_test_split, RandomizedSearchCV


# Define Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500],  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],        # Maximum depth of trees
    'min_samples_split': [2, 5, 10, 15, 20],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4, 8, 12],           # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],       # Number of features to consider at each split
    'bootstrap': [True, False]                      # Whether bootstrap samples are used
}


# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, verbose=1
)
random_search.fit(X_trf, y)

# Best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Train the final model using best parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_trf, y)

# # Make predictions
# y_pred = best_rf.predict(X_test_trf)

# # Evaluate the model
# print("Test Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

prediction_probability = best_rf.predict_proba(test_trf)
# prediction_probability = np.round(prediction_probability,2)
prediction_probability = np.round(pd.Series(prediction_probability[:,1]),2) ### Probability of 1 upto 2 point digits

result = pd.DataFrame({
    'id':id_column,
    'rainfall':prediction_probability
})  

result.to_csv("RF_Randomized_Tuned_prediction.csv",index=False)
print("File Saved as RF_Randomized_Tuned_prediction.csv")
print(f"\n{'-'*50}\n")
print()


























