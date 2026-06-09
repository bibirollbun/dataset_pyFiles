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


import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.head() ## first top 5 rows of train data


train.columns ## columns of dataset


test.head()


print(f"Shape of train data: {train.shape}")
print(f"Shape of test data: {test.shape}")


sample_submission


print("Information of train data")
train.info()


train.describe()


## is there any duplicate values
train.duplicated().sum()


## checking is there any missing values
train.isnull().sum()


## checking is there any missing values in test data
test.isnull().sum()


numerical_features = [feature for feature in train.columns if train[feature].dtype!='O']
categorical_features = [feature for feature in train.columns if train[feature].dtype=='O']


print("List of Numerical Features: ",numerical_features)
print()
print("List of Categorical Features",categorical_features)


# ##  function to plot some of the graphs for categorical and numerical features 
# def plot_graphs(df,feature):
#     if df[feature].dtype=='O':
#         print(f"--------- {feature} is a CATEGORICAL feature --------")
#         print(f"Total Missing Values: {df[feature].isnull().sum()}")
#         print(f"Total Unique Categories: {df[feature].nunique()}\n")
#         print("Unique Values:\n", df[feature].unique())
#         print()

#         ## bar plot for top 10 categoris in categorical feature
#         plt.figure(figsize=(15,6))

#         plt.subplot(1,2,1)
#         plt.title("Bar Plot for {}".format(feature))
#         plt.ylabel("Count")
#         df[feature].value_counts().plot(kind='bar')

#         plt.subplot(1,2,2)
#         plt.title("Pie Chart for {}".format(feature))
#         df[feature].value_counts().plot(kind='pie', autopct='%.2f%%')
#         plt.show()


#     elif df[feature].dtype!='O':
#         print(f"------- {feature} is a NUMERICAL feature ------")
#         print(f"Total Missing Values: {df[feature].isnull().sum()}")
#         print(f"Summary Statistics:\n{df[feature].describe()}\n")
#         df[feature].describe()
        
#         plt.figure(figsize=(18,25))

#         plt.subplot(3,2,1)
#         plt.title("Histogram for '{}'".format(feature))
#         df[feature].plot(kind='hist')

#         plt.subplot(3,2,2)
#         plt.title("KDE plot for '{}'".format(feature))
#         df[feature].plot(kind='kde')

#         plt.subplot(3,2,3)
#         plt.title("Box Plot for '{}'".format(feature))
#         df[feature].plot(kind='box')

#         plt.subplot(3,2,4)
#         plt.title("Distplot for '{}'".format(feature))
#         sns.distplot(df[feature])

#         # plt.subplot(3,2,5)
#         # plt.title("Lineplot for '{}'".format(feature))
#         # df[feature].value_counts().sort_index().plot.line()

#         plt.show()


#     else:
#         print("Datatype of feature is neither numerical nor categorical...")

#     print()


sns.heatmap(train[numerical_features].corr(),annot=True)


## function to find and print all the rows where outlier is present
def check_outlier(df,feature):
    if df[feature].dtype!='O':
        print("Feature Name : {}".format(feature))
        df_col_mean = df[feature].mean()
        df_col_std = df[feature].std()

        df_col_lower_limit = df_col_mean - 3*df_col_std 
        df_col_upper_limit = df_col_mean + 3*df_col_std 
        
        outliers = df[(df[feature]<df_col_lower_limit) | (df[feature]>df_col_upper_limit)]
        if outliers.shape[0]==0:
            print("There are no any outliers")
        else:
            display(outliers)
            print(f"Total outlier containing rows: {outliers.shape}")
    else:
        print("Feature Name : {}".format(feature))
        print("This is a categorical Feature...")

    print()


for i in train.columns:
    check_outlier(train,i)





X = train.drop(columns=['id','Personality'])
y = train['Personality']


X.head()


y.head()


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder


cat_col_list = [feature for feature in X.columns if X[feature].dtype=='O']
num_col_list = [feature for feature in X.columns if X[feature].dtype!='O']


def min_max_in_num_features(df,feature):
    print(f"Feature : {feature}")
    print(f"Minimum Value : {df[feature].min()}")
    print(f"Maximum Value : {df[feature].max()}")
    print()

for feature in num_col_list:
    min_max_in_num_features(X,feature)


## encoding target feature
personality_encoder = LabelEncoder()
y = personality_encoder.fit_transform(y)


y


# Pipelines for categorical and numerical features
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('encoder', OneHotEncoder(drop='first', dtype=int, sparse_output=False))
    ('encoder', OrdinalEncoder())
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  
    ('min_max_scaler', MinMaxScaler())
])

## trying with min max scalling 


# Column Transformer
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipeline, cat_col_list),
    ('num', num_pipeline, num_col_list)
], remainder='passthrough') 


X_trf = preprocessor.fit_transform(X)


X_trf


type(X_trf)


pd.DataFrame(X_trf)


test = test.drop('id',axis=1)
test_trf = preprocessor.transform(test)


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score

from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold


# Creating a function to evaluate the model
def evaluate_model(true, predicted):
    print("- Accuracy Score: {:.4f}".format(accuracy_score(true,predicted)))
    print("- F1 Score: {:.4f}".format(f1_score(true,predicted)))
    print("- Precision Score: {:.4f}".format(precision_score(true,predicted)))
    print("- Recall Score: {:.4f}".format(recall_score(true,predicted)))
    
    # print("- Roc Auc Score: {:.4f}".format(roc_auc_score(true,predicted)))
    # print("- Confusion Matrix: \n",confusion_matrix(true,predicted))


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier 

from catboost import CatBoostClassifier


sample_submission


## Model Training -->
models = {
    "LogisticRegression": LogisticRegression(),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    # "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        bootstrap=False,
    ),
    # "AdaBoostClassifier": AdaBoostClassifier(),
    # "GradientBoostingClassifier": GradientBoostingClassifier(),
    "XGBClassifier": XGBClassifier(
        max_depth=10,
        n_estimators=200,
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='error'
    ),
    # "SVC": SVC(),
    # "LGBMClassifier": LGBMClassifier(),
    "CatBoostClassifier":CatBoostClassifier(
        learning_rate=None,
        max_depth=10,
        n_estimators=200,
        silent=True)
}

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

model_name_list=[]
score_list = []
avg_cv_score_list=[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model_name = list(models.keys())[i]
    print
    print(f"\n<==============={model_name}===============>\n")

    ## applying StratifiedKFold cross validation
    n=1
    for train_idx,test_idx in skf.split(X_trf,y):
        X_train_fold,X_test_fold = X_trf[train_idx],X_trf[test_idx]
        y_train_fold,y_test_fold = y[train_idx],y[test_idx]

        model.fit(X_train_fold,y_train_fold)
        y_test_fold_pred = model.predict(X_test_fold)
        score = accuracy_score(y_test_fold,y_test_fold_pred)
        print(f"Fold {n}: Accuracy : {score:.4f}\n")
        score_list.append(score)
        n+=1
    print(f"Average Score : {np.mean(score_list):.4f}\n")
    
    avg_cv_score_list.append(round(np.mean(score_list),4))
    model_name_list.append(model_name)
     
    ## prediction for test dataset
    prediction = model.predict(test_trf)
    prediction = personality_encoder.inverse_transform(prediction)

    ## saving prediction in submission file
    sample_submission['Personality'] = prediction
    sample_submission.to_csv(f"{model_name}_prediction.csv",index=False)
    print(f"File saved as {model_name}_prediction.csv.....\n")

performace_df = pd.DataFrame({
    'ML Algorithm': model_name_list,
    'Mean CV Score (StratifiedKFold)': avg_cv_score_list
})


performace_df.sort_values('Mean CV Score (StratifiedKFold)',ascending=False)




