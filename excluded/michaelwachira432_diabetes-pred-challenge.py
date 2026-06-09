#importing all dependencies

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# reading train data and test data
train_data=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


# view the first five rows from train data and test data
train_data.head()
test_data.head()


#getting all traindata rows and columns
train_data.shape


#getting all testdata rows and columns
test_data.shape


#getting info about traindata
test_data.info()


#getting describtive statistics
train_data.describe(include="all")


#checking missing values
train_data.isnull().sum()


#checking duplicates
train_data.duplicated().sum()


#dropping id column
train_data.drop(columns=['id'], inplace=True)


#getting the numerical and categorical columns
num=[]
cat=[]
for col in train_data.columns:
    if train_data[col].dtypes==int or train_data[col].dtypes==float:
        num.append(col)
    else:
        cat.append(col)
print(num)
print(cat)


#getting correction for all numerical columns

corr_data=train_data[['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'diagnosed_diabetes']]
corr_data.corr()


#converting object type to category type

cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

for col in cat_cols:
    train_data[col] = train_data[col].astype("category")

#get only categorical columns

cat_cols=[col for col in train_data.columns if train_data[col].dtype.name=="category"]

#visualize category columns on bar chart

distribution="distribution"
plt.style.use("fivethirtyeight")
fig,axes=plt.subplots(3,2,figsize=(20,17))

axes=axes.flatten()

for i,col in enumerate(cat_cols):
   ay=sns.countplot(x=train_data[col],data=train_data,ax=axes[i])
    
   for p in ay.patches: 
        x=p.get_x()+ p.get_width()/2
        y=p.get_height()
        ay.text(x,y,f"{int(p.get_height()):,}",ha="center",va="bottom",fontsize=15,fontweight="bold")
   ay.set_title(f"{col.upper()} {distribution.upper()}",fontsize=20)
# hide unused subplots

for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])   

plt.tight_layout()
plt.show()


#visualizing outliers
plt.style.use("fivethirtyeight")
fig,axes=plt.subplots(5,4,figsize=(20,15))

axes=axes.flatten()

for i,col in enumerate(num):
    sns.boxplot(x=train_data[col],ax=axes[i])
    #axes[i].set_title(col)

# Hide unused subplots if num < 5*4
for j in range(i+1, 5*4):
    fig.delaxes(axes[j])
    
plt.tight_layout()
plt.show()
    


#Importing all the algorithms
!pip install xgboost
from sklearn.preprocessing import RobustScaler,StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB



#splitting x and y from train data

X=train_data.drop(['diagnosed_diabetes'],axis=1)
Y=train_data['diagnosed_diabetes']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=0)


#get only categorical columns

cat_cols=[col for col in X.columns if X[col].dtype.name=="category"]

#get only numerical columns

num_cols=[col for col in X.columns if X[col].dtypes==int or X[col].dtypes==float]

print(num_cols)

print(cat_cols)

#List of all algorithms

algorithms=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),XGBClassifier(),GaussianNB()]

preprocessor = ColumnTransformer(
    transformers=[
        ('robust',RobustScaler(), num_cols),
        ('ordinal', OrdinalEncoder(), ['education_level','income_level','employment_status']),
        ('one', OneHotEncoder(), ['gender','ethnicity','smoking_status']),
    ],
    remainder='passthrough'  # Keep 'other_feature' as is
)

#list model names

model_names=["LogisticRegression","DecisionTreeClassifier","RandomForestClassifier","GBClassifier","GaussianNB"]
results=[]
param_grid=[{"pca__n_components":[6,9,12]}]
for name, alg in zip(model_names,algorithms):
    model=Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("pca",PCA()),
        ("alg",alg)
    ])
    grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring="accuracy",cv=5)
    
    grid.fit(Xtrain,Ytrain)
    
      # Save results
    results.append({
        "Model": name,
        "Best PCA": grid.best_params_["pca__n_components"],
        "Best Score": grid.best_score_
    })

# Print results
for r in results:
    print(r)

  



   

