!pip install plotly


!pip install lightgbm


!pip install --upgrade pip


!pip install --upgrade lightgbm


!pip install xgboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle as pk
import warnings
warnings.simplefilter(action='ignore',category=Warning)
%matplotlib inline


import os
os.getcwd()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


train = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv").rename(columns={'Weight Capacity (kg)':'Weight'})
train.head()


train_ex = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv').rename(columns={'Weight Capacity (kg)':'Weight'})
train_ex.head()


test = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv').rename(columns={'Weight Capacity (kg)':'Weight'})
test.head()


print("training data shape is",train.shape)
print("training extraa data shape is",train_ex.shape)
print("testing data shape is",test.shape)



# Function for EDA visualizations
def plot_eda(df, target_col=None):
    """Generate exploratory data analysis plots"""
    # Missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Missing values percentages
    missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(12, 6))
    missing_percent[missing_percent > 0].plot(kind='bar')
    plt.title('Percentage of Missing Values by Feature')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.show()
    
    if target_col and target_col in df.columns:
        # Target distribution (for regression, use histogram)
        plt.figure(figsize=(10, 5))
        sns.histplot(df[target_col], bins=30, kde=True)
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap for numerical features
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        plt.figure(figsize=(12, 10))
        correlation = df[numerical_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        # Scatter plots for numerical features against target
        for col in numerical_cols:
            if col != target_col:
                plt.figure(figsize=(12, 6))
                sns.scatterplot(x=df[col], y=df[target_col])
                plt.title(f'{col} vs {target_col}')
                plt.xlabel(col)
                plt.ylabel(target_col)
                plt.tight_layout()
                plt.show()

        # Feature relationships with target for categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col and df[col].nunique() < 10]
        
        for col in categorical_cols[:3]:  # Limit to first 3 categorical features
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y=target_col, data=df)
            plt.title(f'{col} vs {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Run EDA on train data
plot_eda(train, 'price')


plot_eda(train_ex, 'price')


(train.isnull().mean())*100


(train_ex.isnull().mean())*100


(test.isnull().mean())*100


train.info()


train.describe()


train.head()


train_ex.head()


train.columns


## getting all different type of feature
num_features = [feature for feature in train.columns if  train[feature].dtype != 'O']
print("Number of numerical features : ",len(num_features))
cat_features = [feature for feature in train.columns if train[feature].dtype == 'O']
print("Number of categorical features : ",len(cat_features))
discrete_features = [feature for feature in num_features if len(train[feature].unique()) <= 25]
print("Number of discrete features : ",len(discrete_features))
continuous_features  = [feature for feature in num_features if len(train[feature].unique()) > 25]
print("Number of continuous features :  ",len(continuous_features))


## check Missing Values 
### these are the features with nan value
features_with_nan = [features for features in train.columns if train[features].isnull().sum() >= 1]
for feature in features_with_nan:
    print(feature,np.round(train[feature].isnull().mean()*100,5), '% missing values')


## statistics on numerical columns (Null cols)
train[features_with_nan].select_dtypes(exclude='object').describe()


train['Brand'].value_counts()


train['Material'].value_counts()


train['Size'].value_counts()


train['Compartments'].value_counts()


train['Laptop Compartment'].value_counts()


train['Waterproof'].value_counts()


train['Style'].value_counts()


train['Color'].value_counts()


train.columns


cat_features


num_features


### Imputing Null Values
## for categorical values
#Brand
train['Brand'] = train['Brand'].fillna(train['Brand'].mode()[0])
#Material
train['Material'] = train['Material'].fillna(train['Material'].mode()[0])
# Size 
train['Size'] = train['Size'].fillna(train['Size'].mode()[0])
#Laptop Compartment
train['Laptop Compartment'] = train['Laptop Compartment'].fillna(train['Laptop Compartment'].mode()[0])
#Waterproof
train['Waterproof'] = train['Waterproof'].fillna(train['Waterproof'].mode()[0])
#style
train['Style'] = train['Style'].fillna(train['Style'].mode()[0])
# Color
train['Color'] = train['Color'].fillna(train['Color'].mode()[0])
################################################
## for numerical feature
train['Weight'] = train['Weight'].fillna(train['Weight'].median())



train.isnull().sum()


## getting all different type of feature
num_features_ex = [feature for feature in train_ex.columns if  train_ex[feature].dtype != 'O']
print("Number of numerical features extending dataset: ",len(num_features_ex))
cat_features_ex = [feature for feature in train_ex.columns if train_ex[feature].dtype == 'O']
print("Number of categorical features extending dataset : ",len(cat_features_ex))
discrete_features_ex = [feature for feature in num_features_ex if len(train_ex[feature].unique()) <= 25]
print("Number of discrete features extending dataset : ",len(discrete_features_ex))
continuous_features_ex  = [feature for feature in num_features_ex if len(train_ex[feature].unique()) > 25]
print("Number of continuous features extending dataset :  ",len(continuous_features_ex))


## check Missing Values 
### these are the features with nan value
features_with_nan_ex = [features for features in train_ex.columns if train_ex[features].isnull().sum() >= 1]
for feature in features_with_nan_ex:
    print(feature,np.round(train_ex[feature].isnull().mean()*100,5), '% missing values in training extending dataset')


## statistics on numerical columns (Null cols)
train_ex[features_with_nan].select_dtypes(exclude='object').describe()


train_ex['Brand'].value_counts()


train_ex['Material'].value_counts()


train_ex['Size'].value_counts()


train_ex['Compartments'].value_counts()


train_ex['Laptop Compartment'].value_counts()


train_ex['Waterproof'].value_counts()


train_ex['Style'].value_counts()


train_ex['Color'].value_counts()


train_ex.columns


num_features_ex


### Imputing Null Values
## for categorical values
#Brand
train_ex['Brand'] = train_ex['Brand'].fillna(train_ex['Brand'].mode()[0])
#Material
train_ex['Material'] = train_ex['Material'].fillna(train_ex['Material'].mode()[0])
# Size 
train_ex['Size'] = train_ex['Size'].fillna(train_ex['Size'].mode()[0])
#Laptop Compartment
train_ex['Laptop Compartment'] = train_ex['Laptop Compartment'].fillna(train_ex['Laptop Compartment'].mode()[0])
#Waterproof
train_ex['Waterproof'] = train_ex['Waterproof'].fillna(train_ex['Waterproof'].mode()[0])
#style
train_ex['Style'] = train_ex['Style'].fillna(train_ex['Style'].mode()[0])
# Color
train_ex['Color'] = train_ex['Color'].fillna(train_ex['Color'].mode()[0])
################################################
## for numerical feature
train_ex['Weight'] = train_ex['Weight'].fillna(train_ex['Weight'].median())



train_ex.isnull().sum()


## getting all different type of feature
num_features_test = [feature for feature in test.columns if  test[feature].dtype != 'O']
print("Number of numerical features : ",len(num_features_test))
cat_features_test = [feature for feature in test.columns if test[feature].dtype == 'O']
print("Number of categorical features : ",len(cat_features_test))
discrete_features_test = [feature for feature in num_features_test if len(test[feature].unique()) <= 25]
print("Number of discrete features : ",len(discrete_features_test))
continuous_features_test  = [feature for feature in num_features_test if len(test[feature].unique()) > 25]
print("Number of continuous features :  ",len(continuous_features_test))


test.isnull().sum()


test.describe()


(test.isnull().mean())*100


### Imputing Null Values
## for categorical values
#Brand
test['Brand'] = test['Brand'].fillna(test['Brand'].mode()[0])
#Material
test['Material'] = test['Material'].fillna(test['Material'].mode()[0])
# Size 
test['Size'] = test['Size'].fillna(test['Size'].mode()[0])
#Laptop Compartment
test['Laptop Compartment'] = test['Laptop Compartment'].fillna(test['Laptop Compartment'].mode()[0])
#Waterproof
test['Waterproof'] = test['Waterproof'].fillna(test['Waterproof'].mode()[0])
#style
test['Style'] = test['Style'].fillna(test['Style'].mode()[0])
# Color
test['Color'] = test['Color'].fillna(test['Color'].mode()[0])
################################################
## for numerical feature
test['Weight'] = test['Weight'].fillna(test['Weight'].median())


(test.isnull().mean())*100


test.head()


## Independent features and  dependent features
x = train.drop(['Price','id'],axis=1)
y = train['Price']
x1 = train_ex.drop(['Price','id'],axis=1)
y1 = train_ex['Price']


## Creating Column Transformer with 3 types of transformer
num_features = x.select_dtypes(exclude='object').columns
onehot_columns  = ['Color','Brand','Material','Size','Laptop Compartment','Waterproof','Style']

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features),
        
        
    ],remainder='passthrough'
    
)


x = preprocessor.fit_transform(x)


pd.DataFrame(x).head()


x1 = preprocessor.transform(x1)


## seperate dataset into train and test dataset 
x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size=0.25,random_state=42)


## seperate dataset into train and test dataset for  train_ex
x1_train,x1_cv,y1_train,y1_cv = train_test_split(x1,y1,test_size=0.25,random_state=42)


x_train.shape


x_cv.shape


x1_train.shape


x1_cv.shape


## Create a function for Evaluation 
def evaluate_model(true,pred):
    mae = mean_absolute_error(true,pred)
    mse = mean_squared_error(true,pred)
    rmse = np.sqrt(mse)
    score = r2_score(true,pred)
    return mae , mse ,rmse ,score


## Beginning  Model training
models = {
    "Random Forest regresssor": RandomForestRegressor(n_jobs= -1),
    "AdaBoost Regressor" : AdaBoostRegressor(),
    "GradientBoost Regressor": GradientBoostingRegressor(),
    "XGBoost Regressor": XGBRegressor(),
    "LightGBM Regression": LGBMRegressor(force_col_wise=True),
}


train_scores = {}
test_scores = {}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train,y_train) # Train model first dataset
    # model.fit(x1_train,y1_train) # Train model second dataset

    # Make Prediction
    y_train_pred = model.predict(x_train)
    y_cv_pred = model.predict(x_cv)   # Evaluate Train and Test dataset 
    model_train_mae , model_train_mse ,model_train_rmse ,model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_cv_mae , model_cv_mse ,model_cv_rmse ,model_cv_r2 =  evaluate_model(y_cv,y_cv_pred)

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Mean Squared Error: {:.4f}".format(model_train_mse))
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    train_scores[list(models.keys())[i]] =  model_train_r2

    print('-'*35)

    print('Model performance for Test set')
    print("- Mean Squared Error: {:.4f}".format(model_cv_mse))
    print("- Root Mean Squared Error: {:.4f}".format(model_cv_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_cv_mae))
    print("- R2 Score: {:.4f}".format(model_cv_r2))
    test_scores[list(models.keys())[i]] = model_cv_r2
    
    print('='*35)
    print('\n')


#Initialize few parameter for Hyperparamter tuning

rf_params = {"max_depth": [ele for ele in range(6,10,2)],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [ele for ele in range(2,10,2)],
             "n_estimators": [ele for ele in range(200,500,50)]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [ele for ele in range(6,10,2)],
                  "n_estimators": [ele for ele in range(200,500,50)],
                  "colsample_bytree": [round(i, 1) for i in np.arange(0.1, 0.6, 0.1)]}

ada_params = {
    "n_estimators": [50,60,70,80],
    'loss':['linear', 'square', 'exponential']
}

gradient_params={"loss": ['squared_error','huber','absolute_error'],
             "criterion": ['friedman_mse','squared_error'],
             "min_samples_split": [2, 8, 15, 12],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8,  None, 10],
            }
light_gbm_params = {
    "n_estimators": [ele for ele in range(200,500,50)],
    "colsample_bytree":[round(i, 1) for i in np.arange(0.1, 0.6, 0.1)],
    
}



## Models list for HyperParameter Tuning 
randomcv_models = [
    ("RF", RandomForestRegressor(),rf_params),
    ("XG",XGBRegressor(),xgboost_params),
    ("ADA",AdaBoostRegressor(),ada_params),
    ("GRA",GradientBoostingRegressor(),gradient_params),
    ("LIGHT",LGBMRegressor(force_col_wise=True),light_gbm_params)
]


randomcv_models


model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(x_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


model_param 


{'RF': {'n_estimators': 500,
  'min_samples_split': 2,
  'max_features': 8,
  'max_depth': 5},
 'XG': {'n_estimators': 300,
  'max_depth': 5,
  'learning_rate': 0.01,
  'colsample_bytree': 0.4},
 'ADA': {'n_estimators': 60, 'loss': 'linear'}}


## Retraining the models with best parameters
### creating variables for parameter of the models 

#1. for RANDOM FOREST
estimator = model_param['RF']['n_estimators']
min_sample_split = model_param['RF']['min_samples_split']
max_feature = model_param['RF']['max_features']
max_depths = model_param['RF']['max_depth']
#2. for XGBOOST
estimate = model_param['XGB']['n_estimators']
learning_rate_xgb = model_param['XGB']['learning_rate']
max_deep = model_param['XGB']['max_depth']
colsample_bytree_xgb = model_param['XGB']['colsample_bytree']
#3 for lightgbm
estimate_light = model_param['lightGBM']['n_estimators']
learning_rate_reg = model_param['lightGBM']['learning_rate ']
max_depth_light = model_param['lightGBM']['max_depth']
colsample_light = model_param['lightGBM']['colsample_bytree']


## creating training score and testing score dictionaries
train_best_score ,test_best_score = {} , {}
models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=estimator,min_samples_split=min_sample_split,
                                                     max_features=max_feature,max_depth=max_depths, 
                                                     n_jobs=-1),
     "XGBBoost Regressor": XGBRegressor(n_estimators=estimate,learning_rate=learning_rate_xgb,
                                                     colsample_bytree = colsample_bytree_xgb,
                                       max_depth=max_deep,n_jobs=-1),
    "LightGBM Regressor":LGBMRegressor(n_estimators=estimate_light,learning_rate=learning_rate_reg,max_depth=max_depth_light,
                                       colsample_bytree=colsample_light,n_jobs = -1)
                                                          
                                                          
    
}
accuracy_score_train,accuracy_score_test, = {},{}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

   # Evaluate Train and Test dataset 
    model_train_mae , model_train_mse ,model_train_rmse ,model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_mse ,model_test_rmse ,model_test_r2 =  evaluate_model(y_test,y_test_pred)

    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Mean Squared Error: {:.4f}".format(model_train_mse))
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    train_best_score[model] = model_train_rmse
    accuracy_score_train[model] = model_train_r2
    

    print('-'*35)

    print('Model performance for Test set')
    print("- Mean Squared Error: {:.4f}".format(model_test_mse))
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    test_best_score[model] = model_test_rmse
    accuracy_score_test[model] = model_test_r2
    
    print('='*35)
    print('\n')

